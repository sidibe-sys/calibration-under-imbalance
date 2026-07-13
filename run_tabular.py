"""
Experience principale, terrain tabulaire.

    python run_tabular.py --dataset diabetes --seeds 3 --epochs 30

Compare, sur le meme reseau et le meme protocole:
    pertes           : CE, Focal, Label Smoothing, CE+MDCA, CE+SoftECE
    corrections      : aucune, temperature scaling, vector scaling, correction prior

et rapporte, pour chaque combinaison:
    exactitude, ECE, ECE adaptatif, SCE (classwise), Brier, NLL, ecart de confiance

La question a laquelle le tableau doit repondre:
    une perte de calibration entrainee de bout en bout rend-elle la correction
    post-hoc inutile ? Et tient-elle quand les classes sont desequilibrees ?
"""

import argparse
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")

from data import load_diabetes
from models import MLP
from losses import build_loss
from train import train_model, evaluate_all, make_tabular_loaders, get_device


LOSSES = ["ce", "focal", "ls", "ce+mdca", "ce+softece"]


def make_synthetic_like_diabetes(n=40000, n_features=40, seed=0):
    """Jeu synthetique reproduisant la structure du jeu UCI: 3 classes,
    proportions 54 / 35 / 11, signal faible et bruite, comme en readmission
    hospitaliere ou l'exactitude plafonne autour de 0.60.

    Sert uniquement de test de bout en bout quand UCI est inaccessible.
    Les resultats scientifiques doivent etre produits sur le vrai jeu.
    """
    rng = np.random.RandomState(seed)
    priors = np.array([0.539, 0.349, 0.112])
    y = rng.choice(3, size=n, p=priors)
    centers = rng.randn(3, n_features) * 0.55
    X = centers[y] + rng.randn(n, n_features) * 1.6

    n_tr, n_val = int(0.65 * n), int(0.15 * n)
    sl = lambda a, b: slice(a, b)
    return ((X[sl(0, n_tr)].astype(np.float32), y[sl(0, n_tr)]),
            (X[sl(n_tr, n_tr + n_val)].astype(np.float32), y[sl(n_tr, n_tr + n_val)]),
            (X[sl(n_tr + n_val, n)].astype(np.float32), y[sl(n_tr + n_val, n)]),
            {"n_classes": 3, "n_features": n_features,
             "train_prior": np.bincount(y[:n_tr], minlength=3) / n_tr})


def run(dataset="diabetes", seeds=3, epochs=30, batch_size=256, lr=1e-3, beta=1.0):
    device = get_device()
    print(f"Peripherique: {device}\n")

    rows = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if dataset == "diabetes":
            tr, va, te, meta = load_diabetes(seed=seed)
        elif dataset == "synthetic":
            tr, va, te, meta = make_synthetic_like_diabetes(seed=seed)
        else:
            raise ValueError(dataset)

        train_loader, val_loader, test_loader = make_tabular_loaders(
            tr[0], tr[1], va[0], va[1], te[0], te[1], batch_size)

        print(f"--- graine {seed} | {tr[0].shape[0]} train, {va[0].shape[0]} val, "
              f"{te[0].shape[0]} test | priors {np.round(meta['train_prior'], 3)}")

        for loss_name in LOSSES:
            model = MLP(meta["n_features"], meta["n_classes"])
            loss_fn = build_loss(loss_name, beta=beta)
            model = train_model(model, loss_fn, train_loader, val_loader,
                                epochs=epochs, lr=lr, device=device, verbose=False)

            res = evaluate_all(model, val_loader, test_loader, meta=meta,
                               device=device, loss_name=loss_name)
            for key, metrics in res.items():
                rows.append({"seed": seed, "config": key, **metrics})
            print(f"    {loss_name:12s} termine")

    df = pd.DataFrame(rows)
    agg = df.groupby("config").agg(["mean", "std"]).drop(columns="seed")

    # tableau lisible: moyenne (ecart-type) sur les graines
    out = pd.DataFrame(index=agg.index)
    for col in ["accuracy", "gap", "ECE", "AdaECE", "SCE", "Brier", "NLL"]:
        if (col, "mean") not in agg.columns:
            continue
        out[col] = [f"{m:.4f} ({s:.4f})" for m, s in
                    zip(agg[(col, "mean")], agg[(col, "std")])]

    print("\n" + "=" * 100)
    print(f"RESULTATS  |  {dataset}  |  moyenne (ecart-type) sur {seeds} graines")
    print("=" * 100)
    print(out.to_string())

    df.to_csv(f"results/raw_{dataset}.csv", index=False)
    out.to_csv(f"results/summary_{dataset}.csv")
    print(f"\nEcrit dans results/summary_{dataset}.csv")
    return df, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="diabetes", choices=["diabetes", "synthetic"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--beta", type=float, default=1.0)
    a = ap.parse_args()
    run(dataset=a.dataset, seeds=a.seeds, epochs=a.epochs, beta=a.beta)
