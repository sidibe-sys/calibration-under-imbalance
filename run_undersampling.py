"""
Experience de sous-echantillonnage.

    python run_undersampling.py --seeds 3 --epochs 30

Pourquoi cette experience existe.

Dans l'experience principale (run_tabular.py), le decoupage est stratifie: les
proportions de classes sont les memes a l'entrainement et au test. La correction
de decalage de priors n'a donc rien a corriger, et elle se comporte comme
l'identite. C'est une limite du protocole, pas un resultat.

Or le sous-echantillonnage est une pratique extremement courante sur donnees
desequilibrees, et Dal Pozzolo, Caelen, Johnson et Bontempi (2015) ont montre
qu'il biaise systematiquement les probabilites a posteriori. Le biais est
analytique, il va toujours dans le sens d'une surestimation du risque, et il ne
disparait pas avec plus de donnees.

Cette experience le reproduit, et teste deux choses:

    1. le sous-echantillonnage detruit-il la calibration, meme si l'exactitude
       ou l'AUC ne bougent pas ?
    2. la correction analytique la restaure-t-elle, et se compose-t-elle avec
       une perte de calibration entrainee de bout en bout, ou les deux
       se contrarient-elles ?

La deuxieme question, a ma connaissance, n'a pas de reponse publiee.
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
from train import train_model, make_tabular_loaders, collect_logits, get_device
from posthoc import softmax_np, prior_shift_correction, TemperatureScaling
from metrics import full_report

LOSSES = ["ce", "focal", "ce+mdca", "ce+softece"]


def undersample(X, y, keep_fraction, seed=0):
    """
    Sous-echantillonne les classes majoritaires (0 et 1) en n'en gardant qu'une
    fraction, et laisse intacte la classe rare (2, readmission sous 30 jours).

    C'est exactement la manoeuvre que fait un praticien confronte a un
    desequilibre: il jette de la classe frequente pour "rebalancer". Le point du
    papier de 2015 est que cette manoeuvre, apparemment anodine, deplace la
    distribution a posteriori.
    """
    rng = np.random.RandomState(seed)
    idx_keep = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if cls == 2:                      # classe rare: on garde tout
            idx_keep.append(idx)
        else:
            n = max(1, int(len(idx) * keep_fraction))
            idx_keep.append(rng.choice(idx, size=n, replace=False))
    idx_keep = np.concatenate(idx_keep)
    rng.shuffle(idx_keep)
    return X[idx_keep], y[idx_keep]


def run(seeds=3, epochs=30, keep_fraction=0.15, batch_size=256):
    device = get_device()
    rows = []

    for seed in range(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        tr, va, te, meta = load_diabetes(seed=seed)
        Xtr, ytr = tr

        # La verite terrain: les proportions reelles de la population,
        # mesurees sur le jeu d'entrainement complet AVANT toute manipulation.
        prior_true = np.bincount(ytr, minlength=3) / len(ytr)

        # On sous-echantillonne l'entrainement ET la validation.
        # La validation doit refleter ce que le modele a vu, sinon le
        # temperature scaling corrigerait deja le biais de prior par accident,
        # et on ne mesurerait plus rien.
        Xtr_u, ytr_u = undersample(Xtr, ytr, keep_fraction, seed)
        Xva_u, yva_u = undersample(va[0], va[1], keep_fraction, seed)
        prior_train = np.bincount(ytr_u, minlength=3) / len(ytr_u)

        # Le test, lui, reste intact. C'est la population reelle.
        print(f"--- graine {seed}")
        print(f"    priors reels        : {np.round(prior_true, 3)}")
        print(f"    priors apres sous-ech: {np.round(prior_train, 3)}")

        train_loader, val_loader, test_loader = make_tabular_loaders(
            Xtr_u, ytr_u, Xva_u, yva_u, te[0], te[1], batch_size)

        for loss_name in LOSSES:
            model = MLP(meta["n_features"], 3)
            model = train_model(model, build_loss(loss_name), train_loader,
                                epochs=epochs, device=device, verbose=False)

            test_logits, test_labels = collect_logits(model, test_loader, device)
            val_logits, val_labels = collect_logits(model, val_loader, device)
            probs = softmax_np(test_logits)

            # a) brut, sur donnees sous-echantillonnees
            rows.append({"seed": seed, "loss": loss_name, "correction": "brut",
                         **full_report(probs, test_labels)})

            # b) temperature scaling seul
            T = TemperatureScaling().fit(val_logits, val_labels)
            rows.append({"seed": seed, "loss": loss_name, "correction": "temperature",
                         "T": T, **full_report(softmax_np(test_logits / T), test_labels)})

            # c) correction de prior (Dal Pozzolo et Caelen, generalisee)
            p_corr = prior_shift_correction(probs, prior_train, prior_true)
            rows.append({"seed": seed, "loss": loss_name, "correction": "correction prior",
                         **full_report(p_corr, test_labels)})

            # d) les deux, dans l'ordre: prior puis temperature
            #    (la question de la composition)
            p_both = prior_shift_correction(softmax_np(test_logits / T),
                                            prior_train, prior_true)
            rows.append({"seed": seed, "loss": loss_name, "correction": "prior + temperature",
                         **full_report(p_both, test_labels)})

            print(f"    {loss_name:12s} termine")

    df = pd.DataFrame(rows)
    agg = df.groupby(["loss", "correction"]).agg(["mean", "std"]).drop(columns="seed")

    out = pd.DataFrame(index=agg.index)
    for col in ["accuracy", "gap", "ECE", "SCE", "Brier"]:
        out[col] = [f"{m:.4f} ({s:.4f})" for m, s in
                    zip(agg[(col, "mean")], agg[(col, "std")])]

    print("\n" + "=" * 95)
    print(f"SOUS-ECHANTILLONNAGE  |  fraction conservee = {keep_fraction}  "
          f"|  moyenne (ecart-type) sur {seeds} graines")
    print("=" * 95)
    print(out.to_string())

    df.to_csv("results/raw_undersampling.csv", index=False)
    out.to_csv("results/summary_undersampling.csv")
    print("\nEcrit dans results/summary_undersampling.csv")
    return df, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--keep", type=float, default=0.15,
                    help="fraction des classes majoritaires conservee")
    a = ap.parse_args()
    run(seeds=a.seeds, epochs=a.epochs, keep_fraction=a.keep)
