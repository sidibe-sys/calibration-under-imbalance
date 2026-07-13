"""
Experience CIFAR-100 long-tailed. Necessite un GPU (une session Colab gratuite suffit).

    python run_cifar.py --imbalance 100 --seeds 3 --epochs 200

C'est ici que la question du depot se joue vraiment: avec 100 classes et un
facteur de desequilibre de 100, la classe la plus rare compte 5 images. La
contrainte MDCA, qui est une moyenne par lot, devient tres bruitee sur ces
classes. La prediction H2 du README est directement testable ici.
"""

import argparse, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from data import load_cifar100_lt
from models import ResNet32
from losses import build_loss
from train import train_model, evaluate_all, get_device
from plots import sce_vs_class_frequency
from posthoc import softmax_np
from train import collect_logits

LOSSES = ["ce", "focal", "ls", "ce+mdca", "ce+softece"]


def run(imbalance=100, seeds=3, epochs=200, batch_size=128, lr=0.1, beta=1.0):
    device = get_device()
    if device.type == "cpu":
        print("ATTENTION: pas de GPU detecte. L'entrainement sera tres long.")

    rows = []
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        tr, va, te, meta = load_cifar100_lt(imbalance_factor=imbalance, seed=seed)
        trl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        val = DataLoader(va, batch_size=256, num_workers=2)
        tel = DataLoader(te, batch_size=256, num_workers=2)

        print(f"--- graine {seed} | {len(tr)} train | classe la plus rare: "
              f"{meta['train_counts'].min()} images")

        for name in LOSSES:
            model = ResNet32(100)
            fn = build_loss(name, beta=beta)
            model = train_model(model, fn, trl, val, epochs=epochs, lr=lr,
                                device=device, verbose=True)
            res = evaluate_all(model, val, tel, meta=meta, device=device, loss_name=name)
            for k, v in res.items():
                rows.append({"seed": seed, "config": k, **v})

            if seed == 0:
                lg, lb = collect_logits(model, tel, device)
                sce_vs_class_frequency(softmax_np(lg), lb, meta["train_counts"],
                                       path=f"figures/sce_par_classe_{name}.png")

    df = pd.DataFrame(rows)
    df.to_csv("results/raw_cifar100lt.csv", index=False)
    agg = df.groupby("config").mean(numeric_only=True).drop(columns="seed")
    agg.to_csv("results/summary_cifar100lt.csv")
    print(agg.to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imbalance", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--beta", type=float, default=1.0)
    a = ap.parse_args()
    run(imbalance=a.imbalance, seeds=a.seeds, epochs=a.epochs, beta=a.beta)
