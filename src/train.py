"""
Entrainement et evaluation.

Protocole, et il faut le tenir strictement:
    - train : le reseau apprend dessus
    - val   : sert a ajuster les methodes post-hoc, et rien d'autre
    - test  : n'est touche qu'une fois, a la toute fin

Calibrer une temperature sur le jeu de test, puis rapporter l'ECE sur ce meme
jeu, est une erreur frequente dans les depots publics. Elle donne des chiffres
flatteurs et sans valeur.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from metrics import full_report
from posthoc import TemperatureScaling, VectorScaling, prior_shift_correction, softmax_np


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, loss_fn, train_loader, val_loader=None,
                epochs=30, lr=1e-3, weight_decay=5e-4, device=None,
                scheduler="cosine", verbose=True):
    device = device or get_device()
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) \
        if scheduler == "cosine" else None

    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
        if sched:
            sched.step()

        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            msg = f"  epoque {ep+1:3d}/{epochs}  perte={total/n:.4f}"
            if val_loader is not None:
                lg, lb = collect_logits(model, val_loader, device)
                acc = (lg.argmax(1) == lb).mean()
                msg += f"  exactitude val={acc:.4f}"
            print(msg)

    return model


@torch.no_grad()
def collect_logits(model, loader, device=None):
    """Recupere les logits bruts (avant softmax) et les etiquettes."""
    device = device or get_device()
    model.eval()
    L, Y = [], []
    for xb, yb in loader:
        L.append(model(xb.to(device)).cpu().numpy())
        Y.append(yb.numpy())
    return np.concatenate(L), np.concatenate(Y)


def evaluate_all(model, val_loader, test_loader, meta=None,
                 device=None, n_bins=15, loss_name=""):
    """
    Evalue un modele entraine sous quatre regimes:
        brut               : softmax des logits, sans correction
        + temperature      : temperature scaling ajuste sur val
        + vector scaling   : vector scaling ajuste sur val
        + correction prior : correction du decalage de priors (Saerens et al.),
                             generalisation multi-classes de la correction de
                             Dal Pozzolo et Caelen

    Retourne un dict de dicts, prets a mettre dans un DataFrame.
    """
    device = device or get_device()
    val_logits, val_labels = collect_logits(model, val_loader, device)
    test_logits, test_labels = collect_logits(model, test_loader, device)

    out = {}
    k = test_logits.shape[1]

    # 1. brut
    out[f"{loss_name} | brut"] = full_report(softmax_np(test_logits), test_labels, n_bins)

    # 2. temperature scaling
    # La temperature apprise est stockee comme une colonne, pas dans le nom de
    # la configuration: sinon deux graines produisent deux noms differents et
    # le regroupement echoue silencieusement.
    ts = TemperatureScaling()
    T = ts.fit(val_logits, val_labels)
    rep = full_report(softmax_np(test_logits / T), test_labels, n_bins)
    rep["T"] = T
    out[f"{loss_name} | + temperature"] = rep

    # 3. vector scaling
    vs = VectorScaling(k)
    vs.fit(val_logits, val_labels)
    with torch.no_grad():
        z = vs(torch.as_tensor(test_logits, dtype=torch.float32)).numpy()
    out[f"{loss_name} | + vector scaling"] = full_report(softmax_np(z), test_labels, n_bins)

    # 4. correction du decalage de priors
    if meta is not None and "train_prior" in meta:
        prior_train = np.asarray(meta["train_prior"], dtype=float)
        prior_true = np.asarray(meta.get("test_prior",
                                np.bincount(test_labels, minlength=k) / len(test_labels)),
                                dtype=float)
        p_corr = prior_shift_correction(softmax_np(test_logits), prior_train, prior_true)
        out[f"{loss_name} | + correction prior"] = full_report(p_corr, test_labels, n_bins)

    return out


def make_tabular_loaders(Xtr, ytr, Xval, yval, Xte, yte, batch_size=256):
    def mk(X, y, shuffle):
        ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                           torch.as_tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)
    return mk(Xtr, ytr, True), mk(Xval, yval, False), mk(Xte, yte, False)
