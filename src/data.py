"""
Jeux de donnees.

Deux terrains, choisis pour des raisons opposees.

1. UCI Diabetes 130-US Hospitals (1999 a 2008).
   Tabulaire, medical, trois classes naturellement desequilibrees.
   101 766 sejours hospitaliers, 47 variables apres nettoyage.
   Cible: readmitted, a trois modalites (<30 jours, >30 jours, jamais).
   Source: UCI Machine Learning Repository, identifiant 296.
   https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
   Licence CC BY 4.0.
   Reference: Clore, Cios, DeShazo, Strack (2014). Voir aussi Strack et al.,
   BioMed Research International, 2014, doi 10.1155/2014/781670.

   Pourquoi celui-ci: c'est un probleme ou la probabilite predite est l'objet
   de la decision. Un service qui doit choisir quels patients suivre apres la
   sortie a besoin de savoir si le 12 % annonce vaut vraiment 12 %. Le
   classement seul ne suffit pas quand le budget de suivi est contraint.

2. CIFAR-100, version desequilibree artificiellement (long-tailed).
   Le terrain canonique de la litterature sur la calibration, ce qui rend les
   resultats comparables a Guo et al. (2017), Mukhoti et al. (2020),
   Hebbalaguppe et al. (2022). Cent classes, donc la distinction entre
   calibration du top-label et calibration du vecteur complet devient visible.
   Source: Krizhevsky (2009), telechargement automatique par torchvision.
   https://www.cs.toronto.edu/~kriz/cifar.html

   Le desequilibre est introduit par decroissance exponentielle des effectifs
   (protocole de Cui et al., 2019, Class-Balanced Loss), avec un facteur de
   desequilibre reglable. C'est ce qui permet de repondre a la question du
   depot: les pertes de calibration tiennent-elles quand les classes ne sont
   plus equilibrees ?
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


DIABETES_UCI_ID = 296


# --------------------------------------------------------------------------- #
# 1. Diabetes 130-US Hospitals
# --------------------------------------------------------------------------- #

def load_diabetes(seed=0, val_size=0.15, test_size=0.20):
    """
    Retourne (X_train, y_train), (X_val, y_val), (X_test, y_test), meta.

    Le jeu de validation sert exclusivement a ajuster les methodes post-hoc
    (temperature scaling, vector scaling). Il n'est jamais vu par le reseau
    pendant l'entrainement. C'est un point de protocole que beaucoup de depots
    negligent, et qui suffit a rendre leurs resultats ininterpretables: calibrer
    la temperature sur le jeu de test revient a se noter soi-meme.
    """
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=DIABETES_UCI_ID)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()

    # Cible: trois classes
    target_col = y.columns[0]
    y = y[target_col].astype(str).str.strip()
    mapping = {"NO": 0, ">30": 1, "<30": 2}
    y = y.map(mapping)
    keep = y.notna()
    X, y = X[keep], y[keep].astype(int)

    # Nettoyage. Les choix sont explicites et discutables, ce qui est le but:
    # ils doivent pouvoir etre defendus.
    drop_cols = [c for c in ["encounter_id", "patient_nbr", "weight",
                             "payer_code", "medical_specialty"] if c in X.columns]
    # weight, payer_code et medical_specialty sont manquantes a plus de 40 %.
    # Les imputer reviendrait a inventer de l'information.
    X = X.drop(columns=drop_cols)
    X = X.replace("?", np.nan)

    # Les codes diagnostiques CIM-9 (diag_1 a diag_3) comptent des centaines de
    # modalites. On les regroupe en grandes familles, comme dans Strack et al.
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in X.columns:
            X[col] = X[col].apply(_group_icd9)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X[cat_cols] = X[cat_cols].fillna("missing").astype(str)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = enc.fit_transform(X[cat_cols])
    X_all = np.hstack([X[num_cols].to_numpy(dtype=np.float32), X_cat.astype(np.float32)])
    y_all = y.to_numpy()

    # Decoupage stratifie: on preserve les proportions de classes, sinon la
    # classe rare (<30 jours, environ 11 %) peut disparaitre du jeu de test.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X_all, y_all, test_size=val_size + test_size,
        stratify=y_all, random_state=seed)
    rel = test_size / (val_size + test_size)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=rel, stratify=y_tmp, random_state=seed)

    scaler = StandardScaler().fit(X_tr)
    X_tr, X_val, X_te = scaler.transform(X_tr), scaler.transform(X_val), scaler.transform(X_te)

    meta = {
        "n_classes": 3,
        "n_features": X_tr.shape[1],
        "class_names": ["pas de readmission", "readmission > 30 j", "readmission < 30 j"],
        "train_prior": np.bincount(y_tr, minlength=3) / len(y_tr),
    }
    return (X_tr.astype(np.float32), y_tr), (X_val.astype(np.float32), y_val), \
           (X_te.astype(np.float32), y_te), meta


def _group_icd9(code):
    """Regroupement des codes CIM-9 en familles cliniques.
    Suit la logique de Strack et al. (2014). Les codes en V et E (facteurs
    externes, motifs de contact) sont regroupes a part."""
    if pd.isna(code):
        return "missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "autre"
    try:
        v = float(code)
    except ValueError:
        return "autre"
    if 390 <= v <= 459 or v == 785:
        return "circulatoire"
    if 460 <= v <= 519 or v == 786:
        return "respiratoire"
    if 520 <= v <= 579 or v == 787:
        return "digestif"
    if 250 <= v < 251:
        return "diabete"
    if 800 <= v <= 999:
        return "trauma"
    if 710 <= v <= 739:
        return "musculosquelettique"
    if 580 <= v <= 629 or v == 788:
        return "genito-urinaire"
    if 140 <= v <= 239:
        return "neoplasie"
    return "autre"


# --------------------------------------------------------------------------- #
# 2. CIFAR-100 long-tailed
# --------------------------------------------------------------------------- #

def make_longtailed_indices(labels, n_classes, imbalance_factor=100, seed=0):
    """
    Construit une distribution a queue longue par decroissance exponentielle.

    L'effectif de la classe k vaut n_max * (1 / IF)^(k / (K-1)).
    Avec IF = 100 sur CIFAR-100: 500 images pour la classe 0, 5 pour la 99.

    Protocole de Cui et al. (2019), Class-Balanced Loss Based on Effective
    Number of Samples, CVPR. C'est le standard du domaine, ce qui rend les
    chiffres comparables a la litterature.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    n_max = np.bincount(labels, minlength=n_classes).max()

    keep = []
    for k in range(n_classes):
        n_k = int(n_max * (1.0 / imbalance_factor) ** (k / (n_classes - 1)))
        n_k = max(n_k, 1)
        idx_k = np.where(labels == k)[0]
        keep.append(rng.choice(idx_k, size=min(n_k, len(idx_k)), replace=False))
    return np.concatenate(keep)


def load_cifar100_lt(root="./data", imbalance_factor=100, val_frac=0.1, seed=0):
    """
    CIFAR-100 desequilibre. Le jeu de test reste equilibre, volontairement:
    on veut mesurer la calibration sur la population reelle, pas sur la
    population biaisee vue a l'entrainement. C'est tout le probleme souleve
    par Dal Pozzolo et Caelen (2015).
    """
    import torch
    from torchvision import datasets, transforms

    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    train_full = datasets.CIFAR100(root, train=True, download=True, transform=tf_train)
    test_set = datasets.CIFAR100(root, train=False, download=True, transform=tf_test)

    idx = make_longtailed_indices(train_full.targets, 100, imbalance_factor, seed)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_set = torch.utils.data.Subset(train_full, train_idx)
    val_set = torch.utils.data.Subset(train_full, val_idx)

    counts = np.bincount(np.asarray(train_full.targets)[train_idx], minlength=100)
    meta = {
        "n_classes": 100,
        "train_counts": counts,
        "train_prior": counts / counts.sum(),
        "test_prior": np.full(100, 1 / 100),
        "imbalance_factor": imbalance_factor,
    }
    return train_set, val_set, test_set, meta
