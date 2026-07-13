"""
Metriques de calibration pour la classification multi-classes.

Toutes les fonctions prennent:
    probs : np.ndarray de forme (N, K), lignes sommant a 1 (sorties softmax)
    labels: np.ndarray de forme (N,), entiers dans [0, K)

Reference des definitions:
    Guo et al. (2017), On Calibration of Modern Neural Networks, ICML.
    Nixon et al. (2019), Measuring Calibration in Deep Learning, CVPR Workshops.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Calibration de la classe predite (top-label)
# --------------------------------------------------------------------------- #

def expected_calibration_error(probs, labels, n_bins=15):
    """
    ECE a intervalles de largeur egale, sur la classe predite.

        ECE = sum_b (n_b / N) * |acc(b) - conf(b)|

    C'est la metrique standard. Elle a deux defauts que ce projet documente:
    elle ne regarde que la classe predite, et elle depend du decoupage en
    intervalles (donc a gradient nul presque partout).
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # intervalle ouvert a gauche, ferme a droite, sauf le premier
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())

    return float(ece)


def adaptive_calibration_error(probs, labels, n_bins=15):
    """
    ECE a intervalles equipeuples (meme nombre d'observations par intervalle).

    Corrige un biais connu de l'ECE a largeur egale: sur un reseau surconfiant,
    la quasi-totalite des observations tombe dans les deux derniers intervalles,
    et les autres sont vides. Le decoupage equipeuple redonne du poids aux
    zones de confiance intermediaire.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]
    splits = np.array_split(np.arange(len(conf)), n_bins)

    n = len(conf)
    ace = 0.0
    for idx in splits:
        if len(idx) == 0:
            continue
        ace += (len(idx) / n) * abs(correct[idx].mean() - conf[idx].mean())
    return float(ace)


def maximum_calibration_error(probs, labels, n_bins=15):
    """Ecart maximal entre confiance et exactitude sur un intervalle non vide."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    gaps = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        gaps.append(abs(correct[mask].mean() - conf[mask].mean()))
    return float(max(gaps)) if gaps else 0.0


# --------------------------------------------------------------------------- #
# Calibration du vecteur complet (multi-classes)
# --------------------------------------------------------------------------- #

def static_calibration_error(probs, labels, n_bins=15):
    """
    Static Calibration Error (SCE), aussi appelee classwise ECE.
    Nixon et al. (2019).

        SCE = (1/K) * sum_k sum_b (n_bk / N) * |acc_k(b) - conf_k(b)|

    Pour chaque classe k, on traite le probleme un-contre-tous: le score
    predit pour k contre l'indicatrice "l'observation est de classe k".

    C'est LA metrique qui compte pour la these. Un modele peut avoir une ECE
    excellente et une SCE mauvaise: il sait quand il a raison sur son premier
    choix, mais les probabilites qu'il attribue aux autres classes sont fausses.
    Toute decision qui utilise le vecteur complet (cout asymetrique, rejet,
    hierarchie de risque) depend de la SCE, pas de l'ECE.
    """
    n, k = probs.shape
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0

    for cls in range(k):
        p_k = probs[:, cls]
        y_k = (labels == cls).astype(float)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (p_k > lo) & (p_k <= hi) if i > 0 else (p_k >= lo) & (p_k <= hi)
            if mask.sum() == 0:
                continue
            total += (mask.sum() / n) * abs(y_k[mask].mean() - p_k[mask].mean())

    return float(total / k)


def classwise_ece_per_class(probs, labels, n_bins=15):
    """SCE detaillee, classe par classe. Utile pour montrer que l'erreur de
    calibration se concentre sur les classes rares."""
    n, k = probs.shape
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = np.zeros(k)

    for cls in range(k):
        p_k = probs[:, cls]
        y_k = (labels == cls).astype(float)
        acc = 0.0
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (p_k > lo) & (p_k <= hi) if i > 0 else (p_k >= lo) & (p_k <= hi)
            if mask.sum() == 0:
                continue
            acc += (mask.sum() / n) * abs(y_k[mask].mean() - p_k[mask].mean())
        out[cls] = acc
    return out


# --------------------------------------------------------------------------- #
# Scores propres
# --------------------------------------------------------------------------- #

def brier_score(probs, labels):
    """
    Score de Brier multi-classes: moyenne de ||p - onehot||^2.
    Score propre: minimise en esperance par la vraie distribution conditionnelle.
    Se decompose (Murphy) en calibration, raffinement et incertitude.
    """
    n, k = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def negative_log_likelihood(probs, labels, eps=1e-12):
    """Log-loss. Score propre lui aussi, mais tres sensible aux probabilites
    proches de zero attribuees a la vraie classe."""
    n = len(labels)
    p = np.clip(probs[np.arange(n), labels], eps, 1.0)
    return float(-np.mean(np.log(p)))


def accuracy(probs, labels):
    return float((probs.argmax(axis=1) == labels).mean())


def mean_confidence(probs):
    return float(probs.max(axis=1).mean())


def overconfidence_gap(probs, labels):
    """Confiance moyenne moins exactitude. Positif = surconfiance.
    Signe le probleme decrit dans l'appel de these."""
    return mean_confidence(probs) - accuracy(probs, labels)


# --------------------------------------------------------------------------- #

def full_report(probs, labels, n_bins=15):
    """Tableau complet, une ligne par modele."""
    return {
        "accuracy":   accuracy(probs, labels),
        "confidence": mean_confidence(probs),
        "gap":        overconfidence_gap(probs, labels),
        "ECE":        expected_calibration_error(probs, labels, n_bins),
        "AdaECE":     adaptive_calibration_error(probs, labels, n_bins),
        "MCE":        maximum_calibration_error(probs, labels, n_bins),
        "SCE":        static_calibration_error(probs, labels, n_bins),
        "Brier":      brier_score(probs, labels),
        "NLL":        negative_log_likelihood(probs, labels),
    }


def reliability_curve(probs, labels, n_bins=15):
    """Retourne (conf_moyenne, exactitude, effectif) par intervalle,
    pour tracer le diagramme de fiabilite."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            xs.append((lo + hi) / 2); ys.append(np.nan); ns.append(0)
        else:
            xs.append(conf[mask].mean()); ys.append(correct[mask].mean()); ns.append(int(mask.sum()))
    return np.array(xs), np.array(ys), np.array(ns)
