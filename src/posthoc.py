"""
Methodes de calibration a posteriori, servant de point de comparaison
aux pertes entrainees de bout en bout.

References:
    Guo et al. (2017), On Calibration of Modern Neural Networks, ICML.
    Dal Pozzolo, Caelen, Johnson, Bontempi (2015), Calibrating Probability with
        Undersampling for Unbalanced Classification, IEEE SSCI.
    Saerens, Latinne, Decaestecker (2002), Adjusting the Outputs of a Classifier
        to New a Priori Probabilities, Neural Computation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #

class TemperatureScaling(nn.Module):
    """
    Un seul parametre T, appris sur un jeu de validation en minimisant la NLL.
    Les logits sont divises par T avant le softmax.

        p = softmax(z / T)

    T > 1 aplatit la distribution et reduit la surconfiance. La methode
    preserve exactement l'ordre des classes, donc l'exactitude ne bouge pas.
    C'est sa force et sa limite: elle ne peut pas corriger une calibration qui
    varie d'une classe a l'autre, puisqu'elle applique le meme facteur partout.
    """

    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))

    @property
    def T(self):
        return self.log_T.exp()

    def forward(self, logits):
        return logits / self.T

    def fit(self, val_logits, val_labels, max_iter=200):
        val_logits = torch.as_tensor(val_logits, dtype=torch.float32)
        val_labels = torch.as_tensor(val_labels, dtype=torch.long)
        opt = torch.optim.LBFGS([self.log_T], lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        opt.step(closure)
        return float(self.T.item())


# --------------------------------------------------------------------------- #
# Vector scaling
# --------------------------------------------------------------------------- #

class VectorScaling(nn.Module):
    """
    Generalisation: un facteur et un biais par classe.

        p = softmax(w * z + b)   avec w, b de dimension K

    Plus expressif que le temperature scaling, donc capable de corriger une
    surconfiance qui differe selon les classes. C'est le comparateur pertinent
    en regime desequilibre. Il peut modifier l'exactitude, contrairement au
    temperature scaling.
    """

    def __init__(self, n_classes):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_classes))
        self.b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, logits):
        return logits * self.w + self.b

    def fit(self, val_logits, val_labels, epochs=500, lr=0.01):
        val_logits = torch.as_tensor(val_logits, dtype=torch.float32)
        val_labels = torch.as_tensor(val_labels, dtype=torch.long)
        opt = torch.optim.Adam([self.w, self.b], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(val_logits), val_labels)
            loss.backward()
            opt.step()
        return self


# --------------------------------------------------------------------------- #
# Correction du biais de sous-echantillonnage
# --------------------------------------------------------------------------- #

def undersampling_correction_binary(p_s, beta):
    """
    Correction analytique de Dal Pozzolo, Caelen, Johnson et Bontempi (2015),
    cas binaire.

    Quand on sous-echantillonne la classe majoritaire en n'en gardant qu'une
    fraction beta, le classifieur apprend la probabilite a posteriori sur la
    distribution echantillonnee, pas sur la vraie. Le lien est:

        p_s = p / (p + beta * (1 - p))

    d'ou, en inversant:

        p = beta * p_s / (beta * p_s - p_s + 1)

    Le point important, et c'est celui du papier: le biais n'est pas un
    detail d'implementation. Il est systematique, il va toujours dans le sens
    d'une surestimation du risque, et il ne disparait pas avec plus de donnees.

    p_s : probabilites predites de la classe minoritaire, sur donnees sous-echantillonnees
    beta: fraction de la classe majoritaire conservee, dans (0, 1]
    """
    p_s = np.asarray(p_s, dtype=float)
    return beta * p_s / (beta * p_s - p_s + 1.0)


def prior_shift_correction(probs_s, prior_train, prior_true):
    """
    Generalisation multi-classes du meme principe (Saerens et al., 2002).

    Si la distribution des classes a l'entrainement (prior_train) differe de
    celle de la population (prior_true), on corrige par le rapport des priors:

        p_k  proportionnel a  p_s_k * (pi_k / pi_s_k)

    puis on renormalise.

    C'est exactement le probleme du papier de 2015, ecrit pour K classes.
    Question ouverte, et l'un des interets de ce depot: cette correction
    a posteriori se compose-t-elle bien avec une perte de calibration
    entrainee de bout en bout, ou les deux se contrarient-elles ?
    """
    probs_s = np.asarray(probs_s, dtype=float)
    ratio = np.asarray(prior_true, dtype=float) / np.asarray(prior_train, dtype=float)
    out = probs_s * ratio[None, :]
    return out / out.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #

def softmax_np(logits):
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
