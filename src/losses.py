"""
Pertes de calibration differentiables, pour la classification multi-classes.

Le probleme central: l'ECE repose sur un decoupage en intervalles, operation
constante par morceaux. Son gradient est nul presque partout. On ne peut donc
pas la minimiser par descente de gradient. Les pertes ci-dessous sont trois
facons de contourner cet obstacle.

References:
    Mukhoti et al. (2020), Calibrating Deep Neural Networks using Focal Loss,
        NeurIPS.
    Muller et al. (2019), When Does Label Smoothing Help?, NeurIPS.
    Hebbalaguppe et al. (2022), A Stitch in Time Saves Nine: A Train-Time
        Regularizing Loss for Improved Neural Network Calibration, CVPR.
    Karandikar et al. (2021), Soft Calibration Objectives for Neural Networks,
        NeurIPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Reference: entropie croisee
# --------------------------------------------------------------------------- #

class CrossEntropy(nn.Module):
    """La ligne de base. Score propre, mais rien dans son gradient ne pousse
    le reseau vers la calibration: elle recompense la confiance sur les
    observations correctes sans jamais la penaliser sur les autres."""
    name = "CE"

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets)


# --------------------------------------------------------------------------- #
# 2. Perte focale
# --------------------------------------------------------------------------- #

class FocalLoss(nn.Module):
    """
        FL = - (1 - p_t)^gamma * log(p_t)

    Le facteur (1 - p_t)^gamma reduit le poids des observations deja bien
    classees. Mukhoti et al. montrent que cela agit comme une regularisation
    implicite de l'entropie: le reseau est moins pousse vers des sorties
    quasi certaines, donc moins surconfiant.

    Ce n'est pas une perte de calibration au sens strict. Elle ne mesure aucun
    ecart entre confiance et frequence. Elle ameliore la calibration par
    effet de bord. C'est justement ce que le sujet de these veut depasser.
    """
    name = "Focal"

    def __init__(self, gamma=3.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        return -((1.0 - p_t) ** self.gamma * logp_t).mean()


# --------------------------------------------------------------------------- #
# 3. Lissage des etiquettes
# --------------------------------------------------------------------------- #

class LabelSmoothing(nn.Module):
    """
    Remplace la cible one-hot par (1 - a) * onehot + a / K.

    Empeche mecaniquement le logit de la vraie classe de partir a l'infini.
    Ameliore l'ECE, mais Muller et al. montrent que cela ecrase aussi
    l'information portee par les classes non predites: la SCE peut se degrader
    pendant que l'ECE s'ameliore. C'est un des points a verifier ici.
    """
    name = "LS"

    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, label_smoothing=self.alpha)


# --------------------------------------------------------------------------- #
# 4. MDCA : Multi-class Difference of Confidence and Accuracy
# --------------------------------------------------------------------------- #

class MDCA(nn.Module):
    """
    Hebbalaguppe et al. (2022).

        L_MDCA = (1/K) * sum_k | mean_i s_i[k] - mean_i q_i[k] |

    ou s_i est le vecteur softmax et q_i la cible one-hot, la moyenne portant
    sur le mini-lot.

    L'idee est simple et elegante: sur chaque lot, la probabilite moyenne
    attribuee a la classe k doit egaler la frequence empirique de la classe k.
    Aucun decoupage en intervalles, donc differentiable partout.

    Le prix a payer: la contrainte est une moyenne sur le lot. Elle est donc
    beaucoup plus faible que la vraie calibration (qui exige l'egalite
    conditionnellement au score). Un modele peut satisfaire MDCA exactement et
    rester mal calibre. Et l'estimation est bruitee sur des lots petits, ce que
    l'appel de these mentionne explicitement.
    """
    name = "MDCA"

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        k = probs.size(1)
        onehot = F.one_hot(targets, num_classes=k).float()
        return torch.abs(probs.mean(dim=0) - onehot.mean(dim=0)).mean()


# --------------------------------------------------------------------------- #
# 5. ECE a decoupage souple (soft binning)
# --------------------------------------------------------------------------- #

class SoftBinnedECE(nn.Module):
    """
    Karandikar et al. (2021).

    On remplace l'appartenance dure a un intervalle par une appartenance souple:
    chaque observation est repartie sur tous les intervalles selon un noyau
    gaussien centre sur les bornes. L'appartenance devient une fonction lisse de
    la confiance, donc differentiable.

        u_ib = softmax_b( -(conf_i - c_b)^2 / T )

    puis on calcule l'ECE avec ces poids souples. Quand T tend vers 0, on
    retrouve le decoupage dur, et le gradient s'evanouit. Le choix de T est
    un compromis entre fidelite a l'ECE et exploitabilite du gradient. C'est,
    a mon sens, le point ou le sujet de these a le plus de marge.
    """
    name = "SoftECE"

    def __init__(self, n_bins=15, temperature=0.01):
        super().__init__()
        self.n_bins = n_bins
        self.T = temperature
        centers = torch.linspace(1.0 / (2 * n_bins), 1.0 - 1.0 / (2 * n_bins), n_bins)
        self.register_buffer("centers", centers)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        correct = (pred == targets).float()

        # appartenance souple: (N, n_bins)
        d = -((conf.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2) / self.T
        u = F.softmax(d, dim=1)

        mass = u.sum(dim=0) + 1e-8                       # effectif souple par intervalle
        conf_b = (u * conf.unsqueeze(1)).sum(dim=0) / mass
        acc_b = (u * correct.unsqueeze(1)).sum(dim=0) / mass
        weight = mass / conf.size(0)

        return (weight * torch.abs(acc_b - conf_b)).sum()


# --------------------------------------------------------------------------- #
# Composition: perte principale + perte auxiliaire
# --------------------------------------------------------------------------- #

class Composite(nn.Module):
    """
        L = L_principale + beta * L_calibration

    C'est la forme que prennent toutes les approches a l'entrainement. Le
    parametre beta est le curseur du compromis exactitude / calibration, et
    l'un des objets d'etude de ce depot.
    """

    def __init__(self, main, aux, beta=1.0):
        super().__init__()
        self.main = main
        self.aux = aux
        self.beta = beta
        self.name = f"{main.name}+{aux.name}(b={beta})"

    def forward(self, logits, targets):
        return self.main(logits, targets) + self.beta * self.aux(logits, targets)


# --------------------------------------------------------------------------- #

def build_loss(name, **kw):
    """Fabrique. Les noms sont ceux utilises dans les tableaux de resultats."""
    name = name.lower()
    if name == "ce":
        return CrossEntropy()
    if name == "focal":
        return FocalLoss(gamma=kw.get("gamma", 3.0))
    if name == "ls":
        return LabelSmoothing(alpha=kw.get("alpha", 0.05))
    if name == "ce+mdca":
        return Composite(CrossEntropy(), MDCA(), beta=kw.get("beta", 1.0))
    if name == "focal+mdca":
        return Composite(FocalLoss(gamma=kw.get("gamma", 3.0)), MDCA(), beta=kw.get("beta", 1.0))
    if name == "ce+softece":
        return Composite(CrossEntropy(),
                         SoftBinnedECE(n_bins=kw.get("n_bins", 15),
                                       temperature=kw.get("temperature", 0.01)),
                         beta=kw.get("beta", 1.0))
    raise ValueError(f"perte inconnue: {name}")
