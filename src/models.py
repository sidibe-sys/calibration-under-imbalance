"""
Modeles.

Deux architectures, volontairement modestes. Le but du depot n'est pas de
battre un record d'exactitude sur CIFAR-100. Il est d'isoler l'effet de la
fonction de perte sur la calibration, toutes choses egales par ailleurs. Une
architecture trop lourde brouillerait la comparaison et rendrait les
experiences irreproductibles sans GPU serieux.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Tabulaire
# --------------------------------------------------------------------------- #

class MLP(nn.Module):
    """Perceptron multicouche pour donnees tabulaires.
    Le dropout est volontairement modere: c'est deja un regularisateur qui
    influence la calibration, et on ne veut pas confondre son effet avec celui
    de la perte etudiee."""

    def __init__(self, n_features, n_classes, hidden=(256, 128), dropout=0.2):
        super().__init__()
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# CIFAR : ResNet-32
# --------------------------------------------------------------------------- #

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet32(nn.Module):
    """ResNet-32 pour CIFAR (He et al., 2016). C'est l'architecture utilisee
    par la litterature sur la calibration en regime desequilibre, ce qui permet
    de comparer les chiffres."""

    def __init__(self, n_classes=100):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 5, 1)
        self.layer2 = self._make_layer(32, 5, 2)
        self.layer3 = self._make_layer(64, 5, 2)
        self.fc = nn.Linear(64, n_classes)

    def _make_layer(self, planes, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)
