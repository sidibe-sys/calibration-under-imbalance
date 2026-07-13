"""
Figures.

Le diagramme de fiabilite est l'outil de diagnostic central. Il porte en
abscisse la confiance annoncee, en ordonnee la frequence observee. La diagonale
est la calibration parfaite. Sous la diagonale: surconfiance. Au-dessus:
sous-confiance.

Un detail qui compte: il faut afficher l'histogramme des effectifs sous le
diagramme. Sans lui, on interprete gravement des intervalles qui contiennent
douze observations sur cinquante mille.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import reliability_curve, full_report, classwise_ece_per_class


def reliability_diagram(probs, labels, n_bins=15, title="", ax=None, show_hist=True):
    r = full_report(probs, labels, n_bins)
    xs, ys, ns = reliability_curve(probs, labels, n_bins)

    if ax is None:
        fig = plt.figure(figsize=(4.6, 5.4))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax = fig.add_subplot(gs[0])
        ax_h = fig.add_subplot(gs[1], sharex=ax) if show_hist else None
    else:
        ax_h = None

    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = 1.0 / n_bins

    valid = ~np.isnan(ys)
    ax.bar(centers[valid], ys[valid], width=width * 0.92,
           edgecolor="#1F3B57", color="#4C7FA8", label="frequence observee")
    ax.bar(centers[valid], (centers - ys)[valid], bottom=ys[valid], width=width * 0.92,
           edgecolor="#B03A2E", color="#E8A9A2", alpha=0.55, hatch="//",
           label="ecart a la calibration")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="calibration parfaite")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_ylabel("frequence observee")
    ax.set_title(f"{title}\nECE={r['ECE']:.3f}   SCE={r['SCE']:.3f}   "
                 f"ecart={r['gap']:+.3f}", fontsize=10)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)

    if ax_h is not None:
        ax_h.bar(centers, ns / max(ns.sum(), 1), width=width * 0.92,
                 color="#7F8C8D", edgecolor="white", lw=0.4)
        ax_h.set_xlabel("confiance annoncee")
        ax_h.set_ylabel("part des\nobservations", fontsize=8)
        ax_h.grid(alpha=0.25, lw=0.5)
        plt.setp(ax.get_xticklabels(), visible=False)

    return ax


def compare_reliability(runs, labels_true, n_bins=15, path="figures/reliability.png"):
    """runs : dict {titre: probs}. Trace une grille de diagrammes."""
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.8), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (title, probs) in zip(axes, runs.items()):
        reliability_diagram(probs, labels_true, n_bins, title, ax=ax, show_hist=False)
        ax.set_ylabel("")
    axes[0].set_ylabel("frequence observee")
    for ax in axes:
        ax.set_xlabel("confiance annoncee")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print("figure ecrite:", path)
    return fig


def sce_vs_class_frequency(probs, labels, train_counts, path="figures/sce_par_classe.png"):
    """
    LA figure du depot.

    En abscisse: l'effectif de la classe a l'entrainement.
    En ordonnee: l'erreur de calibration propre a cette classe.

    Si le nuage monte quand l'effectif descend, cela veut dire que l'erreur de
    calibration se concentre sur les classes rares, celles-la meme que l'ECE
    globale ne voit pas, puisqu'elle ne regarde que la classe predite, et que
    la classe predite est presque toujours une classe frequente.

    C'est l'argument central: en regime desequilibre, l'ECE ment par omission.
    """
    per_class = classwise_ece_per_class(probs, labels)
    counts = np.asarray(train_counts, dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(counts, per_class, s=26, alpha=0.75, color="#1F3B57", edgecolor="white", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("effectif de la classe a l'entrainement (echelle log)")
    ax.set_ylabel("erreur de calibration de la classe")

    if len(counts) > 2:
        lc = np.log10(counts + 1)
        b, a = np.polyfit(lc, per_class, 1)
        xx = np.linspace(lc.min(), lc.max(), 50)
        ax.plot(10 ** xx, a + b * xx, "--", color="#B03A2E", lw=1.6,
                label=f"tendance (pente = {b:+.4f})")
        rho = np.corrcoef(lc, per_class)[0, 1]
        ax.legend(fontsize=9)
        ax.set_title(f"L'erreur de calibration se concentre sur les classes rares\n"
                     f"correlation (log-effectif, erreur) = {rho:+.3f}", fontsize=10)

    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print("figure ecrite:", path)
    return fig
