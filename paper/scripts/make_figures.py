"""Regenerate the data-derived figures for the paper from files committed in the repo.

Usage (from repo root): python -X utf8 paper/scripts/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
XLSX = ROOT / "data" / "raw" / "oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"
CONN = ROOT / "data" / "processed" / "template_connectome.npy"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
})
GROUP_COLORS = {"Nondemented": "#4c78a8", "Demented": "#e45756", "Converted": "#f2a93b"}


def fig_cohort():
    df = pd.read_excel(XLSX)
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0))

    visits = df.groupby("Subject ID").size().value_counts().sort_index()
    ax[0].bar(visits.index.astype(str), visits.values, color="#4c78a8")
    for x, v in zip(visits.index.astype(str), visits.values):
        ax[0].text(x, v + 1, str(v), ha="center", fontsize=7)
    ax[0].set(xlabel="Visits per subject", ylabel="Subjects", title="(a) Follow-up length")

    cdr = df["CDR"].value_counts().sort_index()
    ax[1].bar([str(c) for c in cdr.index], cdr.values, color="#72b7b2")
    for x, v in zip([str(c) for c in cdr.index], cdr.values):
        ax[1].text(x, v + 3, str(v), ha="center", fontsize=7)
    ax[1].set(xlabel="CDR", ylabel="Sessions", title="(b) CDR over 373 sessions")

    base = df[df["Visit"] == 1]
    bins = np.arange(60, 100, 4)
    for g, c in GROUP_COLORS.items():
        ax[2].hist(base.loc[base["Group"] == g, "Age"], bins=bins, alpha=0.65,
                   color=c, label=g)
    ax[2].set(xlabel="Baseline age (years)", ylabel="Subjects", title="(c) Baseline age by group")
    ax[2].legend(frameon=False, loc="upper left", fontsize=6)

    fig.tight_layout()
    fig.savefig(OUT / "fig_cohort.pdf")
    plt.close(fig)


def fig_ndm_spectrum():
    # Same construction as compute_laplacian() in OASIS2_Colab.py
    A = np.load(CONN)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    A = np.maximum(A, 0)
    L = np.diag(A.sum(axis=1)) - A
    lam = np.maximum(np.linalg.eigvalsh(L), 0)

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    ax[0].plot(np.arange(1, len(lam) + 1), lam, "o", ms=3, color="#4c78a8")
    ax[0].set(xlabel="Mode index $i$", ylabel=r"Eigenvalue $\lambda_i$",
              title="(a) Laplacian spectrum of the 56-region template")

    # Slowest non-constant mode (lambda_2): how fast spatial contrast is lost
    lam2 = lam[1]
    t = np.linspace(0, 4, 400)
    for beta, c in [(7.233, "#e45756"), (9.101, "#4c78a8")]:
        ax[1].semilogy(t, np.exp(-beta * lam2 * t), color=c, label=r"$\beta=%s$" % beta)
    ax[1].axvline(3.3, color="gray", ls="--", lw=0.8)
    ax[1].set(xlabel="Elapsed time $t$ (years)", ylabel=r"$e^{-\beta\lambda_2 t}$",
              ylim=(1e-30, 2),
              title=r"(b) Slowest non-constant mode ($\lambda_2=%.2f$)" % lam2)
    ax[1].legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "fig_ndm_spectrum.pdf")
    plt.close(fig)
    return lam


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_cohort()
    lam = fig_ndm_spectrum()
    print("OK; lambda range", lam.min().round(3), lam.max().round(3),
          "n>1e-6:", int((lam > 1e-6).sum()))
