#!/usr/bin/env python3
"""Script per la visualizzazione del Dalitz plot e delle proiezioni invarianti."""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dalitz_mc.kinematics import dalitz_s23_limits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plotting e analisi dei dati Dalitz.")
    parser.add_argument(
        "-i", "--input", type=str, default="dalitz_events.npz",
        help="Percorso del file .npz generato (default: dalitz_events.npz)."
    )
    parser.add_argument(
        "-o", "--output-plot", type=str, default="dalitz_plot.png",
        help="Nome del file grafico di output (default: dalitz_plot.png)."
    )
    parser.add_argument(
        "--bins", type=int, default=120,
        help="Numero di bin per asse negli istogrammi (default: 120)."
    )
    return parser.parse_args()


def calculate_kinematic_boundary(
    m_mother: float, m1: float, m2: float, m3: float, n_points: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """Costruisce il contorno cinematico chiuso (s12, s23) dello spazio delle fasi."""
    s12_min = (m1 + m2) ** 2 + 1e-6
    s12_max = (m_mother - m3) ** 2 - 1e-6

    s12_vals = np.linspace(s12_min, s12_max, n_points)
    s23_lower = np.empty(n_points)
    s23_upper = np.empty(n_points)

    for i, s12 in enumerate(s12_vals):
        s23_lower[i], s23_upper[i] = dalitz_s23_limits(s12, m_mother, m1, m2, m3)

    # Chiudiamo il poligono concatenando la curva inferiore con quella superiore invertita
    boundary_s12 = np.concatenate([s12_vals, s12_vals[::-1], [s12_vals[0]]])
    boundary_s23 = np.concatenate([s23_lower, s23_upper[::-1], [s23_lower[0]]])
    return boundary_s12, boundary_s23


def main() -> None:
    args = parse_args()

    # Caricamento del dataset
    loaded = np.load(args.input)
    s12 = loaded["s12"]
    s23 = loaded["s23"]
    s13 = loaded["s13"]
    m_mother, m1, m2, m3 = loaded["metadata"]

    m12 = np.sqrt(s12)
    m23 = np.sqrt(s23)
    m13 = np.sqrt(s13)

    # Calcolo del contorno teorico
    bound_s12, bound_s23 = calculate_kinematic_boundary(m_mother, m1, m2, m3)

    # Setup figura (layout 2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    # 1. Dalitz Plot 2D (s23 vs s12)
    ax_dalitz = axes[0, 0]
    h2d = ax_dalitz.hist2d(
        s12, s23, bins=args.bins, cmap="viridis", cmin=1
    )
    ax_dalitz.plot(bound_s12, bound_s23, color="crimson", linewidth=1.8, label="Contorno cinematico")
    ax_dalitz.set_xlabel(r"$s_{12} = m_{K^-\pi^+}^2 \; [\mathrm{GeV}^2]$", fontsize=12)
    ax_dalitz.set_ylabel(r"$s_{23} = m_{\pi^+\pi^0}^2 \; [\mathrm{GeV}^2]$", fontsize=12)
    ax_dalitz.set_title("Dalitz Plot $(s_{23} \ \mathrm{vs} \ s_{12})$", fontsize=13)
    ax_dalitz.legend(loc="upper right")
    fig.colorbar(h2d[3], ax=ax_dalitz, label="Conteggi / Bin")

    # 2. Proiezione 1D: m(K- pi+)
    ax_m12 = axes[0, 1]
    ax_m12.hist(m12, bins=args.bins, color="steelblue", edgecolor="black", linewidth=0.5)
    ax_m12.set_xlabel(r"$m_{12} = m_{K^-\pi^+} \; [\mathrm{GeV}]$", fontsize=12)
    ax_m12.set_ylabel("Conteggi", fontsize=12)
    ax_m12.set_title(r"Distribuzione $m(K^-\pi^+)$ - Picco $\bar{K}^*(892)^0$", fontsize=13)
    ax_m12.grid(True, linestyle="--", alpha=0.5)

    # 3. Proiezione 1D: m(pi+ pi0)
    ax_m23 = axes[1, 0]
    ax_m23.hist(m23, bins=args.bins, color="darkorange", edgecolor="black", linewidth=0.5)
    ax_m23.set_xlabel(r"$m_{23} = m_{\pi^+\pi^0} \; [\mathrm{GeV}]$", fontsize=12)
    ax_m23.set_ylabel("Conteggi", fontsize=12)
    ax_m23.set_title(r"Distribuzione $m(\pi^+\pi^0)$ - Picco $\rho^+(770)$", fontsize=13)
    ax_m23.grid(True, linestyle="--", alpha=0.5)

    # 4. Proiezione 1D: m(K- pi0)
    ax_m13 = axes[1, 1]
    ax_m13.hist(m13, bins=args.bins, color="forestgreen", edgecolor="black", linewidth=0.5)
    ax_m13.set_xlabel(r"$m_{13} = m_{K^-\pi^0} \; [\mathrm{GeV}]$", fontsize=12)
    ax_m13.set_ylabel("Conteggi", fontsize=12)
    ax_m13.set_title(r"Distribuzione $m(K^-\pi^0)$ - Picco $K^*(892)^-$", fontsize=13)
    ax_m13.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(args.output_plot, dpi=300)
    print(f"Grafico salvato con successo: {args.output_plot}")


if __name__ == "__main__":
    main()