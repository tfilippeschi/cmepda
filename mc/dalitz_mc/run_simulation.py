#!/usr/bin/env python3

from __future__ import annotations
import argparse
import sys
import time
import numpy as np

from dynamics import DecayMatrixElement, NonResonant, RelativisticBreitWigner
from generator import DalitzMonteCarlo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo generator (Dalitz Plot)."
    )
    parser.add_argument(
        "-n", "--nevents", type=int, default=100_000,
        help="Number of events (default: 100000)."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="dalitz_events.npz",
        help="Path of output file NumPy .npz (default: dalitz_events.npz)."
    )
    parser.add_argument(
        "--seed", type=int, default=1337,
        help="Seed for generator(default: 1337)."
    )
    parser.add_argument(
        "--pure-phase-space", action="store_true",
        help="If active, disables resonances."
    )
    # Parametri cinematici (default: D0 -> K- pi+ pi0)
    parser.add_argument("--m-mother", type=float, default=1.8648, help="Mass of decaying particle [GeV].")
    parser.add_argument("--m1", type=float, default=0.4937, help="Massa product 1 (K-) [GeV].")
    parser.add_argument("--m2", type=float, default=0.1396, help="Massa product 2 (pi+) [GeV].")
    parser.add_argument("--m3", type=float, default=0.1350, help="Massa product 3 (pi0) [GeV].")
    return parser.parse_args()


def build_dynamics(args: argparse.Namespace) -> DecayMatrixElement:
    components = []
    if args.pure_phase_space:
        components.append(NonResonant(amplitude=1.0, phase_rad=0.0))
    else:
        # 1. Fondo continuo non-risonante
        components.append(NonResonant(amplitude=0.35, phase_rad=0.0))
        # 2. Risonanza rho+(770) -> pi+ pi0 (canale 23), J=1
        components.append(
            RelativisticBreitWigner(
                m0=0.7753, gamma0=0.1491, spin=1, channel="23",
                amplitude=1.0, phase_rad=1.15
            )
        )
        # 3. Risonanza K*(892)- -> K- pi0 (canale 13), J=1
        components.append(
            RelativisticBreitWigner(
                m0=0.8917, gamma0=0.0508, spin=1, channel="13",
                amplitude=0.65, phase_rad=-0.45
            )
        )
        # 4. Risonanza Kbar*(892)0 -> K- pi+ (canale 12), J=1
        components.append(
            RelativisticBreitWigner(
                m0=0.8955, gamma0=0.0473, spin=1, channel="12",
                amplitude=0.50, phase_rad=2.30
            )
        )

    return DecayMatrixElement(
        m_mother=args.m_mother,
        m1=args.m1,
        m2=args.m2,
        m3=args.m3,
        components=components,
    )


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Simulatore Monte Carlo - Decadimenti a 3 Corpi")
    print(f"Madre: M = {args.m_mother:.4f} GeV")
    print(f"Figlie: m1 = {args.m1:.4f}, m2 = {args.m2:.4f}, m3 = {args.m3:.4f} GeV")
    print(f"Eventi richiesti: {args.nevents:,} | Output: {args.output}")
    print("=" * 60)

    matrix_element = build_dynamics(args)

    print("Inizializzazione generatore e calcolo del peso massimo...")
    t0 = time.perf_counter()
    mc = DalitzMonteCarlo(matrix_element, seed=args.seed)
    t_init = time.perf_counter() - t0
    print(f"Inizializzazione completata in {t_init:.2f} s.")

    print(f"Generazione di {args.nevents:,} eventi in corso...")
    t1 = time.perf_counter()
    events, efficiency = mc.generate(args.nevents)
    t_gen = time.perf_counter() - t1

    print(f"Generazione conclusa in {t_gen:.2f} s.")
    print(f"Velocità: {args.nevents / t_gen:.1f} eventi/s")
    print(f"Efficienza di rigetto (hit/trials): {efficiency * 100:.2f}%")

    # Conversione e salvataggio compresso
    data_dict = mc.to_numpy(events)
    # Salviamo anche i parametri fisici come metadati
    data_dict["metadata"] = np.array([args.m_mother, args.m1, args.m2, args.m3])

    np.savez_compressed(args.output, **data_dict)
    print(f"Eventi salvati con successo in: {args.output}")


if __name__ == "__main__":
    main()