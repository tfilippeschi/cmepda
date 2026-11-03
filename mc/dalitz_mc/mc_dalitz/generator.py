"""Module for fast vectorized Monte Carlo sampling of 3-body decay."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import time
import numpy as np

from mc_dalitz.four_vector import FourVector
from mc_dalitz.kinematics import dalitz_s23_limits, generate_3body_decay
from mc_dalitz.dynamics import DecayMatrixElement


@dataclass(frozen=True)
class Event:
    """Representation of a 3-body decay event."""
    s12: float
    s23: float
    s13: float
    p1: FourVector
    p2: FourVector
    p3: FourVector


class DalitzMonteCarlo:
    """Monte Carlo generator for 3-body phase space using chunked rejection sampling."""

    def __init__(
        self,
        matrix_element: DecayMatrixElement,
        seed: int | None = None,
    ):
        self.me = matrix_element
        self.rng = np.random.default_rng(seed)

        self.M = self.me.m_mother
        self.m1 = self.me.m1
        self.m2 = self.me.m2
        self.m3 = self.me.m3

        self.s12_min = (self.m1 + self.m2) ** 2
        self.s12_max = (self.M - self.m3) ** 2

        self.s_sum = self.M**2 + self.m1**2 + self.m2**2 + self.m3**2
        self._max_weight = self._estimate_max_weight()
    
    def _weights(self, s12: np.ndarray, s23: np.ndarray, delta_s23: np.ndarray) -> np.ndarray:
        return delta_s23 * self.me.prob_density_array(s12, s23)

    def _estimate_max_weight(self, n_grid: int = 800, safety: float = 1.25) -> float:
        """Estimates maximum weight across the Dalitz plot."""
        eps = 1e-4
        s12 = np.linspace(self.s12_min, self.s12_max, n_grid + 2)[1:-1]
        s23_min, s23_max = dalitz_s23_limits_array(s12, self.M, self.m1, self.m2, self.m3)
        delta = s23_max - s23_min

        t = np.linspace(eps, 1.0 - eps, n_grid)
        s12_2d = np.broadcast_to(s12[:, None], (n_grid, n_grid))
        s23_2d = s23_min[:, None] + t[None, :] * delta[:, None]
        delta_2d = np.broadcast_to(delta[:, None], (n_grid, n_grid))
 
        max_w = float(np.max(self._weights(s12_2d, s23_2d, delta_2d)))
        if max_w == 0.0:
            raise RuntimeError("Phase space or amplitude null everywhere.")

        return max_w * safety

    def generate(
        self, n_events: int, chunk_size: int = 200_000, verbose: bool = True
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """Generates n_events using vectorized batch sampling with live feedback."""
        t_start = time.perf_counter()
        last_log = t_start
 
        while True:  # outer loop only repeats if max_weight turns out to be too low
            s12_acc, s23_acc = [], []
            n_acc = 0          # accepted so far (may exceed n_events by < 1 chunk)
            total_trials = 0
            restart = False
 
            while n_acc < n_events:
                total_trials += chunk_size
 
                s12 = self.rng.uniform(self.s12_min, self.s12_max, size=chunk_size)
                s23_lo, s23_hi = dalitz_s23_limits_array(s12, self.M, self.m1, self.m2, self.m3)
                delta = s23_hi - s23_lo
                s23 = s23_lo + self.rng.random(chunk_size) * delta
 
                w = self._weights(s12, s23, delta)
 
                batch_max = float(w.max())
                if batch_max > self._max_weight:
                    # Events accepted so far used a too-low bound -> biased. Start over.
                    self._max_weight = batch_max * 1.25
                    restart = True
                    if verbose:
                        print("Peso massimo sottostimato: aggiornato e riavvio la generazione.")
                    break
 
                keep = self.rng.uniform(0.0, self._max_weight, size=chunk_size) <= w
                s12_acc.append(s12[keep])
                s23_acc.append(s23[keep])
                n_acc += int(keep.sum())
 
                t_now = time.perf_counter()
                if verbose and (t_now - last_log > 1.5 or n_acc >= n_events):
                    last_log = t_now
                    elapsed = t_now - t_start
                    print(
                        f"Progresso: {min(n_acc, n_events):,}/{n_events:,} "
                        f"[{min(n_acc, n_events) / n_events * 100:.1f}%] "
                        f"| Rate: {min(n_acc, n_events) / elapsed:.0f} ev/s "
                        f"| Eff: {n_acc / total_trials * 100:.2f}%"
                    )
 
            if not restart:
                break
 
        efficiency = n_acc / total_trials
        s12_all = np.concatenate(s12_acc)[:n_events]
        s23_all = np.concatenate(s23_acc)[:n_events]
 
        # Full kinematic reconstruction of accepted events only, in one shot
        p1, p2, p3 = generate_3body_decay_array(
            self.M, self.m1, self.m2, self.m3, s12_all, s23_all, self.rng
        )
        data = {
            "s12": s12_all,
            "s23": s23_all,
            "s13": self.s_sum - s12_all - s23_all,
            "p1": p1,
            "p2": p2,
            "p3": p3,
        }
        return data, efficiency