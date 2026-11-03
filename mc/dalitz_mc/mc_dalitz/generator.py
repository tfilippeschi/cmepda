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

    def _estimate_max_weight(self, n_grid: int = 300) -> float:
        """Estimates maximum weight across the Dalitz plot."""
        s12_vals = np.linspace(self.s12_min + 1e-5, self.s12_max - 1e-5, n_grid)
        max_w = 0.0

        for s12 in s12_vals:
            s23_min, s23_max = dalitz_s23_limits(s12, self.M, self.m1, self.m2, self.m3)
            delta_s23 = s23_max - s23_min
            if delta_s23 <= 0.0:
                continue

            s23_vals = np.linspace(s23_min + 1e-5, s23_max - 1e-5, n_grid // 2)
            for s23 in s23_vals:
                prob = self.me.prob_density(s12, s23)
                weight = delta_s23 * prob
                if weight > max_w:
                    max_w = weight

        if max_w == 0.0:
            raise RuntimeError("Phase space or amplitude null everywhere.")

        return max_w * 1.25

    def generate(
        self, n_events: int, chunk_size: int = 50_000
    ) -> Tuple[List[Event], float]:
        """Generates n_events using vectorized batch sampling with live feedback."""
        accepted_events: List[Event] = []
        total_trials = 0
        t_start = time.perf_counter()
        last_log = t_start

        while len(accepted_events) < n_events:
            total_trials += chunk_size

            # 1. Genera un blocco di s12 uniformi
            s12_batch = self.rng.uniform(self.s12_min, self.s12_max, size=chunk_size)

            # 2. Calcola limiti cinematici s23 per il blocco
            # Ottimizzazione: calcolo vettorizzato rapido dei limiti di Källén
            m12 = np.sqrt(s12_batch)
            e2 = (s12_batch - self.m1**2 + self.m2**2) / (2.0 * m12)
            e3 = (self.M**2 - s12_batch - self.m3**2) / (2.0 * m12)

            p2 = np.sqrt(np.maximum(0.0, e2**2 - self.m2**2))
            p3 = np.sqrt(np.maximum(0.0, e3**2 - self.m3**2))

            term_e = (e2 + e3)**2
            s23_min = term_e - (p2 + p3)**2
            s23_max = term_e - (p2 - p3)**2
            delta_s23 = s23_max - s23_min

            # 3. Estrai s23 uniforme all'interno dei limiti
            s23_batch = s23_min + self.rng.uniform(0.0, 1.0, size=chunk_size) * delta_s23

            # 4. Calcola la densità di probabilità |M|^2
            # Per evitare overhead di lista, valutiamo il vettore di probabilità
            weights = np.empty(chunk_size, dtype=np.float64)
            for i in range(chunk_size):
                weights[i] = delta_s23[i] * self.me.prob_density(s12_batch[i], s23_batch[i])

            # Protezione dinamica su max_weight
            batch_max = np.max(weights)
            if batch_max > self._max_weight:
                self._max_weight = batch_max * 1.15

            # 5. Test Hit-or-Miss vettorizzato
            u = self.rng.uniform(0.0, self._max_weight, size=chunk_size)
            accepted_mask = np.where(u <= weights)[0]

            # 6. Ricostruzione cinematica dei soli eventi accettati
            for idx in accepted_mask:
                s12_val = float(s12_batch[idx])
                s23_val = float(s23_batch[idx])
                s13_val = float(self.s_sum - s12_val - s23_val)

                p1, p2_v, p3_v = generate_3body_decay(
                    self.M, self.m1, self.m2, self.m3, s12_val, s23_val, self.rng
                )
                accepted_events.append(
                    Event(s12=s12_val, s23=s23_val, s13=s13_val, p1=p1, p2=p2_v, p3=p3_v)
                )

                if len(accepted_events) >= n_events:
                    break

            # 7. Feedback periodico ogni 1.5 secondi
            t_now = time.perf_counter()
            if t_now - last_log > 1.5 or len(accepted_events) >= n_events:
                last_log = t_now
                n_acc = len(accepted_events)
                elapsed = t_now - t_start
                rate = n_acc / elapsed if elapsed > 0 else 0
                eff = (n_acc / total_trials) * 100
                pct = (n_acc / n_events) * 100
                print(
                    f"Progresso: {n_acc:,}/{n_events:,} [{pct:.1f}%] "
                    f"| Rate: {rate:.1f} ev/s | Eff: {eff:.2f}%"
                )

        efficiency = len(accepted_events) / total_trials
        return accepted_events, efficiency

    def to_numpy(self, events: List[Event]) -> dict[str, np.ndarray]:
        """Converts a list of events to NumPy arrays for disk storage."""
        n = len(events)
        data = {
            "s12": np.empty(n, dtype=np.float64),
            "s23": np.empty(n, dtype=np.float64),
            "s13": np.empty(n, dtype=np.float64),
            "p1": np.empty((n, 4), dtype=np.float64),
            "p2": np.empty((n, 4), dtype=np.float64),
            "p3": np.empty((n, 4), dtype=np.float64),
        }

        for i, ev in enumerate(events):
            data["s12"][i] = ev.s12
            data["s23"][i] = ev.s23
            data["s13"][i] = ev.s13
            data["p1"][i] = [ev.p1.e, ev.p1.px, ev.p1.py, ev.p1.pz]
            data["p2"][i] = [ev.p2.e, ev.p2.px, ev.p2.py, ev.p2.pz]
            data["p3"][i] = [ev.p3.e, ev.p3.px, ev.p3.py, ev.p3.pz]

        return data