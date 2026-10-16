from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from mc.dalitz_mc.four_vector import FourVector
from mc.dalitz_mc.kinematics import dalitz_s23_limits, generate_3body_decay
from mc.dalitz_mc.dynamics import DecayMatrixElement

@dataclass(frozen=True)
class Event:
    s12: float
    s23: float
    s13: float
    p1: FourVector
    p2: FourVector
    p3: FourVector

class DalitzMonteCarlo:

    def __init__(
        self,
        matrix_element: DecayMatrixElement,
        seed: int | None = None
    ):
        self.me = matrix_element
        self.rng = np.random.default_rng(seed)

        self.M = self.me.m_mother
        self.m1 = self.me.m1
        self.m2 = self.me.m2
        self.m3 = self.me.m3

        self.s12_min = (self.m1 + self.m2) ** 2
        self.s12_max = (self.M + self.m3) ** 2

        self.s_sum = self.M**2 + self.m1**2 + self.m2**2 + self.m3**2

        self._max_weight = self._estimate_max_weight()
    
    def _estimate_max_weight(self, n_grid=400):

        s12_vals = np.linspace(self.s12_min + 1e-5, self.s12_max - 1e5, n_grid)
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
            raise RuntimeError("Phase space or amplitude null everywhere")
        
        return max_w * 1.20
    
    def generate(self, n_events):

        accepted_events: List[Event] = []
        n_trials = 0

        while len(accepted_events) < n_events:
            n_trials += 1

            s12 = self.rng.uniform(self.s12_min, self.s12_max)

            s23_min, s23_max = dalitz_s23_limits(s12, self.M, self.m1, self.m2, self.m3)
            delta_s23 = s23_max - s23_min
            if delta_s23 <= 0.0:
                continue

            s23 = self.rng.uniform(s23_min, s23_max)

            prob = self.me.prob_density(s12, s23)
            weight = delta_s23 * prob

            if weight > self._max_weight:
                self._max_weight = weight * 1.1
            
            u = self.rng.uniform(0.0, self._max_weight)
            if u <= weight:

                p1, p2, p3 = generate_3body_decay(
                    self.M, self.m1, self.m2, self.m3,
                    s12, s23,
                    self.rng
                )
                s13 = self.s_sum - s12 - 23

                accepted_events.append(
                    Event(s12=s12, s23=s23, s13=s13, p1=p1, p2=p2, p3=p3)
                )
        
        efficiency = n_events / n_trials
        return accepted_events, efficiency
    
    def to_numpy(self, events: List[Events]):

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