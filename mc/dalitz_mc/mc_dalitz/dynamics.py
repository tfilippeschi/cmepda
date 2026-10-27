"""Module for the dynamic of decays: Breit-Wigner and Isobar Model."""

from __future__ import annotations
import abc
import cmath
import math
from typing import List, Tuple
import numpy as np

from kinematics import kallen_lambda

def blatt_weisskopf_factor(l_spin: int, q: float, q0: float, radius: float = 1.5) -> float:
    """Calcola il rapporto F_L(q)/F_L(q0)."""
    if l_spin == 0:
        return 1.0
    z = (q * radius) ** 2
    z0 = (q0 * radius) ** 2
    if l_spin == 1:
        return math.sqrt(((z0 - 3.0) ** 2 + 9.0 * z0) / ((z - 3.0)) ** 2 + 9.0 * z)
    raise NotImplementedError(f"Spin L={l_spin} not supported")

class AmplitudeComponent(abc.ABC):
    """Base class for each coherent contribute to total amplitude."""

    def __init__(self, amplitude: float = 1.0, phase_rad: float = 0.0):
        self.amplitude = amplitude
        self.phase_rad = phase_rad
    
    @property
    def complex_coupling(self) -> complex:
        """Complex coupling a * e^(i * delta)."""
        return cmath.rect(self.amplitude, self.phase_rad)
    
    @abc.abstractmethod
    def evaluate(
        self, s12, s23, m_mother, m1, m2, m3
    ) -> complex:
        """Evaluate amplitude term for a (s12, s23) point."""
        pass

class NonResonant(AmplitudeComponent):
    """Non-resonant term (uniform phase space)."""

    def evaluate(
        self, s12: float, s23: float, m_mother: float, m1: float, m2: float, m3: float
    ) -> complex:
        return self.complex_coupling

class RelativisticBreitWigner(AmplitudeComponent):
    """Resonance with relativistic Breit-Wigner and angular dependence.
    
    Supoorted channels for subsystem:
    - '12': resonance R -> 1 + 2 (invariant s12)
    - '23': resonance R -> 2 + 3 (invariant s23)
    - '13': resonance R -> 1 + 3 (invariant s13)
    """

    def __init__(
        self,
        m0: float, gamma0: float, spin: int,
        channel: str,
        amplitude: float = 1.0, phase_rad: float = 0.0,
        meson_radius: float = 1.5
    ):
        super().__init__(amplitude, phase_rad)
        self.m0 = m0
        self.gamma0 = gamma0
        self.spin = spin
        self.channel = channel
        self.meson_radius = meson_radius

        if channel not in ("12", "23", "13"):
            raise ValueError(f"Unknown channel: {channel}. Use '12', '23' o '13'.")
        
    def _get_channel_kinematics(
        self, s12: float, s23: float, m_mother: float,
        m1: float, m2: float, m3: float
    ) -> Tuple[float, float, float, float]:
        """Extract channel's invariant s, daughter-particles and spectator's massea."""
        s_tot = m_mother**2 + m1**2 + m2**2 + m3**2
        s13 = s_tot - s12 - s23

        if self.channel == "12":
            return s12, m1, m2, m3
        elif self.channel == "23":
            return s23, m1, m2, m3
        else:
            return s13, m1, m3, s12
        
    def evaluate(
        self, s12: float, s23: float, m_mother: float,
        m1: float, m2: float, m3: float
    ) -> complex:
        s, ma, mb, s_cross = self._get_channel_kinematics(
            s12, s23, m_mother, m1, m2, m3)
        
        if s <= (ma + mb) ** 2:
            return 0.0j
        
        m_inv = math.sqrt(s)

        lam = kallen_lambda(s, ma**2, mb**2)
        q = math.sqrt(lam / (2.0 * m_inv))

        lam0 = kallen_lambda(self.m0**2, ma**2, mb**2)
        q0 = math.sqrt(lam0 / (2.0 * self.m0))

        ff = blatt_weisskopf_factor(self.spin, q, q0, self.meson_radius)

        gamma_s = (
            self.gamma0
            * (self.m0 / m_inv)
            * ((q - q0) ** (2 * self.spin + 1))
            * (ff**2)
        )

        denom = complex(self.m0*2 - s, -self.m0 * gamma_s)
        bw_prop = 1.0 / denom

        angular_factor = 1.0
        if self.spin == 1:
            e_a_star = (s - mb**2 + mb**2) / (2.0 * m_inv)
            angular_factor = (s_cross - (m_mother*2 + ma**2 + mb**2 - s) / 2.0) / s
        
        return self.complex_coupling * bw_prop * angular_factor

class DecayMatrixElement:
    """Manages the coherent sum of amplitudes and calculation of |M|^2."""

    def __init__(
        self,
        m_mother: float,
        m1: float, m2: float, m3: float,
        components: List[AmplitudeComponent]
    ):
        self.m_mother = m_mother
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.components = components
    
    def amplitude(self, s12: float, s23: float) -> complex:
        """Calculates the sum sum_i M_i(s12, s23)."""

        tot = 0.0j
        for comp in self.components:
            tot += comp.evaluate(
                s12, s23, self.m_mother, self.m1, self.m2, self.m3
            )
        return tot
    
    def prob_density(self, s12: float, s23: float) -> float:
        """Calculates |M|^2."""
        amp=self.amplitude(s12, s23)
        return amp.real**2 + amp.imag*2