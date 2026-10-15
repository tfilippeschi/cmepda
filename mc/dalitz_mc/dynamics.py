import abc
import cmath
import math
from typing import List, Tuple
import numpy as np

from mc.dalitz_mc.kinematics import kallen_lambda

def blatt_weisskopf_factor(l_spin, q, q0, radius=1.5):
    if l_spin == 0:
        return 1.0
    z = (q * radius) ** 2
    z0 = (q0 * radius) ** 2
    if l_spin == 1:
        return math.sqrt(((z0 - 3.0) ** 2 + 9.0 * z0) / ((z - 3.0)) ** 2 + 9.0 * z)
    raise NotImplementedError(f"Spin L={l_spin} not supported")

class AmplitudeComponent(abc.ABC):

    def __init__(self, amplitude=1.0, phase_rad=0.0):
        self.amplitude = amplitude
        self.phase_rad = phase_rad
    
    @property
    def complex_coupling(self):
        return cmath.rect(self.amplitude, self.phase_rad)
    
    @abc.abstractmethod
    def evaluate(
        self, s12, s23, m_mother, m1, m2, m3
    ):
        pass

class NonResonant(AmplitudeVomponent):

    def evaluate(
        self, s12, s23, m_mother, m1, m2, m3
    ):
        return self.complex_coupling

class RelativisticBreitWigner(AmplitudeComponent):

    def __init__(
        self,
        m0, gamma0, spin,
        channel, amplitude=1.0, phase_rad=0.0, meson_radius=1.5
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
        self, s12, s23, m_mother, m1, m2, m3
    ):
        s_tot = m_mother**2 + m1**2 + m2**2 + m3**2
        s13 = s_tot - s12 - s23

        if self.channel == "12":
            return s12, m1, m2, m3
        elif self.channel == "23":
            return s23, m1, m2, m3
        else:
            return s13, m1, m3, s12
        
    def evaluate(
        self, s12, s23, m_mother, m1, m2, m3
    ):
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

    def __init__(
        self,
        m_mother, m1, m2, m3, components: List[AmplitudeComponent]
    ):
        self.m_mother = m_mother
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.components = components
    
    def amplitude(self, s12, s23):

        tot = 0.0j
        for comp in self.components:
            tot += comp.evaluate(
                s12, s23, self.m_mother, self.m1, self.m2, self.m3
            )
        return tot
    
    def prob_density(self, s12, s23):
        amp=self.amplitude(s12, s23)
        return amp.real**2 + amp.imag*2