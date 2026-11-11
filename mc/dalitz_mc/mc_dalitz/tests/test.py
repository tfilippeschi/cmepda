import pytest
import math
import numpy as np

from four_vector import FourVector
from kinematics import dalitz_s23_limits, generate_3body_decay
from dynamics import DecayMatrixElement, NonResonant, RelativisticBreitWigner

# four_vector tests

def test_mass_invariance_under_boost():
    m0 = 1.8648
    p = FourVector.from_spatial(m=m0, px=1.5, py=-0.8, pz=3.2,)
    assert math.isclose(p.m(), m0, rel_tol=1e-9)

    boosted = p.boost(bx=0.2, by=-0.4, bz=0.3)
    assert math.isclose(boosted.m(), m0, rel_tol=1e-9)

def test_rest_frame_transformation():
    p = FourVector.from_spatial(m=3.096, px=2.0, py=1.0, pz=4.0)
    p_rest = p.boost_to_rest_frame_of(p)

    assert math.isclose(p_rest.p_spatial, 0.0, abs_tol=1e-12)
    assert math.isclose(p_rest.e, p.m(), rel_tol=1e-9)

    p_back = p_rest.boost_from_rest_frame_of(p)
    assert math.isclose(p_back.px, p.px, rel_tol=1e-9)
    assert math.isclose(p_back.py, p.py, rel_tol=1e-9)
    assert math.isclose(p_back.pz, p.pz, rel_tol=1e-9)
    assert math.isclose(p_back.e, p.e, rel_tol=1e-9)

def test_mandelstam_s_addition():
    
    p1 = FourVector(e=5.0, px=0.0, py=0.0, pz=5.0)  # ultra-relativistica
    p2 = FourVector(e=5.0, px=0.0, py=0.0, pz=-5.0)

    p_tot = p1 + p2
    s = p_tot.m2()
    assert math.isclose(s, 100.0, rel_tol=1e-9)
    assert math.isclose(p_tot.p_spatial, 0.0, abs_tol=1e-12)

# kinematics tests

def test_conservation_and_invariants():
    rng = np.random.default_rng(seed=42)
    m_d0 = 1.8648
    m_k = 0.4937
    m_pip = 0.1396
    m_pi0 = 0.1350

    s12_min = (m_k + m_pip)**2
    s12_max = (m_d0 - m_pi0)**2

    s12_val = (s12_min + s12_max) / 2.0
    s23_min, s23_max = dalitz_s23_limits(s12_val, m_d0, m_k, m_pip, m_pi0)
    s23_val = (s23_min + s23_max) / 2.0

    p1, p2, p3 = generate_3body_decay(m_d0, m_k, m_pip, m_pi0, s12_val, s23_val, rng)

    p_tot = p1 + p2 + p3
    assert np.isclose(p_tot.m(), m_d0, atol=1e-6)
    assert np.isclose(p_tot.p_spatial, 0.0, atol=1e-6)

    assert np.isclose(p1.m(), m_k, atol=1e-5)
    assert np.isclose(p2.m(), m_pip, atol=1e-5)
    assert np.isclose(p3.m(), m_pi0, atol=1e-5)

    assert np.isclose((p1 + p2).m2(), s12_val, atol=1e-5)
    assert np.isclose((p2 + p3).m2(), s23_val, atol=1e-5)

# dynamics tests

nr = NonResonant(amplitude=0.3, phase_rad=0.0)
rho = RelativisticBreitWigner(
    m0=0.775, gamma0=0.149, spin=1, channel="23", amplitude=1.0, phase_rad=1.2
)
k_star = RelativisticBreitWigner(
    m0=0.892, gamma0=0.050, spin=1, channel="12", amplitude=0.8, phase_rad=-0.5
)

matrix_elem = DecayMatrixElement(
    m_mother=1.8648,
    m1=0.4937,   # K-
    m2=0.1396,   # pi+
    m3=0.1350,   # pi0
    components=[nr, rho, k_star]
)

# Valutazione in un punto dello spazio delle fasi
prob = matrix_elem.prob_density(s12=0.892**2, s23=0.775**2)
assert prob > 0.0
print(f"|M|^2 nel punto di quasi-intersezione dei picchi: {prob:.4e}")
