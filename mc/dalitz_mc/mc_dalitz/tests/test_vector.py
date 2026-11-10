"""Tests for the vectorized generator (run with: python -m  pytest)."""
import numpy as np
import pytest

from mc_dalitz.dynamics import DecayMatrixElement, NonResonant, RelativisticBreitWigner
from mc_dalitz.generator import DalitzMonteCarlo
from mc_dalitz.kinematics import dalitz_s23_limits, dalitz_s23_limits_array

M, M1, M2, M3 = 1.8648, 0.4937, 0.1396, 0.1350


def _matrix_element():
    return DecayMatrixElement(M, M1, M2, M3, [
        NonResonant(0.35, 0.0),
        RelativisticBreitWigner(0.7753, 0.1491, 1, "23", 1.0, 1.15),
        RelativisticBreitWigner(0.8917, 0.0508, 1, "13", 0.65, -0.45),
        RelativisticBreitWigner(0.8955, 0.0473, 1, "12", 0.50, 2.30),
    ])


def _random_points(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s12 = rng.uniform((M1 + M2) ** 2, (M - M3) ** 2, n)
    lo, hi = dalitz_s23_limits_array(s12, M, M1, M2, M3)
    return s12, lo + rng.random(n) * (hi - lo)


def test_limits_vectorized_match_scalar():
    s12, _ = _random_points()
    lo, hi = dalitz_s23_limits_array(s12, M, M1, M2, M3)
    ref = np.array([dalitz_s23_limits(x, M, M1, M2, M3) for x in s12])
    assert np.allclose(lo, ref[:, 0], atol=1e-12)
    assert np.allclose(hi, ref[:, 1], atol=1e-12)


def test_prob_density_vectorized_match_scalar():
    me = _matrix_element()
    s12, s23 = _random_points()
    ref = np.array([me.prob_density(a, b) for a, b in zip(s12, s23)])
    assert np.allclose(me.prob_density_array(s12, s23), ref, rtol=1e-10)


def test_generated_events_are_physical_and_consistent():
    data, eff = DalitzMonteCarlo(_matrix_element(), seed=1).generate(5000, verbose=False)
    assert 0.0 < eff < 1.0
    assert data["s12"].shape == (5000,)

    lo, hi = dalitz_s23_limits_array(data["s12"], M, M1, M2, M3)
    assert np.all(data["s23"] >= lo - 1e-9) and np.all(data["s23"] <= hi + 1e-9)

    p1, p2, p3 = data["p1"], data["p2"], data["p3"]
    dot = lambda a, b: a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1)
    assert np.allclose(data["s12"], M1**2 + M2**2 + 2 * dot(p1, p2), atol=1e-9)
    assert np.allclose(data["s23"], M2**2 + M3**2 + 2 * dot(p2, p3), atol=1e-9)

    tot = p1 + p2 + p3
    assert np.allclose(tot[:, 0], M, atol=1e-9)
    assert np.allclose(tot[:, 1:], 0.0, atol=1e-9)
    for p, m in ((p1, M1), (p2, M2), (p3, M3)):
        m2 = p[:, 0] ** 2 - np.sum(p[:, 1:] ** 2, axis=1)
        assert np.allclose(m2, m**2, atol=1e-9)