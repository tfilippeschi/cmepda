import pytest
import math
from four_vector import FourVector

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