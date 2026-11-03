"""Module for the implementation of relativistic cinematics and phase space generation"""

from __future__ import annotations
import math
from typing import Tuple
import numpy as np

from mc_dalitz.four_vector import FourVector

def kallen_lambda(x: float, y: float, z: float) -> float:
    """lambda(x, y, z) = x^2 + y^2 + z^2 - 2xy -2xz - 2yz."""
    return x**2 + y**2 + z**2 - 2.0 * (x * y + x * z + y * z)

def two_body_momentum(m_parent: float, m_d1: float, m_d2: float) -> float:
    """Momentum in cm rest frame for 2-body decay: parent -> d1 + d2."""
    if m_parent < (m_d1 + m_d2):
        raise ValueError(f"{m_parent:.4f} < {m_d1 + m_d2:.4f}.")
    lam = kallen_lambda(m_parent**2, m_d1**2, m_d2**2)
    return math.sqrt(max(0.0, lam)) / (2.0 * m_parent)

def dalitz_s23_limits(
    s12: float, m_mother: float, m1: float, m2: float, m3: float, tol: float = 1e-5
    ) -> Tuple[float, float]:
    """Kinematical limits [s23_min, s23_max] for a given value of s12 = m12^2."""
    s12_min = (m1 + m2) **2
    s12_max = (m_mother - m3) **2

    if not (s12_min - tol <= s12 <= s12_max + tol):
        raise ValueError(f"ERROR: s12={s12:.4f} GeV^2 mass value outside limits")
    
    s12_clamped = max(s12_min, min(s12, s12_max))
    m12 = math.sqrt(s12_clamped)

    e2_star = (s12_clamped - m1**2 + m2**2) / (2.0 * m12)
    e3_star = (m_mother**2 - s12_clamped - m3**2) / (2.0 * m12)

    p2_star = math.sqrt(max(0.0, e2_star**2 - m2**2))
    p3_star = math.sqrt(max(0.0, e3_star**2 - m3**2))

    term_e = (e2_star + e3_star)**2
    s23_min = term_e - (p2_star + p3_star)**2
    s23_max = term_e - (p2_star - p3_star)**2

    return s23_min, s23_max

def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generates a uniform random 3D rotation matrix via Euler angles."""
    phi = rng.uniform(0.0, 2.0 * math.pi)
    costheta = rng.uniform(-1.0, 1.0)
    psi = rng.uniform(0.0, 2.0 * math.pi)

    sintheta = math.sqrt(max(0.0, 1.0 - costheta**2))

    c1, s1 = math.cos(phi), math.sin(phi)
    c2, s2 = costheta, sintheta
    c3, s3 = math.cos(psi), math.sin(psi)

    return np.array([
        [c1 * c2 * c3 - s1 * s3, -c1 * c2 * s3 - s1 * c3, c1 * s2],
        [s1 * c2 * c3 + c1 * s3, -s1 * c2 * s3 + c1 * c3, s1 * s2],
        [-s2 * c3,               s2 * s3,                c2     ]
    ], dtype=np.float64)

def rotate_vector(p: FourVector, rot: np.ndarray) -> FourVector:
    """Applies a 3x3 spatial rotation matrix to a FourVector."""
    p_rot = rot @ p.p_vec
    return FourVector(p.e, p_rot[0], p_rot[1], p_rot[2])

def generate_3body_decay(
    m_mother: float,
    m1: float, m2: float, m3: float,
    s12: float, s23: float,
    rng: np.random.Generator
    ) -> Tuple[FourVector, FourVector, FourVector]:
    """Generates 3 FourVector (p1, p2, p3) in the mother particle rest frame.
    
    Given the invariants s12, s23, spatial momentum is isotropically oriented
    in 3-D space.
    """
    s12_min = (m1 + m2) ** 2
    s12_max = (m_mother - m3) ** 2
    s12_clamped = max(s12_min, min(s12, s12_max))
    m12 = math.sqrt(s12_clamped)

    p3_cm_mag = two_body_momentum(m_mother, m12, m3)
    p2_star = two_body_momentum(m12, m1, m2)

    e2_star = (s12_clamped - m1**2 + m2**2) / (2.0 * m12)
    e3_star = (m_mother**2 - s12_clamped - m3**2) / (2.0 * m12)
    p3_star = math.sqrt(max(0.0, e3_star**2 - m3**2))

    num = (e2_star + e3_star)**2 - s23 - p2_star**2 - p3_star**2
    costheta_star = float(np.clip(num / (2.0 * p2_star * p3_star), -1.0, 1.0))
    
    sintheta_star = math.sqrt(max(0.0, 1.0 - costheta_star**2))
    phi_star = rng.uniform(0.0, 2.0 * math.pi)

    p2_star_x = p2_star * sintheta_star * math.cos(phi_star)
    p2_star_y = p2_star * sintheta_star * math.sin(phi_star)
    p2_star_z = p2_star * costheta_star
    p2_star_vec = FourVector(e2_star, p2_star_x, p2_star_y, p2_star_z)

    e1_star = (s12_clamped + m1**2 - m2**2) / (2.0 * m12)
    p1_star_vec = FourVector(e1_star, -p2_star_x, -p2_star_y, -p2_star_z)

    e12 = math.sqrt(m12**2 + p3_cm_mag**2)
    p12 = FourVector(e12, 0.0, 0.0, -p3_cm_mag)

    p1_aligned = p1_star_vec.boost_from_rest_frame_of(p12)
    p2_aligned = p2_star_vec.boost_from_rest_frame_of(p12)

    e3 = math.sqrt(m3**2 + p3_cm_mag**2)
    p3_aligned = FourVector(e3, 0.0, 0.0, p3_cm_mag)

    rot = _random_rotation_matrix(rng)
    p1 = rotate_vector(p1_aligned, rot)
    p2 = rotate_vector(p2_aligned, rot)
    p3 = rotate_vector(p3_aligned, rot)

    return p1, p2, p3

# ---------------------------------------------------------------------------
# Vectorized (NumPy) versions: operate on whole arrays of events at once.
# ---------------------------------------------------------------------------
 
def dalitz_s23_limits_array(
    s12: np.ndarray, m_mother: float, m1: float, m2: float, m3: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized [s23_min, s23_max] for an array of s12 values.
 
    Same formula as `dalitz_s23_limits`, but without the range check
    (the caller is responsible for sampling s12 inside its physical range).
    """
    m12 = np.sqrt(s12)
    e2 = (s12 - m1**2 + m2**2) / (2.0 * m12)
    e3 = (m_mother**2 - s12 - m3**2) / (2.0 * m12)
 
    p2 = np.sqrt(np.maximum(0.0, e2**2 - m2**2))
    p3 = np.sqrt(np.maximum(0.0, e3**2 - m3**2))
 
    term_e = (e2 + e3) ** 2
    return term_e - (p2 + p3) ** 2, term_e - (p2 - p3) ** 2
 
 
def _random_rotation_matrices(n: int, rng: np.random.Generator) -> np.ndarray:
    """n uniform random 3D rotation matrices, shape (n, 3, 3) (Euler z-y-z)."""
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n)
    cos_t = rng.uniform(-1.0, 1.0, size=n)
    psi = rng.uniform(0.0, 2.0 * math.pi, size=n)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t**2))
 
    c1, s1 = np.cos(phi), np.sin(phi)
    c2, s2 = cos_t, sin_t
    c3, s3 = np.cos(psi), np.sin(psi)
 
    rot = np.empty((n, 3, 3), dtype=np.float64)
    rot[:, 0, 0] = c1 * c2 * c3 - s1 * s3
    rot[:, 0, 1] = -c1 * c2 * s3 - s1 * c3
    rot[:, 0, 2] = c1 * s2
    rot[:, 1, 0] = s1 * c2 * c3 + c1 * s3
    rot[:, 1, 1] = -s1 * c2 * s3 + c1 * c3
    rot[:, 1, 2] = s1 * s2
    rot[:, 2, 0] = -s2 * c3
    rot[:, 2, 1] = s2 * s3
    rot[:, 2, 2] = c2
    return rot
 
 
def generate_3body_decay_array(
    m_mother: float, m1: float, m2: float, m3: float,
    s12: np.ndarray, s23: np.ndarray,
    rng: np.random.Generator,
    cos_tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized `generate_3body_decay`.
 
    Returns three arrays of shape (n, 4) with columns (E, px, py, pz), in the
    mother rest frame, with a uniformly random overall orientation.
    Raises ValueError if some (s12, s23) lies outside the Dalitz region
    (instead of silently clipping cos(theta) as the scalar version does).
    """
    s12 = np.asarray(s12, dtype=np.float64)
    s23 = np.asarray(s23, dtype=np.float64)
    n = s12.size
    m12 = np.sqrt(s12)
 
    # Momentum of the 12 system / of particle 3 in the mother rest frame
    p3_cm = np.sqrt(np.maximum(0.0, kallen_lambda(m_mother**2, s12, m3**2))) / (2.0 * m_mother)
    # Momentum of 1 and 2 in the 12 rest frame
    p2s = np.sqrt(np.maximum(0.0, kallen_lambda(s12, m1**2, m2**2))) / (2.0 * m12)
 
    e1s = (s12 + m1**2 - m2**2) / (2.0 * m12)
    e2s = (s12 - m1**2 + m2**2) / (2.0 * m12)
    e3s = (m_mother**2 - s12 - m3**2) / (2.0 * m12)
    p3s = np.sqrt(np.maximum(0.0, e3s**2 - m3**2))
 
    num = (e2s + e3s) ** 2 - s23 - p2s**2 - p3s**2
    den = 2.0 * p2s * p3s
    ok = den > 0.0
    cos_star = np.where(ok, num / np.where(ok, den, 1.0), 0.0)
 
    if np.any(np.abs(cos_star) > 1.0 + cos_tol):
        n_bad = int(np.sum(np.abs(cos_star) > 1.0 + cos_tol))
        raise ValueError(f"{n_bad} events outside the Dalitz region (|cos theta*| > 1).")
    cos_star = np.clip(cos_star, -1.0, 1.0)
    sin_star = np.sqrt(1.0 - cos_star**2)
    phi_star = rng.uniform(0.0, 2.0 * math.pi, size=n)
 
    # Particles 1, 2 in the 12 rest frame (back to back)
    p2x = p2s * sin_star * np.cos(phi_star)
    p2y = p2s * sin_star * np.sin(phi_star)
    p2z = p2s * cos_star
 
    # Boost along z to the mother frame: the 12 system moves with beta_z = -p3_cm / E12
    e12 = np.sqrt(m12**2 + p3_cm**2)
    gamma = e12 / m12
    beta_z = -p3_cm / e12
 
    p1 = np.empty((n, 4)); p2 = np.empty((n, 4)); p3 = np.empty((n, 4))
    p1[:, 0] = gamma * (e1s + beta_z * (-p2z));  p1[:, 1] = -p2x;  p1[:, 2] = -p2y
    p1[:, 3] = gamma * (-p2z + beta_z * e1s)
    p2[:, 0] = gamma * (e2s + beta_z * p2z);     p2[:, 1] = p2x;   p2[:, 2] = p2y
    p2[:, 3] = gamma * (p2z + beta_z * e2s)
    p3[:, 0] = np.sqrt(m3**2 + p3_cm**2);        p3[:, 1] = 0.0;   p3[:, 2] = 0.0
    p3[:, 3] = p3_cm
 
    # Uniform random orientation of the whole event
    rot = _random_rotation_matrices(n, rng)
    for p in (p1, p2, p3):
        p[:, 1:] = np.einsum("nij,nj->ni", rot, p[:, 1:])
 
    return p1, p2, p3
 