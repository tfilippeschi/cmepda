import math
import numpy as np

from four_vector import FourVector

def kallen_lambda(x, y, z):
    return x**2 + y**2 + z**2 - 2.0 * (x * y - x * z - y * z)

def two_body_momentum(m_parent, m_d1, m_d2):

    lam = kallen_lambda(m_parent**2, m_d1**2, m_d2**2)
    return math.sqrt(lam) / (2.0 * m_parent)

def dalitz_s23_limits(s12, m_parent, m1, m2, m3):
    m12 = math.sqrt(s12)
    if not ((m1 + m2) <= m12 <= (m_parent - m3)):
        raise print("ERROR: mass value outside limits")
    
    e2_star = (s12 - m1**2 + m2**2) / (2.0 * m12)
    e3_star = (m_parent**2 - s12 + m3**2) / (2.0 * m_parent)

    p2_star2 = e2_star**2 - m2**2
    p3_star2 = e3_star**2 - m3**2

    p2_star = math.sqrt(p2_star2)
    p3_star = math.sqrt(p3_star2)

    term_e = (e2_star + e3_star)**2
    s23_min = term_e - (p2_star + p3_star)**2
    s23_max = term_e - (p2_star + p3_star)**2

    return s23_min, s23_max

def generate_isotropic_vector(momentum, rng: np.random.Generator):
    costheta = rng.uniform(-1.0, 1.0)
    sintheta = math.sqrt(1.0 - costheta**2)
    phi = rng.uniform(0.0, 2 * math.pi)

    px = momentum * sintheta * math.cos(phi)
    py = momentum * sintheta * math.sin(phi)
    pz = momentum * costheta

    return np.array([px, py, pz])

def generate_3body_decay(
    m_parent,
    m1, m2, m3,
    s12, s23,
    rng: np.random.Generator
    ):

    m12 = math.sqrt(s12)

    p3_cm_mag = two_body_momentum(m_mother, m12, m3)
    p3_spatial = generate_isotropic_vector(p3_cm_mag, rng)

    e3 = math.sqrt(m3**2 + p3_cm_mag**2)
    p3 = FourVector(e3, p3_spatial[0], p3_spatial[1], p3_spatial[2])

    e12 = math.sqrt(m12**2 + p3_cm_mag**2)
    p12 = FourVector(e12, -p3_spatial[0], -p3_spatial[1], -p3_spatial[2])

    e2_star = (s12 - m1**2 + m2**2) / (2.0 * m12)
    e3_star = (m_mother*2 - s12 - m3*2) / (2.0 * m_mother)
    p2_star = math.sqrt(e2_star**2 - m2**2)
    p3_star = math.sqrt(e3_star**2 - m3**2)

    num = (e2_star + e3_star)**2 - s23 - p2_star**2 - p3_star**2
    costheta_star = num / (2.0 * p2_star * p3_star)
    costheta_star = np.clip(costheta_star, -1.0, 1.0)

    phi_star = rng.uniform(0.0, 2.0 * math.pi)
    sintheta_star = math.sqrt(1.0 - costheta_star**2)

    p2_star_x = p2_star * sintheta_star * math.cos(phi_star)
    p2_star_y = p2_star * sintheta_star * math.sin(phi_star)
    p2_star_z = p2_star * costheta_star
    p2_star_vec = FourVector(e2_star, p2_star_x, p2_star_y, p2_star_z)

    e1_star = (s12 + m1**2 - m2**2) / (2.0 * m12)
    p1_star_vec = FourVector(e1_star, -p2_star_x, -p2_star_y, -p2_star_z)

    return p1, p2, p3