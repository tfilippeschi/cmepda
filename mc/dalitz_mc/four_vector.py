import math
import numpy as np

class FourVector:
    def __init__(self, E, px, py, pz):
        self._E = E
        self._px = px
        self._py = py
        self._pz = pz
    
    @property
    def E(self):
        return self._E
    
    @property
    def px(self):
        return self._px
    
    @property
    def py(self):
        return self._py
    
    @property
    def pz(self):
        return self._pz
    
    @property
    def p_vec(self):
        return np.array([self._px, self._py, self._pz])
    
    @property
    def p2_spatial(self):
        return self._px ** 2 + self._py ** 2 + self._pz ** 2
    
    @property
    def p_spatial(self):
        return math.sqrt(self.p2_spatial)
    
    @property
    def beta_vec(self):
        return self.p_vec / self._E
    
    def dot(self, other : FourVector):
        return (
            self._E * other._E
            - self._px * other._px
            - self._py * other._py
            - self._pz * other._pz
            )
    
    def m2(self):
        return self.dot(self)
    
    def m(self):
        return math.sqrt(self.m2)
    
    def boost(self, bx, by, bz):
        b2 = bx ** 2 + by **2 + bz **2
        gamma = 1.0 / math.sqrt(1.0 - b2)
        gamma2 = gamma - 1.0
        bp = bx * self._px + by * self._py + bz * self._pz

        E_prime = gamma * (self._E + bp)
        px_prime = self._px + ((gamma2 / b2) * bp - gamma * self._E)
        py_prime = self._py + ((gamma2 / b2) * bp - gamma * self._E)
        pz_prime = self._pz + ((gamma2 / b2) * bp - gamma * self._E)

        return FourVector(E_prime, px_prime, py_prime, pz_prime)
    
    def boost_to_rest_frame_of(self, parent: FourVector):

        b = parent.beta_vec
        return self.boost(-b[0], -b[1], -b[2])
    
    def boost_from_rest_rest_of(self, parent: FourVector):

        b = parent.beta_vec
        return self.boost(b[0], b[1], b[2])
    
    def __add__(self, other: FourVector):
        return FourVector(
            self._E + other._E,
            self._px + other._px,
            self._py + other._py,
            self._pz + other._pz
        )
    
    def __sub__(self, other: FourVector):
        return FourVector(
            self._E - other._E,
            self._px - other._px,
            self._py - other._py,
            self._pz - other._pz
        )
    
    def __mul__(self, scalar):
        return FourVector(
            self._E * scalar,
            self._px * scalar,
            self._py * scalar,
            self._pz * scalar
        )

    __rmul__ = __mul__

    def __neg__(self):
        return FourVector(self._E, -self._px, -self._py, -self._pz)