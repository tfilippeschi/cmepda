"""Module for the implementation of 4-vector algebra with metrics (+, -, -, -)."""

from __future__ import annotations
import math
from typing import Union
import numpy as np

class FourVector:
    """Representation of a 4-vector p = (E, px, py, pz).

    Metrics used is (+, -, -, -).
    Natural units (c = 1).
    """

    __slots__ = ("_e", "_px", "_py", "_pz")

    def __init__(self, e: float, px: float, py: float, pz: float):
        self._e = float(e)
        self._px = float(px)
        self._py = float(py)
        self._pz = float(pz)
    
    @classmethod
    def from_spatial(cls, m: float, px: float, py: float, pz: float) -> FourVector:
        """Builds a FourVector on-shell starting from mass and spatial momentum."""
        p2 = px**2 + py**2 + pz**2
        e = math.sqrt(m**2 + p2)
        return cls(e, px, py, pz)
    
    @property
    def e(self) -> float:
        return self._e
    
    @property
    def px(self) -> float:
        return self._px
    
    @property
    def py(self) -> float:
        return self._py
    
    @property
    def pz(self) -> float:
        return self._pz
    
    @property
    def p_vec(self) -> np.darray:
        """Spatial momentum 3-D vector."""
        return np.array([self._px, self._py, self._pz], dtype=np.float64)
    
    @property
    def p2_spatial(self) -> float:
        """Square modulus of spatial momentum."""
        return self._px**2 + self._py**2 + self._pz**2
    
    @property
    def p_spatial(self) -> float:
        """Modulus of spatial momentum."""
        return math.sqrt(self.p2_spatial)
    
    @property
    def pt(self) -> float:
        """Transverse momentum."""
        return math.hypot(self._px, self._py)
    
    @property
    def beta_vec(self) -> np.darray:
        """Velocity vector beta = v / c = p / E."""
        if self._e == 0.0:
            raise ValueError("Null energy: cannot compute beta vector.")
        return self.p_vec / self._e
    
    def dot(self, other : FourVector) -> float:
        """Lorentz scalar product p . q = E_p * E_q - vec(p) . vec(q)."""
        return (
            self._e * other._e
            - self._px * other._px
            - self._py * other._py
            - self._pz * other._pz
            )
    
    def m2(self) -> float:
        """Square invariant mass: m^2 = E^2 - |p|^2"""
        return self.dot(self)
    
    def m(self) -> float:
        "Invariant mass. Numerical tollerance around zero."
        val = self.m2()
        if val < 0.0:
            if val > -1e-12:
                return 0.0
            raise ValueError(
                f"m^2 = {val:.4e} < 0 : mass not defined."
            )
        return math.sqrt(val)
    
    def boost(self, bx: float, by: float, bz: float) -> FourVector:
        """Applies Lorentz boost with velocity (bx, by, bz)."""
        b2 = bx**2 + by**2 + bz**2
        if b2 == 0.0:
            return FourVector(self._e, self._px, self._py, self._pz)
        if b2 >= 1.0:
            raise ValueError(f"beta cannot be > 1.0.")
        
        gamma = 1.0 / math.sqrt(1.0 - b2)
        bp = bx * self._px + by * self._py + bz * self._pz
        gamma2 = (gamma - 1.0) / b2

        e_prime = gamma * (self._e + bp)
        px_prime = self._px + (gamma2 * bp + gamma * self._e) * bx
        py_prime = self._py + (gamma2 * bp + gamma * self._e) * by
        pz_prime = self._pz + (gamma2 * bp + gamma * self._e) * bz

        return FourVector(e_prime, px_prime, py_prime, pz_prime)
    
    def boost_to_rest_frame_of(self, parent: FourVector) -> FourVector:
        """Transorms this 4-vector in the rest frame of FourVector 'parent'."""
        b = parent.beta_vec
        return self.boost(-b[0], -b[1], -b[2])
    
    def boost_from_rest_frame_of(self, parent: FourVector) -> FourVector:
        """Transforms this 4-vector from the rest frame of 'parent' to lab."""
        b = parent.beta_vec
        return self.boost(b[0], b[1], b[2])
    
    def __add__(self, other: FourVector) -> FourVector:
        return FourVector(
            self._e + other._e,
            self._px + other._px,
            self._py + other._py,
            self._pz + other._pz
        )
    
    def __sub__(self, other: FourVector) -> FourVector:
        return FourVector(
            self._e - other._e,
            self._px - other._px,
            self._py - other._py,
            self._pz - other._pz
        )
    
    def __mul__(self, scalar) -> FourVector:
        return FourVector(
            self._e * scalar,
            self._px * scalar,
            self._py * scalar,
            self._pz * scalar
        )

    __rmul__ = __mul__

    def __neg__(self) -> FourVector:
        return FourVector(self._e, -self._px, -self._py, -self._pz)
    
    def __repr__(self) -> str:
        return f"FourVector(E={self._e:.4f}, px={self._px:.4f}, py={self._py:.4f}, pz={self._pz:.4f}, M={self.m():.4f})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FourVector):
            return NotImplemented
        return math.isclose(self._e, other._e) and \
               math.isclose(self._px, other._px) and \
               math.isclose(self._py, other._py) and \
               math.isclose(self._pz, other._pz)