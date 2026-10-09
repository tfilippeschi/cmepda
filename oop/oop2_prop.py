import math

class Vector2d:
    def __init__(self, r, phi):
        self._r = float(r)
        self._phi = float(phi)
    
    def __str__(self):
        return f"Cartesian coordinates: ({self.x}, {self.y})"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(x={self.x}, y={self.y})"
    
    @property
    def r(self):
        return self._r
    
    @r.setter
    def r(self, new_r):
        self._r = float(new_r)
    
    @property
    def y(self):
        return self._r * math.sin(self._phi)
    
    @property
    def x(self):
        return self._r * math.cos(self._phi)
    
    @x.setter
    def x(self, x):
        new_r = math.sqrt(x**2 + self.y**2)
        new_phi = math.atan2(self.y, x)
        self._r = new_r
        self._phi = new_phi

v = Vector2d(3., math.pi/2)
v.x = 5.
print(v)
print(repr(v))