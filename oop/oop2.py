import math

class Vector2d:
    def __init__(self, r, phi):
        self._r = float(r)
        self._phi = float(phi)
    
    def r(self):
        return self._r
    
    def phi(self):
        return self._phi
    
    def x(self):
        return self._r * math.cos(self._phi)
    
    def y(self):
        return self._r * math.sin(self._phi)

    def set_r(self, new_r):
        new_r = float(new_r)
        if new_r < 0:
            print("Error: r cannot be negative.")
        else:
            self._r = new_r

    def rotate_counter_clock(self, theta):
        self._phi += theta

v = Vector2d(3., 0.5 * math.pi)
print(v.r())
print(v.x())
print(v.y())
v.set_r(-5.)
v.rotate_counter_clock(0.5 * math.pi)
print(v.phi())