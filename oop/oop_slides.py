import math

class Vector2d:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    
    def __abs__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def module(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def info(self):
        print(f"Vector2d(x={self.x}, y={self.y})")
    
    def add(self, other):
        return Vector2d(self.x + other.x, self.y + other.y)
    
    v = Vector2d(2., 1.)
    v.info()
    print(abs(v))
    print(v.module())
    z = Vector2d(1., 2.)
    t = v.add(z)
    t.info()