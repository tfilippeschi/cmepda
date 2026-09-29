import numpy as np

class Vector2d:

    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def norm(self):
        return np.sqrt((self.x ** 2 + self.y ** 2))
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.x}, {self.y})"


if __name__ == 'main':
    v = Vector2d(1., 1.)
    print(v)
    print(v.x, v.y)
    print(v.norm())