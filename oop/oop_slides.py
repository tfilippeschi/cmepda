import math

class Vector2d:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    
    def info(self):
            print(f"Vector2d(x={self.x}, y={self.y})")
    
    # Better than typing v.info()
    def __repr__(self):
        return f"{self.__class__.__name__} ({self.x}, {self.y})"
    
    # Better than typing v.info(), similar to __repr__ but for end users
    def __str__(self):
        return str((self.x, self.y))
    
    def module(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    # Better than typing v.module()
    def __abs__(self):
            return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def add(self, other):
        return Vector2d(self.x + other.x, self.y + other.y)
    
    # just type v1 + v2 instead of v1.add(v2)
    def __add__(self, other):
        return Vector2d(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector2d(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar):
        return self * scalar
    
    # In-place addition, modifies the current instance instead of creating a new one
    # This is useful for performance reasons when you want to update the vector without creating a new object.
    # Can be used like this: v1 += v2
    # Can do the same with imul.
    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self
    
    # Implement the == operator
    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y)
    
    # Implement the >= operator
    def __ge__(self, other):
        return abs(self) >= abs(other)
    
    def __lt__(self, other):
        return abs(self) < abs(other)