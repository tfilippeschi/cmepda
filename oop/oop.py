import numpy as np

class Vector3d:

    def __init__(self, x: float, y: float, z: float):
        print("calling __init__")
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def norm(self):
        return np.sqrt((self.x ** 2 + self.y ** 2 + self.z ** 2))
    
    def dot(self, other: "Vector3d"):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def vector_product(self, other: "Vector3d"):
        return Vector3d(self.y * other.z - self.z * other.y,
                        self.x * other.z - self.z * other.x,
                        self.x * other.y - self.y * other.x)

    def __str__(self):
        return f"{self.__class__.__name__}({self.x}, {self.y}, {self.z})"

if __name__ == "__main__":
    n1 = Vector3d(1., 1., 0.)
    print(n1)
    n2 = Vector3d(1., -1., 0.)
    print(n2)
    print(n1.dot(n2))
    print(n1.vector_product(n2))
    # print(v.x, v.y, v.z)
    # print(v.norm())