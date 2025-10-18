import numpy as np
class Point():
    x:int
    y:int
    z:int
    b:float

    def __init__(self, x:int, y:int, z:int, brightness:float):
        self.x = x
        self.y = y
        self.z = z
        self.b = brightness
    
    def __str__(self):
        return f"({self.x}, {self.y})"

class Canva():
    array:np.ndarray
    d:int
    C:tuple
    V:tuple

    def __init__(self, V:tuple, d:int, C:tuple):
        WHITE = 255
        self.array = np.full((C[0], C[1], 3), 255, dtype=np.uint8)
        self.d = d
        self.C = C
        self.V = V
    
    def __str__(self):
        return f"V: {self.V}\nd: {self.d}\nC: {self.C}"