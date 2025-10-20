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
        return f"({self.x}, {self.y}, {self.z})"

class Canva():
    array:np.ndarray
    d:int
    C:tuple
    V:tuple
    z_inv_buf:np.ndarray
    light_srouce:list
    ambient:float
    s:float
    def __init__(self, V:tuple, d:int, C:tuple, ambient:float, light_source:list, s=10):
        WHITE = 255
        self.array = np.full((C[0], C[1], 3), 255, dtype=np.uint8)
        self.z_inv_buf = np.full((C[0], C[1]), 0, dtype=np.float32)
        self.d = d
        self.C = C
        self.V = V
        self.ambient = ambient
        self.light_srouce = light_source
        self.s = s
    
    def __str__(self):
        return f"V: {self.V}\nd: {self.d}\nC: {self.C}"

class Triangles():
    points:list
    def __init__(self, p0, p1, p2):
        self.points = [p0, p1, p2]

    def __str__(self):
        return f"{str(self.points[0])}, {str(self.points[1])}, {str(self.points[2])}"

class ThreeDimensionObject():
    triangles:list
    points:list

    def __init__(self, triangles, points):
        self.triangles = triangles
        self.points = points

    def transform(self, location:tuple, rotation:tuple, scale):
        """
        input rotation: degree
        """
        alpha, beta, gamma = rotation
        a, b, g = np.deg2rad([alpha, beta, gamma])
        r_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(a), -np.sin(a)],
                [0, np.sin(a), np.cos(a)]
            ],
            dtype=float
        )
        r_y = np.array(
            [
                [np.cos(b), 0, np.sin(b)],
                [0, 1, 0],
                [-np.sin(b), 0, np.cos(b)]
            ],
            dtype=float
        )
        r_z = np.array(
            [
                [np.cos(g), -np.sin(g), 0],
                [np.sin(g), np.cos(g), 0],
                [0, 0, 1]
            ],
        )
        R = r_z @ r_y @ r_x
        for p in self.points:
            offset_loc = np.array([location[0], location[1], location[2]], dtype=float)
            original_loc = np.array([p.x, p.y, p.z], dtype=float)
            transformed_loc = scale * original_loc
            transformed_loc = R @ transformed_loc
            transformed_loc = transformed_loc + offset_loc
            p.x, p.y, p.z = transformed_loc[0], transformed_loc[1], transformed_loc[2]
            # TODO brightness change



