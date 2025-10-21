import numpy as np
class Point():
    location:np.ndarray
    b:float

    def __init__(self, location:list, brightness:float):
        self.location = np.array(location, dtype=np.float32)
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
    WHITE:int
    dpi:int
    def __init__(self, V:tuple, d:int, C:tuple, dpi:int, ambient:float, light_source:list, s=10):
        self.WHITE = 255
        self.array = np.full((C[0], C[1], 3), self.WHITE, dtype=np.uint8)
        self.z_inv_buf = np.full((C[0], C[1]), 0, dtype=np.float32)
        self.d = d
        self.C = C
        self.V = V
        self.ambient = ambient
        self.light_srouce = light_source
        self.s = s
        self.dpi = dpi
    
    def clear(self):
        self.array = np.full((self.C[0], self.C[1], 3), self.WHITE, dtype=np.uint8)
        self.z_inv_buf = np.full((self.C[0], self.C[1]), 0, dtype=np.float32)

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
        offset_loc = np.array(location, dtype=float)
        for p in self.points:
            original_loc = p.location
            transformed_loc = scale * original_loc
            transformed_loc = R @ transformed_loc
            transformed_loc = transformed_loc + offset_loc
            p.location = transformed_loc
            # TODO brightness change

    def reset(self):
        for p, base in zip(self.points, self.base_coords):
            p.x, p.y, p.z = base

    def transform_from_base(self, location:tuple, rotation:tuple, scale:float):
        self.reset()
        self.transform(location, rotation, scale)



