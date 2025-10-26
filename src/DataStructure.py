import numpy as np
class Point():
    loc:np.ndarray
    world_loc:np.ndarray
    b:float

    def __init__(self, loc, brightness:float, world_loc=None):
        if isinstance(loc, list):
            self.loc = np.array(loc, dtype=np.float32, copy=True)
        elif isinstance(loc, np.ndarray):
            self.loc = loc.astype(np.float32, copy=True)
        else:
            self.loc = np.array(loc, dtype=np.float32, copy=True)

        if world_loc is None:
            self.world_loc = self.loc.copy()
        elif isinstance(world_loc, list):
            self.world_loc = np.array(world_loc, dtype=np.float32, copy=True)
        elif isinstance(world_loc, np.ndarray):
            self.world_loc = world_loc.astype(np.float32, copy=True)
        else:
            self.world_loc = np.array(world_loc, dtype=np.float32, copy=True)

        self.b = brightness
    
    def __str__(self):
        return str(self.loc)

class Light(Point):
    li_dir:np.ndarray
    ltype:str

    def __init__(self, loc, brightness, ltype):
        super().__init__(loc, brightness)
        assert(ltype == "point" or ltype == "directional")
        
        self.ltype = ltype
        if ltype == "directional":
            if isinstance(loc, list):
                self.li_dir = np.array(loc, dtype=np.float32, copy=True)
            elif isinstance(loc, np.ndarray):
                self.li_dir = loc.astype(np.float32, copy=True)
        else:
            self.li_dir = np.array([0, 0, 0], dtype=np.float32)

class Canva():
    array:np.ndarray
    d:int
    C:tuple
    V:tuple
    z_inv_buf:np.ndarray
    light_srouce:list
    ambient:float
    s:float
    BACKGROUND:int
    dpi:int
    def __init__(self, V:tuple, d:int, C:tuple, dpi:int, ambient:float, light_source:list, s=10):
        self.BACKGROUND = 0
        self.array = np.full((C[0], C[1], 3), self.BACKGROUND, dtype=np.uint8)
        self.z_inv_buf = np.full((C[0], C[1]), 0, dtype=np.float32)
        self.d = d
        self.C = C
        self.V = V
        self.ambient = ambient
        self.light_srouce = light_source
        self.s = s
        self.dpi = dpi
    
    def clear(self):
        self.array = np.full((self.C[0], self.C[1], 3), self.BACKGROUND, dtype=np.uint8)
        self.z_inv_buf = np.full((self.C[0], self.C[1]), 0, dtype=np.float32)

    def __str__(self):
        return f"V: {self.V}\nd: {self.d}\nC: {self.C}"

class Triangle():
    points:list
    vns:list
    def __init__(self, points, vts, vns):
        self.points = points
        self.vns = vns

    def __str__(self):
        return f"{str(self.points[0])}, {str(self.points[1])}, {str(self.points[2])}"

class ThreeDimensionObject():
    triangles:list
    points:list
    s:float
    def __init__(self, triangles, points):
        self.triangles = triangles
        self.points = points
        self.s = 10

    def transform(self, location:tuple, rotation:tuple, scale):
        """
        input rotation: degree
        """
        # self.s = 10.0
        alpha, beta, gamma = rotation
        a, b, g = np.deg2rad([alpha, beta, gamma])
        r_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(a), -np.sin(a)],
                [0, np.sin(a), np.cos(a)]
            ],
            dtype=np.float32
        )
        r_y = np.array(
            [
                [np.cos(b), 0, np.sin(b)],
                [0, 1, 0],
                [-np.sin(b), 0, np.cos(b)]
            ],
            dtype=np.float32
        )
        r_z = np.array(
            [
                [np.cos(g), -np.sin(g), 0],
                [np.sin(g), np.cos(g), 0],
                [0, 0, 1]
            ],
            dtype=np.float32
        )
        R = r_z @ r_y @ r_x
        offset_loc = np.array(location, dtype=float)
        for p in self.points:
            original_loc = p.loc
            transformed_loc = scale * original_loc
            transformed_loc = R @ transformed_loc
            transformed_loc = transformed_loc + offset_loc
            p.loc = transformed_loc
            # TODO brightness change

        # if np.isscalar(scale):
        #     normal_matrix = R
        # else:
        #     normal_matrix = R

        for tri in self.triangles:
            transformed_normals = []
            for vn in tri.vns:
                vn_vec = np.asarray(vn, dtype=np.float32)
                rotated = R @ vn_vec
                norm = np.linalg.norm(rotated)
                if norm > 1e-6:
                    rotated = rotated / norm
                transformed_normals.append(rotated.astype(np.float32))
            tri.vns = transformed_normals
    
    def set_s(self, s:float):
        self.s = s



