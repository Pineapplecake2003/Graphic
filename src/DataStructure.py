import numpy as np
import copy
class Point():
    canva_loc:np.ndarray
    world_loc:np.ndarray
    b:float

    def __init__(self, world_loc, brightness:float):
        if isinstance(world_loc, list):
            self.world_loc = np.array(world_loc, dtype=np.float32, copy=True)
        elif isinstance(world_loc, np.ndarray):
            self.world_loc = world_loc.astype(np.float32, copy=True)
        else:
            self.world_loc = np.array(world_loc, dtype=np.float32, copy=True)

        self.canva_loc = np.array([0., 0., 0.])

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
    points_idx:list
    vns_idx:list
    def __init__(self, points_idx, vts_idx, vns_idx):
        self.points_idx = points_idx
        self.vns_idx = vns_idx

    def __str__(self):
        return f"points index: {self.points_idx}\n vns index: {self.vns_idx}"

class ThreeDimensionObject():
    """
    triangles: list of triangles
    points_original: nparray (3, N)
    vns_original: nparray (3, N)
    """
    triangles:list
    points_original:list
    points_transformed:list
    vns_original:list
    vns_transformed:list
    s:float
    def __init__(self, triangles:list, points:list, vns:list):
        self.triangles= triangles
        
        self.points_original = points
        self.points_transformed = copy.deepcopy(self.points_original)

        self.vns_original = vns
        self.vns_transformed = copy.deepcopy(self.vns_original)

        self.s = 10

    def transform(self, location:tuple, rotation:tuple, scale):
        """
        input rotation: degree
        """
        alpha, beta, gamma = rotation
        a, b, g = np.deg2rad([alpha, beta, gamma])
        r_x = np.array(
            [
                [1., 0.         , 0.        ,0.],
                [0., np.cos(a)  , -np.sin(a),0.],
                [0., np.sin(a)  , np.cos(a) ,0.],
                [0., 0.         , 0.        ,1.]
            ]
        )
        r_y = np.array(
            [
                [np.cos(b)  , 0., np.sin(b) ,0.],
                [0.         , 1., 0.        ,0.],
                [-np.sin(b) , 0., np.cos(b) ,0.],
                [0.         , 0., 0.        ,1.]
            ]
        )
        r_z = np.array(
            [
                [np.cos(g)  , -np.sin(g)    , 0., 0.],
                [np.sin(g)  , np.cos(g)     , 0., 0.],
                [0.         , 0.            , 1., 0.],
                [0.         , 0.            , 0., 1.]
            ]
        )
        scale_a = np.array(
            [
                [scale, 0.    , 0.    , 0.],
                [0.   , scale , 0.    , 0.],
                [0.   , 0.    , scale , 0.],
                [0.   , 0.    , 0.    , 1.]
            ]
        )
        offset_a = np.array(
            [
                [1., 0., 0., location[0]],
                [0., 1., 0., location[1]],
                [0., 0., 1., location[2]],
                [0., 0., 0., 1.         ]
            ]
        )
        transform_mat = offset_a @ r_z @ r_y @ r_x @ scale_a
        
        for p in self.points_transformed:
            original_loc = original_loc = np.append(p.world_loc, 1.0)
            transformed_loc = transform_mat @ original_loc
            p.world_loc = transformed_loc[:3].copy()
        
        R_only = r_z @ r_y @ r_x
        for i, vn in enumerate(self.vns_transformed):
            rotated_vn = R_only @ np.append(vn, 0.0) 
            norm = np.linalg.norm(rotated_vn)
            if norm > 1e-6:
                rotated_vn = rotated_vn[:3] / norm
            else:
                rotated_vn = rotated_vn[:3]
            self.vns_transformed[i] = rotated_vn
    
    def set_s(self, s:float):
        self.s = s



