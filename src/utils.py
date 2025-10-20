import numpy as np
from DataStructure import *
import copy
import math
def PutPixel(x:int, y:int, z:int, canva:Canva, color:tuple):
    canva_height = canva.array.shape[0] - 1
    canva_weight = canva.array.shape[1] - 1
    x_idx = x + canva_weight // 2
    y_idx = canva_height - y - canva_height // 2
    if((y_idx > canva_height or y_idx < 0) or (x_idx > canva_weight or x_idx < 0)): return
    z_inv = 1 / z
    if(z_inv > canva.z_inv_buf[y_idx][x_idx]):
        canva.z_inv_buf[y_idx][x_idx] = z_inv
        canva.array[y_idx][x_idx][0] = color[0]
        canva.array[y_idx][x_idx][1] = color[1]
        canva.array[y_idx][x_idx][2] = color[2]

def get_light_for_point(p:Point, canva:Canva):
    pass

def Interpolate(i0, d0, i1, d1):
    if(i0 == i1):
        return [d0]
    values = []
    a = (d1 - d0) / (i1 - i0)
    d = d0
    for i in range(i0, i1):
        values.append(d)
        d = d + a
    return values

def DrawLine(P0:Point, P1:Point, canva:Canva, color:tuple):
    """
    input: P0, P1, canva
        pass by value
    return: drawn canva
    """
    canva_height = canva.array.shape[0] - 1
    canva_weight = canva.array.shape[1] - 1
    points = [copy.deepcopy(P0), copy.deepcopy(P1)]
    if(abs(points[1].x - points[0].x) > abs(points[1].y - points[0].y)):
        # Horizontal line
        # Make sure x0 < x1
        if(points[0].x > points[1].x):
            temp = points[1]
            points[1] = points[0]
            points[0] = temp
        ys = Interpolate(points[0].x, points[0].y, points[1].x, points[1].y)
        for x in range(points[0].x, points[1].x):
            canva.array[canva_height - int(ys[x - points[0].x])][x][0] = color[0] # R
            canva.array[canva_height - int(ys[x - points[0].x])][x][1] = color[1] # G
            canva.array[canva_height - int(ys[x - points[0].x])][x][2] = color[2] # B
    else:
        # Vertical line
        # Make sure y0 < y1
        if(points[0].y > points[1].y):
            temp = points[1]
            points[1] = points[0]
            points[0] = temp
        xs = Interpolate(points[0].y, points[0].x, points[1].y, points[1].x)
        for y in range(points[0].y, points[1].y):
            # print(canva_height - y)
            # print(canva_width - int(xs[y - P0.y]))
            canva.array[canva_height - y][int(xs[y - points[0].y])][0] = color[0] # R
            canva.array[canva_height - y][int(xs[y - points[0].y])][1] = color[1] # G
            canva.array[canva_height - y][int(xs[y - points[0].y])][2] = color[2] # B

def DrawShadedLine(P0:Point, P1:Point, canva:Canva, color:tuple):
    """
    input: P0, P1, canva
        pass by value
    return: drawn canva
    """
    points = [copy.deepcopy(P0), copy.deepcopy(P1)]
    if(abs(points[1].x - points[0].x) > abs(points[1].y - points[0].y)):
        # Horizontal line
        # Make sure x0 < x1
        if(points[0].x > points[1].x):
            temp = points[1]
            points[1] = points[0]
            points[0] = temp
        ys = Interpolate(points[0].x, points[0].y, points[1].x, points[1].y)
        zs = Interpolate(points[0].x, points[0].z, points[1].x, points[1].z)
        hs = Interpolate(points[0].x, points[0].b, points[1].x, points[1].b)
        for x in range(points[0].x, points[1].x):
            PutPixel(x, int(ys[x - points[0].x]), int(zs[x - points[0].x]), canva, 
                        (
                            int(color[0] * hs[x - points[0].x]),
                            int(color[1] * hs[x - points[0].x]),
                            int(color[2] * hs[x - points[0].x])
                        )
                    )
    else:
        # Vertical line
        # Make sure y0 < y1
        if(points[0].y > points[1].y):
            temp = points[1]
            points[1] = points[0]
            points[0] = temp
        xs = Interpolate(points[0].y, points[0].x, points[1].y, points[1].x)
        zs = Interpolate(points[0].y, points[0].z, points[1].y, points[1].z)
        hs = Interpolate(points[0].y, points[0].b, points[1].y, points[1].b)
        for y in range(points[0].y, points[1].y):
            PutPixel(int(xs[y - points[0].y]), y, int(zs[y - points[0].y]), canva,
                        (
                            int(color[0] * hs[y - points[0].y]),
                            int(color[1] * hs[y - points[0].y]),
                            int(color[2] * hs[y - points[0].y])
                        )
                    )


def DrawFilledTriangle(P0:Point, P1:Point, P2:Point, canva:Canva, color):
    points =[copy.deepcopy(P0), copy.deepcopy(P1), copy.deepcopy(P2)]
    canva_height = canva.array.shape[0]
    # Sort points depended on y
    if(points[1].y < points[0].y):
        temp = points[1]
        points[1] = points[0]
        points[0] = temp
    
    if(points[2].y < points[0].y):
        temp = points[2]
        points[2] = points[0]
        points[0] = temp
    
    if(points[2].y < points[1].y):
        temp = points[2]
        points[2] = points[1]
        points[1] = temp
    
    x01 = Interpolate(points[0].y, points[0].x, points[1].y, points[1].x)
    x12 = Interpolate(points[1].y, points[1].x, points[2].y, points[2].x)
    x02 = Interpolate(points[0].y, points[0].x, points[2].y, points[2].x)
    x012 = x01
    x012.extend(x12)

    m = math.floor(len(x012) / 2)
    if(x02[m] < x012[m]):
        x_left = x02
        x_right = x012
    else:
        x_left = x012
        x_right = x02
    
    for y in range(points[0].y, points[2].y):
        for x in range(int(x_left[y - points[0].y]) + 1, int(x_right[y - points[0].y])):
            pass #PutPixel(x, y, canva, color)

def DrawShadedTriangle (P0:Point, P1:Point, P2:Point, canva:Canva, color):
    points = [copy.deepcopy(P0), copy.deepcopy(P1), copy.deepcopy(P2)]

    if(points[0].y == points[1].y and points[1].y == points[2].y):
        return
    
    canva_height = canva.array.shape[0]
    # Sort points depended on y
    if(points[1].y < points[0].y):
        temp = points[1]
        points[1] = points[0]
        points[0] = temp
    
    if(points[2].y < points[0].y):
        temp = points[2]
        points[2] = points[0]
        points[0] = temp
    
    if(points[2].y < points[1].y):
        temp = points[2]
        points[2] = points[1]
        points[1] = temp
    
    x01 = Interpolate(points[0].y, points[0].x, points[1].y, points[1].x)
    x12 = Interpolate(points[1].y, points[1].x, points[2].y, points[2].x)
    x02 = Interpolate(points[0].y, points[0].x, points[2].y, points[2].x)
    
    h01 = Interpolate(points[0].y, points[0].b, points[1].y, points[1].b)
    h12 = Interpolate(points[1].y, points[1].b, points[2].y, points[2].b)
    h02 = Interpolate(points[0].y, points[0].b, points[2].y, points[2].b)
    x012 = x01
    x012.extend(x12)
    h012 = h01
    h012.extend(h12)

    m = math.floor(len(x012) / 2)
    if(x02[m] < x012[m]):
        x_left = x02
        x_right = x012
        h_left = h02
        h_right = h012
    else:
        x_left = x012
        x_right = x02
        h_left = h012
        h_right = h02
    
    for y in range(points[0].y, points[2].y):
        x_l = x_left[y - points[0].y]
        x_r = x_right[y - points[0].y]
        h_segment = Interpolate(int(x_l), h_left[y - points[0].y], 
                                int(x_r)+1, h_right[y - points[0].y])
        for x in range(int(x_l), int(x_r)+1):
            b = h_segment[x - int(x_l)]
            shaded_color = (
                int(color[0] * b), 
                int(color[1] * b), 
                int(color[2] * b)
            )
            # PutPixel(x, y, canva, shaded_color)
            # canva.array[canva_height - y][x][0] = shaded_color[0]
            # canva.array[canva_height - y][x][1] = shaded_color[1]
            # canva.array[canva_height - y][x][2] = shaded_color[2]

def ProjectToCanvas(P:Point, canva:Canva):
    x = P.x * canva.d / P.z
    y = P.y * canva.d / P.z
    z = canva.d
    projected_p = Point(x, y, z, P.b)
    projected_p.x = int(projected_p.x * canva.C[1] / canva.V[1])
    projected_p.y = int(projected_p.y * canva.C[0] / canva.V[0])
    return projected_p

def shift_location(P:Point, shifts:tuple, scale):
    return Point(scale * (P.x + shifts[0]), scale * (P.y + shifts[1]), scale * (P.z + shifts[2]), P.b)

def DrawWireframeTriangle(
        P0:Point, 
        P1:Point,
        P2:Point, 
        canva:Canva, 
        line_color:tuple, 
        filled_color:tuple
    ):
    mid_point = np.array(
        [
            (P0.x + P1.x + P2.x) / 3,
            (P0.y + P1.y + P2.y) / 3,
            (P0.z + P1.z + P2.z) / 3
        ],
        dtype=float
    )
    n_vector = np.cross(
        np.array(
            [
                P1.x - P0.x,
                P1.y - P0.y,
                P1.z - P0.z
            ],
            dtype=float
        ),
        np.array(
            [
                P2.x - P0.x,
                P2.y - P0.y,
                P2.z - P0.z
            ],
            dtype=float
        )
    )
    if (np.dot(n_vector, -mid_point) <= 0):
        return

    p0 = copy.deepcopy(P0)
    p1 = copy.deepcopy(P1)
    p2 = copy.deepcopy(P2)
    
    p0 = ProjectToCanvas(p0, canva)
    p1 = ProjectToCanvas(p1, canva)
    p2 = ProjectToCanvas(p2, canva)
    
    DrawShadedLine(p0, p1, canva, line_color)
    DrawShadedLine(p1, p2, canva, line_color)
    DrawShadedLine(p2, p0, canva, line_color)
    # DrawShadedTriangle(P0, P1, P2, canva, filled_color)

def load_objs(path:str):
    with open(path) as f:
        lines = f.readlines()
    
    points = []
    tris = []
    for line in lines:
        if(line[0] == "v" and line[1] == ' '):
            splited = line.split(' ')
            if '' in splited:
                splited.remove('')
            point = Point(float(splited[1]), float(splited[2]), float(splited[3]), 1.0)
            points.append(point)
        elif(line[0] == 'f'):
            splited = line.split(' ')
            if '' in splited:
                splited.remove('')
            tri = Triangles(
                points[int(splited[1].split('/')[0]) - 1], 
                points[int(splited[2].split('/')[0]) - 1], 
                points[int(splited[3].split('/')[0]) - 1])
            tris.append(tri)
    print(f"Number of triangles: {len(tris)}")
    obj = ThreeDimensionObject(tris, points)
    return obj