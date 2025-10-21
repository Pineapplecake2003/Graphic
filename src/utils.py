import numpy as np
from DataStructure import *
import copy
import math
def PutPixel(x:int, y:int, z:float, canva:Canva, color:tuple):
    canva_height = canva.array.shape[0] - 1
    canva_weight = canva.array.shape[1] - 1
    x_idx = x + canva_weight // 2
    y_idx = canva_height - y - canva_height // 2
    if((y_idx > canva_height or y_idx < 0) or (x_idx > canva_weight or x_idx < 0)): return
    if z <= 0:
        return
    z_inv = 1.0 / z
    if(z_inv > canva.z_inv_buf[y_idx][x_idx]):
        canva.z_inv_buf[y_idx][x_idx] = z_inv
        canva.array[y_idx][x_idx][0] = 255 if color[0] >= 255 else color[0]
        canva.array[y_idx][x_idx][1] = 255 if color[1] >= 255 else color[1]
        canva.array[y_idx][x_idx][2] = 255 if color[2] >= 255 else color[2]

def get_light_for_triangle(t: Triangles, canva: Canva):
    # Ambient term
    I_p = canva.ambient
    for p in t.points:
        p.b = I_p

    # Face normal
    n_vector = np.cross(
        t.points[1].loc - t.points[0].loc,
        t.points[2].loc - t.points[0].loc
    )
    n_vector /= np.linalg.norm(n_vector)

    # Lighting loop
    for li in canva.light_srouce:
        for p in t.points:
            l_vector = li.loc - p.loc
            l_vector /= np.linalg.norm(l_vector)

            v_vector = -p.loc
            v_vector /= np.linalg.norm(v_vector)

            r_vector = 2 * np.dot(n_vector, l_vector) * n_vector - l_vector
            r_vector /= np.linalg.norm(r_vector)

            # Diffuse + Specular
            shooted = max(np.dot(n_vector, l_vector), 0.0)
            reflected = max(np.dot(r_vector, v_vector), 0.0) ** canva.s

            p.b += li.b * (shooted + reflected)

def Interpolate(i0, d0, i1, d1):
    if(i0 == i1):
        return [d0]
    values = []
    a = (d1 - d0) / (i1 - i0)
    d = d0
    for i in range(int(i0), int(i1)):
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
            canva.array[canva_height - y][int(xs[y - points[0].y])][0] = color[0] # R
            canva.array[canva_height - y][int(xs[y - points[0].y])][1] = color[1] # G
            canva.array[canva_height - y][int(xs[y - points[0].y])][2] = color[2] # B

def DrawShadedLine(P0:Point, P1:Point, canva:Canva, color:tuple):
    """
    input: P0, P1, canva
        pass by value
    return: drawn canva
    """
    points = [Point(P0.loc, P0.b), Point(P1.loc, P1.b)]
    if abs(points[1].loc[0] - points[0].loc[0]) > abs(points[1].loc[1] - points[0].loc[1]):
        # Horizontal line
        # Make sure x0 < x1
        if points[0].loc[0] > points[1].loc[0]:
            points[0], points[1] = points[1], points[0]
        x0 = int(round(points[0].loc[0]))
        x1 = int(round(points[1].loc[0]))
        if x0 == x1:
            return
        ys = Interpolate(x0, points[0].loc[1], x1, points[1].loc[1])
        zs = Interpolate(x0, points[0].loc[2], x1, points[1].loc[2])
        hs = Interpolate(x0, points[0].b,     x1, points[1].b)
        for idx, x in enumerate(range(x0, x1)):
            PutPixel(
                x,
                int(round(ys[idx])),
                zs[idx],
                canva,
                (
                    int(color[0] * hs[idx]),
                    int(color[1] * hs[idx]),
                    int(color[2] * hs[idx])
                )
            )
    else:
        # Vertical line
        if points[0].loc[1] > points[1].loc[1]:
            points[0], points[1] = points[1], points[0]
        y0 = int(round(points[0].loc[1]))
        y1 = int(round(points[1].loc[1]))
        if y0 == y1:
            return
        xs = Interpolate(y0, points[0].loc[0], y1, points[1].loc[0])
        zs = Interpolate(y0, points[0].loc[2], y1, points[1].loc[2])
        hs = Interpolate(y0, points[0].b,     y1, points[1].b)
        for idx, y in enumerate(range(y0, y1)):
            PutPixel(
                int(round(xs[idx])),
                y,
                zs[idx],
                canva,
                (
                    int(color[0] * hs[idx]),
                    int(color[1] * hs[idx]),
                    int(color[2] * hs[idx])
                )
            )


def DrawShadedTriangle (p0, p1, p2, canva:Canva, color):
    points = [p0, p1, p2]

    if(points[0].loc[1] == points[1].loc[1] and points[1].loc[1] == points[2].loc[1]):
        return
    
    # Sort points depended on y
    if(points[1].loc[1] < points[0].loc[1]):
        temp = points[1]
        points[1] = points[0]
        points[0] = temp
    
    if(points[2].loc[1] < points[0].loc[1]):
        temp = points[2]
        points[2] = points[0]
        points[0] = temp
    
    if(points[2].loc[1] < points[1].loc[1]):
        temp = points[2]
        points[2] = points[1]
        points[1] = temp
    
    y0 = int(round(points[0].loc[1]))
    y1 = int(round(points[1].loc[1]))
    y2 = int(round(points[2].loc[1]))

    x01 = Interpolate(y0, points[0].loc[0], y1, points[1].loc[0])
    x12 = Interpolate(y1, points[1].loc[0], y2, points[2].loc[0])
    x02 = Interpolate(y0, points[0].loc[0], y2, points[2].loc[0])
    if len(x02) <= 1:
        return
    x012 = x01
    x012.extend(x12)

    h01 = Interpolate(y0, points[0].b, y1, points[1].b)
    h12 = Interpolate(y1, points[1].b, y2, points[2].b)
    h02 = Interpolate(y0, points[0].b, y2, points[2].b)
    h012 = h01
    h012.extend(h12)


    z01 = Interpolate(y0, points[0].loc[2], y1, points[1].loc[2])
    z12 = Interpolate(y1, points[1].loc[2], y2, points[2].loc[2])
    z02 = Interpolate(y0, points[0].loc[2], y2, points[2].loc[2])
    z012 = z01
    z012.extend(z12)

    m = math.floor(len(x012) / 2)
    if(x02[m] < x012[m]):
        x_left = x02
        x_right = x012
        h_left = h02
        h_right = h012
        z_left = z02
        z_right = z012
    else:
        x_left = x012
        x_right = x02
        h_left = h012
        h_right = h02
        z_left = z012
        z_right = z02
    
    for y in range(y0, y2):
        x_l = x_left[y - y0]
        x_r = x_right[y - y0]
        x_start = int(round(x_l))
        x_end = int(round(x_r))
        if x_end < x_start:
            x_start, x_end = x_end, x_start
        h_segment = Interpolate(x_start, h_left[y - y0], x_end + 1, h_right[y - y0])
        z_segment = Interpolate(x_start, z_left[y - y0], x_end + 1, z_right[y - y0])
        for x in range(x_start, x_end + 1):
            b = h_segment[x - x_start]
            shaded_color = (
                int(color[0] * b), 
                int(color[1] * b), 
                int(color[2] * b)
            )
            PutPixel(x=x, y=y, z=z_segment[x - x_start], canva=canva, color=shaded_color)
            # canva.array[canva_height - y][x][0] = shaded_color[0]
            # canva.array[canva_height - y][x][1] = shaded_color[1]
            # canva.array[canva_height - y][x][2] = shaded_color[2]

def ProjectToCanvas(P:Point, canva:Canva):
    z = P.loc[2]
    if z <= 0:
        return None
    scale = canva.d / z
    x = P.loc[0] * scale * canva.dpi
    y = P.loc[1] * scale * canva.dpi
    projected_loc = np.array([x, y, z], dtype=np.float32)
    projected_p = Point(projected_loc, P.b)
    return projected_p

def DrawWireframeTriangle(
        tri:Triangles,
        canva:Canva, 
        line_color:tuple, 
        filled_color:tuple
    ):
    minus_p0_loc = -tri.points[0].loc
    minus_p1_loc = -tri.points[1].loc
    minus_p2_loc = -tri.points[2].loc
    n_vector = np.cross(tri.points[1].loc - tri.points[0].loc,
                        tri.points[2].loc - tri.points[0].loc)
    if (
        np.dot(n_vector, minus_p0_loc) <= 0 or
        np.dot(n_vector, minus_p1_loc) <= 0 or
        np.dot(n_vector, minus_p2_loc) <= 0
    ):
        return
    
    get_light_for_triangle(tri, canva)


    p0 = ProjectToCanvas(tri.points[0], canva)
    p1 = ProjectToCanvas(tri.points[1], canva)
    p2 = ProjectToCanvas(tri.points[2], canva)
    if p0 is None or p1 is None or p2 is None:
        return
    
    DrawShadedLine(p0, p1, canva, line_color)
    DrawShadedLine(p1, p2, canva, line_color)
    DrawShadedLine(p2, p0, canva, line_color)
    DrawShadedTriangle(p0, p1, p2, canva, filled_color)

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
            point = Point([float(splited[i]) for i in range(1, 4)], 1.0)
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