import numpy as np
from DataStructure import *
import copy
import math
from tqdm import tqdm
def PutPixel(x:int, y:int, z:float, canva:Canva, color:tuple):
    if z < canva.d:
        return
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
        clamped = np.clip(np.round(color).astype(np.int32), 0, 255)
        canva.array[y_idx][x_idx][0] = clamped[0]
        canva.array[y_idx][x_idx][1] = clamped[1]
        canva.array[y_idx][x_idx][2] = clamped[2]

def get_light_for_triangle(t: Triangle, canva: Canva, s:float):
    # Ambient term
    I_p = canva.ambient
    for p in t.points:
        p.b = I_p

    # Face normal
    n_vector = np.cross(
        t.points[1].world_loc - t.points[0].world_loc,
        t.points[2].world_loc - t.points[0].world_loc
    )
    n_vector /= np.maximum(np.linalg.norm(n_vector), 1e-6)

    # Lighting loop
    for li in canva.light_srouce:
        for i, p in enumerate(t.points):
            if li.ltype == "point":
                l_vector = li.loc - p.world_loc
            elif li.ltype == "directional":
                l_vector = -li.li_dir
            l_vector /= np.maximum(np.linalg.norm(l_vector), 1e-6)

            v_vector = -p.world_loc
            v_vector /= np.maximum(np.linalg.norm(v_vector), 1e-6)

            r_vector = 2 * np.dot(n_vector, l_vector) * n_vector - l_vector
            r_vector /= np.maximum(np.linalg.norm(r_vector), 1e-6)

            # Diffuse + Specular
            shooted = max(np.dot(n_vector, l_vector), 0.0)
            reflected = max(np.dot(r_vector, v_vector), 0.0) ** s

            t.points[i].b += li.b * (shooted + reflected)

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

def DrawFlatShadedLine(P0:Point, P1:Point, canva:Canva, color:tuple):
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

def getPhongHs(locs:np.ndarray, vns:np.ndarray, canva:Canva, s:float):
    locs = np.asarray(locs, dtype=np.float32)
    vns = np.asarray(vns, dtype=np.float32)

    if locs.shape[0] != 3 or vns.shape[0] != 3:
        raise ValueError("locs and vns must be shaped (3, N)")

    sample_count = locs.shape[1]
    if sample_count == 0:
        return np.zeros((0,), dtype=np.float32)

    n_vector = vns.copy()
    n_norm = np.linalg.norm(n_vector, axis=0, keepdims=True)
    n_vector /= np.maximum(n_norm, 1e-6)

    v_vector = -locs.copy()
    v_norm = np.linalg.norm(v_vector, axis=0, keepdims=True)
    v_vector /= np.maximum(v_norm, 1e-6)

    hs = np.full((sample_count,), canva.ambient, dtype=np.float32)

    for li in canva.light_srouce:
        if li.ltype == "point":
            light_pos = np.asarray(li.loc, dtype=np.float32).reshape(3, 1)
            l_vector = light_pos - locs
        elif li.ltype == "directional":
            light_dir = -np.asarray(li.li_dir, dtype=np.float32).reshape(3, 1)
            l_vector = np.broadcast_to(light_dir, locs.shape).copy()
        else:
            continue

        l_norm = np.linalg.norm(l_vector, axis=0, keepdims=True)
        l_vector /= np.maximum(l_norm, 1e-6)

        ndotl = np.clip(np.sum(n_vector * l_vector, axis=0), 0.0, None)
        r_vector = 2.0 * n_vector * ndotl[np.newaxis, :] - l_vector
        r_norm = np.linalg.norm(r_vector, axis=0, keepdims=True)
        r_vector /= np.maximum(r_norm, 1e-6)

        rdotv = np.clip(np.sum(r_vector * v_vector, axis=0), 0.0, None)
        hs += li.b * (ndotl + rdotv ** s)
    
    return np.clip(hs, 0.0, None)


def DrawPhongShadedLine(
        P0:Point, P1:Point, 
        vn0:np.ndarray, vn1:np.ndarray, 
        canva:Canva, color:tuple, s:float
    ):
    points = [P0, P1]
    world_points = [
        np.asarray(P0.world_loc, dtype=np.float32),
        np.asarray(P1.world_loc, dtype=np.float32)
    ]
    vns = [np.array(vn0, dtype=np.float32, copy=True), np.array(vn1, dtype=np.float32, copy=True)]
    if abs(points[1].loc[0] - points[0].loc[0]) > abs(points[1].loc[1] - points[0].loc[1]):
        # Horizontal line
        # Make sure x0 < x1
        if points[0].loc[0] > points[1].loc[0]:
            points[0], points[1] = points[1], points[0]
            world_points[0], world_points[1] = world_points[1], world_points[0]
            vns[0], vns[1] = vns[1], vns[0]
        x0 = int(round(points[0].loc[0]))
        x1 = int(round(points[1].loc[0]))
        if x0 == x1:
            return
        xs = range(x0, x1)
        ys = Interpolate(x0, points[0].loc[1], x1, points[1].loc[1])
        zs = Interpolate(x0, points[0].loc[2], x1, points[1].loc[2])
        sample_count = len(ys)
        if sample_count == 0:
            return
        t = np.linspace(0.0, 1.0, sample_count, endpoint=False, dtype=np.float32)
        world_locs = (
            world_points[0].reshape(3, 1) * (1.0 - t.reshape(1, -1)) +
            world_points[1].reshape(3, 1) * t.reshape(1, -1)
        )
        vnxs = Interpolate(x0, vns[0][0], x1, vns[1][0])
        vnys = Interpolate(x0, vns[0][1], x1, vns[1][1])
        vnzs = Interpolate(x0, vns[0][2], x1, vns[1][2])
        vns = np.array([
            vnxs,
            vnys, 
            vnzs
        ], dtype=np.float32)
        hs = getPhongHs(world_locs, vns, canva, s)
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
            world_points[0], world_points[1] = world_points[1], world_points[0]
            vns[0], vns[1] = vns[1], vns[0]
        y0 = int(round(points[0].loc[1]))
        y1 = int(round(points[1].loc[1]))
        if y0 == y1:
            return
        xs = Interpolate(y0, points[0].loc[0], y1, points[1].loc[0])
        ys = range(y0, y1)
        zs = Interpolate(y0, points[0].loc[2], y1, points[1].loc[2])
        sample_count = len(xs)
        if sample_count == 0:
            return
        t = np.linspace(0.0, 1.0, sample_count, endpoint=False, dtype=np.float32)
        world_locs = (
            world_points[0].reshape(3, 1) * (1.0 - t.reshape(1, -1)) +
            world_points[1].reshape(3, 1) * t.reshape(1, -1)
        )
        vnxs = Interpolate(y0, vns[0][0], y1, vns[1][0])
        vnys = Interpolate(y0, vns[0][1], y1, vns[1][1])
        vnzs = Interpolate(y0, vns[0][2], y1, vns[1][2])
        vns = np.array([
            vnxs,
            vnys, 
            vnzs
        ], dtype=np.float32)
        hs = getPhongHs(world_locs, vns, canva, s)
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

def toInt(val):
    #return math.floor(val)
    if val > 0:
        return math.floor(val)
    else:
        return math.ceil(val)

def update_p(toP1_p, toP2_p, toP1_v, toP2_v, toB1_v, toB2_v):
    toP1_p_return = copy.deepcopy(toP1_p)
    toP2_p_return = copy.deepcopy(toP2_p)
    toP1_p_return.loc += toP1_v
    toP2_p_return.loc += toP2_v
    toP1_p_return.b += toB1_v[1]
    toP2_p_return.b += toB2_v[1]

    return (toP1_p_return, toP2_p_return)

def DrawFlatShadedTriangle (p0, p1, p2, canva:Canva, color):
    points = [p0, p1, p2]
    # Sort points depended on y
    # sort to
    #        0
    #       /|
    #      / |
    #     1  |
    #      \ |
    #       \|
    #        2
    # y based
    points[0].loc[1] = toInt(points[0].loc[1])
    points[1].loc[1] = toInt(points[1].loc[1])
    points[2].loc[1] = toInt(points[2].loc[1])

    # x
    points[0].loc[0] = toInt(points[0].loc[0])
    points[1].loc[0] = toInt(points[1].loc[0])
    points[2].loc[0] = toInt(points[2].loc[0])
    if(points[1].loc[1] > points[0].loc[1]):
        temp = points[1]
        points[1] = points[0]
        points[0] = temp
    
    if(points[2].loc[1] > points[0].loc[1]):
        temp = points[2]
        points[2] = points[0]
        points[0] = temp
    
    if(points[2].loc[1] > points[1].loc[1]):
        temp = points[2]
        points[2] = points[1]
        points[1] = temp
    
    # # y based
    # points[0].loc[1] = toInt(points[0].loc[1])
    # points[1].loc[1] = toInt(points[1].loc[1])
    # points[2].loc[1] = toInt(points[2].loc[1])
# 
    # # x
    # points[0].loc[0] = toInt(points[0].loc[0])
    # points[1].loc[0] = toInt(points[1].loc[0])
    # points[2].loc[0] = toInt(points[2].loc[0])

    if points[0].loc[1] != points[1].loc[1]:
        toP1_p = copy.deepcopy(points[0])
        toP2_p = copy.deepcopy(points[0])
    else:
        toP1_p = copy.deepcopy(points[1]) # 0 ________1          1 ________0
        toP2_p = copy.deepcopy(points[0]) #   \      /             \      /
                                          #    \    /               \    /
                                          #     \  /                 \  /
                                          #      \/                   \/
                                          #       2         or         2



    toP1_v_U = points[1].loc - points[0].loc
    toP1_v_U = toP1_v_U / abs(toP1_v_U[1]) if not np.isclose(toP1_v_U[1], 0.) else toP1_v_U
    toP2_v_U = points[2].loc - points[0].loc
    toP2_v_U = toP2_v_U / abs(toP2_v_U[1]) if not np.isclose(toP2_v_U[1], 0.) else toP2_v_U
    
    toB1_v_U = np.array([points[1].loc[1] - points[0].loc[1], points[1].b - points[0].b])
    toB1_v_U = toB1_v_U / abs(toB1_v_U[0]) if not np.isclose(toB1_v_U[0], 0.) else toB1_v_U
    toB2_v_U = np.array([points[2].loc[1] - points[0].loc[1], points[2].b - points[0].b])
    toB2_v_U = toB2_v_U / abs(toB2_v_U[0]) if not np.isclose(toB2_v_U[0], 0.) else toB2_v_U

    toP1_v_D = points[2].loc - points[1].loc
    toP1_v_D = toP1_v_D / abs(toP1_v_D[1]) if not np.isclose(toP1_v_D[1], 0.) else toP1_v_D
    toP2_v_D = points[2].loc - points[0].loc
    toP2_v_D = toP2_v_D / abs(toP2_v_D[1]) if not np.isclose(toP2_v_D[1], 0.) else toP2_v_D
    
    toB1_v_D = np.array([points[2].loc[1] - points[1].loc[1], points[2].b - points[1].b])
    toB1_v_D = toB1_v_D / abs(toB1_v_D[0]) if not np.isclose(toB1_v_D[0], 0.) else toB1_v_D
    toB2_v_D = np.array([points[2].loc[1] - points[0].loc[1], points[2].b - points[0].b])
    toB2_v_D = toB2_v_D / abs(toB2_v_D[0]) if not np.isclose(toB2_v_D[0], 0.) else toB2_v_D
    
    det = (points[1].loc[0] - points[0].loc[0]) * (points[2].loc[1] - points[0].loc[1]) - (points[2].loc[0] - points[0].loc[0]) * (points[1].loc[1] - points[0].loc[1])
    if det == 0:
        return
    a_coff_for_brig = (points[1].b - points[0].b)*(points[2].loc[1] - points[0].loc[1]) - (points[2].b - points[0].b)*(points[1].loc[1] - points[0].loc[1])
    a_coff_for_brig /= det

    b_coff_for_brig = (points[2].b - points[0].b)*(points[1].loc[0] - points[0].loc[0]) - (points[1].b - points[0].b)*(points[2].loc[0] - points[0].loc[0])
    b_coff_for_brig /= det

    a_coff_for_z = (points[1].loc[2] - points[0].loc[2])*(points[2].loc[1] - points[0].loc[1]) - (points[2].loc[2] - points[0].loc[2])*(points[1].loc[1] - points[0].loc[1])
    a_coff_for_z /= det

    b_coff_for_z = (points[2].loc[2] - points[0].loc[2])*(points[1].loc[0] - points[0].loc[0]) - (points[1].loc[2] - points[0].loc[2])*(points[2].loc[0] - points[0].loc[0])
    b_coff_for_z /= det


    while(toP1_p.loc[1] > points[2].loc[1]):
        if toP1_p.loc[0] > toP2_p.loc[0]:
            left_p, right_p = toP2_p, toP1_p
        else:
            left_p, right_p = toP1_p, toP2_p
        
        z_px_left = a_coff_for_z * (left_p.loc[0] - points[0].loc[0]) + \
                b_coff_for_z * (left_p.loc[1] - points[0].loc[1]) + points[0].loc[2]
        z_px = z_px_left

        b_px_left = a_coff_for_brig * (left_p.loc[0] - points[0].loc[0]) + \
                b_coff_for_brig * (left_p.loc[1] - points[0].loc[1]) + points[0].b
        b_px = b_px_left
        
        for x in range(int(left_p.loc[0]), int(right_p.loc[0])+1):
            draw_color = list(color)
            draw_color[0] = color[0] * b_px
            draw_color[1] = color[1] * b_px
            draw_color[2] = color[2] * b_px
            PutPixel(x, toInt(left_p.loc[1]), z_px, canva, draw_color)
            b_px += a_coff_for_brig
            z_px += a_coff_for_z

        if toP1_p.loc[1] > points[1].loc[1]:
            # Above P1
            #print("U")
            toP1_v = toP1_v_U
            toP2_v = toP2_v_U
            
            toB1_v = toB1_v_U
            toB2_v = toB2_v_U
        else:
            # below P1
            #print("D")
            toP1_v = toP1_v_D
            toP2_v = toP2_v_D
            
            toB1_v = toB1_v_D
            toB2_v = toB2_v_D
        toP1_p, toP2_p = update_p(toP1_p, toP2_p, toP1_v, toP2_v, toB1_v, toB2_v)



def DrawPhongShadedTriangle(p0, p1, p2, vn0, vn1, vn2, canva:Canva, color:tuple, s:float):
    vertices = [
        (p0, np.asarray(vn0, dtype=np.float32)),
        (p1, np.asarray(vn1, dtype=np.float32)),
        (p2, np.asarray(vn2, dtype=np.float32))
    ]
    vertices.sort(key=lambda item: item[0].loc[1])

    points = [item[0] for item in vertices]
    normals = [item[1] for item in vertices]

    y0 = int(round(points[0].loc[1]))
    y1 = int(round(points[1].loc[1]))
    y2 = int(round(points[2].loc[1]))
    if y0 == y2:
        return

    def _span(i0, d0, i1, d1):
        if i0 == i1:
            return np.empty(0, dtype=np.float32)
        return np.array(Interpolate(i0, d0, i1, d1), dtype=np.float32)

    def _stack(x_vals, y_vals, z_vals):
        if x_vals.size == 0:
            return np.empty((3, 0), dtype=np.float32)
        return np.vstack((x_vals, y_vals, z_vals))

    x01 = _span(y0, points[0].loc[0], y1, points[1].loc[0])
    x12 = _span(y1, points[1].loc[0], y2, points[2].loc[0])
    x02 = _span(y0, points[0].loc[0], y2, points[2].loc[0])
    if x02.size == 0:
        return

    y02 = np.arange(y0, y2, dtype=np.float32)
    if y02.size == 0:
        return

    z01 = _span(y0, points[0].loc[2], y1, points[1].loc[2])
    z12 = _span(y1, points[1].loc[2], y2, points[2].loc[2])
    z02 = _span(y0, points[0].loc[2], y2, points[2].loc[2])

    x012 = np.concatenate((x01, x12)) if x01.size or x12.size else np.empty(0, dtype=np.float32)
    z012 = np.concatenate((z01, z12)) if z01.size or z12.size else np.empty(0, dtype=np.float32)
    if x012.size == 0:
        return

    vnxs01 = _span(y0, normals[0][0], y1, normals[1][0])
    vnys01 = _span(y0, normals[0][1], y1, normals[1][1])
    vnzs01 = _span(y0, normals[0][2], y1, normals[1][2])
    vnxs12 = _span(y1, normals[1][0], y2, normals[2][0])
    vnys12 = _span(y1, normals[1][1], y2, normals[2][1])
    vnzs12 = _span(y1, normals[1][2], y2, normals[2][2])
    vnxs02 = _span(y0, normals[0][0], y2, normals[2][0])
    vnys02 = _span(y0, normals[0][1], y2, normals[2][1])
    vnzs02 = _span(y0, normals[0][2], y2, normals[2][2])

    vns01 = _stack(vnxs01, vnys01, vnzs01)
    vns12 = _stack(vnxs12, vnys12, vnzs12)
    vns02 = _stack(vnxs02, vnys02, vnzs02)
    vns012 = np.concatenate((vns01, vns12), axis=1) if vns01.size or vns12.size else np.empty((3, 0), dtype=np.float32)

    worldx01 = _span(y0, points[0].world_loc[0], y1, points[1].world_loc[0])
    worldy01 = _span(y0, points[0].world_loc[1], y1, points[1].world_loc[1])
    worldz01 = _span(y0, points[0].world_loc[2], y1, points[1].world_loc[2])
    worldx12 = _span(y1, points[1].world_loc[0], y2, points[2].world_loc[0])
    worldy12 = _span(y1, points[1].world_loc[1], y2, points[2].world_loc[1])
    worldz12 = _span(y1, points[1].world_loc[2], y2, points[2].world_loc[2])
    worldx02 = _span(y0, points[0].world_loc[0], y2, points[2].world_loc[0])
    worldy02 = _span(y0, points[0].world_loc[1], y2, points[2].world_loc[1])
    worldz02 = _span(y0, points[0].world_loc[2], y2, points[2].world_loc[2])

    world01 = _stack(worldx01, worldy01, worldz01)
    world12 = _stack(worldx12, worldy12, worldz12)
    world02 = _stack(worldx02, worldy02, worldz02)
    world012 = np.concatenate((world01, world12), axis=1) if world01.size or world12.size else np.empty((3, 0), dtype=np.float32)

    m = x012.shape[0] // 2
    if x02[m] < x012[m]:
        x_left = x02
        x_right = x012
        z_left = z02
        z_right = z012
        vns_left = vns02
        vns_right = vns012
        world_left = world02
        world_right = world012
    else:
        x_left = x012
        x_right = x02
        z_left = z012
        z_right = z02
        vns_left = vns012
        vns_right = vns02
        world_left = world012
        world_right = world02

    for idx, y in enumerate(range(y0, y2)):
        x_l = x_left[idx]
        x_r = x_right[idx]

        x_start = int(round(x_l))
        x_end = int(round(x_r))

        z_start = z_left[idx]
        z_end = z_right[idx]
        n_start = vns_left[:, idx]
        n_end = vns_right[:, idx]
        world_start = world_left[:, idx]
        world_end = world_right[:, idx]

        if x_end < x_start:
            x_start, x_end = x_end, x_start
            z_start, z_end = z_end, z_start
            n_start, n_end = n_end, n_start
            world_start, world_end = world_end, world_start

        x_segment = np.arange(x_start, x_end + 1, dtype=np.float32)
        y_segment = np.full(x_segment.shape, y, dtype=np.float32)
        z_segment = np.array(Interpolate(x_start, z_start, x_end + 1, z_end), dtype=np.float32)

        vnx_segment = np.array(Interpolate(x_start, n_start[0], x_end + 1, n_end[0]), dtype=np.float32)
        vny_segment = np.array(Interpolate(x_start, n_start[1], x_end + 1, n_end[1]), dtype=np.float32)
        vnz_segment = np.array(Interpolate(x_start, n_start[2], x_end + 1, n_end[2]), dtype=np.float32)
        vn_segment = _stack(vnx_segment, vny_segment, vnz_segment)

        worldx_segment = np.array(Interpolate(x_start, world_start[0], x_end + 1, world_end[0]), dtype=np.float32)
        worldy_segment = np.array(Interpolate(x_start, world_start[1], x_end + 1, world_end[1]), dtype=np.float32)
        worldz_segment = np.array(Interpolate(x_start, world_start[2], x_end + 1, world_end[2]), dtype=np.float32)
        world_segment = _stack(worldx_segment, worldy_segment, worldz_segment)
        if world_segment.shape[1] == 0:
            continue

        h_segment = getPhongHs(world_segment, vn_segment, canva, s)

        for px_idx, x in enumerate(range(x_start, x_end + 1)):
            b = h_segment[px_idx]
            shaded_color = (
                int(color[0] * b),
                int(color[1] * b),
                int(color[2] * b)
            )
            PutPixel(x=x, y=y, z=z_segment[px_idx], canva=canva, color=shaded_color)


def ProjectToCanvas(P:Point, canva:Canva):
    z = P.loc[2]
    if z <= 0:
        return None
    scale = canva.d / z
    x = P.loc[0] * scale * canva.dpi
    y = P.loc[1] * scale * canva.dpi
    projected_loc = np.array([x, y, z], dtype=np.float32)
    projected_p = Point(projected_loc, P.b, world_loc=P.world_loc)
    return projected_p

def DrawWireframeTriangle(
        tri:Triangle,
        canva:Canva, 
        line_color:tuple, 
        filled_color:tuple,
        shade_type:str,
        s:float
    ):
    assert(shade_type == "Flat" or shade_type == "Phong" or shade_type == "None"), "Shadaw type must be 'Flat' or 'Phong'."

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
    if shade_type == "Flat":
        get_light_for_triangle(tri, canva, s)
    elif shade_type == "Phong":
        pass # ...

    p0 = ProjectToCanvas(tri.points[0], canva)
    p1 = ProjectToCanvas(tri.points[1], canva)
    p2 = ProjectToCanvas(tri.points[2], canva)
    if p0 is None or p1 is None or p2 is None:
        return
    
    if shade_type == "Flat":
        # DrawFlatShadedLine(p0, p1, canva, line_color)
        # DrawFlatShadedLine(p1, p2, canva, line_color)
        # DrawFlatShadedLine(p2, p0, canva, line_color)
        DrawFlatShadedTriangle(p0, p1, p2, canva, filled_color)
    elif shade_type == "Phong":
        vns = tri.vns
        DrawPhongShadedLine(p0, p1, vns[0], vns[1], canva, line_color, s)
        DrawPhongShadedLine(p1, p2, vns[1], vns[2], canva, line_color, s)
        DrawPhongShadedLine(p2, p0, vns[2], vns[0], canva, line_color, s)
        DrawPhongShadedTriangle(p0, p1, p2, vns[0], vns[1], vns[2], canva, filled_color, s)
    elif shade_type == "None":
        vns = tri.vns
        DrawPhongShadedLine(p0, p1, vns[0], vns[1], canva, line_color, s)
        DrawPhongShadedLine(p1, p2, vns[1], vns[2], canva, line_color, s)
        DrawPhongShadedLine(p2, p0, vns[2], vns[0], canva, line_color, s)

def load_objs(path:str):
    print(f"Loading {path}...")
    with open(path) as f:
        lines = f.readlines()
    
    points = []
    tris = []
    vns = []
    for line in tqdm(lines, ncols=50):
        if line.startswith("v "):
            splited = line.split(' ')
            if '' in splited:
                splited.remove('')
            point = Point([float(splited[i]) for i in range(1, 4)], 1.0)
            points.append(point)
        elif line.startswith("vn "):
            splited = line.split(' ')
            if '' in splited:
                splited.remove('')
            vn = np.array([float(splited[i]) for i in range(1, 4)], np.float32)
            vns.append(vn)
        elif line.startswith("f "):
            splited = line.split(' ')
            if '' in splited:
                splited.remove('')
            
            point_infos = []
            for s in splited[1:]:
                s = s.replace('\n', '')
                point_infos.append(s.split('/'))
            tri = Triangle(
                [
                    points[int(point_infos[0][0]) - 1], 
                    points[int(point_infos[1][0]) - 1], 
                    points[int(point_infos[2][0]) - 1]
                ],
                [],# Not support vt yet
                [
                    vns[int(point_infos[0][2]) - 1], 
                    vns[int(point_infos[1][2]) - 1], 
                    vns[int(point_infos[2][2]) - 1]
                ]
            )
            tris.append(tri)
    print(f"Number of triangles: {len(tris)}")
    obj = ThreeDimensionObject(tris, points)
    return obj