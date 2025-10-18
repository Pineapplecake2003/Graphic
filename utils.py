import numpy as np
from DataStructure import *
import copy
import math
def PutPixel(x:int, y:int, canva:Canva, color:tuple):
    canva_height = canva.array.shape[0] - 1
    canva_weight = canva.array.shape[1] - 1
    canva.array[canva_height - y - canva_height // 2][x + canva_weight // 2][0] = color[0]
    canva.array[canva_height - y - canva_height // 2][x + canva_weight // 2][1] = color[1]
    canva.array[canva_height - y - canva_height // 2][x + canva_weight // 2][2] = color[2]



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
        hs = Interpolate(points[0].x, points[0].b, points[1].x, points[1].b)
        for x in range(points[0].x, points[1].x):
            PutPixel(x, int(ys[x - points[0].x]), canva, 
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
        hs = Interpolate(points[0].y, points[0].b, points[1].y, points[1].b)
        for y in range(points[0].y, points[1].y):
            PutPixel(int(xs[y - points[0].y]), y, canva,
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
            PutPixel(x, y, canva, color)

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
            PutPixel(x, y, canva, shaded_color)
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


def DrawWireframeTriangle(
        P0:Point, 
        P1:Point,
        P2:Point, 
        canva:Canva, 
        line_color:tuple, 
        filled_color:tuple
    ):
    p0 = copy.deepcopy(P0)
    p1 = copy.deepcopy(P1)
    p2 = copy.deepcopy(P2)
    p0.x = p0.x - 1.5
    p0.z = p0.z + 7

    p1.x = p1.x - 1.5
    p1.z = p1.z + 7

    p2.x = p2.x - 1.5
    p2.z = p2.z + 7
    
    p0 = ProjectToCanvas(p0, canva)
    p1 = ProjectToCanvas(p1, canva)
    p2 = ProjectToCanvas(p2, canva)
    DrawShadedLine(p0, p1, canva, line_color)
    DrawShadedLine(p1, p2, canva, line_color)
    DrawShadedLine(p2, p0, canva, line_color)
    # DrawShadedTriangle(P0, P1, P2, canva, filled_color)
