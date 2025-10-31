import numpy as np
from utils import *
from vedo import *
from PIL import Image
import matplotlib.pyplot as plt
import time
import math
import random
import copy

class PointData:
    real_x:float
    real_y:float
    canva_x:int
    canva_y:int
    def __init__(self, x, y):
        self.real_x = x
        self.real_y = y
        self.canva_x = int(x)
        self.canva_x = int(y)

def update_p(toP1_p, toP2_p, toP1_v, toP2_v):
    toP1_p_return = toP1_p.copy() + toP1_v
    toP2_p_return = toP2_p.copy() + toP2_v

    return (toP1_p_return, toP2_p_return)

def update_b(toB1_b, toB2_b, toB1_v, toB2_v):
    toB1_b_return = toB1_b.copy() + toB1_v
    toB2_b_return = toB2_b.copy() + toB2_v

    return (toB1_b_return, toB2_b_return)

def putPixel(canva_x, canva_y, canva, color):
    canva_height = canva.shape[0] - 1
    canva_weight = canva.shape[1] - 1
    x_idx = canva_x + canva_weight // 2
    y_idx = canva_height - canva_y - canva_height // 2
    
    canva[y_idx][x_idx][0] = color[0]
    canva[y_idx][x_idx][1] = color[1]
    canva[y_idx][x_idx][2] = color[2]

def toInt(val):
    #return math.floor(val)
    if val > 0:
        return math.floor(val)
    else:
        return math.ceil(val)

def main():
    color = [0x88, 0x00, 0x00]
    draw_color = copy.deepcopy(color)
    canva = np.full((100, 100, 3), 0xFF, dtype=np.uint8)
    # [canva_x, canva_y, real_z]
    # after projection to canva
    P0 = np.array([random.uniform(-50.0, 50.0), random.uniform(-50.0, 50.0), random.uniform(-50.0, 50.0)])
    P1 = np.array([random.uniform(-50.0, 50.0), random.uniform(-50.0, 50.0), random.uniform(-50.0, 50.0)])
    P2 = np.array([random.uniform(-50.0, 50.0), random.uniform(-50.0, 50.0), random.uniform(-50.0, 50.0)])
    # P0 = np.array([-20.741988809231838, -48.516322365112075, 1.152533334951876])
    # P1 = np.array([12.224992344053078, -0.3945586536833048, -10.674345760264714])
    # P2 = np.array([-24.769264965801373, 38.890515158047265, 14.21442370213272])

    B0 = np.array([*P0, 0.5])
    B1 = np.array([*P1, 1.0])
    B2 = np.array([*P2, 0.3])

    # Used for debug.
    # If you found bugs, copy this file and reproduce bugs you found.
    with open("log.txt", "w") as f:
        f.write(f"    P0 = np.array([{P0[0]}, {P0[1]}, {P0[2]}])\n")
        f.write(f"    P1 = np.array([{P1[0]}, {P1[1]}, {P1[2]}])\n")
        f.write(f"    P2 = np.array([{P2[0]}, {P2[1]}, {P2[2]}])\n")

        f.write(f"({P0[0]}, {P0[1]})\n")
        f.write(f"({P1[0]}, {P1[1]})\n")
        f.write(f"({P2[0]}, {P2[1]})\n")

    #########################################
    #           Drawing Line state          #
    #########################################
    points = [P0, P1, P2]
    bs =  [B0, B1, B2]
    for i in range(len(points)):
        if i < len(points) - 1:
            for j in range(i+1, len(points)):
                if abs(points[i][0] - points[j][0]) > abs(points[i][1] - points[j][1]):
                    # Horizontal line
                    p_start = points[i].copy()
                    p_end = points[j].copy()
                    if p_start[0] > p_end[0]:# TODO This comparison may be improved
                        p_start, p_end = p_end, p_start
                    
                    p_start[0] = toInt(p_start[0])
                    p_end[0] = toInt(p_end[0])
                    p_vec = p_end - p_start
                    p_vec /= abs(p_vec[0])
                    p = p_start
                    
                    b_start = bs[i]
                    b_end = bs[j]
                    # [x, b]
                    b_start = np.array([b_start[0],b_start[3]])
                    b_end = np.array([b_end[0],b_end[3]])
                    if b_start[0] > b_end[0]:# TODO This comparison may be improved
                        b_start, b_end = b_end, b_start
                    b_vec = b_end - b_start
                    # brightness per x
                    b_vec /= b_vec[0]
                    b = b_start
                    for x in range(toInt(p_start[0]), toInt(p_end[0])):
                        draw_color[0] = color[0] * b[1]
                        draw_color[1] = color[1] * b[1]
                        draw_color[2] = color[2] * b[1]
                        putPixel(x, toInt(p[1]), canva, draw_color)
                        p += p_vec
                        b += b_vec
                else:
                    # Vertical line
                    p_start = points[i].copy()
                    p_end = points[j].copy()
                    if p_start[1] > p_end[1]:# TODO This comparison may be improved
                        p_start, p_end = p_end, p_start
                    p_start[1] = toInt(p_start[1])
                    p_end[1] = toInt(p_end[1])
                    p_vec = p_end - p_start
                    p_vec /= abs(p_vec[1])
                    p = p_start

                    b_start = bs[i]
                    b_end = bs[j]
                    # [y, b]
                    b_start = np.array([b_start[1],b_start[3]])
                    b_end = np.array([b_end[1],b_end[3]])
                    if b_start[0] > b_end[0]:# TODO This comparison may be improved
                        b_start, b_end = b_end, b_start
                    # brightness per y
                    b_vec = b_end - b_start
                    b_vec /= b_vec[0]
                    b = b_start
                    for y in range(toInt(p_start[1]), toInt(p_end[1])):
                        draw_color[0] = color[0] * b[1]
                        draw_color[1] = color[1] * b[1]
                        draw_color[2] = color[2] * b[1]
                        putPixel(toInt(p[0]), y, canva, draw_color)
                        p += p_vec
                        b += b_vec

    #########################################
    #        Filling triangle state         #
    #########################################
    # [y, b] ==> brightness per y
    B0 = np.array([P0[1], 0.5])
    B1 = np.array([P1[1], 1.0])
    B2 = np.array([P2[1], 0.3])
    # sort to
    #        0
    #       /|
    #      / |
    #     1  |
    #      \ |
    #       \|
    #        2
    # sort y
    if(P1[1] > P0[1]):
        P0, P1 = P1, P0
        B0, B1 = B1, B0
    
    if(P2[1] > P0[1]):
        P0, P2 = P2, P0
        B0, B2 = B2, B0
    
    if(P2[1] > P1[1]):
        P1, P2 = P2, P1
        B1, B2 = B2, B1
    
    # y based
    P0[1] = toInt(P0[1])
    P1[1] = toInt(P1[1])
    P2[1] = toInt(P2[1])
    B0[0] = toInt(B0[0])
    B1[0] = toInt(B1[0])
    B2[0] = toInt(B2[0])
    # big to small
    # P0, P1, P2
    if P0[1] != P1[1]:
        toP1_p = copy.deepcopy(P0)
        toP2_p = copy.deepcopy(P0)

        toB1_b = copy.deepcopy(B0)
        toB2_b = copy.deepcopy(B0)
    else:
        toP1_p = copy.deepcopy(P1)
        toP2_p = copy.deepcopy(P0)

        toB1_b = copy.deepcopy(B1)
        toB2_b = copy.deepcopy(B0)
        

    draw_color[0] = draw_color[0] * toB1_b[1]
    draw_color[1] = draw_color[1] * toB1_b[1]
    draw_color[2] = draw_color[2] * toB1_b[1]
    print(P0, P1, P2)
    print(B0, B1, B2)
    putPixel(toInt(toP1_p[0]), toInt(toP1_p[1]), canva, draw_color)
    #print(toP1_p, toP2_p)
    #print(toB1_b, toB2_b)
    #print("--------------------")

    # Combinational circuit
    toP1_v_U = P1 - P0
    toP1_v_U = toP1_v_U / abs(toP1_v_U[1]) if not np.isclose(toP1_v_U[1], 0.) else toP1_v_U
    toP2_v_U = P2 - P0
    toP2_v_U = toP2_v_U / abs(toP2_v_U[1]) if not np.isclose(toP2_v_U[1], 0.) else toP2_v_U
    
    toB1_v_U = B1 - B0
    toB1_v_U = toB1_v_U / abs(toB1_v_U[0]) if not np.isclose(toB1_v_U[0], 0.) else toB1_v_U
    toB2_v_U = B2 - B0
    toB2_v_U = toB2_v_U / abs(toB2_v_U[0]) if not np.isclose(toB2_v_U[0], 0.) else toB2_v_U

    toP1_v_D = P2 - P1
    toP1_v_D = toP1_v_D / abs(toP1_v_D[1]) if not np.isclose(toP1_v_D[1], 0.) else toP1_v_D
    toP2_v_D = P2 - P0
    toP2_v_D = toP2_v_D / abs(toP2_v_D[1]) if not np.isclose(toP2_v_D[1], 0.) else toP2_v_D
    
    toB1_v_D = B2 - B1
    toB1_v_D = toB1_v_D / abs(toB1_v_D[0]) if not np.isclose(toB1_v_D[0], 0.) else toB1_v_D
    toB2_v_D = B2 - B0
    toB2_v_D = toB2_v_D / abs(toB2_v_D[0]) if not np.isclose(toB2_v_D[0], 0.) else toB2_v_D

    while(toP1_p[1] > P2[1]):
        time.sleep(0.0)
        print(toP1_p, toP2_p)
        if toP1_p[1] > P1[1]:
            # Above P1
            print("U")
            toP1_v = toP1_v_U
            toP2_v = toP2_v_U
            
            toB1_v = toB1_v_U
            toB2_v = toB2_v_U
        else:
            # below P1
            print("D")
            toP1_v = toP1_v_D
            toP2_v = toP2_v_D
            
            toB1_v = toB1_v_D
            toB2_v = toB2_v_D
        if(np.isclose(toP1_v[1], 0.) or np.isclose(toP2_v[1], 0.)):
            # P0 and P1 in same horizontal pixel line
            continue
        toP1_p, toP2_p = update_p(toP1_p, toP2_p, toP1_v, toP2_v)
        toB1_b, toB2_b = update_b(toB1_b, toB2_b, toB1_v, toB2_v)

        if toP1_p[0] > toP2_p[0]:
            left_p, right_p = toP2_p, toP1_p
            left_b, right_b = toB2_b, toB1_b
        else:
            left_p, right_p = toP1_p, toP2_p
            left_b, right_b = toB1_b, toB2_b
        # [x, b] ==> brightness per x
        b_x_dir_v = np.array([right_p[0] - left_p[0], right_b[1] - left_b[1]])
        b_x_dir_v /= abs(b_x_dir_v[0])
        b_px = left_b[1]

        for x in range(toInt(left_p[0]), toInt(right_p[0])+1):
            draw_color[0] = color[0] * b_px
            draw_color[1] = color[1] * b_px
            draw_color[2] = color[2] * b_px
            b_px += b_x_dir_v[1]
            putPixel(x, toInt(left_p[1]), canva, draw_color)

    img = Image.fromarray(canva, mode="RGB")
    img.save("./images/test_have_line.png")

    canva = np.full_like(canva, 0xFF)
    
    if P0[1] != P1[1]:
        toP1_p = copy.deepcopy(P0)
        toP2_p = copy.deepcopy(P0)

        toB1_b = copy.deepcopy(B0)
        toB2_b = copy.deepcopy(B0)
    else:
        toP1_p = copy.deepcopy(P1)
        toP2_p = copy.deepcopy(P0)

        toB1_b = copy.deepcopy(B1)
        toB2_b = copy.deepcopy(B0)
    while(toP1_p[1] > P2[1]):
        time.sleep(0.0)
        print(toP1_p, toP2_p)
        if toP1_p[1] > P1[1]:
            # Above P1
            print("U")
            toP1_v = toP1_v_U
            toP2_v = toP2_v_U
            
            toB1_v = toB1_v_U
            toB2_v = toB2_v_U
        else:
            # below P1
            print("D")
            toP1_v = toP1_v_D
            toP2_v = toP2_v_D
            
            toB1_v = toB1_v_D
            toB2_v = toB2_v_D
        if(np.isclose(toP1_v[1], 0.) or np.isclose(toP2_v[1], 0.)):
            # P0 and P1 in same horizontal pixel line
            continue
        toP1_p, toP2_p = update_p(toP1_p, toP2_p, toP1_v, toP2_v)
        toB1_b, toB2_b = update_b(toB1_b, toB2_b, toB1_v, toB2_v)

        if toP1_p[0] > toP2_p[0]:
            left_p, right_p = toP2_p, toP1_p
            left_b, right_b = toB2_b, toB1_b
        else:
            left_p, right_p = toP1_p, toP2_p
            left_b, right_b = toB1_b, toB2_b
        # [x, b] ==> brightness per x
        b_x_dir_v = np.array([right_p[0] - left_p[0], right_b[1] - left_b[1]])
        b_x_dir_v /= abs(b_x_dir_v[0])
        b_px = left_b[1]

        for x in range(toInt(left_p[0]), toInt(right_p[0])+1):
            draw_color[0] = color[0] * b_px
            draw_color[1] = color[1] * b_px
            draw_color[2] = color[2] * b_px
            b_px += b_x_dir_v[1]
            putPixel(x, toInt(left_p[1]), canva, draw_color)

    img = Image.fromarray(canva, mode="RGB")
    img.save("./images/test_no_line.png")



if __name__ == "__main__":
    main()

