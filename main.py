import numpy as np
from utils import *
from DataStructure import *
from PIL import Image
import matplotlib.pyplot as plt

def main():
    canva_height_px = 10000
    canva_width_px = 10000

    canva_height = 10
    canva_width = 10
    canva_d = 1

    picture = Canva(
        (canva_height, canva_width),
        canva_d, 
        (canva_height_px, canva_width_px)
    )
    # p0 = Point(100, 200, 200, 1.0)
    # p1 = Point(150, 400, 300, 0.5)
    # p2 = Point(300, 200, 200, 0.3)
    # DrawWireframeTriangle(
    #     ProjectToCanvas(p0, picture),
    #     ProjectToCanvas(p1, picture),
    #     ProjectToCanvas(p2, picture),
    #     picture, 
    #     (0xFF, 0x8E, 0xFF),
    #     (0xFF, 0x8E, 0xFF))
    p0 = Point( 1,  1,  1, 1.0)
    p1 = Point(-1,  1,  1, 1.0)
    p2 = Point(-1, -1,  1, 1.0)
    p3 = Point( 1, -1,  1, 1.0)
    p4 = Point( 1,  1, -1, 1.0)
    p5 = Point(-1,  1, -1, 1.0)
    p6 = Point(-1, -1, -1, 1.0)
    p7 = Point( 1, -1, -1, 1.0)
    t0 = [p0, p1, p2]
    t1 = [p0, p2, p3]
    t2 = [p4, p0, p3]
    t3 = [p4, p3, p7]
    t4 = [p5, p4, p7]
    t5 = [p5, p7, p6]
    t6 = [p1, p5, p6]
    t7 = [p1, p6, p2]
    t8 = [p4, p5, p1]
    t9 = [p4, p1, p0]
    t10 = [p2, p6, p7]
    t11 = [p2, p7, p3]

    trangles = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11]
    for t in trangles:
        DrawWireframeTriangle(t[0], t[1],t [2], picture, (0x00, 0x00, 0x00), (0x00, 0x00, 0x00))

    img = Image.fromarray(picture.array, mode="RGB")
    img.save("./img/result.png")
    plt.imshow(img)

if __name__ == "__main__":
    main()
