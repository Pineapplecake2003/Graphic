import numpy as np
from utils import *
from DataStructure import *
# from vedo import *
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    obj_file = "./models/Rushia.obj"

    canva_height_px = 2700
    canva_width_px = 1200

    canva_height = 450
    canva_width = 200
    canva_d = 1000

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

    
    objects = load_objs(obj_file)
    # with open("./models/Rushia.obj") as iif:
    #     lines = iif.readlines()
    
    # with open("./models/Rushia_part.obj", 'w') as f:
    #     flag_head = 0
    #     for l in lines:
    #         # print(lines[0:3])
    #         # break
    #         if(l[0] == "#"):
    #             if(l == "# Head_1\n"):
    #                 flag_head = 1
    #             else:
    #                 flag_head = 0
    #         if(flag_head == 1):
    #             f.write(l)
    #     f.close()


    # objects = [objects[0][1]]
# 
    for t in tqdm(objects.triangles, ncols=80):
        DrawWireframeTriangle(
            t.points[0],
            t.points[1],
            t.points[2],
            picture, (0x00, 0x00, 0x00), (0x00, 0x00, 0x00)
        )

    img = Image.fromarray(picture.array, mode="RGB")
    img.save("./images/result.png")
    # mesh = Mesh("./models/Anyaoutline.obj")
    # mesh.show()

if __name__ == "__main__":
    main()

# from vedo import *
# mesh = Mesh("./models/Rushia.obj")
# mesh.show()