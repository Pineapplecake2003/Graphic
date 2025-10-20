import numpy as np
from utils import *
from vedo import *
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import DataStructure


def main():
    obj_file = "./models/teapot.obj"

    dpi = 5
    canva_height = 800
    canva_width = 200

    canva_height_px = canva_height * dpi
    canva_width_px = canva_width * dpi

    canva_d = 1000

    ambient = 0.5
    light_src = DataStructure.Point(10, 10, 1050, 1.0)

    picture = Canva(
        (canva_height, canva_width),
        canva_d, 
        (canva_height_px, canva_width_px),
        ambient,
        [light_src]
    )
    
    object = load_objs(obj_file)
    object.transform((0, 0, 1020), (45, 45, 45), 1.0)

    for t in tqdm(object.triangles, ncols=80):
        DrawWireframeTriangle(
            t.points[0],
            t.points[1],
            t.points[2],
            picture, (0x00, 0x00, 0x00), (0x00, 0x00, 0x00)
        )

    img = Image.fromarray(picture.array, mode="RGB")
    img.save("./images/result.png")
    #plt.imshow(img)
    # mesh = Mesh(obj_file)
    # mesh.show()

if __name__ == "__main__":
    main()

