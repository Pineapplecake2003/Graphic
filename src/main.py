import numpy as np
from utils import *
from vedo import *
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import DataStructure
# import pygame

def main():
    obj_file = "./models/Cube.obj"

    dpi = 1
    canva_height = 300
    canva_width = 300

    canva_height_px = canva_height * dpi
    canva_width_px = canva_width * dpi

    canva_d = 1000

    ambient = 0.1
    light_src0 = DataStructure.Light([600, 800, 1500], 0.7, "point")
    light_src1 = DataStructure.Light([1, 1, 0], 0.5, "directional")

    picture = Canva(
        (canva_height, canva_width),
        canva_d, 
        (canva_height_px, canva_width_px),
        dpi,
        ambient,
        [light_src0, light_src1]
    )
    
    object = load_objs(obj_file)
    object.transform((0, 0, 1000), (25, 160, 0), 100)
    object.set_s(5)
    print("Render with Flat shading.")
    for t in tqdm(object.triangles, ncols=50):
    #for t in object.triangles:
        DrawWireframeTriangle(
            object,
            t,
            picture, 
            (0xFF, 0xFF, 0xFF), 
            (0xFF, 0xFF, 0xFF),
            "Flat",
            s=object.s,
        )
    img = Image.fromarray(picture.array, mode="RGB")
    img.save("./images/result_Flat.png")

if __name__ == "__main__":
    main()

