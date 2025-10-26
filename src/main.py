import numpy as np
from utils import *
from vedo import *
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import DataStructure
# import pygame

def main():
    obj_file = "./models/Rushia.obj"

    dpi = 30
    canva_height = 400
    canva_width = 400

    canva_height_px = canva_height * dpi
    canva_width_px = canva_width * dpi

    canva_d = 1000

    ambient = 0.2
    light_src0 = DataStructure.Light([300, -500, 6200], 0.7, "point")
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
    object.transform((0, -200, 5015), (0, 225, 45), 1.0)
    object.set_s(10)
    print("Render with Phong shading.")
    for t in tqdm(object.triangles, ncols=50):
        DrawWireframeTriangle(
            t,
            picture, 
            (0x4E, 0xFE, 0xB3), 
            (0x4E, 0xFE, 0xB3),
            "Phong",
            s=object.s,
        )
    img = Image.fromarray(picture.array, mode="RGB")
    img.save("./images/result_Phong.png")
    
    picture.clear()

    print("Render with Flat shading.")
    for t in tqdm(object.triangles, ncols=50):
        DrawWireframeTriangle(
            t,
            picture, 
            (0x4E, 0xFE, 0xB3), 
            (0x4E, 0xFE, 0xB3),
            "Flat",
            s=object.s,
        )
    img = Image.fromarray(picture.array, mode="RGB")
    img.save("./images/result_Flat.png")

if __name__ == "__main__":
    main()

