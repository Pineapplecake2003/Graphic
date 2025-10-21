import numpy as np
from utils import *
from vedo import *
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import DataStructure
import pygame

def main():
    obj_file = "./models/Rushia.obj"

    dpi = 30
    canva_height = 300
    canva_width = 300

    canva_height_px = canva_height * dpi
    canva_width_px = canva_width * dpi

    canva_d = 1000

    ambient = 0.4
    light_src = DataStructure.Point([300, 0, 4800], 0.7)

    picture = Canva(
        (canva_height, canva_width),
        canva_d, 
        (canva_height_px, canva_width_px),
        dpi,
        ambient,
        [light_src]
    )
    
    object = load_objs(obj_file)
    object.transform((0, -200, 5015), (30, 180, 0), 1.0)
    for t in tqdm(object.triangles, ncols=50):
        DrawWireframeTriangle(
            t,
            picture, 
            (0x4E, 0xFE, 0xB3), 
            (0x4E, 0xFE, 0xB3)
        )

    img = Image.fromarray(picture.array, mode="RGB")
    img.save("./images/result.png")

if __name__ == "__main__":
    main()

