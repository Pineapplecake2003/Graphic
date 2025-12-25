import numpy as np
from utils import *
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import DataStructure
def main():
    obj_file = "./models/teapot.obj"

    dpi = 1
    canva_height = 480
    canva_width = 680

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
    object.transform((0, 0, 2000), (-45, 20, 0), 300.0)
    object.set_s(1)
    
    # Clear the triangle info file
    with open("triangle_info.txt", "w") as f:
        pass

    print("Render with Flat shading.")
    for t in tqdm(object.triangles, ncols=50):
    #for t in object.triangles:
        DrawWireframeTriangle(
            t,
            picture, 
            (0xFF, 0xFF, 0xFF), 
            (0xFF, 0xFF, 0xFF),
            "Flat",
            s=object.s,
        )
    img = Image.fromarray(picture.array, mode="RGB")
    img.save("./images/result_Flat.png")
    exit()
    picture.clear()
# 
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
# 
    picture.clear()
# 
    print("Render vertices and lines only.")
    for t in tqdm(object.triangles, ncols=50):
        DrawWireframeTriangle(
            t,
            picture, 
            (0xFF, 0xFF, 0xFF), 
            (0xFF, 0xFF, 0xFF),
            "None",
            s=object.s,
        )
    img = Image.fromarray(picture.array, mode="RGB")
    img.save("./images/result_Vertices_and_lines.png")

if __name__ == "__main__":
    main()
    

