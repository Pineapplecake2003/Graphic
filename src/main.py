import numpy as np
from utils import *
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import DataStructure
def main():
    obj_file = "./models/Rushia's_head.obj"

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
    with open("obj.dump", "w") as f:
        rotation = (25, 160, 0)
        location = (-200, -1250, 1800)
        scale = 1.5
        alpha, beta, gamma = rotation
        a, b, g = np.deg2rad([alpha, beta, gamma])
        r_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(a), -np.sin(a)],
                [0, np.sin(a), np.cos(a)]
            ],
            dtype=np.float32
        )
        r_y = np.array(
            [
                [np.cos(b), 0, np.sin(b)],
                [0, 1, 0],
                [-np.sin(b), 0, np.cos(b)]
            ],
            dtype=np.float32
        )
        r_z = np.array(
            [
                [np.cos(g), -np.sin(g), 0],
                [np.sin(g), np.cos(g), 0],
                [0, 0, 1]
            ],
            dtype=np.float32
        )
        R = r_z @ r_y @ r_x
        offset_loc = np.array(location, dtype=float)
        S = object.scale_matrix(scale)
        R4 = object.rotation_matrix_4x4(R)
        T = object.translation_matrix(location)
        M = T @ R4 @ S
        # N
        f.write(f"{to_hex32(len(object.triangles) * 3)}\n")

        # MV matrix (4x4)
        for row in range(4):
            for col in range(4):
                f.write(f"{to_hex32(M[row, col])}\n")
        
        # Lp (Lpx,Lpy,Lpz) 
        f.write(f"{to_hex32(np.float32(600.0))}\n")
        f.write(f"{to_hex32(np.float32(800.0))}\n")
        f.write(f"{to_hex32(np.float32(1500.0))}\n")
        # Ld (Ldx,Ldy,Ldz
        f.write(f"{to_hex32(light_src1.li_dir[0])}\n")
        f.write(f"{to_hex32(light_src1.li_dir[1])}\n")
        f.write(f"{to_hex32(light_src1.li_dir[2])}\n")

        # L_intensity
        f.write(f"{to_hex32(np.float32(light_src0.b))}\n")
        f.write(f"{to_hex32(np.float32(light_src1.b))}\n")
        f.write(f"{to_hex32(np.float32(ambient))}\n")

        # P scale x, y
        f.write(f"{to_hex32(np.float32(1.5))}\n")
        f.write(f"{to_hex32(np.float32(1.5))}\n")

        # Records (repeat N):
        for t in object.triangles:
            for p, vn in zip(t.points, t.vns):
                f.write(f"{to_hex32(p.loc[0])}\n")
                f.write(f"{to_hex32(p.loc[1])}\n")
                f.write(f"{to_hex32(p.loc[2])}\n")
                f.write(f"{to_hex32(np.float32(1.0))}\n")
                f.write(f"{to_hex32(vn[0])}\n")
                f.write(f"{to_hex32(vn[1])}\n")
                f.write(f"{to_hex32(vn[2])}\n")


    object.transform((-200, -1250, 1800), (25, 160, 0), 1.5)
    object.set_s(1)
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
    

