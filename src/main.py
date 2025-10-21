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

    dpi = 2
    canva_height = 360
    canva_width = 640

    canva_height_px = canva_height * dpi
    canva_width_px = canva_width * dpi

    canva_d = 1000

    ambient = 0.5
    light_src = DataStructure.Point([10, 10, 1050], 1.0)

    picture = Canva(
        (canva_height, canva_width),
        canva_d, 
        (canva_height_px, canva_width_px),
        dpi,
        ambient,
        [light_src]
    )
    
    object = load_objs(obj_file)
    object.transform((0, -500, 4000), (0, 0, 0), 0.75)
    print("sd")
    for t in tqdm(object.triangles, ncols=80):
        DrawWireframeTriangle(
            t.points[0],
            t.points[1],
            t.points[2],
            picture, (0x00, 0x00, 0x00), (0x00, 0x00, 0x00)
        )

    img = Image.fromarray(picture.array, mode="RGB")
    img.save("./images/result.png")

    # pygame.init()
    # try:
    #     screen = pygame.display.set_mode((canva_width_px, canva_height_px))
    #     pygame.display.set_caption("Graphic Renderer")
    #     surface_array = np.ascontiguousarray(np.transpose(picture.array, (1, 0, 2)))
    #     pygame_surface = pygame.surfarray.make_surface(surface_array)
    #     clock = pygame.time.Clock()
    #     running = True
    #     while running:
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 running = False
    #             elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
    #                 running = False
    #         screen.blit(pygame_surface, (0, 0))
    #         pygame.display.flip()
    #         clock.tick(60)
    # finally:
    #     pygame.quit()

if __name__ == "__main__":
    main()

