import os
import glob
import csv
import ast

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import openslide

PAIP_PATH = os.path.join(os.path.sep, "nas-ctm01", "datasets", "public", "PAIP_CRC")
PAIP_TILES_CSV = "PAIP_tiles_512_tissuethr_100.csv"
PAIP_TILES_RAW_OTSU_CSV = "PAIP_tiles_512_tissuethr_100_raw_otsu.csv"
SLIDES_PATH = os.path.join(PAIP_PATH, "slides")
SLIDE_EXT = ".svs"

PATCH_SIZE = 512

df = pd.read_csv(os.path.join(PAIP_PATH, PAIP_TILES_CSV))
df["100_tiles_grid"] = df["100_tiles_grid"].apply(ast.literal_eval)
df["100_tiles_coords"] = df["100_tiles_coords"].apply(ast.literal_eval)


example = 0
print(df["slide_name"][example], df["100_ntiles"][example], len(df["100_tiles_grid"][example]), len(df["100_tiles_coords"][example]))


slide_name = df["slide_name"][example] + SLIDE_EXT
slide = openslide.open_slide(os.path.join(SLIDES_PATH, slide_name))
print(slide.level_count, slide.dimensions, slide.level_dimensions, slide.level_downsamples)

tile_id = 0
print("loading tile... ", df["100_tiles_grid"][example][tile_id])

tile_grid = np.array(df["100_tiles_grid"][example])
print("tile grid... ", tile_grid.shape)
plt.scatter(tile_grid[:, 0], tile_grid[:, 1])
plt.savefig("tile-grid.png")
plt.close()

tile = slide.read_region(df["100_tiles_coords"][example][tile_id], level=0, size=(PATCH_SIZE, PATCH_SIZE))
tile = np.array(tile)
print(tile.shape)

plt.imshow(tile)
plt.savefig("my-first-tile.png")
plt.close()