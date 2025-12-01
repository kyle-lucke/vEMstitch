import os
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import tile_grid
from stitching import three_stitching, n_stitching # two_stitching

parser = ArgumentParser(description="Autostitch")
parser.add_argument('--input_path', type=str, default="", help='input file path')
parser.add_argument('--store_path', type=str, default="", help='store res path')
parser.add_argument('--pattern', type=str, default=3, help='two or three')
parser.add_argument('--refine', action='store_true', default=False, help='refine or not')
parser.add_argument('--file-ext', default='.tif', choices=['.tif', '.bmp', '.png'])
parser.add_argument('--output-file-ext', default='.png', choices=['.png', '.tif', '.bmp'])

args = parser.parse_args()
pattern = args.pattern
refine_flag = args.refine

data_path = args.input_path
store_path = args.store_path
file_ext = args.file_ext

if pattern != 'n':
    pattern = int(pattern)

image_list = []
for root, dirs, files in os.walk(data_path):
    for file in files:
        first, _, _ = file.split(".")[0].split("_")
        image_list.append(first)
image_list = list(set(image_list))
image_list.sort()

# file_ext = '.tif'

if not os.path.exists(store_path):
    os.mkdir(store_path)

for top_num in tqdm(image_list):
    if pattern == 3:
        # try:
            three_stitching(data_path, store_path, top_num, file_ext, args.output_file_ext, refine_flag=refine_flag)
        # except Exception:
        #     final_res = np.zeros((1000, 1000))
        #     cv2.imwrite(os.path.join(store_path, "".join([str(top_num), "-res", file_ext])), final_res)
    elif pattern == 2:
        try:
            two_stitching(data_path, store_path, top_num, file_ext, refine_flag=refine_flag)
        except Exception:
            final_res = np.zeros((1000, 1000))
            cv2.imwrite(os.path.join(store_path, "".join([str(top_num), "-res", args.output_file_ext])), final_res)

    elif pattern == 'n':

        df = pd.DataFrame(
            [
                ["test_1_1.tif",	"test_1_2.tif",	"test_1_3.tif"],
                ["test_2_1.tif",	"test_2_2.tif", "test_2_3.tif"],
                ["test_3_1.tif",	"test_3_2.tif",	"test_3_3.tif"]
            ]
        )
        
        tile_grid = tile_grid.TileGridFromDataFrame(data_path, df)

        n_stitching(data_path, store_path, top_num, file_ext, args.output_file_ext,
                    tile_grid, refine_flag=refine_flag)
