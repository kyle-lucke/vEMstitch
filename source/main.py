import os
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import tile_grid
from stitching import three_stitching, n_stitching, n_stitching_parallel

parser = ArgumentParser(description="Autostitch")
parser.add_argument('tile_grid')
parser.add_argument('input_path', type=str, help='input file path')
parser.add_argument('store_path', type=str, help='store res path')
parser.add_argument('--pattern', type=str, default='n', help='two or three')

parser.add_argument('--refine', action='store_true', default=False, help='refine or not')

parser.add_argument('--file-ext', default='.tif', choices=['.tif', '.bmp', '.png'])
parser.add_argument('--output-file-ext', default='.png', choices=['.png', '.tif', '.bmp'])

parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n-jobs', type=int, default=4)

args = parser.parse_args()
pattern = args.pattern
refine_flag = args.refine

data_path = args.input_path
store_path = args.store_path
file_ext = args.file_ext
output_file_ext = args.output_file_ext

if pattern != 'n':
    pattern = int(pattern)

# image_list = []
# for root, dirs, files in os.walk(data_path):
#     for file in files:
#         first, _, _ = file.split(".")[0].split("_")
#         image_list.append(first)
# image_list = list(set(image_list))
# image_list.sort()

# file_ext = '.tif'

if not os.path.exists(store_path):
    os.mkdir(store_path)

# for top_num in tqdm(image_list):
if pattern == 3:
        # try:
    stitched_image = three_stitching(data_path, store_path, file_ext, args.output_file_ext, refine_flag=refine_flag)
    
elif pattern == 'n':

    df = pd.read_csv(args.tile_grid, header=None)
            
    tile_grid = tile_grid.TileGridFromDataFrame(data_path, df)
    
    if args.parallel:
        n_jobs = args.n_jobs
        if args.n_jobs > tile_grid.n_rows:
            print(f'args.n_jobs > number of rows in image grid. Using {tile_grid.n_rows} jobs.')
                    
            n_jobs = tile_grid.n_rows
        
        stitched_image = n_stitching_parallel(tile_grid, n_jobs, refine_flag=refine_flag)
        
    else:
        stitched_image = n_stitching(tile_grid, refine_flag=refine_flag)

cv2.imwrite(os.path.join(store_path, f"test-res{output_file_ext}"), stitched_image)
