import os
import logging
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import tile_grid
from stitching import two_stitching, three_stitching, n_stitching, n_stitching_parallel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = ArgumentParser(description="Autostitch")
parser.add_argument('tile_grid')
parser.add_argument('input_path', type=str, help='input file path')
parser.add_argument('store_path', type=str, help='store res path')
parser.add_argument('--pattern', type=str, default='n', help='two, three, or n')

parser.add_argument('--refine', action='store_true', default=False, help='refine or not')

parser.add_argument('--input-file-ext', default='tif', choices=['tif', 'bmp', 'png'])
# parser.add_argument('--output-file-ext', default='.png', choices=['png', 'tif', 'bmp'])

parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n-jobs', type=int, default=4)

args = parser.parse_args()
pattern = args.pattern
refine_flag = args.refine

data_path = args.input_path
store_path = args.store_path
file_ext = f".{args.input_file_ext}"
# output_file_ext = f".{args.output_file_ext}"

if pattern != 'n':
    pattern = int(pattern)

if not os.path.exists(Path(store_path).parent):
    os.mkdir(store_path)

df = pd.read_csv(args.tile_grid, header=None)
            
tile_grid = tile_grid.TileGridFromDataFrame(data_path, df)

logger.info("TILE GRID:")
logger.info("\n" + str(tile_grid))

# tile_grid.plot_grid(color=True)
# exit()

if pattern == 3:
    stitched_image = three_stitching(tile_grid, refine_flag=refine_flag)

if pattern == 2:
    two_stitching(tile_grid, refine_flag=refine_flag)
    
elif pattern == 'n':

    # limit jobs to number of rows
    if args.parallel:
        n_jobs = args.n_jobs
        if args.n_jobs > tile_grid.n_rows:
            logger.info(f'args.n_jobs > number of rows in image grid. Using {tile_grid.n_rows} jobs.')
                    
            n_jobs = tile_grid.n_rows
        
        stitched_image = n_stitching_parallel(tile_grid, n_jobs, refine_flag=refine_flag)
        
    else:
        stitched_image = n_stitching(tile_grid, refine_flag=refine_flag)

# save stitched image
im_out_fname = store_path
logging.info(f"Saving result to: {im_out_fname}")
cv2.imwrite(im_out_fname, stitched_image)
