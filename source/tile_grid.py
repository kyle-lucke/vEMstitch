import os

import cv2
import pandas as pd

def read_image(fname, grayscale=True):
    image = cv2.imread(fname)

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


class TileGridFromDataFrame:
    def __init__(self, image_dirpath, df):

        self.image_dirpath = image_dirpath
        self.df = df
        
        self.n_rows = df.shape[0]
        self.n_cols = df.shape[1]

    def get_tile(self, r, c, grayscale=True):

      tile_path = self.get_tile_fname(r, c)

      return read_image(tile_path, grayscale)

    def get_tile_fname(self, r, c):
      return os.path.join(self.image_dirpath, self.df.iloc[r, c])
    
    def __str__(self):
      str_rep = f"{self.image_dirpath}\n"
      str_rep += self.df.to_string(header=False, index=False)

      return str_rep
