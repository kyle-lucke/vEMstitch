import os

import cv2
import pandas as pd

def read_image(fname, grayscale=True):
    image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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

  # debugging function
    def plot_grid(self, color=False, save_fname=''):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(nrows=self.n_rows, ncols=self.n_cols)

        for ax in axs.flat:
            ax.axis('off')

        cmap = None if color else 'gray'
            
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                axs[r][c].imshow(self.get_tile(r, c, grayscale=not color), cmap=cmap)

        plt.tight_layout()

        if save_fname:
            plt.savefig(save_fname, dpi=1200, bbox_inches='tight', pad_inches=0.05)
            plt.close()
        else:
            plt.show()
