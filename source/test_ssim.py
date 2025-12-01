import argparse

import cv2
from skimage.metrics import structural_similarity

def main():

  cli = argparse.ArgumentParser()

  cli.add_argument("im1_pth")
  cli.add_argument("im2_pth")

  args = cli.parse_args()
  
  im1 = cv2.cvtColor(cv2.imread(args.im1_pth), cv2.COLOR_BGR2GRAY)
  im2 = cv2.cvtColor(cv2.imread(args.im2_pth), cv2.COLOR_BGR2GRAY)

  if im1.shape != im2.shape:
    h, w = im1.shape[:2]
    im2 = cv2.resize(im2, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
  (score, diff) = structural_similarity(im1, im2, full=True)
  print(f"SSIM: {score:.4f}")

  import matplotlib.pyplot as plt
  plt.imshow(diff)
  plt.show()
  
if __name__ == '__main__':
  main()
