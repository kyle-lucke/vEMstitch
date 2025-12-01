import os
from joblib import Parallel, delayed

import cv2
import numpy as np

from Utils import SIFT, direct_stitch
from elastic_transform import local_TPS
from rigid_transform import rigid_transform
from refinement import refinement_local, fast_brief

def stitching_pair(im1, im2, im1_color, im2_color, im1_mask, im2_mask, mode, overlap=0.15):
    kp1, dsp1, kp2, dsp2 = SIFT(im1, im2)
    im1_shape = im1.shape
    im2_shape = im2.shape
    H, ok, X1, X2 = rigid_transform(kp1, dsp1, kp2, dsp2, im1_mask, im2_mask, mode)

    if H is None:
        X1, X2, height, im1_region, im2_region = None, None, None, None, None
        height = int(im2_shape[1] * 0.15)
        im1_region = [0, im1_shape[0]]
        im2_region = [0, im2_shape[0]]
        H, ok, X1, X2 = fast_brief(im1, im2, im1_mask, im2_mask, X1, X2, height, im1_region, im2_region, mode)
        stitching_res, stitching_res_color, _, _, mass, overlap_mass = local_TPS(im1, im2, im1_color, im2_color, H, X1.T[:, ok], X2.T[:, ok], im1_mask,
                                                            im2_mask, mode)
        return stitching_res, stitching_res_color, mass, overlap_mass

    stitching_res, stitching_res_color, _, _, mass, overlap_mass = local_TPS(im1, im2, im1_color, im2_color, H, X1.T[:, ok], X2.T[:, ok], im1_mask, im2_mask, mode)
    return stitching_res, stitching_res_color, mass, overlap_mass


def stitching_rows(im1, im2, im1_color, im2_color, im1_mask, im2_mask, mode, refine_flag):
    kp1, dsp1, kp2, dsp2 = SIFT(im1, im2)
    H, ok, X1, X2 = rigid_transform(kp1, dsp1, kp2, dsp2, im1_mask, im2_mask, mode)
    if refine_flag:
        stitching_res, mass, overlap_mass = refinement_local(im1, im2, H, X1, X2, ok, im1_mask, im2_mask, mode)
        if stitching_res is None:
            stitching_res, stitching_res_color, _, _, mass, overlap_mass = local_TPS(im1, im2, im1_color, im2_color, H, X1.T[:, ok], X2.T[:, ok], im1_mask, im2_mask, mode)
    else:
        stitching_res, stitching_res_color, _, _, mass, overlap_mass = local_TPS(im1, im2, im1_color, im2_color, H, X1.T[:, ok], X2.T[:, ok], im1_mask, im2_mask,
                                                            mode)
    return stitching_res, stitching_res_color, mass, overlap_mass

def preprocess(im1, im2, im1_mask, im2_mask, mode):
    if mode == "r":
        half_w = int(im2.shape[1] // 2)
        half_h = int(im2.shape[0] // 2)
        im1_shape = im1.shape
        im2_shape = im2.shape
        if np.std(im1[:, -half_w:]) <= 12.0:
            h = im2_shape[0]
            extra_w = int(im1_shape[1] * 0.9)
            w = im2_shape[1] + extra_w
            stitching_res = np.zeros((h, w))
            mass = np.ones((h, w))
            stitching_res[:, -im2_shape[1]:] = im2
            stitching_res[:im1_shape[0], :extra_w] = im1[:, :extra_w]
            return stitching_res, mass, None

        if np.std(im2[:, :half_w]) <= 12.0:
            return direct_stitch(im1, im2, im1_mask, im2_mask)
        return True, True, True
    else:
        return True, True, True


def read_image(fname, grayscale=True):
    image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image

def post_process(image, cvt_color=True):
    # clip before casting to uint8
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    if cvt_color:    
        # convert to uint8, then RGB -> BGR for openCV saving
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

# Algorithm:
# 1) stitch together the 3 images in each row, this is what the "for i in range(3)" loop does
# 2) stitch together each image row
def three_stitching(data_path, store_path, top_num, file_ext, output_file_ext, refine_flag=False):

    tier_list = []
    tier_mask_list = []
    tier_list_color = []
    # for i in range(num_cols):
    for i in range(3):
        # img_1 = cv2.imread(os.path.join(data_path, "".join([top_num, "_" + str(i + 1), "_1", ".bmp"])))
        # img_2 = cv2.imread(os.path.join(data_path, "".join([top_num, "_" + str(i + 1), "_2", ".bmp"])))

        img_1_pth = os.path.join(data_path, "".join([top_num, "_" + str(i + 1), "_1", file_ext]))
        img_2_pth = os.path.join(data_path, "".join([top_num, "_" + str(i + 1), "_2", file_ext]))

        img_1 = read_image(img_1_pth)
        img_2 = read_image(img_2_pth)

        img1_color = read_image(img_1_pth, grayscale=False)
        img2_color = read_image(img_2_pth, grayscale=False)
        
        print(f"stitching {img_1_pth} and {img_2_pth}")
        print()
        
        if img_1 is not None and img_2 is not None:

            mode = "r"
            stitching_res_temp, mass_temp, process_flag = preprocess(img_1, img_2, None, None, mode)
            if process_flag:
                img_1_mask = np.ones(img_1.shape)
                img_2_mask = np.ones(img_2.shape)
                stitching_res, stitching_res_color, mass, _ = stitching_pair(img_1, img_2, img1_color, img2_color, img_1_mask, img_2_mask, mode)
                stitching_res = np.uint8(stitching_res)
                
            else:
                stitching_res, mass = stitching_res_temp, mass_temp
                stitching_res = np.uint8(stitching_res)
                
        elif img_1 is None:
            stitching_res = img_2
            mass = np.ones(img_2.shape)
            
        else:
            stitching_res = img_1
            mass = np.ones(img_1.shape)

        img_3_pth = os.path.join(data_path, "".join([top_num, "_" + str(i + 1), "_3", file_ext]))
        img_3 = read_image(img_3_pth)
        img3_color = read_image(img_3_pth, grayscale=False)
        if img_3 is None:
            tier_list.append(stitching_res)
            tier_mask_list.append(mass)
            tier_list_color.append(stitching_res_color)
            continue

        img_3_mask = np.ones(img_3.shape)
        mode = "r"
        stitching_res_temp, mass_temp, process_flag = preprocess(stitching_res, img_3, mass, None, mode)
        if process_flag:
            stitching_res, stitching_res_color, mass, _ = stitching_pair(stitching_res, img_3, stitching_res_color, img3_color, mass, img_3_mask, mode)
            stitching_res = np.uint8(stitching_res)
        else:
            stitching_res, mass = stitching_res_temp, mass_temp
            stitching_res = np.uint8(stitching_res)

            
        tier_list.append(stitching_res)
        tier_mask_list.append(mass)
        tier_list_color.append(stitching_res_color)

        ### DEBUG ###
        import matplotlib.pyplot as plt
        plt.axis('off')
        plt.imshow(post_process(stitching_res_color, cvt_color=False))
        plt.tight_layout()
        plt.show()
        # exit()
        ############
        
    while len(tier_list) >= 2:
        im1 = tier_list[0]
        im2 = tier_list[1]
        im1_mask = tier_mask_list[0]
        im2_mask = tier_mask_list[1]

        im1_color = tier_list_color[0]
        im2_color = tier_list_color[1]
        
        mode = "d"
        stitching_res, stitching_res_color, mass, overlap_mass = stitching_rows(im1, im2, im1_color, im2_color, im1_mask, im2_mask, mode, refine_flag)
        stitching_res = np.uint8(stitching_res)
        
        tier_list[1] = stitching_res
        tier_mask_list[1] = mass
        tier_list_color[1] = stitching_res_color
        
        tier_list = tier_list[1:]
        tier_mask_list = tier_mask_list[1:]
        tier_list_color = tier_list_color[1:]
        
    final_res = tier_list[0]
    
    final_res = np.uint8(final_res)

    final_res_color = tier_list_color[0]
    
    final_res_color = post_process(final_res_color)
    
    print(f"shape: {final_res_color.shape}, min/max: {final_res_color.min()}/{final_res_color.max()}")
    
    # cv2.imwrite(os.path.join(store_path, "".join([str(top_num), "-res", output_file_ext])), final_res_color)
    return final_res_color

# Algorithm:
# 1) stitch together the 3 images in each row, this is what the "for i in range(3)" loop does
# 2) stitch together each image row
def n_stitching(tile_grid, refine_flag=False):

    tier_list = []
    tier_mask_list = []
    tier_list_color = []

    # step 1) stitch together the images in each row across all columns
    for r in range(tile_grid.n_rows):
        for c in range(tile_grid.n_cols-1):

            if c == 0:
        
                img_1 = tile_grid.get_tile(r, c)
                img_2 = tile_grid.get_tile(r, c+1)
        
                img1_color = tile_grid.get_tile(r, c, grayscale=False)
                img2_color = tile_grid.get_tile(r, c+1, grayscale=False)

            else:

                img_1 = stitching_res
                img_2 = tile_grid.get_tile(r, c+1)

                img1_color = stitching_res_color
                img2_color = tile_grid.get_tile(r, c+1, grayscale=False)
        
            mode = "r"

            if c == 0:
            
                stitching_res_temp, mass_temp, process_flag = preprocess(img_1, img_2, None, None, mode)

            else:
                stitching_res_temp, mass_temp, process_flag = preprocess(img_1, img_2, mass, None, mode)
                
            if process_flag:

                if c == 0:
                    img_1_mask = np.ones(img_1.shape)
                else:
                    img_1_mask = mass
                    
                img_2_mask = np.ones(img_2.shape)
                stitching_res, stitching_res_color, mass, _ = stitching_pair(img_1, img_2, img1_color, img2_color, img_1_mask, img_2_mask, mode)
                stitching_res = np.uint8(stitching_res)
                
            else:
                stitching_res, mass = stitching_res_temp, mass_temp
                stitching_res = np.uint8(stitching_res)
               
        # append stitching result from this row, over all columns
        tier_list.append(stitching_res)
        tier_mask_list.append(mass)
        tier_list_color.append(stitching_res_color)

        ### DEBUG ###
        # import matplotlib.pyplot as plt
        # plt.axis('off')
        # plt.imshow(post_process(stitching_res_color, cvt_color=False))
        # plt.tight_layout()
        # plt.show()
        # exit()
        ############

    # stitch together image rows:
    while len(tier_list) >= 2:
        im1 = tier_list[0]
        im2 = tier_list[1]
        im1_mask = tier_mask_list[0]
        im2_mask = tier_mask_list[1]

        im1_color = tier_list_color[0]
        im2_color = tier_list_color[1]
        
        mode = "d"
        stitching_res, stitching_res_color, mass, overlap_mass = stitching_rows(im1, im2, im1_color, im2_color, im1_mask, im2_mask, mode, refine_flag)
        stitching_res = np.uint8(stitching_res)
        
        tier_list[1] = stitching_res
        tier_mask_list[1] = mass
        tier_list_color[1] = stitching_res_color
        
        tier_list = tier_list[1:]
        tier_mask_list = tier_mask_list[1:]
        tier_list_color = tier_list_color[1:]

    final_res_color = post_process(tier_list_color[0])
        
    return final_res_color


def stitch_columns_for_row(r, tile_grid, refine_flag):

    for c in range(tile_grid.n_cols-1):

        if c == 0:
        
            img_1 = tile_grid.get_tile(r, c)
            img_2 = tile_grid.get_tile(r, c+1)
        
            img1_color = tile_grid.get_tile(r, c, grayscale=False)
            img2_color = tile_grid.get_tile(r, c+1, grayscale=False)

        else:

            img_1 = stitching_res
            img_2 = tile_grid.get_tile(r, c+1)

            img1_color = stitching_res_color
            img2_color = tile_grid.get_tile(r, c+1, grayscale=False)
        
        mode = "r"

        if c == 0:
            
            stitching_res_temp, mass_temp, process_flag = preprocess(img_1, img_2, None, None, mode)

        else:
            stitching_res_temp, mass_temp, process_flag = preprocess(img_1, img_2, mass, None, mode)
            
        if process_flag:

            if c == 0:
                img_1_mask = np.ones(img_1.shape)
            else:
                img_1_mask = mass
                    
            img_2_mask = np.ones(img_2.shape)
            stitching_res, stitching_res_color, mass, _ = stitching_pair(img_1, img_2, img1_color, img2_color, img_1_mask, img_2_mask, mode)
            stitching_res = np.uint8(stitching_res)
                
        else:
            stitching_res, mass = stitching_res_temp, mass_temp
            stitching_res = np.uint8(stitching_res)
            
    return {'r': r,
            'stitching_res': stitching_res,
            "mass": mass,
            "stitching_res_color": stitching_res_color}

def n_stitching_parallel(tile_grid, n_jobs, refine_flag=False):

    tier_list = []
    tier_mask_list = []
    tier_list_color = []

    # step 1) stitch together the images in each row across all columns
    row_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(stitch_columns_for_row)(
            r, tile_grid, refine_flag
        ) for r in range(tile_grid.n_rows) 
    )
    
    tier_list = []
    tier_mask_list = []
    tier_list_color = []

    # Put stitched rows into correct order for further stitching
    for res in sorted(row_results, key=lambda x: x['r']):
        tier_list.append(res['stitching_res'])
        tier_mask_list.append(res['mass'])
        tier_list_color.append(res['stitching_res_color'])

    # stitch together image rows:
    while len(tier_list) >= 2:
        im1 = tier_list[0]
        im2 = tier_list[1]
        im1_mask = tier_mask_list[0]
        im2_mask = tier_mask_list[1]

        im1_color = tier_list_color[0]
        im2_color = tier_list_color[1]
        
        mode = "d"
        stitching_res, stitching_res_color, mass, overlap_mass = stitching_rows(im1, im2, im1_color, im2_color, im1_mask, im2_mask, mode, refine_flag)
        stitching_res = np.uint8(stitching_res)
        
        tier_list[1] = stitching_res
        tier_mask_list[1] = mass
        tier_list_color[1] = stitching_res_color
        
        tier_list = tier_list[1:]
        tier_mask_list = tier_mask_list[1:]
        tier_list_color = tier_list_color[1:]

    final_res_color = post_process(tier_list_color[0])
    
    return final_res_color

