import os
from joblib import Parallel, delayed

import cv2
import numpy as np

from .vis_utils import post_process_image, plot_single_image
from .elastic_transform import local_TPS
from .rigid_transform import rigid_transform, similarity_transform, estimate_similarity_transform
from .refinement import refinement_local, fast_brief

import logging
logger = logging.getLogger(__name__)

# Constants:
FLANN_RATIO_PAIR = 0.5
FLANN_RATIO_ROWS = 0.5

SUBSET_FLANN_PAIR = True
SUBSET_FLANN_ROWS = True

PLOT_KP_MATCHES_PAIR =False
PLOT_KP_MATCHES_ROWS = False

PLOT_KP_MATCHES_RANSAC_PAIR = False
PLOT_KP_MATCHES_RANSAC_ROWS = False

FEATURE_EXTRACTORS = ['sift', 'orb', 'freak']

def stitching_pair(im1, im2, im1_color, im2_color, im1_mask, im2_mask, mode, overlap=0.15, sift_mask_percent=0.1):
    
    # mask everything but RHS edge for im1:
    im1_sift_mask = np.zeros_like(im1, dtype=np.uint8)
    im1_sift_mask_start = im1.shape[1] - int(im1.shape[1] * sift_mask_percent)
    im1_sift_mask[:, im1_sift_mask_start:] = 1

    # mask everything but LHS edge for im2:
    im2_sift_mask = np.zeros_like(im2, dtype=np.uint8)
    im2_sift_mask_end = int(im2.shape[1] * sift_mask_percent)
    im2_sift_mask[:, :im2_sift_mask_end] = 1

    # plot_single_image(im1 * im1_sift_mask)
    # plot_single_image(im2 * im2_sift_mask)
    # exitt()

    # kp1, kp2 = detect_and_match(im1, im2, im1_sift_mask, im2_sift_mask)
        
    # kp1, dsp1, kp2, dsp2 = SIFT(im1, im2,
    #                             im1_mask=im1_sift_mask, im2_mask=im2_sift_mask)

    # H, ok, X1, X2 = estimate_similarity_transform(
    #     kp1, dsp1, kp2, dsp2, im1_mask, im2_mask,
    #     mode, flann_ratio=FLANN_RATIO_PAIR,
    #     subset_flann=SUBSET_FLANN_PAIR,
    #     kwargs={
    #         'im1': im1, 'im2': im2,
    #         'im1_color': im1_color,
    #         'im2_color': im2_color,
    #         'plot_kp_matches': PLOT_KP_MATCHES_PAIR,
    #         'plot_kp_matches_ransac': PLOT_KP_MATCHES_RANSAC_PAIR
    #                                      }
    # )

    H, ok, X1, X2 = estimate_similarity_transform(
        im1, im2,
        im1_mask, im2_mask,
        im1_sift_mask, im2_sift_mask,
        FEATURE_EXTRACTORS, mode,
        flann_ratio=FLANN_RATIO_PAIR,
        subset_flann=SUBSET_FLANN_PAIR,
        kwargs={
            'im1': im1, 'im2': im2,
            'im1_color': im1_color,
            'im2_color': im2_color,
            'plot_kp_matches': PLOT_KP_MATCHES_PAIR,
            'plot_kp_matches_ransac': PLOT_KP_MATCHES_RANSAC_PAIR
        }
    )
    
    if H is None:
        raise RuntimeError("Error estimating similiarity matrix.")
        
    stitching_res, stitching_res_color, _, _, mass, overlap_mass = local_TPS(im1, im2, im1_color, im2_color, H, X1.T[:, ok], X2.T[:, ok], im1_mask, im2_mask, mode)

    # stitching_res, stitching_res_color, _, _, mass, overlap_mass = local_TPS_stable(im1, im2, im1_color, im2_color, H, X1.T[:, ok], X2.T[:, ok], im1_mask, im2_mask, mode)
        
    return stitching_res, stitching_res_color, mass, overlap_mass


def stitching_rows(im1, im2, im1_color, im2_color, im1_mask, im2_mask, mode, refine_flag, sift_mask_percent=0.15):

    # mask bottom part of upper image
    im1_sift_mask = np.zeros_like(im1, dtype=np.uint8)
    im1_sift_mask_start = im1.shape[0] - int(im1.shape[0] * sift_mask_percent)
    im1_sift_mask[im1_sift_mask_start:] = 1

    # mask lower part of lower image
    im2_sift_mask = np.zeros_like(im2, dtype=np.uint8)
    im2_sift_mask_end = int(im2.shape[0] * sift_mask_percent)
    im2_sift_mask[:im2_sift_mask_end] = 1

    # plot_single_image(im1 * im1_sift_mask)
    # plot_single_image(im2 * im2_sift_mask)
    # exit()
        
    # kp1, dsp1, kp2, dsp2 = SIFT(im1, im2, im1_sift_mask, im2_sift_mask)

    # H, ok, X1, X2 = similarity_transform(kp1, dsp1,
    #                                      kp2, dsp2,
    #                                      im1_mask, im2_mask, mode,
    #                                      flann_ratio=FLANN_RATIO_ROWS,
    #                                      subset_flann=SUBSET_FLANN_ROWS,
    #                                      kwargs={
    #                                          'im1': im1, 'im2': im2,
    #                                          'im1_color': im1_color,
    #                                          'im2_color': im2_color,
    #                                          'plot_kp_matches': PLOT_KP_MATCHES_ROWS,
    #                                          'plot_kp_matches_ransac': PLOT_KP_MATCHES_RANSAC_ROWS
    #                                      })

    H, ok, X1, X2 = estimate_similarity_transform(
        im1, im2,
        im1_mask, im2_mask,
        im1_sift_mask, im2_sift_mask,
        FEATURE_EXTRACTORS, mode,
        flann_ratio=FLANN_RATIO_ROWS,
        subset_flann=SUBSET_FLANN_ROWS,
        kwargs={
            'im1': im1, 'im2': im2,
            'im1_color': im1_color,
            'im2_color': im2_color,
            'plot_kp_matches': PLOT_KP_MATCHES_ROWS,
            'plot_kp_matches_ransac': PLOT_KP_MATCHES_RANSAC_ROWS
        }
    )
    
    if H is None:
        raise RuntimeError("Error estimating similiarity matrix.")
    
    if refine_flag:
        stitching_res, stitching_res_color, mass, overlap_mass = refinement_local(im1, im2, im1_color, im2_color, H, X1, X2, ok, im1_mask, im2_mask, mode)
        if stitching_res is None:
            stitching_res, stitching_res_color, _, _, mass, overlap_mass = local_TPS(im1, im2, im1_color, im2_color, H, X1.T[:, ok], X2.T[:, ok], im1_mask, im2_mask, mode)
            # stitching_res, stitching_res_color, _, _, mass, overlap_mass = local_TPS_stable(im1, im2, im1_color, im2_color, H, X1.T[:, ok], X2.T[:, ok], im1_mask, im2_mask, mode)
    else:
        stitching_res, stitching_res_color, _, _, mass, overlap_mass = local_TPS(im1, im2, im1_color, im2_color, H, X1.T[:, ok], X2.T[:, ok], im1_mask, im2_mask, mode)
        
        # stitching_res, stitching_res_color, _, _, mass, overlap_mass = local_TPS_stable(im1, im2, im1_color, im2_color, H, X1.T[:, ok], X2.T[:, ok], im1_mask, im2_mask, mode)
    return stitching_res, stitching_res_color, mass, overlap_mass

def preprocess(im1, im2, im1_color, im2_color, im1_mask, im2_mask, mode):
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
            stitching_res_color = np.zeros((h, w, 3))
            mass = np.ones((h, w))

            stitching_res[:, -im2_shape[1]:] = im2
            stitching_res[:im1_shape[0], :extra_w] = im1[:, :extra_w]

            stitching_res_color[:, -im2_shape[1]:] = im2_color
            stitching_res_color[:im1_shape[0], :extra_w] = im1_color[:, :extra_w]
            
            return stitching_res, stitching_res_color, mass, None

        if np.std(im2[:, :half_w]) <= 12.0:
            return direct_stitch(im1, im2, im1_color, im2_color, im1_mask, im2_mask)
        return True, True, True, True
    else:
        return True, True, True, True

# Algorithm:
# 1) stitch together the images in each row
# 2) stitch together each image row
def n_stitching(tile_grid, refine_flag=False):

    tier_list = []
    tier_mask_list = []
    tier_list_color = []

    # tile_grid.plot_grid(color=True)
    # exit()
    
    # special case: there is only 1 column in the tile grid
    if tile_grid.n_cols == 1:
        for r in range(tile_grid.n_rows):
            stitching_res = tile_grid.get_tile(r, 0)
            stitching_res_color = tile_grid.get_tile(r, 0, grayscale=False)

            mass = np.ones(stitching_res.shape)

            tier_list.append(stitching_res)
            tier_mask_list.append(mass)
            tier_list_color.append(stitching_res_color)
            
    else:
            
        # step 1) stitch together the images in each row across all columns
        for r in range(tile_grid.n_rows):
            logger.info(f"Stitching columns for row {r+1} / {tile_grid.n_rows}")
            for c in range(tile_grid.n_cols-1):
                logger.info(f"Stitching column {c+1} / {tile_grid.n_cols-1}")
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
            
                    stitching_res_temp, stitching_res_color_temp, mass_temp, process_flag = preprocess(img_1, img_2, img1_color, img2_color, None, None, mode)

                else:
                    stitching_res_temp, stitching_res_color_temp, mass_temp, process_flag = preprocess(img_1, img_2, img1_color, img2_color, mass, None, mode)
                
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
                    stitching_res_color = stitching_res_color_temp
                    stitching_res = np.uint8(stitching_res)

                ### DEBUG ###

                # import matplotlib.pyplot as plt
                # plt.axis('off')
                # plt.imshow(post_process_image(stitching_res_color, cvt_color=False))
                # plt.tight_layout()
                # plt.show()
                # # exit()

                ############
                
            # append stitching result from this row, over all columns
            tier_list.append(stitching_res)
            tier_mask_list.append(mass)
            tier_list_color.append(stitching_res_color)

            ### DEBUG ###
            # import matplotlib.pyplot as plt
            # plt.axis('off')
            # plt.title(f"Stitched (Row {r+1} / {tile_grid.n_rows})")
            # plt.imshow(post_process_image(stitching_res_color, cvt_color=False))
            # mng = plt.get_current_fig_manager()
            # mng.resize(*mng.window.maxsize())
            # plt.tight_layout()
            # plt.show()
            # # exit()

            ############

    # stitch together image rows:
    logger.info(f"Stitching rows")
    
    row_number = 0
    while len(tier_list) >= 2:
        logging.info(f"Stitching row {row_number+1} / {tile_grid.n_rows}")
        row_number += 1
        
        im1 = tier_list[0]
        im2 = tier_list[1]
        im1_mask = tier_mask_list[0]
        im2_mask = tier_mask_list[1]

        im1_color = tier_list_color[0]
        im2_color = tier_list_color[1]
        
        mode = "d"
        stitching_res, stitching_res_color, mass, overlap_mass = stitching_rows(im1, im2, im1_color, im2_color, im1_mask, im2_mask, mode, refine_flag)
        stitching_res = np.uint8(stitching_res)

        ### DEBUG ###
        # import matplotlib.pyplot as plt
        # plt.axis('off')
        # plt.title(f"row {row_number+1} -> {row_number+2}")
        # plt.imshow(post_process_image(stitching_res_color, cvt_color=False))
        # plt.tight_layout()
        # plt.show()
        # exit()
        ##############
        
        tier_list[1] = stitching_res
        tier_mask_list[1] = mass
        tier_list_color[1] = stitching_res_color
        
        tier_list = tier_list[1:]
        tier_mask_list = tier_mask_list[1:]
        tier_list_color = tier_list_color[1:]

    # clip image to [0, 255] range and convert to BGR
    final_res_color = post_process_image(tier_list_color[0])

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
            
            stitching_res_temp, stitching_res_color_temp, mass_temp, process_flag = preprocess(img_1, img_2, img1_color, img2_color, None, None, mode)

        else:
            stitching_res_temp, stitching_res_color_temp, mass_temp, process_flag = preprocess(img_1, img_2, img1_color, img2_color, mass, None, mode)

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
            stitching_res_color = stitching_res_color_temp
            stitching_res = np.uint8(stitching_res)
            
    return {'row_idx': r,
            'stitching_res': stitching_res,
            'mass': mass,
            'stitching_res_color': stitching_res_color}

def n_stitching_parallel(tile_grid, n_jobs, refine_flag=False):

    tier_list = []
    tier_mask_list = []
    tier_list_color = []

    # special case: there is only 1 column in the tile grid
    if tile_grid.n_cols == 1:

        row_results = []
        for r in range(tile_grid.n_rows):
            stitching_res = tile_grid.get_tile(r, 0)
            stitching_res_color = tile_grid.get_tile(r, 0, grayscale=False)

            mass = np.ones(stitching_res.shape)
            
            row_results.append({
                'row_idx': r,
                'stitching_res': stitching_res,
                'mass': mass,
                'stitching_res_color': stitching_res_color
            })
        
    else:
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
    for res in sorted(row_results, key=lambda x: x['row_idx']):
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

    final_res_color = post_process_image(tier_list_color[0])
    
    return final_res_color

