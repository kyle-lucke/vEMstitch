from Utils import SIFT, direct_stitch
from rigid_transform import rigid_transform
import numpy as np
from elastic_transform import local_TPS
import cv2
import os
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
        stitching_res, _, _, mass, overlap_mass = local_TPS(im1, im2, im1_color, im2_color, H, X1.T[:, ok], X2.T[:, ok], im1_mask,
                                                            im2_mask, mode)
        return stitching_res, mass, overlap_mass

    stitching_res, _, _, mass, overlap_mass = local_TPS(im1, im2, im1_color, im2_color, H, X1.T[:, ok], X2.T[:, ok], im1_mask, im2_mask, mode)
    return stitching_res, mass, overlap_mass


def stitching_rows(im1, im2, im1_mask, im2_mask, mode, refine_flag):
    kp1, dsp1, kp2, dsp2 = SIFT(im1, im2)
    H, ok, X1, X2 = rigid_transform(kp1, dsp1, kp2, dsp2, im1_mask, im2_mask, mode)
    if refine_flag:
        stitching_res, mass, overlap_mass = refinement_local(im1, im2, H, X1, X2, ok, im1_mask, im2_mask, mode)
        if stitching_res is None:
            stitching_res, _, _, mass, overlap_mass = local_TPS(im1, im2, H, X1.T[:, ok], X2.T[:, ok], im1_mask, im2_mask, mode)
    else:
        stitching_res, _, _, mass, overlap_mass = local_TPS(im1, im2, H, X1.T[:, ok], X2.T[:, ok], im1_mask, im2_mask,
                                                            mode)
    return stitching_res, mass, overlap_mass


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


def two_stitching(data_path, store_path, top_num, file_ext, refine_flag=False):
    tier_list = []
    tier_mask_list = []
    for i in range(2):
        img_1 = cv2.imread(os.path.join(data_path, "".join([top_num, "_" + str(i + 1), "_1", file_ext])))
        img_2 = cv2.imread(os.path.join(data_path, "".join([top_num, "_" + str(i + 1), "_2", file_ext])))

        if img_1 is not None and img_2 is not None:
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

            mode = "r"
            stitching_res_temp, mass_temp, process_flag = preprocess(img_1, img_2, None, None, mode)
            if process_flag:
                img_1_mask = np.ones(img_1.shape)
                img_2_mask = np.ones(img_2.shape)
                stitching_res, mass, _ = stitching_pair(img_1, img_2, img_1_mask, img_2_mask, mode)
                stitching_res = np.uint8(stitching_res)
            else:
                stitching_res, mass = stitching_res_temp, mass_temp
                stitching_res = np.uint8(stitching_res)
        elif img_1 is None:
            img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            stitching_res = img_2
            mass = np.ones(img_2.shape)
        else:
            img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            stitching_res = img_1
            mass = np.ones(img_1.shape)

        tier_list.append(stitching_res)
        tier_mask_list.append(mass)

    im1 = tier_list[0]
    im2 = tier_list[1]
    im1_mask = tier_mask_list[0]
    im2_mask = tier_mask_list[1]
    mode = "d"
    stitching_res, _, _ = stitching_rows(im1, im2, im1_mask, im2_mask, mode, refine_flag)

    final_res = np.uint8(stitching_res)
    cv2.imwrite(os.path.join(store_path, "".join([str(top_num), "-res", file_ext])), final_res)
    return

def read_image(fname, grayscale=True):
    image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image

        
def three_stitching(data_path, store_path, top_num, file_ext, refine_flag=False, save_intermediate=False):
    
    tier_list = []
    tier_mask_list = []
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
                stitching_res, mass, _ = stitching_pair(img_1, img_2, img1_color, img2_color, img_1_mask, img_2_mask, mode)
                stitching_res = np.uint8(stitching_res)
                
            else:
                stitching_res, mass = stitching_res_temp, mass_temp
                stitching_res = np.uint8(stitching_res)
                
        elif img_1 is None:
            # img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            stitching_res = img_2
            mass = np.ones(img_2.shape)
            
        else:
            # img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            stitching_res = img_1
            mass = np.ones(img_1.shape)

        if save_intermediate:
            cv2.imwrite(os.path.join(store_path, "".join([str(top_num), "_"+str(i+1), "_1+2", file_ext])), np.uint8(stitching_res))
            cv2.imwrite(os.path.join(store_path, "".join([str(top_num), "_"+str(i+1), "_1+2_mask", file_ext])), np.uint8(mass * 255))

        img_3_pth = os.path.join(data_path, "".join([top_num, "_" + str(i + 1), "_3", file_ext]))
        img_3 = read_image(img_3_pth)
        if img_3 is None:
            tier_list.append(stitching_res)
            tier_mask_list.append(mass)
            continue
        # img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)
        img_3_mask = np.ones(img_3.shape)
        mode = "r"
        stitching_res_temp, mass_temp, process_flag = preprocess(stitching_res, img_3, mass, None, mode)
        if process_flag:
            stitching_res, mass, _ = stitching_pair(stitching_res, img_3, mass, img_3_mask, mode)
            stitching_res = np.uint8(stitching_res)
        else:
            stitching_res, mass = stitching_res_temp, mass_temp
            stitching_res = np.uint8(stitching_res)
        tier_list.append(stitching_res)
        tier_mask_list.append(mass)

        if save_intermediate:
            cv2.imwrite(os.path.join(store_path, "".join([str(top_num), "_" + str(i + 1), "_1+2+3", file_ext])), np.uint8(stitching_res))
            cv2.imwrite(os.path.join(store_path, "".join([str(top_num), "_" + str(i + 1), "_1+2+3_mask", file_ext])), np.uint8(mass * 255))


    while len(tier_list) >= 2:
        im1 = tier_list[0]
        im2 = tier_list[1]
        im1_mask = tier_mask_list[0]
        im2_mask = tier_mask_list[1]

        mode = "d"
        stitching_res, mass, overlap_mass = stitching_rows(im1, im2, im1_mask, im2_mask, mode, refine_flag)
        stitching_res = np.uint8(stitching_res)
        tier_list[1] = stitching_res
        tier_mask_list[1] = mass
        tier_list = tier_list[1:]
        tier_mask_list = tier_mask_list[1:]

        if save_intermediate:
            cv2.imwrite(os.path.join(store_path, "".join([str(top_num), "_row_", str(len(tier_list)), file_ext])),
                        np.uint8(stitching_res))
            cv2.imwrite(os.path.join(store_path, "".join([str(top_num), "_row_", str(len(tier_list)), "_mask", file_ext])), np.uint8(mass * 255))

    final_res = tier_list[0]
    final_res = np.uint8(final_res)
    cv2.imwrite(os.path.join(store_path, "".join([str(top_num), "-res", file_ext])), final_res)
    return


