import random
import logging

import numpy as np
from scipy import linalg

from Utils import flann_match, flann_match_subset, generate_None_list, rigidity_cons

logger = logging.getLogger(__name__)

def RANSAC(ps1, ps2, iter_num, min_dis):

    '''

    ps1: (n, 2) np.array of KPs from source image.
    ps2: (n, 2) np.array of KPs from target image.
    iter_num: int that specifies the number of RANSAC iterations to run.
    min_dis: float specifying the maximum allowed distance for a point
    to be considered an inlier.
    
    '''
    
    point_num = ps1.shape[0]

    x1 = ps1[:, 0].reshape(-1, 1)
    y1 = ps1[:, 1].reshape(-1, 1)
    x2 = ps2[:, 0].reshape(-1, 1)
    y2 = ps2[:, 1].reshape(-1, 1)

    # Scales the keypoints so that their average value is 1. This
    # helps improve numerical stability during matrix operations.
    scale = 1 / np.mean(np.vstack([x1, y1, x2, y2]))
    x1 *= scale
    y1 *= scale
    x2 *= scale
    y2 *= scale

    X = np.hstack([np.zeros((point_num, 3)), x1, y1, np.ones((point_num, 1)), -y2 * x1, -y2 * y1, -y2])
    Y = np.hstack([x1, y1, np.ones((point_num, 1)), np.zeros((point_num, 3)), -x2 * x1, -x2 * y1, -x2])

    # computed homography at each iteration
    H = generate_None_list(iter_num, 1)

    # number of inliers for each iteration
    score = generate_None_list(iter_num, 1)

    # inlier mask for each iteration
    ok = generate_None_list(iter_num, 1)

    # matrix used to compute homography at each iteration.
    A = generate_None_list(iter_num, 1)

    for it in range(iter_num):
        subset = random.sample(list(range(point_num)), 4)

        # skip subsets that do meet the rigidity constraint
        if not rigidity_cons(x1[subset, :], y1[subset, :],
                             x2[subset, :], y2[subset, :]):
            
            ok[it] = False
            score[it] = 0
            continue

        # compute homography:

        # 1) compute Singular Value Decomposition for current subset
        # of points
        A[it] = np.vstack([X[subset, :], Y[subset, :]])
        U, S, V = linalg.svd(A[it])

        # extract homography solution
        h = V.T[:, 8]
        H[it] = h.reshape(3, 3)

        # check number of inliers less than min_dis
        dis = np.dot(X, h)**2 + np.dot(Y, h)**2
        ok[it] = dis < min_dis * min_dis
        score[it] = sum(ok[it])

    # get best score (number of inliers) and corresponding iteration.
    score, best = max(score), np.argmax(score)

    # compute homography using iterion that results in most inliers.
    ok = ok[best]
    A = np.vstack([X[ok, :], Y[ok, :]])
    U, S, V = linalg.svd(A, 0)    
    h = V.T[:, 8]
    H = h.reshape(3, 3)

    # scale homography back to original scale.
    H = np.dot(np.dot(np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]]), H),
               np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]]))
    return H, ok


def rigid_transform(kp1, dsp1, kp2, dsp2, im1_mask, im2_mask, mode, flann_ratio=0.4, subset_flann=False, **kwargs):
    dis = 0.0
    if mode == "d":
        dis = im1_mask.shape[0]
    elif mode == "l" or "r":
        dis = im1_mask.shape[1]
    shifting = (mode, dis)

    if subset_flann:
        X1, X2 = flann_match_subset(kp1, dsp1, kp2, dsp2, mode, ratio=flann_ratio, im1_mask=im1_mask, im2_mask=im2_mask, shifting=shifting, **kwargs)
        
    else:
        X1, X2 = flann_match(kp1, dsp1, kp2, dsp2, ratio=flann_ratio, im1_mask=im1_mask, im2_mask=im2_mask, shifting=shifting, **kwargs)
    if len(X1) == 0:
        logger.info("len(X1) == 0. Falling back to fast_brief routine.")
        return None, None, None, None

    try:
        H, ok = RANSAC(X1.copy(), X2.copy(), 2000, 0.1)
    except Exception as e:
        logger.info(f"exception in RANSAC: {e}.\nFalling back to fast_brief routine.")
        ok = [True for _ in X1]
        return None, None, None, None

    point_num = X1.shape[0]
    centroid_1 = np.mean(X1, axis=0)
    centroid_2 = np.mean(X2, axis=0)
    X = X1 - np.tile(centroid_1, (point_num, 1))
    Y = X2 - np.tile(centroid_2, (point_num, 1))
    H = np.matmul(np.transpose(X[ok, :]), Y[ok, :])
    U, S, VT = np.linalg.svd(H)
    R = np.matmul(VT.T, U.T)
    if np.linalg.det(R) < 0:
        VT[1, :] *= -1
        R = np.matmul(VT.T, U.T)
    t = -np.matmul(R, centroid_1) + centroid_2
    H = np.zeros((3, 3))
    H[2, 2] = 1.0
    H[:2, 2] = t
    H[:2, :2] = R
    return H, ok, X1, X2
