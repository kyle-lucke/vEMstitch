import random
import logging

import numpy as np
from scipy import linalg

from Utils import flann_match, flann_match_subset, generate_None_list, rigidity_cons

logger = logging.getLogger(__name__)

def normalize_points(pts):
    """
    Hartley normalization for 2D points.

    pts: (n, 2)
    returns:
        pts_norm: (n, 2)
        T: (3, 3) normalization matrix
    """
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid

    mean_dist = np.mean(np.linalg.norm(pts_centered, axis=1))
    scale = np.sqrt(2) / mean_dist

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm_h = (T @ pts_h.T).T

    return pts_norm_h[:, :2], T


def dlt_homography(ps1, ps2):
    """
    Normalized DLT homography estimation.

    ps1, ps2: (n, 2), n >= 4
    returns:
        H: (3, 3) homography
    """
    ps1_n, T1 = normalize_points(ps1)
    ps2_n, T2 = normalize_points(ps2)

    n = ps1.shape[0]
    A = np.zeros((2 * n, 9))

    for i in range(n):
        x, y = ps1_n[i]
        xp, yp = ps2_n[i]

        A[2*i]   = [0, 0, 0, -x, -y, -1, yp*x, yp*y, yp]
        A[2*i+1] = [x, y, 1,  0,  0,  0, -xp*x, -xp*y, -xp]

    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)

    # Denormalize
    H = np.linalg.inv(T2) @ Hn @ T1
    H /= H[2, 2]

    return H


def RANSAC(ps1, ps2, num_iter=2000, thresh=3.0):
    """
    RANSAC + normalized DLT for homography estimation.

    ps1, ps2: (n, 2) matched points
    num_iter: RANSAC iterations
    thresh: reprojection error threshold (pixels)

    returns:
        H: estimated homography
        inliers: boolean mask (n,)
    """
    n = ps1.shape[0]
    if n < 4:
        raise ValueError("Need at least 4 point correspondences")

    best_inliers = None
    best_count = 0
    best_H = None

    ps1_h = np.hstack([ps1, np.ones((n, 1))])

    for _ in range(num_iter):

        subset = random.sample(range(n), 4)
        p1_sub = ps1[subset]
        p2_sub = ps2[subset]

        # skip subsets that do meet the rigidity constraint
        if not rigidity_cons(p1_sub[:, 0], p1_sub[:, 1],
                             p2_sub[:, 0], p2_sub[:, 1]):
            continue

        # Estimate homography from minimal set
        try:
            H = dlt_homography(p1_sub, p2_sub)
        except np.linalg.LinAlgError:
            continue

        # Project all points
        p2_proj = (H @ ps1_h.T).T
        p2_proj = p2_proj[:, :2] / p2_proj[:, 2:3]

        # Reprojection error
        errors = np.linalg.norm(p2_proj - ps2, axis=1)

        inliers = errors < thresh
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_H = H

    # Recompute homography using all inliers
    if best_inliers is None or best_count < 4:
        raise RuntimeError("RANSAC failed to find a valid homography")

    H = dlt_homography(ps1[best_inliers], ps2[best_inliers])

    return H, best_inliers

########
## OLD
########
# def RANSAC(ps1, ps2, iter_num, min_dis):

#     '''

#     ps1: (n, 2) np.array of (matched) KPs from source image.
#     ps2: (n, 2) np.array of (matched) KPs from target image.
#     iter_num: int that specifies the number of RANSAC iterations to run.
#     min_dis: float specifying the maximum allowed distance for a point
#     to be considered an inlier.
    
#     '''

#     point_num = ps1.shape[0]

#     if point_num < 4:
#         raise ValueError("ERROR: must have atleast 4 keypoint matches to estimate homogrpahy matrix.")

#     x1 = ps1[:, 0].reshape(-1, 1)
#     y1 = ps1[:, 1].reshape(-1, 1)
#     x2 = ps2[:, 0].reshape(-1, 1)
#     y2 = ps2[:, 1].reshape(-1, 1)

#     # Scales the keypoints so that their average value is 1. This
#     # helps improve numerical stability during matrix operations.
#     scale = 1 / np.mean(np.vstack([x1, y1, x2, y2]))
#     x1 *= scale
#     y1 *= scale
#     x2 *= scale
#     y2 *= scale

#     X = np.hstack([np.zeros((point_num, 3)), x1, y1, np.ones((point_num, 1)), -y2 * x1, -y2 * y1, -y2])
#     Y = np.hstack([x1, y1, np.ones((point_num, 1)), np.zeros((point_num, 3)), -x2 * x1, -x2 * y1, -x2])

#     # List of homography computed at each iteration
#     H = generate_None_list(iter_num, 1)

#     # number of inliers found for each iteration
#     score = generate_None_list(iter_num, 1)

#     # inlier mask for each iteration
#     ok = generate_None_list(iter_num, 1)

#     # matrix used to compute homography at each iteration.
#     A = generate_None_list(iter_num, 1)

#     for it in range(iter_num):

#         # randomly sample 4 matched keypoints
#         subset = random.sample(list(range(point_num)), 4)

#         # skip subsets that do meet the rigidity constraint
#         if not rigidity_cons(x1[subset, :], y1[subset, :],
#                              x2[subset, :], y2[subset, :]):
            
#             ok[it] = False
#             score[it] = 0
#             continue

#         # compute homography:

#         # 1) compute Singular Value Decomposition (SVD) for current
#         # subset of points
#         A[it] = np.vstack([X[subset, :], Y[subset, :]])
#         U, S, V = linalg.svd(A[it])

#         # extract homography solution
#         h = V.T[:, 8]
#         H[it] = h.reshape(3, 3)
        
#         dis = np.dot(X, h)**2 + np.dot(Y, h)**2
        
#         # check number of inliers less than min_dis
#         ok[it] = dis < min_dis * min_dis
#         score[it] = sum(ok[it])

#     # get best score (number of inliers) and corresponding iteration.
#     score, best = max(score), np.argmax(score)

#     # compute homography using iterion that results in most inliers.
#     ok = ok[best]
#     A = np.vstack([X[ok, :], Y[ok, :]])
#     U, S, V = linalg.svd(A, 0)    
#     h = V.T[:, 8]
#     H = h.reshape(3, 3)

#     # scale homography back to original scale.
#     H = np.dot(np.dot(np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]]), H),
#                np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]]))
#     return H, ok


def rigid_transform(kp1, dsp1, kp2, dsp2, im1_mask, im2_mask, mode, flann_ratio=0.4, subset_flann=False, **kwargs):
    dis = 0.0
    if mode == "d":
        dis = im1_mask.shape[0]
    elif mode in ("l", "r"):
        dis = im1_mask.shape[1]
    shifting = (mode, dis)

    if subset_flann:
        X1, X2 = flann_match_subset(kp1, dsp1, kp2, dsp2, mode, ratio=flann_ratio, im1_mask=im1_mask, im2_mask=im2_mask, shifting=shifting, **kwargs)
        
    else:
        X1, X2 = flann_match(kp1, dsp1, kp2, dsp2, ratio=flann_ratio, im1_mask=im1_mask, im2_mask=im2_mask, shifting=shifting, **kwargs)

    # fallback to ORB-based homography estimation
    if len(X1) == 0:
        logger.info("len(X1) == 0. Falling back to fast_brief routine.")
        return None, None, None, None

    try:
        # w/ proper reprojection error computation + Hartley
        # normalization (for numerical stability)
        H_ransac, ok = RANSAC(X1.copy(), X2.copy(), 4)
       
        # w/o proper reprojection error computation (original implementation)
        # H_ransac, ok = RANSAC(X1.copy(), X2.copy(), 2000, 0.1)
    except Exception as e:
        logger.info(f"exception in RANSAC: {e}.\nFalling back to fast_brief routine.")
        ok = [True for _ in X1]
        return None, None, None, None

    logger.info(f"Inliers from RANSAC computation: {np.sum(ok)}")
    
    # X1, X2 represent the matched keypoint from the source and target
    # images, respectively.
    point_num = X1.shape[0]

    # compute centroids of matched points:
    centroid_1 = np.mean(X1, axis=0)
    centroid_2 = np.mean(X2, axis=0)

    # center the points
    X = X1 - np.tile(centroid_1, (point_num, 1))
    Y = X2 - np.tile(centroid_2, (point_num, 1))
    
    # compute rotation using SVD (Kabasch algorithm). Computes optimal
    # rotation matrix that minimizes the least-squares error
    H_cov = np.matmul(np.transpose(X[ok, :]), Y[ok, :])
    U, S, VT = np.linalg.svd(H_cov)
    R = np.matmul(VT.T, U.T)

    # Ensure proper rotation (e.g., no reflection) 
    if np.linalg.det(R) < 0:
        VT[1, :] *= -1
        R = np.matmul(VT.T, U.T)

    # compute translation
    t = -np.matmul(R, centroid_1) + centroid_2
    H_final = np.zeros((3, 3))
    H_final[2, 2] = 1.0
    H_final[:2, 2] = t
    H_final[:2, :2] = R
    
    return H_final, ok, X1, X2


def similiarity_transform(kp1, dsp1, kp2, dsp2, im1_mask, im2_mask, mode, flann_ratio=0.4, subset_flann=False, **kwargs):
    dis = 0.0
    if mode == "d":
        dis = im1_mask.shape[0]
    elif mode in ("l", "r"):
        dis = im1_mask.shape[1]
    shifting = (mode, dis)

    if subset_flann:
        X1, X2 = flann_match_subset(kp1, dsp1, kp2, dsp2, mode, ratio=flann_ratio, im1_mask=im1_mask, im2_mask=im2_mask, shifting=shifting, **kwargs)
        
    else:
        X1, X2 = flann_match(kp1, dsp1, kp2, dsp2, ratio=flann_ratio, im1_mask=im1_mask, im2_mask=im2_mask, shifting=shifting, **kwargs)

    # fallback to ORB-based homography estimation
    if len(X1) == 0:
        logger.info("len(X1) == 0. Falling back to fast_brief routine.")
        return None, None, None, None

    try:
        H_ransac, ok = RANSAC(X1.copy(), X2.copy(), 2000, 0.1)
    except Exception as e:
        logger.info(f"exception in RANSAC: {e}.\nFalling back to fast_brief routine.")
        ok = [True for _ in X1]
        return None, None, None, None

    logger.info(f"Inliers from RANSAC computation: {len(ok)}")

    # X1, X2 represent the matched keypoint from the source and target
    # images, respectively.
    point_num = X1.shape[0]

    # Use only RANSAC inliers
    Xin = X1[ok, :]
    Yin = X2[ok, :]
    
    # Compute centroids
    centroid_1 = np.mean(Xin, axis=0)
    centroid_2 = np.mean(Yin, axis=0)

    # Center the points
    X = Xin - centroid_1
    Y = Yin - centroid_2

    # Compute covariance
    H = X.T @ Y

    # SVD
    U, S, VT = np.linalg.svd(H)

    # Rotation
    R = VT.T @ U.T
    if np.linalg.det(R) < 0:
        VT[-1, :] *= -1
        R = VT.T @ U.T

    # --- NEW: estimate scale ---
    var_X = np.sum(np.sum(X ** 2, axis=1))
    scale = np.sum(S) / var_X

    if abs(scale - 1.0) > 0.02:
        logger.warning("Suspicious scale estimate, falling back to rigid transform.")
        scale = 1.0
    
    # Translation
    t = centroid_2 - scale * R @ centroid_1

    # Build homogeneous transform
    H_sim = np.eye(3)
    H_sim[:2, :2] = scale * R
    H_sim[:2, 2] = t
    
    return H_sim, ok, X1, X2
