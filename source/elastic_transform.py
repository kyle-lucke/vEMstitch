import cv2
import numpy as np
from Utils import unique
from scipy import linalg
from scipy.ndimage import map_coordinates
from Utils import stitch_add_mask_linear_border, normalize_img, stitch_add_mask_linear_per_border
from matplotlib import pyplot as plt

EPS = 1e-12

def build_mosaic_canvas(im1, im2, H):
    box1 = np.array([
        [0, im1.shape[1]-1, im1.shape[1]-1, 0],
        [0, 0, im1.shape[0]-1, im1.shape[0]-1],
        [1, 1, 1, 1]
    ])

    box2 = np.array([
        [0, im2.shape[1]-1, im2.shape[1]-1, 0],
        [0, 0, im2.shape[0]-1, im2.shape[0]-1],
        [1, 1, 1, 1]
    ])

    box2p = np.linalg.solve(H, box2)
    box2p /= box2p[2]

    u0 = min(0, box2p[0].min())
    u1 = max(im1.shape[1]-1, box2p[0].max())
    v0 = min(0, box2p[1].min())
    v1 = max(im1.shape[0]-1, box2p[1].max())

    ur = np.arange(u0, u1 + 1)
    vr = np.arange(v0, v1 + 1)

    return ur, vr, (u0, v0)

def compute_local_region(box2p, ur, vr, imsize1, margin_ratio=0.1):
    margin = margin_ratio * min(imsize1)

    u0, v0 = ur[0], vr[0]

    u0_loc = max(box2p[0].min() - margin, u0)
    u1_loc = min(box2p[0].max() + margin, ur[-1])
    v0_loc = max(box2p[1].min() - margin, v0)
    v1_loc = min(box2p[1].max() + margin, vr[-1])

    off_u0 = int(np.ceil(u0_loc - u0))
    off_u1 = int(np.floor(u1_loc - u0))
    off_v0 = int(np.ceil(v0_loc - v0))
    off_v1 = int(np.floor(v1_loc - v0))

    return off_u0, off_u1, off_v0, off_v1

def compute_overlap_region(imsize2, H, margin):
    box1 = np.array([
        [0, imsize2[1]-1, imsize2[1]-1, 0],
        [0, 0, imsize2[0]-1, imsize2[0]-1],
        [1, 1, 1, 1]
    ])

    box1p = H @ box1
    box1p /= box1p[2]

    sub_u0 = max(0, box1p[0].min())
    sub_u1 = min(imsize2[1]-1, box1p[0].max())
    sub_v0 = max(0, box1p[1].min()) - margin
    sub_v1 = min(imsize2[0]-1, box1p[1].max())

    return sub_u0, sub_u1, sub_v0, sub_v1

def select_tps_control_points(X1_ok, X2_ok):
    _, idx1 = np.unique(np.round(X1_ok), axis=1, return_index=True)
    _, idx2 = np.unique(np.round(X2_ok), axis=1, return_index=True)

    ok = np.zeros(X1_ok.shape[1], dtype=bool)
    ok[idx1] = True
    ok[idx2] &= True

    return X1_ok[:, ok], X2_ok[:, ok]

def project_points_homography(H, x, y):
    z = H[2,0]*x + H[2,1]*y + H[2,2]
    xp = (H[0,0]*x + H[0,1]*y + H[0,2]) / z
    yp = (H[1,0]*x + H[1,1]*y + H[1,2]) / z
    return xp, yp

def build_tps_system(xp, yp, dx, dy, lambd):
    n = len(xp)

    dxm = xp[:,None] - xp[None,:]
    dym = yp[:,None] - yp[None,:]

    r2 = dxm*dxm + dym*dym
    np.fill_diagonal(r2, 1.0)

    K = 0.5 * r2 * np.log(r2)
    np.fill_diagonal(K, lambd * 8 * np.pi)

    P = np.vstack([xp, yp, np.ones(n)]).T

    A = np.zeros((n+3, n+3))
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.T

    B = np.zeros((n+3, 2))
    B[:n,0] = dx
    B[:n,1] = dy

    return A, B

# Solve using Iteratively Reweighted Least Squares
# (IRLS) with a Tukey biweight influence function:
def solve_tps_robust(A, B, max_iter=5, eps=1e-8):
    n = B.shape[0] - 3
    W = np.ones(n)

    for _ in range(max_iter):
        Wmat = np.diag(np.r_[W, 1, 1, 1])
        Aw = Wmat @ A @ Wmat + 1e-5*np.eye(n+3)
        Bw = Wmat @ B

        sol = np.linalg.solve(Aw, Bw)

        r = np.sqrt(
            (A[:n,:n] @ sol[:n,0] - B[:n,0])**2 +
            (A[:n,:n] @ sol[:n,1] - B[:n,1])**2
        )

        # use MAD to compute roubst estimate of standard deviation
        s = np.median(np.abs(r)) / 0.6745 + eps

        # 4.685 = Tukey 95% efficiency constant
        u = r / (4.685 * s)

        # remove outliers
        W = (1 - u*u)**2
        W[np.abs(u) >= 1] = 0

    return sol

def warp_identity(im, u, v):
    return map_coordinates(im, [v, u])


def warp_homography(H, u, v):
    z = H[2,0]*u + H[2,1]*v + H[2,2]
    uh = (H[0,0]*u + H[0,1]*v + H[0,2]) / z
    vh = (H[1,0]*u + H[1,1]*v + H[1,2]) / z
    return uh, vh

def eval_tps_field(u, v, xp, yp, wx, wy, a, b, eps=1e-8):
    dx = u[...,None] - xp
    dy = v[...,None] - yp
    r2 = np.clip(dx*dx + dy*dy, eps, None)

    U = 0.5 * r2 * np.log(r2)

    gx = np.sum(U * wx, axis=-1) + a[0]*u + a[1]*v + a[2]
    gy = np.sum(U * wy, axis=-1) + b[0]*u + b[1]*v + b[2]

    return gx, gy

def local_TPS_stable(
    im1, im2,
    im1_color, im2_color,
    H,
    X1_ok, X2_ok,
    im1_mask=None, im2_mask=None,
    mode=None
):

    # EPS = 1e-8

    # ---------------------------------------------------------
    # Default masks
    # ---------------------------------------------------------
    if im1_mask is None:
        im1_mask = np.ones(im1.shape[:2])
    if im2_mask is None:
        im2_mask = np.ones(im2.shape[:2])

    imsize1 = im1.shape[:2]
    imsize2 = im2.shape[:2]

    # ---------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------
    if mode == "d":
        lambd = 0.001 * imsize1[0]
    else:
        lambd = 0.001 * imsize1[0] * imsize1[1]

    intv_mesh = 3
    K_smooth = 5
    margin = 0.1 * min(imsize1)

    # ---------------------------------------------------------
    # 1. Mosaic canvas
    # ---------------------------------------------------------
    ur, vr, (u0, v0) = build_mosaic_canvas(im1, im2, H)
    mosaic_w = len(ur)
    mosaic_h = len(vr)

    # ---------------------------------------------------------
    # 2. Local TPS computation region
    # ---------------------------------------------------------
    box2 = np.array([
        [0, im2.shape[1]-1, im2.shape[1]-1, 0],
        [0, 0, im2.shape[0]-1, im2.shape[0]-1],
        [1, 1, 1, 1]
    ])
    box2p = np.linalg.solve(H, box2)
    box2p /= box2p[2]

    off_u0, off_u1, off_v0, off_v1 = \
        compute_local_region(box2p, ur, vr, imsize1)

    imw_loc = off_u1 - off_u0 + 1
    imh_loc = off_v1 - off_v0 + 1

    # ---------------------------------------------------------
    # 3. Overlap region in image 2
    # ---------------------------------------------------------
    sub_u0, sub_u1, sub_v0, sub_v1 = \
        compute_overlap_region(imsize2, H, margin)

    # ---------------------------------------------------------
    # 4. TPS control points
    # ---------------------------------------------------------
    X1, X2 = select_tps_control_points(X1_ok, X2_ok)
    x1, y1 = X1
    x2, y2 = X2
    n = len(x1)

    # ---------------------------------------------------------
    # 5. Fixed TPS coordinate frame
    # ---------------------------------------------------------
    xp, yp = project_points_homography(H, x1, y1)
    dx = xp - x2
    dy = yp - y2

    # ---------------------------------------------------------
    # 6. Build TPS system (ONCE)
    # ---------------------------------------------------------
    A, B = build_tps_system(xp, yp, dx, dy, lambd)

    # ---------------------------------------------------------
    # 7. Robust TPS solve
    # ---------------------------------------------------------
    sol = solve_tps_robust(A, B)

    wx = sol[:n, 0]
    wy = sol[:n, 1]
    a  = sol[n:, 0]
    b  = sol[n:, 1]

    # ---------------------------------------------------------
    # 8. Warp image 1 (identity)
    # ---------------------------------------------------------
    u, v = np.meshgrid(ur, vr)

    im1_p = warp_identity(im1, u, v)
    mask1_p = warp_identity(im1_mask, u, v)

    im1_color_p = np.stack([
        warp_identity(im1_color[..., c], u, v)
        for c in range(im1_color.shape[2])
    ], axis=-1)

    # ---------------------------------------------------------
    # 9. Warp image 2: homography + TPS
    # ---------------------------------------------------------
    uh, vh = warp_homography(H, u, v)

    uh_sub = uh[off_v0:off_v1+1:intv_mesh,
                off_u0:off_u1+1:intv_mesh]
    vh_sub = vh[off_v0:off_v1+1:intv_mesh,
                off_u0:off_u1+1:intv_mesh]

    gx_sub, hy_sub = eval_tps_field(
        uh_sub, vh_sub, xp, yp,
        wx, wy, a, b
    )

    gx_sub = cv2.resize(gx_sub, (imw_loc, imh_loc))
    hy_sub = cv2.resize(hy_sub, (imw_loc, imh_loc))

    gx = np.zeros((mosaic_h, mosaic_w))
    hy = np.zeros((mosaic_h, mosaic_w))

    gx[off_v0:off_v1+1, off_u0:off_u1+1] = gx_sub
    hy[off_v0:off_v1+1, off_u0:off_u1+1] = hy_sub

    # ---------------------------------------------------------
    # 10. Smooth transition to global homography
    # ---------------------------------------------------------
    eta_max = K_smooth * max(abs(np.r_[dx, dy]))

    dist_h = np.maximum(sub_u0 - uh, uh - sub_u1)
    dist_v = np.maximum(sub_v0 - vh, vh - sub_v1)
    dist = np.maximum(0, np.maximum(dist_h, dist_v))

    eta = np.clip((eta_max - dist) / eta_max, 0, 1)

    gx *= eta
    hy *= eta

    uf = uh - gx
    vf = vh - hy

    im2_p = map_coordinates(im2, [vf, uf])
    mask2_p = map_coordinates(im2_mask, [vf, uf])

    im2_color_p = np.stack([
        map_coordinates(im2_color[..., c], [vf, uf])
        for c in range(im2_color.shape[2])
    ], axis=-1)

    # ---------------------------------------------------------
    # 11. Mask blending
    # ---------------------------------------------------------
    mask1_p = (mask1_p > 0.8).astype(float)
    mask2_p = (mask2_p > 0.8).astype(float)
    
    # # Enforce mutually exclusive support outside overlap
    # overlap = (mask1_p > 0) & (mask2_p > 0)

    # # Outside overlap: ensure exclusivity
    # mask1_p = mask1_p * (~overlap)
    # mask2_p = mask2_p * (~overlap)

    # # Restore overlap explicitly
    # mask1_p = mask1_p + overlap.astype(float)
    # mask2_p = mask2_p + overlap.astype(float)
    
    if mode == "d":
        mask1_p, mask2_p, mass, overlap_mass = \
            stitch_add_mask_linear_per_border(mask1_p, mask2_p)
    else:
        mask1_p, mask2_p, mass, overlap_mass = \
            stitch_add_mask_linear_border(mask1_p, mask2_p, mode=mode)

    # ---------------------------------------------------------
    # 12. Final stitching
    # ---------------------------------------------------------
    stitched = im1_p * mask1_p + im2_p * mask2_p
    stitched_color = (
        im1_color_p * mask1_p[..., None] +
        im2_color_p * mask2_p[..., None]
    )

    return (
        stitched,
        stitched_color,
        [v, u],
        [vf, uf],
        mass,
        overlap_mass
    )


def local_TPS(
        im1, im2,
        im1_color, im2_color,
        H, X1_ok, X2_ok,
        im1_mask=None, im2_mask=None,
        mode=None
):
    if im1_mask is None:
        im1_mask = np.ones((im1.shape[0], im1.shape[1]))
    if im2_mask is None:
        im2_mask = np.ones((im2.shape[0], im2.shape[1]))
    imsize1 = im1.shape[:2]
    imsize2 = im2.shape[:2]

    # Parameters
    if mode == "d":
        lambd = 0.001 * imsize1[0]  
    else:
        lambd = 0.001 * imsize1[0] * imsize1[1]

    # normalize lambda according to estimated scale
    # logger.warning("No lambda normalization")
    s = np.sqrt(H[0,0]**2 + H[1,0]**2)  # approximate similarity scale
    lambd = lambd * s * s
        
    # spacing of TPS control grid (performs subsampling for increased
    # speed).
    intv_mesh = 3

    # the smooth transition width in the non-overlapping region is set
    # to K_smooth times the maximum bias.
    K_smooth = 5  

    # Mosaic
    box1 = np.array([[0, im1.shape[1] - 1, im1.shape[1] - 1, 0],
                     [0, 0, im1.shape[0] - 1, im1.shape[0] - 1],
                     [1, 1, 1, 1]])
    
    box2 = np.array([[0, im2.shape[1] - 1, im2.shape[1] - 1, 0],
                     [0, 0, im2.shape[0] - 1, im2.shape[0] - 1],
                     [1, 1, 1, 1]])

    box2_ = linalg.solve(H, box2)
    
    box2_[0, :] = box2_[0, :] / box2_[2, :]
    box2_[1, :] = box2_[1, :] / box2_[2, :]

    u0 = min(0, min(box2_[0, :]))
    u1 = max(im1.shape[1] - 1, max(box2_[0, :]))
    ur = np.arange(u0, u1 + 1)
    v0 = min(0, min(box2_[1, :]))
    v1 = max(im1.shape[0] - 1, max(box2_[1, :]))
    vr = np.arange(v0, v1 + 1)

    mosaicw = len(ur)
    mosaich = len(vr)

    # align the sub coordinates with the mosaic coordinates
    margin = 0.1 * min(imsize1[0], imsize1[1])  # additional margin of the reprojected image region
    u0_im_ = max(min(box2_[0, :]) - margin, u0)
    u1_im_ = min(max(box2_[0, :]) + margin, u1)
    v0_im_ = max(min(box2_[1, :]) - margin, v0)
    v1_im_ = min(max(box2_[1, :]) + margin, v1)
    offset_u0_ = int(np.ceil(u0_im_ - u0))
    offset_u1_ = int(np.floor(u1_im_ - u0))
    offset_v0_ = int(np.ceil(v0_im_ - v0))
    offset_v1_ = int(np.floor(v1_im_ - v0))
    imw_ = int(np.floor(offset_u1_ - offset_u0_ + 1))
    imh_ = int(np.floor(offset_v1_ - offset_v0_ + 1))

    # boundaries of the overlapping region in the image coordiantes of image 2
    box1_2 = np.dot(H, box1)
    box1_2[0, :] = box1_2[0, :] / box1_2[2, :]
    box1_2[1, :] = box1_2[1, :] / box1_2[2, :]
    sub_u0_ = max([0, min(box1_2[0, :])])
    sub_u1_ = min([imsize2[1] - 1, max(box1_2[0, :])])
    sub_v0_ = max([0, min(box1_2[1, :])]) - margin
    sub_v1_ = min([imsize2[0] - 1, max(box1_2[1, :])])

    # TPS
    # merge the coincided points（重合点）

    # remove duplicated matches (this helps stabilize TPS fitting).
    ok_nd1 = np.full(X1_ok.shape[1], False)
    _, idx1 = unique(np.round(X1_ok))
    ok_nd1[idx1] = True
    
    ok_nd2 = np.full(X2_ok.shape[1], False)
    _, idx2 = unique(np.round(X2_ok))
    ok_nd2[idx2] = True

    ok_nd = ok_nd1 & ok_nd2
    X1_nd = X1_ok[:, ok_nd]
    X2_nd = X2_ok[:, ok_nd]

    # form the linear system
    x1 = X1_nd[0, :]
    y1 = X1_nd[1, :]
    x2 = X2_nd[0, :]
    y2 = X2_nd[1, :]

    z1_ = H[2, 0] * x1 + H[2, 1] * y1 + H[2, 2]
    x1_ = (H[0, 0] * x1 + H[0, 1] * y1 + H[0, 2]) / z1_
    y1_ = (H[1, 0] * x1 + H[1, 1] * y1 + H[1, 2]) / z1_

    # Measure error of KP matches after applying homography (e.g.,
    # deviation between global transformed img2 and img1)
    gxn = x1_ - x2   
    hyn = y1_ - y2

    n = len(x1_)
    xx = np.repeat(x1_, n).reshape(n, n).T
    yy = np.repeat(y1_, n).reshape(n, n).T
    dist2 = (xx - xx.T) ** 2 + (yy - yy.T) ** 2
    dist2.ravel()[::dist2.shape[1] + 1] = 1
    K = 0.5 * dist2 * np.log(dist2)
    K.ravel()[::dist2.shape[1] + 1] = lambd * 8 * np.pi
    K_ = np.zeros((n + 3, n + 3))
    K_[0:n, 0:n] = K
    K_[n, 0:n] = x1_
    K_[n + 1, 0:n] = y1_
    K_[n + 2, 0:n] = np.ones(n)
    K_[0:n, n] = x1_
    K_[0:n, n + 1] = y1_
    K_[0:n, n + 2] = np.ones(n)
    G_ = np.zeros((n + 3, 2))
    G_[0:n, 0] = gxn
    G_[0:n, 1] = hyn

    # apply Tikhonov regularization to improve conditioning
    c = 1e-5
    K_ += c*np.eye(K_.shape[0], M=K_.shape[1])

    # solve the linear system
    W_ = linalg.solve(K_, G_)
    
    wx = W_[0:n, 0]
    wy = W_[0:n, 1]
    a = W_[n:n + 3, 0]
    b = W_[n:n + 3, 1]

    # remove outliers based on the distribution of weights
    # (i.e. removes feature points causing extreme warps).
    outlier = (abs(wx) > 3 * np.std(wx)) | (abs(wy) > 3 * np.std(wy))

    inlier_idx = np.arange(len(x1_))
    for kiter in range(10):
        if sum(outlier) < 0.0027 * n:
            break
        
        ok = ~outlier
        inlier_idx = inlier_idx[ok]
        K_ = K_[np.concatenate((ok, [True, True, True])), :][:, np.concatenate((ok, [True, True, True]))]
        G_ = G_[np.concatenate((ok, [True, True, True])), :]

        W_ = linalg.solve(K_, G_)
        
        n = len(inlier_idx)
        wx = W_[0:n, 0]
        wy = W_[0:n, 1]
        a = W_[n:n + 3, 0]
        b = W_[n:n + 3, 1]
        outlier = (abs(wx) > 3 * np.std(wx)) | (abs(wy) > 3 * np.std(wy))
        
    ok = np.full(len(x1), False)
    ok[inlier_idx] = True
    x1 = x1[ok]
    y1 = y1[ok]
    x2 = x2[ok]
    y2 = y2[ok]
    x1_ = x1_[ok]
    y1_ = y1_[ok]
    gxn = gxn[ok]
    hyn = hyn[ok]

    # deform image
    gx = np.zeros((mosaich, mosaicw))
    hy = np.zeros((mosaich, mosaicw))
    u, v = np.meshgrid(ur, vr)

    # place im1 into mosiac 
    im1_p = map_coordinates(im1, [v, u])
    warped_mask1 = map_coordinates(im1_mask, [v, u])
    
    channels = []
    for i in range(im1_color.shape[2]):
        channel_data = im1_color[:, :, i]
        
        channel_data_p = map_coordinates(channel_data, [v, u])

        channels.append(channel_data_p)

    im1_color_p = np.stack(channels, axis=-1)
        
    z_ = H[2, 0] * u + H[2, 1] * v + H[2, 2]
    u_ = (H[0, 0] * u + H[0, 1] * v + H[0, 2]) / z_
    v_ = (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / z_
    u_im_ = u_[offset_v0_:offset_v1_ + 1:intv_mesh][:, offset_u0_:offset_u1_ + 1:intv_mesh]
    v_im_ = v_[offset_v0_:offset_v1_ + 1:intv_mesh][:, offset_u0_:offset_u1_ + 1:intv_mesh]
    gx_sub = np.zeros((int(np.ceil(imh_ / intv_mesh)), int(np.ceil(imw_ / intv_mesh))))
    hy_sub = np.zeros((int(np.ceil(imh_ / intv_mesh)), int(np.ceil(imw_ / intv_mesh))))
    for kf in range(n):
        dist2 = (u_im_ - x1_[kf]) ** 2 + (v_im_ - y1_[kf]) ** 2
        # clip zeros to small value so log is numerically stable:
        dist2 = np.clip(dist2, EPS, None)
        rbf = 0.5 * dist2 * np.log(dist2)
        gx_sub = gx_sub + wx[kf] * rbf
        hy_sub = hy_sub + wy[kf] * rbf
    gx_sub = gx_sub + a[0] * u_im_ + a[1] * v_im_ + a[2]
    hy_sub = hy_sub + b[0] * u_im_ + b[1] * v_im_ + b[2]
    gx_sub = cv2.resize(gx_sub, (imw_, imh_))
    hy_sub = cv2.resize(hy_sub, (imw_, imh_))
    gx[offset_v0_:offset_v1_ + 1][:, offset_u0_:offset_u1_ + 1] = gx_sub
    hy[offset_v0_:offset_v1_ + 1][:, offset_u0_:offset_u1_ + 1] = hy_sub

    # smooth tansition to global transform
    eta_d0 = 0  # lower boundary for smooth transition area
    eta_d1 = K_smooth * max(abs(np.concatenate([gxn, hyn])))  # upper boundary for smooth transition area
    sub_u0_ = sub_u0_ + min(gxn)
    sub_u1_ = sub_u1_ + max(gxn)
    sub_v0_ = sub_v0_ + min(hyn)
    sub_v1_ = sub_v1_ + max(hyn)
    dist_horizontal = np.maximum(sub_u0_ - u_, u_ - sub_u1_)
    dist_vertical = np.maximum(sub_v0_ - v_, v_ - sub_v1_)
    dist_sub = np.maximum(dist_horizontal, dist_vertical)
    dist_sub = np.maximum(0, dist_sub)
    eta = (eta_d1 - dist_sub) / (eta_d1 - eta_d0)
    eta[dist_sub < eta_d0] = 1
    eta[dist_sub > eta_d1] = 0
    gx = gx * eta
    hy = hy * eta

    u_ = u_ - gx
    v_ = v_ - hy

    im2_p = map_coordinates(im2, [v_, u_])
    warped_mask2 = map_coordinates(im2_mask, [v_, u_])

    channels = []
    for i in range(im2_color.shape[2]):
        channel_data = im2_color[:, :, i]
        
        channel_data_p = map_coordinates(channel_data, [v_, u_])

        channels.append(channel_data_p)

    im2_color_p = np.stack(channels, axis=-1)
    
    warped_mask1 = np.where(warped_mask1 > 0.8, 1.0, 0)
    warped_mask2 = np.where(warped_mask2 > 0.8, 1.0, 0)

    if mode == "d":
        warped_mask1, warped_mask2, mass, overelap_mass = stitch_add_mask_linear_per_border(warped_mask1, warped_mask2)
    else:
        warped_mask1, warped_mask2, mass, overelap_mass = stitch_add_mask_linear_border(warped_mask1, warped_mask2,
                                                                                        mode=mode)
    stitching_res = im1_p * warped_mask1 + im2_p * warped_mask2

    stitching_res_color = im1_color_p * warped_mask1[:, :, np.newaxis] + im2_color_p * warped_mask2[:, :, np.newaxis]
    
    
    # fig, axs = plt.subplots(nrows=2, ncols=1)

    # for ax in axs.flat:
    #     ax.axis('off')
    
    # axs[0].imshow(stitching_res)
    # axs[1].imshow(stitching_res_color.astype(np.uint8))
    # plt.show()
    # exit()
    
    return stitching_res, stitching_res_color, [v, u], [v_, u_], mass, overelap_mass
