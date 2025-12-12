import logging 
import numpy as np
import cv2
from collections import defaultdict
import os

logger = logging.getLogger(__name__)

COLORS = [(0, 0, 255), # Blue
          (255, 0, 0), # Red
          (0, 255, 0), # Green
          (255, 0, 214) # Grey(ish)
          ]

def draw_matches_vertical(img1, kp1, img2, kp2, matches, draw_matched_kps=False):
    """
    Draws matches between two images, stacking them vertically.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # kp objects are mutable, so make copy here so the input
    # parameters are not modified.
    kp1 = kp1.copy()
    kp2 = kp2.copy()
    
    # Create a new canvas with appropriate height and max width
    vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)
    
    # Place images onto the new canvas
    vis[:h1, :w1] = img1
    vis[h1:, :w2] = img2
    
    # Adjust keypoint coordinates for the second image
    # cv2.drawMatchesKnn expects kp2 to be based on the *original* image
    # We need to adjust the 'matches' data to point to the correct vertical location
    # Note: cv2.drawMatchesKnn actually handles the coordinate offset internally 
    # when given two separate images and a single output canvas (which is not how we set this up).
    # We need a custom drawing logic or use the original `drawMatches` which handles canvas creation.

    # A simpler approach using drawMatches with a custom combined image
    # This requires manually adjusting all matched keypoint coordinates in kp2
    adjusted_kp2 = []
    for kp in kp2:
        kp.pt = (kp.pt[0], kp.pt[1] + h1)
        adjusted_kp2.append(kp)

    # Use a loop to draw lines manually on the combined image for better control
    for match in matches:
            p1 = tuple(map(int, kp1[match[0]].pt)) # queryIdx
            p2 = tuple(map(int, adjusted_kp2[match[1]].pt)) # trainIdx
            cv2.line(vis, p1, p2, (0, 255, 0), 1) # Green lines

            if draw_matched_kps:
                cv2.circle(vis, tuple(map(int, kp1[match[0]].pt)), 4, (0, 0, 255), 1)

                cv2.circle(vis, tuple(map(int, adjusted_kp2[match[1]].pt)), 4, (0, 0, 255), 1)

            
    # Draw keypoints (optional)
    if not draw_matched_kps:
    
        for kp in kp1:
            cv2.circle(vis, tuple(map(int, kp.pt)), 4, (0, 0, 255), 1)
            
        for kp in adjusted_kp2:
            cv2.circle(vis, tuple(map(int, kp.pt)), 4, (0, 0, 255), 1)
        
    return vis

# for drawing after filter_isolate or filter_geometry
def draw_matches_after_filter(img1, kp1, img2, kp2):
    """
    Draws matches between two images, stacking them vertically.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create a new canvas with appropriate height and max width
    vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)
    
    # Place images onto the new canvas
    vis[:h1, :w1] = img1
    vis[h1:, :w2] = img2
    
    # Adjust keypoint coordinates for the second image
    # cv2.drawMatchesKnn expects kp2 to be based on the *original* image
    # We need to adjust the 'matches' data to point to the correct vertical location
    # Note: cv2.drawMatchesKnn actually handles the coordinate offset internally 
    # when given two separate images and a single output canvas (which is not how we set this up).
    # We need a custom drawing logic or use the original `drawMatches` which handles canvas creation.

    # A simpler approach using drawMatches with a custom combined image
    # This requires manually adjusting all matched keypoint coordinates in kp2
    adjusted_kp2 = []
    for kp in kp2:
        kp = (kp[0], kp[1] + h1)
        adjusted_kp2.append(kp)

    adjusted_kp2 = np.float32(adjusted_kp2)
        
    # Use a loop to draw lines manually on the combined image for better control
    for p1, p2 in zip(kp1, adjusted_kp2):
            # p1 = tuple(map(int, kp1[match[0]].pt)) # queryIdx
            p1 = tuple(p1.astype(int)) # queryIdx
            
            # p2 = tuple(map(int, adjusted_kp2[match[1]].pt)) # trainIdx
            p2 = tuple(p2.astype(int)) # trainIdx
            cv2.line(vis, p1, p2, (0, 255, 0), 1) # Green lines

            # draw KPs
            cv2.circle(vis, p1, 4, (0, 0, 255), 1)
            cv2.circle(vis, p2, 4, (0, 0, 255), 1)
        
    return vis

def draw_keypoints(img, kp, kp_color):

    # # Create a new canvas with appropriate height and max width
    # vis = np.zeros_like(img, np.uint8)

    # vis[:, :] = img

    # draw KPs
    for p in kp:
        cv2.circle(img, tuple(map(int, p.pt)), 4, kp_color, 1) 

    return img
        
def generate_None_list(m, n):
    a = []
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(None)
        a.append(tmp)
    return a


def normalize_img(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def unique(A):
    ar, idx = np.unique(A, return_index=True, axis=1)
    return ar, idx


def appendimages(im1, im2, vertical=False):
    if vertical:
        col1 = im1.shape[1]
        col2 = im2.shape[1]
        if col1 < col2:
            im1 = np.hstack((im1, np.zeros((im1.shape[0], col2 - col1))))
        elif col1 > col2:
            im2 = np.hstack((im2, np.zeros((im2.shape[0], col1 - col2))))
        return np.concatenate((im1, im2), axis=0)
    else:
        rows1 = im1.shape[0]
        rows2 = im2.shape[0]
        if rows1 < rows2:
            im1 = np.vstack((im1, np.zeros((rows2 - rows1, im1.shape[1]))))
        elif rows1 > rows2:
            im2 = np.vstack((im2, np.zeros((rows1 - rows2, im2.shape[1]))))
        return np.concatenate((im1, im2), axis=1)


def draw_matches(im1, im2, locs1, locs2, ok, vertical=False):
    im3 = appendimages(im1, im2, vertical)
    if vertical:
        for i in range(locs1.shape[0]):
            center1 = (int(round(locs1[i, 0])), int(round(locs1[i, 1])))
            center2 = (int(round(locs2[i, 0])), int(round(locs2[i, 1]) + im1.shape[0]))
            if ok[i] == 0:
                cv2.circle(im3, center1, 3, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(im3, center2, 3, (255, 0, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(im3, center1, 3, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(im3, center2, 3, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.line(im3, center1, center2, (255, 0, 0), 1, cv2.LINE_AA)
    else:
        for i in range(locs1.shape[0]):
            center1 = (int(round(locs1[i, 0])), int(round(locs1[i, 1])))
            center2 = (int(round(locs2[i, 0] + im1.shape[1])), int(round(locs2[i, 1])))
            if ok[i] == 0:
                cv2.circle(im3, center1, 3, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(im3, center2, 3, (255, 0, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(im3, center1, 3, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(im3, center2, 3, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.line(im3, center1, center2, (255, 0, 0), 1, cv2.LINE_AA)
    return im3


# filter isolated points
def filter_isolate(src, tgt, shifting=None):
    # filter by x-axis
    row_index = np.argsort(src[:, 0])
    src_row = src[row_index, 0]
    dis = src_row[4:] - src_row[:-4]
    mean_dis = np.mean(dis) / 2
    index = []
    i = 0
    while i < src_row.shape[0]:
        if i > src_row.shape[0] - 3:
            if abs(src_row[i] - src_row[i - 2]) <= mean_dis * 3:
                index.append(i)
            i += 1
        else:
            if abs(src_row[i] - src_row[i + 2]) <= mean_dis * 3:
                index = index + [i, i + 1, i + 2]
                i = i + 3
            else:
                i += 1
    src = src[row_index, :][index, :]
    tgt = tgt[row_index, :][index, :]

    # filter by y-axis
    col_index = np.argsort(src[:, 1])
    src_col = src[col_index, 1]
    dis = src_col[4:] - src_col[:-4]
    mean_dis = np.mean(dis) / 2
    index = []
    i = 0
    while i < src.shape[0]:
        if i > src.shape[0] - 3:
            if abs(src_col[i] - src_col[i - 2]) <= mean_dis * 3:
                index.append(i)
            i += 1
        else:
            if abs(src_col[i] - src_col[i + 2]) <= mean_dis * 3:
                index = index + [i, i + 1, i + 2]
                i = i + 3
            else:
                i += 1
    return src[col_index, :][index, :], tgt[col_index, :][index, :]


# filter by corresponding
def filter_geometry(src, tgt, window_size=3, index_flag=False, shifting=None):
    new_tgt = tgt.copy()
    if shifting:
        mode, d = shifting
        if mode == "l":
            new_tgt[:, 0] = new_tgt[:, 0] - d
        elif mode == "r":
            new_tgt[:, 0] = new_tgt[:, 0] + d
        elif mode == "d":
            new_tgt[:, 1] = new_tgt[:, 1] + d
    else:
        new_tgt = tgt[:, :]

    dis = np.sqrt(np.square(src[:, 0] - new_tgt[:, 0]) + np.square(src[:, 1] - new_tgt[:, 1]))
    global_mean_dis = np.mean(dis)
    radius = window_size // 2
    index = []
    for i in range(src.shape[0]):
        if i <= radius - 1:
            dis_m = np.mean(dis[:window_size])
            if dis[i] <= dis_m * 1.5 and dis[i] <= global_mean_dis * 1.5:
                index.append(i)
        else:
            dis_m = np.mean(dis[i - radius: i + radius + 1])
            if dis[i] <= dis_m * 1.5 and dis[i] <= global_mean_dis * 1.5:
                index.append(i)
    if not index_flag:
        return src[index, :], tgt[index, :]
    else:
        return index


def rigidity_cons(x, y, x_, y_):

    '''

    x: 1D np.array containing x positions of good KPs from first (i.e., source) image.
    y: 1D np.array containing y positions of good KPs from first (i.e., source) image.

    x_: 1D np.array containing x positions of good KPs from second (i.e., target) image.
    y_: 1D np.array containing y positions of good KPs from second (i.e., target) image.

    Invariant: all input parameters have 4 elements and the KPs from
    the source and target image have been matched.
    
    Checks that the quadrilaterial formed by the four points in the
    source image are transformed rigidly to the second image (i.e.,
    the shape and orientation of the quadrilateral are preserved).

    '''
    
    flag = True
    for i in range(4):
        V = (x[(i + 1) % 4] - x[i]) * (y[(i + 2) % 4] - y[(i + 1) % 4]) - (y[(i + 1) % 4] - y[i]) * (
             x[(i + 2) % 4] - x[(i + 1) % 4])
        V_ = (x_[(i + 1) % 4] - x_[i]) * (y_[(i + 2) % 4] - y_[(i + 1) % 4]) - (y_[(i + 1) % 4] - y_[i]) * (
              x_[(i + 2) % 4] - x_[(i + 1) % 4])
        V_s = np.sign(V)
        V_s_ = np.sign(V_)
        if V_s != V_s_:
            flag = False
            break
    return flag

def SIFT(im1, im2, im1_mask=None, im2_mask=None, filtering='add_weighted', mb_ksize=5):

    if filtering == 'sharpen':
        # Create the sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        # Sharpen the image
        im1 = cv2.filter2D(im1, -1, kernel)
        im2 = cv2.filter2D(im2, -1, kernel)
            
    elif filtering == 'median':
        im1 = cv2.medianBlur(im1, mb_ksize)
        im2 = cv2.medianBlur(im2, mb_ksize)

    elif filtering == 'add_weighted':

        im1_gb = cv2.GaussianBlur(im1, (3,3),0)
        im2_gb = cv2.GaussianBlur(im2, (3,3),0)

        im1 = cv2.addWeighted(im1, 1.5, im1_gb, -0.5, 0)
        im2 = cv2.addWeighted(im2, 1.5, im2_gb, -0.5, 0)
        
        
    sift = cv2.SIFT_create()

    kp1, dsp1 = sift.detectAndCompute(im1, im1_mask)  # None --> mask
    kp2, dsp2 = sift.detectAndCompute(im2, im2_mask)
    
    return kp1, dsp1, kp2, dsp2

def flann_match(kp1, dsp1, kp2, dsp2, ratio=0.4, im1_mask=None, im2_mask=None, shifting=None, **kwargs):
    """
    return DMatch (queryIdx, trainIdx, distance)
    queryIdx: index of query keypoint
    trainIdx: index of target keypoint
    distance: Euclidean distance
    """
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(dsp1, dsp2, k=2)

    good = []
    good_matches = [[0, 0] for i in range(len(matches))]

    logger.info(f"Overall matches: {len(matches)}")
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            if im1_mask is not None and im2_mask is not None:
                im1_x, im1_y = np.int32(np.round(kp1[m.queryIdx].pt))
                im2_x, im2_y = np.int32(np.round(kp2[m.trainIdx].pt))
                if im1_mask[im1_y][im1_x] and im2_mask[im2_y][im2_x]:
                    good.append([m.queryIdx, m.trainIdx])
                    good_matches[i] = [1, 0]
            else:
                good.append([m.queryIdx, m.trainIdx])
                good_matches[i] = [1, 0]    

    srcdsp = np.float32([kp1[m[0]].pt for m in good])
    tgtdsp = np.float32([kp2[m[1]].pt for m in good])
    
    logger.info(f"Matches after ratio test: {len(good)}")

    # DEBUG: plot good KPs that pass the ratio test
    if kwargs and 'plot_kp_matches' in kwargs['kwargs'] and kwargs['kwargs']['plot_kp_matches']:

        im1, im2 = kwargs['kwargs']['im1_color'], kwargs['kwargs']['im2_color']
        
        if kwargs['kwargs']['plot_kp_vertical']:

            matched_im = draw_matches_vertical(im1, kp1, im2, kp2, good)
            
        else:        
            matched_im = cv2.drawMatchesKnn(im1, kp1, im2, kp2, matches,
                                            outImg=None, matchesMask=good_matches,
                                            matchColor=(0,255,0),
                                            singlePointColor=(0,255,255),
                                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        import matplotlib.pyplot as plt
        plt.imshow(matched_im)
        plt.axis('off')
        
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        plt.tight_layout()
        plt.show()

    kp_length = len(srcdsp)
    if kp_length <= 2:
        logger.info("feature number = %d" % len(srcdsp))
        if len(srcdsp) == 1:
            return [], []
        return srcdsp, tgtdsp

    _, index = np.unique(srcdsp[:, 0], return_index=True)
    srcdsp = srcdsp[np.sort(index), :]
    tgtdsp = tgtdsp[np.sort(index), :]

    if len(srcdsp) >= 8:
        srcdsp, tgtdsp = filter_isolate(srcdsp, tgtdsp)
        tgtdsp, srcdsp = filter_isolate(tgtdsp, srcdsp)
        logger.info(f"matches after filter_isolate: {len(srcdsp)}")

    srcdsp, tgtdsp = filter_geometry(srcdsp, tgtdsp, shifting=shifting)
    logger.info(f"matches after filter_geometry: {len(srcdsp)}")

    if kwargs and 'plot_kp_matches' in kwargs['kwargs'] and kwargs['kwargs']['plot_kp_matches']:

        im1, im2 = kwargs['kwargs']['im1_color'], kwargs['kwargs']['im2_color']
        
        if kwargs['kwargs']['plot_kp_vertical']:
            matched_im = draw_matches_after_filter(im1, srcdsp, im2, tgtdsp)
            
        else:
            logger.info('Drawing horizontal KP matches after filtering not currently implemented')
            # matched_im = cv2.drawMatchesKnn(im1, kp1, im2, kp2, matches,
            #                                 outImg=None, matchesMask=good_matches,
            #                                 matchColor=(0,255,0),
            #                                 singlePointColor=(0,255,255),
            #                                 flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        import matplotlib.pyplot as plt
        plt.imshow(matched_im)
        plt.axis('off')
        
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        plt.tight_layout()
        plt.show()

    
    
    return srcdsp, tgtdsp

def _generate_spatial_subsets(kp, dsp, n_subsets, im_axis_shape, mode):

    if mode == 'r':
        axis_index = 1
    elif mode == 'd':
        axis_index = 0
    
    # convert to numpy for fast vectorized subsetting
    kp_np = np.float32([kp_i.pt for kp_i in kp])

    # note: in openCV the x, y positions index columns and rows,
    # respectively.
    subset_width = im_axis_shape // n_subsets
    subset_axis_points = np.arange(0, im_axis_shape + subset_width, im_axis_shape // n_subsets)
    # print(f"subset y points: {subset_x_points}")

    # print(len(kp_np))
    
    subset_idxs = []
    for i in range(len(subset_axis_points)-1):

        subset_axis_start = subset_axis_points[i]
        subset_axis_end = subset_axis_points[i+1]

        # less than condition should capture all KPs in most cases
        # since OpenCV's keypoint detectors do not produce KPs near
        # image edges (by default).
        condition = np.logical_and(kp_np[:, axis_index] >= subset_axis_start,
                                   kp_np[:, axis_index] < subset_axis_end)
        
        current_subset_idxs = np.where(condition)[0]
        subset_idxs.append(current_subset_idxs)
        
        # print(current_subset_idxs)
        
        # print(f"range: [{subset_axis_points[i]}, {subset_axis_points[i+1]}]")

    assert sum([len(s) for s in subset_idxs]) == len(kp), f"ERROR: number of subset indices and KPs not equal."

    subset_kps = []
    subset_dsps = []
    for s in subset_idxs:
        subset_kps.append( [ kp[s_i] for s_i in s ] )
        subset_dsps.append( dsp[s] )

    return subset_kps, subset_dsps

# [ ] TODO: generalize this to vertical + horizontal stitching by
#     modifying _generate_spatial_subsets accordingly.

# [ ] TODO: refactor into single function w/ flann_match
def flann_match_subset(kp1, dsp1, kp2, dsp2, mode, ratio=0.4, n_subsets=8, im1_mask=None, im2_mask=None, shifting=None, **kwargs):
    """
    return DMatch (queryIdx, trainIdx, distance)
    queryIdx: index of query keypoint
    trainIdx: index of target keypoint
    distance: Euclidean distance
    """

    if mode == 'r':
        im1_axis_shape = kwargs['kwargs']['im1'].shape[0]
        im2_axis_shape = kwargs['kwargs']['im2'].shape[0]
        
    elif mode == 'd':
        im1_axis_shape = kwargs['kwargs']['im1'].shape[1]
        im2_axis_shape = kwargs['kwargs']['im2'].shape[1]
        
        
    # seperate KPs into subsets based on spatial position:
    kp1_subsets, dsp1_subsets = _generate_spatial_subsets(kp1, dsp1, n_subsets,
                                                          im1_axis_shape, mode)
    
    kp2_subsets, dsp2_subsets = _generate_spatial_subsets(kp2, dsp2, n_subsets,
                                                          im2_axis_shape, mode)

    # DEBUG: draw subset keypoints
    # im1_draw = kwargs['kwargs']['im1_color'].copy()
    # for i, kp_subset in enumerate(kp1_subsets):
    #     im1_draw = draw_keypoints(im1_draw, kp_subset, COLORS[i % len(COLORS)])
        
    # import matplotlib.pyplot as plt
    # plt.imshow(im1_draw)
    # plt.axis('off')
        
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())

    # plt.tight_layout()
    # plt.show()

    # im2_draw = kwargs['kwargs']['im2_color'].copy()
    # for i, kp_subset in enumerate(kp2_subsets):
    #     im2_draw = draw_keypoints(im2_draw, kp_subset, COLORS[i % len(COLORS)])
        
    # import matplotlib.pyplot as plt
    # plt.imshow(im2_draw)
    # plt.axis('off')
        
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())

    # plt.tight_layout()
    # plt.show()
    
    # exit()
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    srcdsp = [] 
    tgtdsp = []
    
    for i in range(n_subsets):

        kp1_subset = kp1_subsets[i] 
        dsp1_subset = dsp1_subsets[i]

        kp2_subset = kp2_subsets[i] 
        dsp2_subset = dsp2_subsets[i]

        # if number of descriptors in subset is less than k=2, skip as
        # we cannot do KP matching
        if not(len(dsp1_subset) >= 2 and len(dsp2_subset) >= 2):
            logging.info(f"dsp subset too small, skipping subset {i}")
            continue
        
        matches = flann.knnMatch(dsp1_subset, dsp2_subset, k=2)

        good = []
        good_matches = [[0, 0] for _ in range(len(matches))]

        logger.info(f"Overall matches: {len(matches)}")
        for j, (m, n) in enumerate(matches):
            if m.distance < ratio * n.distance:
                if im1_mask is not None and im2_mask is not None:
                    im1_x, im1_y = np.int32(np.round(kp1_subset[m.queryIdx].pt))
                    im2_x, im2_y = np.int32(np.round(kp2_subset[m.trainIdx].pt))
                    if im1_mask[im1_y][im1_x] and im2_mask[im2_y][im2_x]:
                        good.append([m.queryIdx, m.trainIdx])
                        good_matches[j] = [1, 0]
                else:
                    good.append([m.queryIdx, m.trainIdx])
                    good_matches[j] = [1, 0]    

        srcdsp_subset = np.float32([kp1_subset[m[0]].pt for m in good])
        tgtdsp_subset = np.float32([kp2_subset[m[1]].pt for m in good])

        # if len(srcdsp_subset) > 0 and len(tgtdsp_subset) > 0: 
        
        #     srcdsp.append(srcdsp_subset)
        #     tgtdsp.append(tgtdsp_subset)

        logger.info(f"Matches after ratio test (subset {i+1} / {n_subsets}): {len(good)}")

        # DEBUG: plot good KPs that pass the ratio test
        if kwargs and 'plot_kp_matches' in kwargs['kwargs'] and kwargs['kwargs']['plot_kp_matches']:

            im1, im2 = kwargs['kwargs']['im1_color'], kwargs['kwargs']['im2_color']
        
            if kwargs['kwargs']['plot_kp_vertical']:

                matched_im = draw_matches_vertical(im1, kp1_subset, im2, kp2_subset, good, draw_matched_kps=True)
            
            else:        
                matched_im = cv2.drawMatchesKnn(im1, kp1_subset, im2, kp2_subset, matches,
                                                outImg=None, matchesMask=good_matches,
                                                matchColor=(0,255,0),
                                                singlePointColor=(0,255,255),
                                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        
                
            import matplotlib.pyplot as plt
            plt.imshow(matched_im)
            plt.axis('off')
        
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())

            plt.tight_layout()
            plt.show()

        # srcdsp = np.vstack(srcdsp)
        # tgtdsp = np.vstack(tgtdsp)

        # logger.info(f"Matches after ratio test over all subsets: {len(srcdsp_subset)}")
    
        # print(srcdsp.shape)
    
        # exit()
            
        # kp_length = len(srcdsp)
        # if kp_length <= 2:
        #     logger.info("feature number = %d" % len(srcdsp))
        #     if len(srcdsp) == 1:
        #         return [], []
        #     return srcdsp, tgtdsp

        if len(srcdsp_subset) == 0:
            continue
        
        _, index = np.unique(srcdsp_subset[:, 0], return_index=True)
        srcdsp_subset = srcdsp_subset[np.sort(index), :]
        tgtdsp_subset = tgtdsp_subset[np.sort(index), :]

        if len(srcdsp_subset) >= 8:
            srcdsp_subset, tgtdsp_subset = filter_isolate(srcdsp_subset, tgtdsp_subset)
            tgtdsp_subset, srcdsp_subset = filter_isolate(tgtdsp_subset, srcdsp_subset)
            logger.info(f"matches after filter_isolate (subset {i+1} / {n_subsets}): {len(srcdsp_subset)}")

        srcdsp_subset, tgtdsp_subset = filter_geometry(srcdsp_subset, tgtdsp_subset, shifting=shifting)
        logger.info(f"matches after filter_geometry (subset {i+1} / {n_subsets}): {len(srcdsp_subset)}")

        if len(srcdsp_subset) > 0 and len(tgtdsp_subset) > 0: 
        
            srcdsp.append(srcdsp_subset)
            tgtdsp.append(tgtdsp_subset)

        
        if kwargs and 'plot_kp_matches' in kwargs['kwargs'] and kwargs['kwargs']['plot_kp_matches']:

            im1, im2 = kwargs['kwargs']['im1_color'], kwargs['kwargs']['im2_color']
        
            if kwargs['kwargs']['plot_kp_vertical']:
                matched_im = draw_matches_after_filter(im1, srcdsp_subset, im2, tgtdsp_subset)
            
            else:
                logger.info('Drawing horizontal KP matches after filtering not currently implemented')
                # matched_im = cv2.drawMatchesKnn(im1, kp1, im2, kp2, matches,
                #                                 outImg=None, matchesMask=good_matches,
                #                                 matchColor=(0,255,0),
                #                                 singlePointColor=(0,255,255),
                #                                 flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

            import matplotlib.pyplot as plt
            plt.imshow(matched_im)
            plt.axis('off')
        
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())

            plt.tight_layout()
            plt.show()

    srcdsp = np.vstack(srcdsp)
    tgtdsp = np.vstack(tgtdsp)

    logger.info(f"Filtered matches over all subsets: {len(srcdsp)}")
    
    kp_length = len(srcdsp)
    if kp_length <= 2:
        logger.info("feature number = %d" % len(srcdsp))
        if len(srcdsp) == 1:
            return [], []
        return srcdsp, tgtdsp
    
    return srcdsp, tgtdsp


def loG(img):
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    img = cv2.convertScaleAbs(img)
    img = normalize_img(img)
    img = np.where(img >= 0.1, img, 0)
    return img


def stitch_add_mask_linear_border(mask1, mask2, mode=None):
    height, width = mask1.shape
    x_map = np.sum(mask1, axis=1)
    y_map = np.sum(mask1, axis=0)
    y_l = min(np.argwhere(y_map >= 1.0))[0]
    y_r = max(np.argwhere(y_map >= 1.0))[0]
    x_u = min(np.argwhere(x_map >= 1.0))[0]
    x_d = max(np.argwhere(x_map >= 1.0))[0]

    mask_added = mask1 + mask2
    mask_super = np.where(mask_added > 0, 1.0, 0)
    mask_overlap = np.where(mask_added > 1.0, 1.0, 0)
    o_x_map = np.sum(mask_overlap, axis=1)
    o_y_map = np.sum(mask_overlap, axis=0)
    o_x_u = min(np.argwhere(o_x_map >= 1.0))[0]
    o_x_d = max(np.argwhere(o_x_map >= 1.0))[0]
    o_y_l = min(np.argwhere(o_y_map >= 1.0))[0]
    o_y_r = max(np.argwhere(o_y_map >= 1.0))[0]

    radius_ratio = 0.15

    x_median = (o_x_u + o_x_d) // 2
    x_radius = int((o_x_d - x_median) * radius_ratio)
    y_median = (o_y_l + o_y_r) // 2
    y_radius = int((o_y_r - y_median) * radius_ratio)

    mass_overlap_1 = np.zeros(mask_overlap.shape)
    if mode is None:
        if abs(o_x_u - x_u) <= 3:
            mass_overlap_1[x_u:o_x_d + 1, :] = np.tile(np.linspace(0, 1, o_x_d - x_u + 1).reshape(-1, 1), (1, width))
        elif abs(o_x_d - x_d) <= 3:
            mass_overlap_1[o_x_u:o_x_d + 1, :] = np.tile(np.linspace(1, 0, o_x_d - o_x_u + 1).reshape(-1, 1),
                                                         (1, width))
        elif abs(o_y_l - y_l) <= 3:
            mass_overlap_1[:, y_l:o_y_r + 1] = np.tile(np.linspace(0, 1, o_y_r - y_l + 1).reshape(1, -1), (height, 1))
        else:
            mass_overlap_1[:, o_y_l:o_y_r + 1] = np.tile(np.linspace(1, 0, o_y_r - o_y_l + 1).reshape(1, -1),
                                                         (height, 1))
    else:
        if mode == 'u':
            mass_overlap_1[x_median - x_radius:x_median + x_radius, :] = np.tile(
                np.linspace(0, 1, 2 * x_radius).reshape(-1, 1), (1, width))
            mass_overlap_1[x_median + x_radius: o_x_d, :] = 1.0
        elif mode == 'd':
            mass_overlap_1[x_median - x_radius:x_median + x_radius, :] = np.tile(
                np.linspace(1, 0, 2 * x_radius).reshape(-1, 1),
                (1, width))
            mass_overlap_1[o_x_u: x_median - x_radius, :] = 1.0

        elif mode == 'l':
            mass_overlap_1[:, y_median - y_radius:y_median + y_radius] = np.tile(
                np.linspace(0, 1, 2 * y_radius).reshape(1, -1), (height, 1))
            mass_overlap_1[:, y_median + y_radius: o_y_r] = 1.0
        else:
            mass_overlap_1[:, y_median - y_radius:y_median + y_radius] = np.tile(
                np.linspace(1, 0, 2 * y_radius).reshape(1, -1),
                (height, 1))
            mass_overlap_1[:, o_y_l: y_median - y_radius] = 1.0
    mass_overlap_1 *= mask_overlap
    mass_overlap_2 = (1 - mass_overlap_1) * mask_overlap
    mass_overlap_1 = mass_overlap_1 + mask1 - mask_overlap
    mass_overlap_2 = mass_overlap_2 + mask2 - mask_overlap
    return mass_overlap_1, mass_overlap_2, mask_super, mask_overlap


def stitch_add_mask_linear_per_border(mask1, mask2):
    height, width = mask1.shape
    mask_added = mask1 + mask2
    mask_super = np.where(mask_added > 0, 1.0, 0)
    mask_overlap = np.where(mask_added > 1.0, 1.0, 0)
    radius_ratio = 0.15

    mass_overlap_1 = np.zeros(mask_overlap.shape)
    patch_width = width // 3
    for i in range(3):
        left_w = i * patch_width
        if i < 2:
            right_w = (i + 1) * patch_width
        else:
            right_w = width
        temp_overlap = mask_overlap[:, left_w: right_w]
        temp_x_map = np.sum(temp_overlap, axis=1)
        temp_x_u = min(np.argwhere(temp_x_map >= 1.0))[0]
        temp_x_d = max(np.argwhere(temp_x_map >= 1.0))[0]
        temp_median = (temp_x_u + temp_x_d) // 2
        temp_radius = int((temp_x_d - temp_median) * radius_ratio)

        mass_overlap_1[temp_median - temp_radius:temp_median + temp_radius, left_w: right_w] = np.tile(
            np.linspace(1, 0, 2 * temp_radius).reshape(-1, 1),
            (1, right_w - left_w))
        mass_overlap_1[temp_x_u: temp_median - temp_radius, left_w: right_w] = 1.0
    mass_overlap_1 *= mask_overlap
    mass_overlap_2 = (1 - mass_overlap_1) * mask_overlap
    mass_overlap_1 = mass_overlap_1 + mask1 - mask_overlap
    mass_overlap_2 = mass_overlap_2 + mask2 - mask_overlap
    return mass_overlap_1, mass_overlap_2, mask_super, mask_overlap


def direct_stitch(im1, im2, im1_color, im2_color, im1_mask, im2_mask):
    im1_shape = im1.shape
    im2_shape = im2.shape

    dis_h = int((im1_shape[0] - im2_shape[0]) // 2)

    if im1_mask is None:
        im1_mask = np.ones((im1.shape[0], im1.shape[1]))

    dis_w = int(im1.shape[1] * 0.1)

    h = im1_shape[0]
    extra_w = im2_shape[1] - dis_w
    w = im1_shape[1] + extra_w

    stitching_im1_res = np.zeros((h, w))
    stitching_im1_color_res = np.zeros((h, w, 3))
    stitch_im1_mask = np.zeros((h, w))

    stitching_im1_res[:, :im1_shape[1]] = im1
    stitching_im1_color_res[:, :im1_shape[1]] = im1_color
    
    stitch_im1_mask[:, :im1_shape[1]] = im1_mask
    
    stitch_im2_res = np.zeros((h, w))
    stitch_im2_color_res = np.zeros((h, w, 3))
    stitch_im2_mask = np.zeros((h, w))

    stitch_im2_mask[dis_h:im2_shape[0] + dis_h, im1_shape[1] - dis_w:] = 1.0
    stitch_im2_mask = stitch_im2_mask * (1 - stitch_im1_mask)

    stitch_im2_res[dis_h:im2_shape[0] + dis_h, im1_shape[1] - dis_w:] = im2
    stitch_im2_color_res[dis_h:im2_shape[0] + dis_h, im1_shape[1] - dis_w:] = im2_color

    stitching_res = stitching_im1_res * stitch_im1_mask + stitch_im2_res * stitch_im2_mask

    stitching_res_color = stitching_im1_color_res * stitch_im1_mask[:, :, np.newaxis] + stitch_im2_color_res * stitch_im2_mask[:, :, np.newaxis]
    
    mass = stitch_im1_mask + stitch_im2_mask

    return stitching_res, stitching_res_color, mass, None
