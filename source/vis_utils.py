import cv2
import numpy as np
import matplotlib.pyplot as plt

COLORS = [(0, 0, 255), # Blue
          (255, 0, 0), # Red
          (0, 255, 0), # Green
          (255, 0, 214) # Grey(ish)
          ]

def post_process_image(image, cvt_color=True):
    # clip before casting to uint8
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    if cvt_color:    
        # convert to uint8, then RGB -> BGR for openCV saving
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


# def draw_matches_vertical(img1, kp1, img2, kp2, matches, draw_matched_kps=False):
#     """
#     Draws matches between two images, stacking them vertically.
#     """
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]

#     # kp objects are mutable, so make copy here so the input
#     # parameters are not modified.
#     kp1 = kp1.copy()
#     kp2 = kp2.copy()
    
#     # Create a new canvas with appropriate height and max width
#     vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)
    
#     # Place images onto the new canvas
#     vis[:h1, :w1] = img1
#     vis[h1:, :w2] = img2
    
#     # Adjust keypoint coordinates for the second image
#     # cv2.drawMatchesKnn expects kp2 to be based on the *original* image
#     # We need to adjust the 'matches' data to point to the correct vertical location
#     # Note: cv2.drawMatchesKnn actually handles the coordinate offset internally 
#     # when given two separate images and a single output canvas (which is not how we set this up).
#     # We need a custom drawing logic or use the original `drawMatches` which handles canvas creation.

#     # A simpler approach using drawMatches with a custom combined image
#     # This requires manually adjusting all matched keypoint coordinates in kp2
#     adjusted_kp2 = []
#     for kp in kp2:
#         kp.pt = (kp.pt[0], kp.pt[1] + h1)
#         adjusted_kp2.append(kp)

#     # Use a loop to draw lines manually on the combined image for better control
#     for match in matches:
#             p1 = tuple(map(int, kp1[match[0]].pt)) # queryIdx
#             p2 = tuple(map(int, adjusted_kp2[match[1]].pt)) # trainIdx
#             cv2.line(vis, p1, p2, (0, 255, 0), 1) # Green lines

#             if draw_matched_kps:
#                 cv2.circle(vis, tuple(map(int, kp1[match[0]].pt)), 2, (0, 0, 255), 1)

#                 cv2.circle(vis, tuple(map(int, adjusted_kp2[match[1]].pt)), 2, (0, 0, 255), 1)

            
#     # Draw keypoints (optional)
#     if not draw_matched_kps:
    
#         for kp in kp1:
#             cv2.circle(vis, tuple(map(int, kp.pt)), 4, (0, 0, 255), 1)
            
#         for kp in adjusted_kp2:
#             cv2.circle(vis, tuple(map(int, kp.pt)), 4, (0, 0, 255), 1)
        
#     return vis

## helper for debugging
def plot_single_image(img, grayscale=True):

    plt.axis('off')
    plt.imshow(img, cmap='gray' if grayscale else None)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.tight_layout()
    plt.show()


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
            p1 = tuple(p1.astype(int)) # queryIdx
            
            p2 = tuple(p2.astype(int)) # trainIdx

            # draw line connecting KPs
            cv2.line(vis, p1, p2, (0, 255, 0), 1)

            # draw KPs
            cv2.circle(vis, p1, 4, (0, 0, 255), 1)
            cv2.circle(vis, p2, 4, (0, 0, 255), 1)
        
    return vis

# for drawing after filter_isolate or filter_geometry
def draw_matches_vertical(img1, kp1, img2, kp2):
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
            p1 = tuple(p1.astype(int)) # queryIdx
            
            p2 = tuple(p2.astype(int)) # trainIdx

            # draw line connecting KPs
            cv2.line(vis, p1, p2, (0, 255, 0), 1)

            # draw KPs
            cv2.circle(vis, p1, 4, (0, 0, 255), 1)
            cv2.circle(vis, p2, 4, (0, 0, 255), 1)
        
    return vis

# for drawing after filter_isolate or filter_geometry
def draw_matches_horizontal(img1, kp1, img2, kp2):
    """
    Draws matches between two images, stacking them vertically.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create a new canvas with appropriate height and max width
    # vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    
    # Place images onto the new canvas
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    
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
        kp = (kp[0] + w1, kp[1])
        adjusted_kp2.append(kp)

    adjusted_kp2 = np.float32(adjusted_kp2)
        
    # Use a loop to draw lines manually on the combined image for better control
    for p1, p2 in zip(kp1, adjusted_kp2):
            p1 = tuple(p1.astype(int)) # queryIdx
            
            p2 = tuple(p2.astype(int)) # trainIdx

            # draw line connecting KPs
            cv2.line(vis, p1, p2, (0, 255, 0), 1)

            # draw KPs
            cv2.circle(vis, p1, 4, (0, 0, 255), 1)
            cv2.circle(vis, p2, 4, (0, 0, 255), 1)
        
    return vis


def draw_keypoints(img, kp, kp_color):

    im_draw = img.copy() 
  
    # draw KPs
    for p in kp:
        cv2.circle(im_draw, tuple(map(int, p.pt)), 2, kp_color, 1) 

    return im_draw

def draw_matches_helper(im1, srckp, im2, tgtkp, plot_vertical, plot_title):

    # ensure uint8 dtype for opencv 
    im1 = post_process_image(im1.copy(), cvt_color=False)
    im2 = post_process_image(im2.copy(), cvt_color=False)
            
    if plot_vertical:
        matched_im = draw_matches_vertical(im1, srckp, im2, tgtkp)
            
    else:        
        matched_im = draw_matches_horizontal(im1, srckp, im2, tgtkp)

    plt.title(plot_title)
    plt.imshow(matched_im)
    plt.axis('off')
        
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    
    plt.tight_layout()
    plt.show()

def draw_keypoint_subsets(im1, kp1_subsets, im2, kp2_subsets, plot_vertical):
  
    # DEBUG: draw subset keypoints
    im1_draw = im1.copy()
    for i, kp_subset in enumerate(kp1_subsets):
        im1_draw = draw_keypoints(im1_draw, kp_subset, COLORS[i % len(COLORS)])

    im2_draw = im2.copy()
    for i, kp_subset in enumerate(kp2_subsets):
        im2_draw = draw_keypoints(im2_draw, kp_subset, COLORS[i % len(COLORS)])

    if plot_vertical:
        h1, w1 = im1_draw.shape[:2]
        h2, w2 = im2_draw.shape[:2]
        
        im_combined_h = im1_draw.shape[0] + im2_draw.shape[0] 
        im_combined_w = max(im1_draw.shape[1], im2_draw.shape[1])
        
        # combine images:
        im_draw_combined = np.zeros((im_combined_h, im_combined_w, 3), dtype=np.uint8)
        
        # Place images onto the new canvas
        im_draw_combined[:h1, :w1] = im1_draw
        im_draw_combined[h1:, :w2] = im2_draw

    else:
        h1, w1 = im1_draw.shape[:2]
        h2, w2 = im2_draw.shape[:2]
        
        im_combined_h = max(im1_draw.shape[0], im2_draw.shape[0])
        im_combined_w = im1_draw.shape[1] + im2_draw.shape[1]
        
        # combine images:
        im_draw_combined = np.zeros((im_combined_h, im_combined_w, 3), dtype=np.uint8)
        
        # Place images onto the new canvas
        im_draw_combined[:h1, :w1] = im1_draw
        im_draw_combined[:h2, w1:w1+w2] = im2_draw
    
    plt.title("Keypoints (RAW)")
    plt.imshow(im_draw_combined)
    plt.axis('off')
        
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.tight_layout()
    plt.show()
    
