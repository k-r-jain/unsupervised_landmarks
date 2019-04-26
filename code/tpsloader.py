import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'tpsrepo/'))
sys.path.append(os.path.join(os.getcwd(), 'code/tpsrepo/'))
from tpsrepo import thinplate as tps
import cv2

def show_warped(img, warped, c_src, c_dst):
    fig, axs = plt.subplots(1, 2, figsize=(16,8))
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].imshow(img[...,::-1], origin='upper')
    axs[0].scatter(c_src[:, 0]*img.shape[1], c_src[:, 1]*img.shape[0], marker='+', color='black')
    axs[1].imshow(warped[...,::-1], origin='upper')
    axs[1].scatter(c_dst[:, 0]*warped.shape[1], c_dst[:, 1]*warped.shape[0], marker='+', color='black')
    plt.show()

def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

def custom_tps_function(path):

    img = cv2.imread(path)
    src_three_points = [[0.0, 0.0],
        [1., 0],
        [1, 1],
        [0, 1]]
    dst_three_points = [[0., 0],
        [1., 0],    
        [1, 1],
        [0, 1]]
    for _ in range(3):
        x = np.random.rand()
        y = x
        delta_x = np.random.rand() / 10.0
        delta_y = delta_x
        delta_x += x
        delta_x %= 1.0
        delta_y += y
        delta_x %= 1.0
        src_three_points.append([x, y])
        dst_three_points.append([delta_x, delta_y])

    # print(np.random.rand())
    c_src = np.array(src_three_points)

    c_dst = np.array(dst_three_points)
        
    warped = warp_image_cv(img, c_src, c_dst, dshape=(512, 512))
    # show_warped(img, warped, c_src, c_dst)
    im_pil = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(im_pil)
    # plt.imshow(im_pil)
    # plt.show()
    return im_pil
    

