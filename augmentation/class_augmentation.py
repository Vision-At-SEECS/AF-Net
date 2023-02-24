from multiprocessing import pool
from operator import index
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from matplotlib import cm
# from torch_cluster import neighbor_sampler
import xlsxwriter
import glob
import os
import math
import skimage.filters
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import (distance_transform_cdt,
                                      distance_transform_edt)
import skimage.filters
from skimage.morphology import disk
import time
from matplotlib.colors import ListedColormap
import random
import tqdm

from math import sqrt as sqrt
import heapq
import numpy as np

# flags
KNOWN = 0
BAND = 1
UNKNOWN = 2
# extremity values
INF = 1e6 # dont use np.inf to avoid inf * 0
EPS = 1e-6

# solves a step of the eikonal equation in order to find closest quadrant
def _solve_eikonal(y1, x1, y2, x2, height, width, dists, flags):
    # check image frame
    if y1 < 0 or y1 >= height or x1 < 0 or x1 >= width:
        return INF

    if y2 < 0 or y2 >= height or x2 < 0 or x2 >= width:
        return INF

    flag1 = flags[y1, x1]
    flag2 = flags[y2, x2]

    # both pixels are known
    if flag1 == KNOWN and flag2 == KNOWN:
        dist1 = dists[y1, x1]
        dist2 = dists[y2, x2]
        d = 2.0 - (dist1 - dist2) ** 2
        if d > 0.0:
            r = sqrt(d)
            s = (dist1 + dist2 - r) / 2.0
            if s >= dist1 and s >= dist2:
                return s
            s += r
            if s >= dist1 and s >= dist2:
                return s
            # unsolvable
            return INF

    # only 1st pixel is known
    if flag1 == KNOWN:
        dist1 = dists[y1, x1]
        return 1.0 + dist1

    # only 2d pixel is known
    if flag2 == KNOWN:
        dist2 = dists[y2, x2]
        return 1.0 + dist2

    # no pixel is known
    return INF

# returns gradient for one pixel, computed on 2 pixel range if possible
def _pixel_gradient(y, x, height, width, vals, flags):
    val = vals[y, x]

    # compute grad_y
    prev_y = y - 1
    next_y = y + 1
    if prev_y < 0 or next_y >= height:
        grad_y = INF
    else:
        flag_prev_y = flags[prev_y, x]
        flag_next_y = flags[next_y, x]

        if flag_prev_y != UNKNOWN and flag_next_y != UNKNOWN:
            grad_y = (vals[next_y, x] - vals[prev_y, x]) / 2.0
        elif flag_prev_y != UNKNOWN:
            grad_y = val - vals[prev_y, x]
        elif flag_next_y != UNKNOWN:
            grad_y = vals[next_y, x] - val
        else:
            grad_y = 0.0

    # compute grad_x
    prev_x = x - 1
    next_x = x + 1
    if prev_x < 0 or next_x >= width:
        grad_x = INF
    else:
        flag_prev_x = flags[y, prev_x]
        flag_next_x = flags[y, next_x]

        if flag_prev_x != UNKNOWN and flag_next_x != UNKNOWN:
            grad_x = (vals[y, next_x] - vals[y, prev_x]) / 2.0
        elif flag_prev_x != UNKNOWN:
            grad_x = val - vals[y, prev_x]
        elif flag_next_x != UNKNOWN:
            grad_x = vals[y, next_x] - val
        else:
            grad_x = 0.0

    return grad_y, grad_x

# compute distances between initial mask contour and pixels outside mask, using FMM (Fast Marching Method)
def _compute_outside_dists(height, width, dists, flags, band, radius):
    band = band.copy()
    orig_flags = flags
    flags = orig_flags.copy()
    # swap INSIDE / OUTSIDE
    flags[orig_flags == KNOWN] = UNKNOWN
    flags[orig_flags == UNKNOWN] = KNOWN

    last_dist = 0.0
    double_radius = radius * 2
    while band:
        # reached radius limit, stop FFM
        if last_dist >= double_radius:
            break

        # pop BAND pixel closest to initial mask contour and flag it as KNOWN
        _, y, x = heapq.heappop(band)
        flags[y, x] = KNOWN

        # process immediate neighbors (top/bottom/left/right)
        neighbors = [(y - 1, x), (y, x - 1), (y + 1, x), (y, x + 1)]
        for nb_y, nb_x in neighbors:
            # skip out of frame
            if nb_y < 0 or nb_y >= height or nb_x < 0 or nb_x >= width:
                continue

            # neighbor already processed, nothing to do
            if flags[nb_y, nb_x] != UNKNOWN:
                continue

            # compute neighbor distance to inital mask contour
            last_dist = min([
                _solve_eikonal(nb_y - 1, nb_x, nb_y, nb_x - 1, height, width, dists, flags),
                _solve_eikonal(nb_y + 1, nb_x, nb_y, nb_x + 1, height, width, dists, flags),
                _solve_eikonal(nb_y - 1, nb_x, nb_y, nb_x + 1, height, width, dists, flags),
                _solve_eikonal(nb_y + 1, nb_x, nb_y, nb_x - 1, height, width, dists, flags)
            ])
            dists[nb_y, nb_x] = last_dist

            # add neighbor to narrow band
            flags[nb_y, nb_x] = BAND
            heapq.heappush(band, (last_dist, nb_y, nb_x))

    # distances are opposite to actual FFM propagation direction, fix it
    dists *= -1.0

# computes pixels distances to initial mask contour, flags, and narrow band queue
def _init(height, width, mask, radius):
    # init all distances to infinity
    dists = np.full((height, width), INF, dtype=float)
    # status of each pixel, ie KNOWN, BAND or UNKNOWN
    flags = mask.astype(int) * UNKNOWN
    # narrow band, queue of contour pixels
    band = []

    mask_y, mask_x = mask.nonzero()
    for y, x in zip(mask_y, mask_x):
        # look for BAND pixels in neighbors (top/bottom/left/right)
        neighbors = [(y - 1, x), (y, x - 1), (y + 1, x), (y, x + 1)]
        for nb_y, nb_x in neighbors:
            # neighbor out of frame
            if nb_y < 0 or nb_y >= height or nb_x < 0 or nb_x >= width:
                continue

            # neighbor already flagged as BAND
            if flags[nb_y, nb_x] == BAND:
                continue

            # neighbor out of mask => mask contour
            if mask[nb_y, nb_x] == 0:
                flags[nb_y, nb_x] = BAND
                dists[nb_y, nb_x] = 0.0
                heapq.heappush(band, (0.0, nb_y, nb_x))


    # compute distance to inital mask contour for KNOWN pixels
    # (by inverting mask/flags and running FFM)
    _compute_outside_dists(height, width, dists, flags, band, radius)

    return dists, flags, band

# returns RGB values for pixel to by inpainted, computed for its neighborhood
def _inpaint_pixel(y, x, img, height, width, dists, flags, radius):
    dist = dists[y, x]
    # normal to pixel, ie direction of propagation of the FFM
    dist_grad_y, dist_grad_x = _pixel_gradient(y, x, height, width, dists, flags)
    pixel_sum = np.zeros((3), dtype=float)
    weight_sum = 0.0

    # iterate on each pixel in neighborhood (nb stands for neighbor)
    for nb_y in range(y - radius, y + radius + 1):
        #  pixel out of frame
        if nb_y < 0 or nb_y >= height:
            continue

        for nb_x in range(x - radius, x + radius + 1):
            # pixel out of frame
            if nb_x < 0 or nb_x >= width:
                continue

            # skip unknown pixels (including pixel being inpainted)
            if flags[nb_y, nb_x] == UNKNOWN:
                continue

            # vector from point to neighbor
            dir_y = y - nb_y
            dir_x = x - nb_x
            dir_length_square = dir_y ** 2 + dir_x ** 2
            dir_length = sqrt(dir_length_square)
            # pixel out of neighborhood
            if dir_length > radius:
                continue

            # compute weight
            # neighbor has same direction gradient => contributes more
            dir_factor = abs(dir_y * dist_grad_y + dir_x * dist_grad_x)
            if dir_factor == 0.0:
                dir_factor = EPS

            # neighbor has same contour distance => contributes more
            nb_dist = dists[nb_y, nb_x]
            level_factor = 1.0 / (1.0 + abs(nb_dist - dist))

            # neighbor is distant => contributes less
            dist_factor = 1.0 / (dir_length * dir_length_square)

            weight = abs(dir_factor * dist_factor * level_factor)

            pixel_sum[0] += weight * img[nb_y, nb_x, 0]
            pixel_sum[1] += weight * img[nb_y, nb_x, 1]
            pixel_sum[2] += weight * img[nb_y, nb_x, 2]

            weight_sum += weight

    return pixel_sum / weight_sum

# main inpainting function
def inpaint(img, mask, radius=5):
    if img.shape[0:2] != mask.shape[0:2]:
        raise ValueError("Image and mask dimensions do not match")

    height, width = img.shape[0:2]
    dists, flags, band = _init(height, width, mask, radius)

    # find next pixel to inpaint with FFM (Fast Marching Method)
    # FFM advances the band of the mask towards its center,
    # by sorting the area pixels by their distance to the initial contour
    while band:
        # pop band pixel closest to initial mask contour
        _, y, x = heapq.heappop(band)
        # flag it as KNOWN
        flags[y, x] = KNOWN

        # process his immediate neighbors (top/bottom/left/right)
        neighbors = [(y - 1, x), (y, x - 1), (y + 1, x), (y, x + 1)]
        for nb_y, nb_x in neighbors:
            # pixel out of frame
            if nb_y < 0 or nb_y >= height or nb_x < 0 or nb_x >= width:
                continue

            # neighbor outside of initial mask or already processed, nothing to do
            if flags[nb_y, nb_x] != UNKNOWN:
                continue

            # compute neighbor distance to inital mask contour
            nb_dist = min([
                _solve_eikonal(nb_y - 1, nb_x, nb_y, nb_x - 1, height, width, dists, flags),
                _solve_eikonal(nb_y + 1, nb_x, nb_y, nb_x + 1, height, width, dists, flags),
                _solve_eikonal(nb_y - 1, nb_x, nb_y, nb_x + 1, height, width, dists, flags),
                _solve_eikonal(nb_y + 1, nb_x, nb_y, nb_x - 1, height, width, dists, flags)
            ])
            dists[nb_y, nb_x] = nb_dist

            # inpaint neighbor
            pixel_vals = _inpaint_pixel(nb_y, nb_x, img, height, width, dists, flags, radius)

            img[nb_y, nb_x, 0] = pixel_vals[0]
            img[nb_y, nb_x, 1] = pixel_vals[1]
            img[nb_y, nb_x, 2] = pixel_vals[2]

            # add neighbor to narrow band
            flags[nb_y, nb_x] = BAND
            # push neighbor on band
            heapq.heappush(band, (nb_dist, nb_y, nb_x))

def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def rotate(image, angle, center=None, scale=1.0):
    h, w = image.shape[:2]
    if center is None:
        center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def size_calculate(major_id, inst_map):
    size_1 = np.sum((inst_map == (major_id + 1))>0)
    return size_1

def pick_minor_index(pool_minor, size_1):
    for basename, minor_class_list in pool_minor.items():
        ann_2 = sio.loadmat('/content/CoNSeP/Train/Labels/' + basename + '.mat')
        inst_map_2 = ann_2['inst_map']
        for minor_class_id in minor_class_list:
            mask_2 = (inst_map_2 == (minor_class_id+1)).astype(np.uint8)
            size_2 = np.sum(mask_2>0)
            if size_1 >= 2.3 * size_2:
                pool_minor[basename].remove(minor_class_id)
                return basename, ann_2, minor_class_id, pool_minor


file_1_list = glob.glob('/content/CoNSeP/Train/Images/*.png')

for file_1 in tqdm.tqdm(file_1_list):
    print(file_1)
    eps = 5
    img_list = glob.glob('/content/CoNSeP/Train/Images/*.png')
    random.shuffle(img_list)
    img_list.remove(file_1)

    file_name = os.path.basename(file_1)
    basename = file_name.split('.')[0]
    # Img Read
    img = cv2.imread('/content/CoNSeP/Train/Images/' + file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Label Read : Label contains 'inst_map', 'type_map', 'inst_type', 'inst_centroid'

    ann = sio.loadmat('/content/CoNSeP/Train/Labels/' + basename + '.mat')
    inst_map = ann['inst_map']
    inst_map_out = inst_map.copy()
    type_map = ann['type_map']
    class_arr = np.squeeze(ann['inst_type'])

    # Combine nuclie classes: you can skip depends upon your data
    class_arr[(class_arr == 3) | (class_arr == 4)] = 3
    class_arr[(class_arr == 5) | (class_arr == 6) | (class_arr == 7)] = 4
    class_arr_copy = class_arr.copy()

    cent_ann = ann['inst_centroid']  # x, y
    for i, cent in enumerate(cent_ann):  # old_value = 30
        if ((cent[1] < 30) or
                (cent[1] > (inst_map.shape[0] - 30)) or
                (cent[0] < 30) or
                (cent[0] > (inst_map.shape[1] - 30))):
            class_arr_copy[i] = 0
    nuc_color = img * (inst_map[..., np.newaxis] > 0)
    # Avg Nuclear color intensities
    avg_color_1 = [
        np.sum(nuc_color[..., 0]) / np.sum(nuc_color[..., 0] > 0),
        np.sum(nuc_color[..., 1]) / np.sum(nuc_color[..., 1] > 0),
        np.sum(nuc_color[..., 2]) / np.sum(nuc_color[..., 2] > 0)
    ]
    # Calculate Major  and Minor class indices
    major_class_idx = list(np.where(class_arr_copy == 2)[0]) + list(np.where(class_arr_copy == 3)[0]) + list(
        np.where(class_arr_copy == 4)[0])
    picked_major_class = list(np.random.choice(major_class_idx, int(0.8 * len(major_class_idx)), replace=False))
    picked_major_class = sorted(picked_major_class, key=lambda x: size_calculate(x, inst_map))
    final = img.copy()
    inpainted = img.copy()

    pool_minor = {}
    class_arr_2 = class_arr_copy.copy()
    cent_ann_2 = cent_ann.copy()

    minor_class_idx = list(np.where(class_arr_2 == 1)[0])
    pool_minor[basename] = minor_class_idx

    final = img.copy()
    inpainted = img.copy()

    pool_minor = {}
    class_arr_2 = class_arr_copy.copy()
    cent_ann_2 = cent_ann.copy()
    minor_class_idx = list(np.where(class_arr_2 == 1)[0])

    pool_minor[basename] = minor_class_idx
    for file in img_list:
        file_name = os.path.basename(file)
        basename_1 = file_name.split('.')[0]
        ann_2 = sio.loadmat('/content/CoNSeP/Train/Labels/' + basename_1 + '.mat')
        inst_map_2 = ann['inst_map']
        class_arr_2 = np.squeeze(ann_2['inst_type'])
        class_arr_2[(class_arr_2 == 3) | (class_arr_2 == 4)] = 3
        class_arr_2[(class_arr_2 == 5) | (class_arr_2 == 6) | (class_arr_2 == 7)] = 4
        cent_ann_2 = ann_2['inst_centroid']

        for i, cent in enumerate(cent_ann_2):
            if ((cent[1] < 30) or
                    (cent[1] > (inst_map_2.shape[0] - 30)) or
                    (cent[0] < 30) or
                    (cent[0] > (inst_map_2.shape[1] - 30))):
                class_arr_2[i] = 0
        minor_class_idx = list(np.where(class_arr_2 == 1)[0])
        pool_minor[basename_1] = minor_class_idx

    for major_class_idx in picked_major_class:
        mask_0 = (inst_map == (major_class_idx + 1)).astype(np.uint8)

        mask = binary_dilation(mask_0, iterations=2).astype(np.uint8)
        cent1 = cent_ann[major_class_idx]
        bbox1 = bounding_box(mask)
        h1, w1 = bbox1[1] - bbox1[0], bbox1[3] - bbox1[2]
        size_1 = np.sum(mask > 0)

        try:
            basename_2, ann_2, index_2, pool_minor = pick_minor_index(pool_minor, size_1)
        except TypeError:
            continue
        img_2_ori = cv2.imread('/content/CoNSeP/Train/Images/' + basename_2 + '.png')
        img_2_ori = cv2.cvtColor(img_2_ori, cv2.COLOR_BGR2RGB)
        img_2 = img_2_ori.copy()
        inst_map_2 = ann_2['inst_map']
        mask_2 = (inst_map_2 == (index_2 + 1)).astype(np.uint8)
        cent_ann_2 = ann_2['inst_centroid']
        cent_2 = cent_ann_2[index_2]
        bbox2 = bounding_box(mask_2)
        h2, w2 = bbox2[1] - bbox2[0], bbox2[3] - bbox2[2]

        img_2[..., 0][mask_2 > 0] = (img_2_ori[..., 0][mask_2 > 0] + avg_color_1[0]) / 2
        img_2[..., 1][mask_2 > 0] = (img_2_ori[..., 1][mask_2 > 0] + avg_color_1[1]) / 2
        img_2[..., 2][mask_2 > 0] = (img_2_ori[..., 2][mask_2 > 0] + avg_color_1[2]) / 2

        class_arr[major_class_idx] = 1

        ## Inapinting

        inpaint(final, mask, eps)
        inpaint(inpainted, mask, eps)

        inst_map_out[inst_map == (major_class_idx + 1)] = 0
        img_copy = img.copy()
        img_copy[bbox1[0]:bbox1[1], bbox1[2]:bbox1[3], :] = img_2[
                                                            int(np.round(cent_2[1]) - h1 / 2):int(
                                                                np.round(cent_2[1]) + h1 / 2),
                                                            int(np.round(cent_2[0]) - w1 / 2):int(
                                                                np.round(cent_2[0]) + w1 / 2),
                                                            :
                                                            ]
        mask_translated = np.zeros_like(mask)
        mask_translated[int(np.round(cent1[1]) - h2 / 2):int(np.round(cent1[1]) + h2 / 2),
        int(np.round(cent1[0]) - w2 / 2):int(np.round(cent1[0]) + w2 / 2)] = mask_2[bbox2[0]:bbox2[1],
                                                                             bbox2[2]:bbox2[3]]
        inst_map_out[mask_translated > 0] = major_class_idx + 1
        mask = ((mask + mask_translated) > 0).astype(np.uint8)
        mask_substract = mask - mask_translated
        cdt_map = distance_transform_cdt(1 - mask_translated).astype('float32')
        cdt_map[mask == 0] = 0
        cdt_map[mask_substract > 0] -= 1
        cdt_map[mask_substract > 0] /= np.amax(cdt_map[mask_substract > 0])
        cdt_map[mask_substract > 0] = 1 - cdt_map[mask_substract > 0]
        cdt_map[mask_translated > 0] = 1

        #
        final = final * (1 - mask[..., np.newaxis]) + img_copy * mask_translated[..., np.newaxis] + (
                    img_copy * (cdt_map * mask_substract)[..., np.newaxis]).astype(np.uint8) + (
                            final * ((1 - cdt_map) * mask_substract)[..., np.newaxis]).astype(np.uint8)
        final = (img_copy * cdt_map[..., np.newaxis]).astype(np.uint8) + (
                    final * (1 - cdt_map)[..., np.newaxis]).astype(np.uint8)
        final_smooth = np.stack(
            [skimage.filters.median(final[..., 0], disk(1)), skimage.filters.median(final[..., 1], disk(1)),
             skimage.filters.median(final[..., 2], disk(5))], axis=2)
        final = (final * (1 - mask_substract[..., np.newaxis])).astype(np.uint8) + (
                    final_smooth.astype(np.float32) * mask_substract[..., np.newaxis]).astype(np.uint8)
        # final = img_copy * mask_translated[...,np.newaxis].astype(np.uint8) + final * (1 - mask_translated)[...,np.newaxis].astype(np.uint8)

    final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
    inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    type_map = np.zeros_like(type_map)
    inst_list = list(np.unique(inst_map_out))
    inst_type = []
    inst_list.remove(0)
    for inst_id in inst_list:
        type_map[inst_map_out == int(inst_id)] = class_arr[int(inst_id) - 1]
        inst_type.append(class_arr[int(inst_id - 1)])
    cv2.imwrite('/content/CoNSeP/Train/Grad_mix_Images/' + basename + '_synthesized.png', final)
    cv2.imwrite('/content/CoNSeP/Train/Grad_mix_Inpainted/' + basename + '_inpainted.png', inpainted)
    sio.savemat('/content/CoNSeP/Train/Grad_mix_Labels/' + basename + '_synthesized.mat',
                {'inst_map': inst_map_out,
                 'type_map': type_map,
                 'inst_type': np.array(class_arr[:, None]),
                 'inst_centroid': cent_ann,
                 })
