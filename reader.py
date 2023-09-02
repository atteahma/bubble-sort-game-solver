import cv2
from numba import njit
import numpy as np
from tqdm import tqdm
from sklearn import cluster
from scipy.ndimage import convolve

import os
from collections import Counter

from utils import mask3dwith2d, segment_image


class Reader:
    def __init__(self, input_dir):
        self.input_dir = os.path.join(input_dir)

    # reads image from disk
    def read_image(self, image_name):
        input_path = os.path.join(self.input_dir, image_name)
        image = cv2.imread(input_path)
        return image

    # takes in an image and outputs bins of balls, grid size, and a mapping of id's to colors
    def analyze_image(self, im):
        # crop to only the bins
        im_crop = im[500:2250]

        # remove background
        bins_mask = self.get_bins_mask(im_crop)
        bins_im = mask3dwith2d(im_crop, bins_mask)

        # get only balls
        block_mask = self.get_clean_by_blocks_mask(bins_im, 15)
        balls_im = mask3dwith2d(bins_im, block_mask)

        # split image into bins/individual balls ims
        bins_ims, grid_widths = self.segment_to_individual(balls_im)

        # get mode colors for each ball
        bins_colors = self.convert_ims_to_colors(bins_ims)

        # fit a clustering function to take care of any small pixel variances
        bins_ids, id2color_map = self.convert_colors_to_ids(bins_colors)

        # add the final two blank bins
        bins_ids.append([-1 for _ in range(4)])
        bins_ids.append([-1 for _ in range(4)])
        grid_widths[-1] += 2

        widths_str = "(" + " ".join(map(str, grid_widths)) + ")"
        print(f"found a grid with height={len(grid_widths)} and widths={widths_str}")

        return bins_ids, grid_widths, id2color_map

    def convert_colors_to_ids(self, bins_colors):
        all_colors = np.array(list(set([ball for bin in bins_colors for ball in bin])))
        num_colors_expected = len(bins_colors)
        bins_ids = []

        print(f"found {all_colors.shape[0]} colors, expected {num_colors_expected}")
        if all_colors.shape[0] < num_colors_expected:
            print(f"found fewer colors than expected, exiting.")
            return None, None

        clustering_func = cluster.KMeans(n_clusters=num_colors_expected).fit(all_colors)

        for bin_colors in bins_colors:
            bin_ids = clustering_func.predict(bin_colors)
            bins_ids.append(list(bin_ids))

        ids_to_bins = [tuple(map(int, c)) for c in clustering_func.cluster_centers_]

        return bins_ids, ids_to_bins

    def convert_ims_to_colors(self, bins_ims):
        bins_colors = []

        for bin_ims in bins_ims:
            curr_bin = []
            for ball_im in bin_ims:
                H, W, _ = ball_im.shape
                ball_vectorized = ball_im.reshape((H * W, 3))
                ball_colors, ball_color_counts = np.unique(
                    ball_vectorized, return_counts=True, axis=0
                )

                ball_color_mode = max(
                    [
                        (col, cnt)
                        for col, cnt in zip(ball_colors, ball_color_counts)
                        if not np.all(col == 0)
                    ],
                    key=lambda p: p[1],
                )[0]

                curr_bin.append(tuple(ball_color_mode))
            bins_colors.append(curr_bin)

        return bins_colors

    def segment_to_individual(self, im):
        cols = segment_image(im, pad=0, axis=1)
        rows = segment_image(im, pad=0, axis=0)
        grid_widths = []

        for row in rows[::4]:
            width = len(segment_image(row, pad=0, axis=1))
            grid_widths.append(width)

        rows_bin_ims = [[] for _ in grid_widths]

        for col in cols:
            rows_in_col = segment_image(col, pad=0, axis=0)
            num_bins_vert = len(rows_in_col) // 4

            for bucket_i in range(num_bins_vert):
                rows_bin_ims[bucket_i].append(
                    rows_in_col[4 * bucket_i : 4 * (bucket_i + 1)][::-1]
                )

        bins_ims = [bin_im for row_bin_ims in rows_bin_ims for bin_im in row_bin_ims]

        return bins_ims, grid_widths

    def get_bins_mask(self, im):
        return (im[:, :, 0] < 225) | ~(
            (np.abs(im[:, :, 0] - im[:, :, 1]) < 5)
            & (np.abs(im[:, :, 1] - im[:, :, 2]) < 5)
        )

    def get_clean_by_blocks_mask(self, im, filter_radius, thresh=25):
        im_mask = np.any(im > thresh, axis=2).astype(int)
        kernel = np.ones((filter_radius * 2 + 1, filter_radius * 2 + 1))

        mask_by_block = (
            convolve(im_mask, kernel, mode="constant", cval=0.0) == kernel.sum()
        )

        return mask_by_block

    def validate_bins(self, bins):
        ball_counts = Counter([ball for bin in bins for ball in bin])

        if sum([count for count in ball_counts.values()]) % 4 != 0:
            print("count of balls not divisible by 4")
            return False

        if any([count != 4 for ball, count in ball_counts.items() if ball != -1]):
            print("a ball type has a count not equal to 4")
            return False

        return True
