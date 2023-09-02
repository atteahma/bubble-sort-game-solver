import cv2

import os

from utils import index_to_ij


class Creator:
    def __init__(self, out_size, out_dir):
        self.out_size = out_size
        self.out_dir = os.path.join(out_dir)

    # takes in a solution, initial bin state, and a color map and outputs a video solution
    def create_walkthrough(self, solution, initial_bins, color_map, grid_widths):
        return None

    def create_guide(self, solution, grid_widths):
        for from_ind, to_ind in solution:
            from_i, from_j = index_to_ij(from_ind, grid_widths)
            to_i, to_j = index_to_ij(to_ind, grid_widths)

            print(
                f"move ball from row {from_i} column {from_j} to row {to_i} column {to_j}"
            )

    # saves a walkthrough to disk
    def save_walkthrough(self, walkthrough, image_name):
        try:
            walkthrough_name = ".".join(image_name.split(".")[0]) + "_walkthrough.avi"
            walkthrough_path = os.path.join(self.out_dir, walkthrough_name)
            num_frames = walkthrough.shape[0]

            out = cv2.VideoWriter(walkthrough_path, -1, 20.0, self.out_size)

            for i in range(num_frames):
                frame = walkthrough[i]
                frame = cv2.flip(frame, 0)
                out.write(frame)

            out.release()

        except Exception as e:
            print("caught an exception while trying to save a walkthrough")
            print(e)
            return False

        return True
