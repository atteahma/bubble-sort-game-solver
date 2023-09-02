from reader import Reader
from solver import Solver
from creator import Creator

import os
import pickle as p
import sys

import pandas as pd
import plotly.express as px


def main():
    out_size = (1, 1)
    input_dir = "./images"
    output_dir = "./walkthroughs"

    reader = Reader(input_dir)
    solver = Solver()
    creator = Creator(out_size, output_dir)

    image_names = os.listdir(input_dir)

    for image_name in image_names:
        if len(sys.argv) > 1 and not image_name.startswith(sys.argv[1]):
            continue

        print(f"analyzing image {image_name}...")
        image = reader.read_image(image_name)
        bins, grid_widths, color_map = reader.analyze_image(image)

        ok = reader.validate_bins(bins)

        if not ok:
            print(f"{image_name} FAILED")
            break

        print("done.\n")

        print(bins)

        print("solving for optimal solution...")
        solution = solver.solve(bins)
        print("done.\n")

        px.line(
            pd.DataFrame(solver.depth_by_call_order, columns=["call index", "depth"]),
            x="call index",
            y="depth",
            title="",
        ).show()

        print(solution)

        creator.create_guide(solution, grid_widths)

        # print("creating walkthrough...")
        # walkthrough = creator.create_walkthrough(solution, bins, color_map, grid_widths)
        # success = creator.save_walkthrough(walkthrough, image_name)
        # if not success:
        #     print(f"{image_name} FAILED")
        #     break
        # print("done.\n")


if __name__ == "__main__":
    main()
