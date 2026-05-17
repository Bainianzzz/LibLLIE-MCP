from _utils import ensure_results_dir, find_example_folder, find_example_image

import libllie as llie


def main():
    image_path = find_example_image()
    input_dir = find_example_folder()
    output_dir = ensure_results_dir("traditional")

    enhanced, single_path = llie.predict(
        "gamma",
        # image_path,
        r"D:\13011\Documents\8.png",
        # color_space='hsv',
        gamma=1.5,
        # output=output_dir / "he_yuv.jpg",
        output=r"D:\13011\Documents\9.png",
    )

    print(f"Single traditional output: {single_path}")
    print(f"Single output shape: {enhanced.shape}")


    # batch_paths = llie.predict(
    #     "rclahe",
    #     input_dir,
    #     color_space='hsv',
    #     clip_limit=2.0,
    #     tile_grid_size=(8, 8),
    #     iterations=3,
    #     output=output_dir / "clahe_batch",
    #     progress_bar=True,
    # )
    #
    # print(f"Batch traditional output dir: {output_dir / 'clahe_batch'}")
    # print(f"Batch image count: {len(batch_paths)}")


if __name__ == "__main__":
    main()
