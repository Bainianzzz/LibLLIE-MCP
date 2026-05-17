from _utils import ensure_results_dir, find_example_folder, find_example_image

import torch

import libllie as llie


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = find_example_image()
    input_dir = find_example_folder()
    output_dir = ensure_results_dir("deep")

    enhanced, single_path = llie.predict(
        "zerodce",
        image_path,
        output=output_dir / "zerodce_single.png",
        device=device,
    )

    print(f"Single deep-learning output: {single_path}")
    print(f"Single output size: {enhanced.size}")

    # batch_paths = llie.predict(
    #     "../checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    #     input_dir,
    #     output=output_dir / "zerodce_batch",
    #     device=device,
    #     progress_bar=True,
    # )
    #
    # print(f"Batch deep-learning output dir: {output_dir / 'zerodce_batch'}")
    # print(f"Batch image count: {len(batch_paths)}")


if __name__ == "__main__":
    main()
