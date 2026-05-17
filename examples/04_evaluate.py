from _utils import ensure_results_dir, find_example_folder

import libllie as llie


def main():
    input_dir = find_example_folder()
    output_dir = ensure_results_dir("evaluation")
    enhanced_dir = output_dir / "enhanced_by_he"
    save_path = output_dir / "eval_psnr_ssim_mse.json"

    llie.predict(
        "he",
        input_dir,
        output=enhanced_dir,
    )

    results = llie.evaluate(
        en_img_dir=enhanced_dir,
        ref_img_dir=input_dir,
        metrics=["PSNR", "SSIM", "MSE", "NIQE"],
        save_path=save_path,
        batch_size=1,
        num_workers=0,
        device="cpu",
    )

    print(type(results))

if __name__ == "__main__":
    main()
