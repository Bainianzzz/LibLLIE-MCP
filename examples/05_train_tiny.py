from _utils import create_tiny_common_dataset, ensure_results_dir

import libllie as llie


def main():
    dataset_root = create_tiny_common_dataset(ensure_results_dir("tiny_dataset"))
    output_dir = ensure_results_dir("ckp", "ZeroDCE_tiny")

    result = llie.train(
        model="ZeroDCE",
        model_params={"input_channels": 3},
        dataset="CommonDataset",
        root_dir=str(dataset_root),
        train_split="train",
        val_split="val",
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        loss="zerodce",
        optimizer="adam",
        lr=1e-4,
        epochs=3,
        device="cuda",
        output_dir=str(output_dir),
        log_every=1,
        save_every=1,
        validate_every=1,
    )

    print(f"type of result: {type(result)}")
    print(f"keys of result: {result.keys()}")
    print(f"History: {result['history']}")


if __name__ == "__main__":
    main()
