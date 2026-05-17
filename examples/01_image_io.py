from _utils import ensure_results_dir, find_example_image

import libllie as llie


def main():
    image_path = find_example_image()
    output_dir = ensure_results_dir("io")

    pil_image = llie.imread(image_path, output_format="pil")
    numpy_image = llie.imread(image_path, output_format="numpy")
    image_bytes = llie.imread(image_path, output_format="bytes")
    image_base64 = llie.imread(image_path, output_format="base64")

    copied_path = llie.imwrite(pil_image, output=output_dir)
    png_path = llie.imwrite(pil_image, output=output_dir / "converted_from_io.png")

    print(f"Input image: {image_path}")
    print(f"PIL size: {pil_image.size}")
    print(f"NumPy shape: {numpy_image.shape}")
    print(f"Bytes length: {len(image_bytes)}")
    print(f"Base64 length: {len(image_base64)}")
    print(f"Copied image: {copied_path}")
    print(f"Converted image: {png_path}")


if __name__ == "__main__":
    main()
