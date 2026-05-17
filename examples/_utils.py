from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RESULTS_DIR = ROOT / "_dataset_tmp"
INPUTS_DIR = ROOT / "Inputs"


def ensure_results_dir(*parts: str) -> Path:
    path = RESULTS_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resize_save_image(input_path, output_path, scale):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(input_path)

    new_size = (
        int(img.width * scale),
        int(img.height * scale)
    )

    resized_img = img.resize(new_size)
    resized_img.save(output_path)


def find_example_image() -> Path:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if INPUTS_DIR.exists():
        for path in sorted(INPUTS_DIR.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                return path

    fallback_dir = ensure_results_dir("assets")
    fallback_path = fallback_dir / "example_input.png"
    if not fallback_path.exists():
        image = np.zeros((96, 128, 3), dtype=np.uint8)
        image[..., 0] = 30
        image[..., 1] = np.linspace(20, 120, image.shape[1], dtype=np.uint8)
        image[..., 2] = np.linspace(40, 180, image.shape[0], dtype=np.uint8)[:, None]
        Image.fromarray(image, mode="RGB").save(fallback_path)
    return fallback_path


def find_example_folder() -> Path:
    if INPUTS_DIR.exists() and any(INPUTS_DIR.iterdir()):
        return INPUTS_DIR

    image_path = find_example_image()
    return image_path.parent


def create_tiny_common_dataset(root: Path, image_size=(32, 32), count: int = 10) -> Path:
    for split in ("train", "val"):
        low_dir = root / split / "low"
        normal_dir = root / split / "normal"
        low_dir.mkdir(parents=True, exist_ok=True)
        normal_dir.mkdir(parents=True, exist_ok=True)

        for index in range(count):
            low_value = 20 + index * 10
            high_value = 100 + index * 10
            Image.new("RGB", image_size, (low_value, low_value, low_value)).save(
                low_dir / f"{index:03d}.png"
            )
            Image.new("RGB", image_size, (high_value, high_value, high_value)).save(
                normal_dir / f"{index:03d}.png"
            )

    return root
