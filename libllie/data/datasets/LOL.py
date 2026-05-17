"""LOLv1 dataset and dataloader helpers for_teach low-light enhancement."""

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

from torch.utils.data import DataLoader

from libllie.data.basetransform import predict_Trans
from libllie.data.datasets.BaseDataset import BaseDataset


__all__ = [
    "LOLv1Dataset",
    "create_lolv1_dataloaders",
]


class LOLv1Dataset(BaseDataset):
    """LOLv1 paired low-light enhancement dataset.

    The dataset resolver supports both the official LOLv1 layout and common
    local variants.

    Example:
        Official LOLv1 layout:

        root_dir/
            our485/
                low/
                high/
            eval15/
                low/
                high/

        Common project layout:

        root_dir/
            Train/
                Low/
                Normal/
            Test/
                Low/
                Normal/

    ``split="train"`` maps to ``our485`` first, then ``Train``.
    ``split="_test"`` maps to ``eval15`` first, then ``Test``.
    Explicit ``low_dir`` and ``high_dir`` override these conventions.
    """

    name = "LOLv1Dataset"
    aliases = ["LOLv1", "lol", "lolv1"]

    split_aliases = {
        "train": ("our485", "train", "Train"),
        "training": ("our485", "train", "Train"),
        "our485": ("our485",),
        "_test": ("eval15", "_test", "Test"),
        "eval": ("eval15", "_test", "Test"),
        "eval15": ("eval15",),
        "val": ("eval15", "val", "Val", "validation"),
        "valid": ("eval15", "val", "Val", "validation"),
        "validation": ("eval15", "val", "Val", "validation"),
    }

    low_dir_names = ("low", "Low", "LOW", "input", "Input")
    high_dir_names = ("high", "High", "HIGH", "normal", "Normal", "target", "Target")

    def _resolve_pair_dirs(
            self,
            low_dir: Optional[Path],
            high_dir: Optional[Path],
    ) -> Tuple[Path, Optional[Path]]:
        """Resolve LOLv1 low-light and normal-light image directories.

        Args:
            low_dir: Optional explicit low-light image directory. When provided,
                it is returned directly with ``high_dir``.
            high_dir: Optional explicit normal-light image directory.

        Returns:
            A tuple containing the resolved low-light directory and optional
            normal-light directory.
        """
        if low_dir is not None:
            return low_dir, high_dir

        split_dirs = self.split_aliases.get(self.split.lower(), (self.split,))
        candidates = []

        for split_dir in split_dirs:
            split_path = self.root_dir / split_dir
            for low_name in self.low_dir_names:
                for high_name in self.high_dir_names:
                    candidates.append((split_path / low_name, split_path / high_name))

        # Some local datasets are already organized as root/low and root/high.
        for low_name in self.low_dir_names:
            for high_name in self.high_dir_names:
                candidates.append((self.root_dir / low_name, self.root_dir / high_name))

        for candidate_low, candidate_high in candidates:
            if candidate_low.exists() and candidate_high.exists():
                return candidate_low, candidate_high

        # Return the most likely layout so BaseDataset raises a clear path error.
        first_split = split_dirs[0]
        return self.root_dir / first_split / "low", self.root_dir / first_split / "high"

    def get_stats(self) -> Dict[str, Union[str, int]]:
        """Get LOLv1 dataset statistics.

        Returns:
            Dictionary containing base dataset statistics and the split aliases
            used when resolving image directories.
        """
        stats = super().get_stats()
        stats["split_aliases"] = ", ".join(self.split_aliases.get(self.split.lower(), (self.split,)))
        return stats


def create_lolv1_dataloaders(
        root_dir: Union[str, Path],
        batch_size: int = 16,
        num_workers: int = 4,
        train_transform_low: Optional[Callable] = None,
        train_transform_high: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        common_transform: Optional[Callable] = None,
        return_filename: bool = True,
        **kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders for_teach LOLv1.

    Args:
        root_dir: LOLv1 root directory.
        batch_size: Batch size for_teach both dataloaders.
        num_workers: Number of dataloader workers.
        train_transform_low: Transform for_teach train low-light images.
        train_transform_high: Transform for_teach train normal-light images.
        test_transform: Transform applied to both _test low/high images.
        common_transform: Optional transform applied before separate transforms.
        return_filename: Whether dataset items include filenames.
        **kwargs: Extra keyword arguments passed to DataLoader.

    Returns:
        A tuple containing the train dataloader and test dataloader.
    """
    if test_transform is None:
        test_transform = predict_Trans

    train_dataset = LOLv1Dataset(
        root_dir=root_dir,
        split="train",
        transform_low=train_transform_low,
        transform_high=train_transform_high,
        common_transform=common_transform,
        return_filename=return_filename,
    )

    test_dataset = LOLv1Dataset(
        root_dir=root_dir,
        split="_test",
        transform_low=test_transform,
        transform_high=test_transform,
        common_transform=None,
        return_filename=return_filename,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )

    return train_loader, test_loader
