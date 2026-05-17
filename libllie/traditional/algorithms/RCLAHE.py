"""Recursive CLAHE enhancer."""

import numpy as np
import cv2

from .BaseEnhancer import LLIEnhancer


class RCLAHE(LLIEnhancer):
    """Recursive CLAHE enhancer.

    The method applies CLAHE multiple times to progressively enhance local
    contrast.
    """

    name = "rclahe"

    _COLOR_SPACE_ALIASES = {
        "rgb": "rgb",
        "bgr": "rgb",
        "hsv": "hsv",
        "hls": "hls",
        "yuv": "yuv",
        "ycbcr": "yuv",
        "lab": "lab",
    }

    def __init__(
        self,
        *,
        color_space: str = "yuv",
        clip_limit: float = 2.0,
        tile_grid_size=(8, 8),
        iterations: int = 3,
        **kwargs,
    ):
        """Initialize recursive CLAHE enhancer.

        Args:
            color_space: Color space where CLAHE is applied.
            clip_limit: CLAHE contrast clipping limit.
            tile_grid_size: CLAHE tile grid size.
            iterations: Number of recursive CLAHE applications.
            **kwargs: Base enhancer parameters.
        """
        super().__init__(**kwargs)

        self.color_space = self._normalize_color_space(color_space)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.iterations = iterations

        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    def _enhance(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply recursive CLAHE enhancement.

        Args:
            image: Input BGR or grayscale image array.
            **kwargs: Unused method-specific parameters.

        Returns:
            Enhanced image array.
        """
        img = self._ensure_uint8(image)

        for _ in range(self.iterations):
            img = self._apply_once(img)

        return img

    def _apply_once(self, img):
        """Apply one CLAHE iteration.

        Args:
            img: BGR or grayscale image array.

        Returns:
            Enhanced image array.

        Raises:
            ValueError: If ``color_space`` is unsupported.
        """
        if img.ndim == 2:
            return self._clahe.apply(img)

        if self.color_space == "rgb":
            return cv2.merge([self._clahe.apply(c) for c in cv2.split(img)])

        elif self.color_space == "hsv":
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = self._clahe.apply(v)
            return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

        elif self.color_space == "hls":
            hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            h, l, s = cv2.split(hls)
            l = self._clahe.apply(l)
            return cv2.cvtColor(cv2.merge([h, l, s]), cv2.COLOR_HLS2BGR)

        elif self.color_space == "yuv":
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            y = self._clahe.apply(y)
            return cv2.cvtColor(cv2.merge([y, u, v]), cv2.COLOR_YUV2BGR)

        elif self.color_space == "lab":
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        raise ValueError

    def _normalize_color_space(self, cs):
        """Normalize a color-space name.

        Args:
            cs: Color-space name or alias.

        Returns:
            Canonical color-space name.

        Raises:
            ValueError: If the color space is unsupported.
        """
        cs = cs.lower()
        if cs not in self._COLOR_SPACE_ALIASES:
            raise ValueError
        return self._COLOR_SPACE_ALIASES[cs]

    @staticmethod
    def _ensure_uint8(img):
        """Convert image to uint8.

        Args:
            img: Input image array.

        Returns:
            Uint8 image array.
        """
        if img.dtype == np.uint8:
            return img
        img = img.astype(np.float32)
        if img.max() <= 1:
            img *= 255
        return np.clip(img, 0, 255).astype(np.uint8)
