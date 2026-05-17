"""Logarithmic transform enhancer."""

from typing import Any, Dict

import numpy as np

from .BaseEnhancer import LLIEnhancer


class Log(LLIEnhancer):
    """
    Logarithmic transform enhancer.

    The transform is:
        output = log(1 + gain * input) / log(1 + gain)

    A larger gain brightens dark regions more aggressively while preserving the
    input dtype range.
    """

    name = "log"
    aliases = ["Log",]

    def __init__(
        self,
        gain: float = 10.0,
        **kwargs: Any,
    ) -> None:
        """Initialize logarithmic enhancer.

        Args:
            gain: Logarithmic gain. Larger values brighten dark regions more
                aggressively.
            **kwargs: Base enhancer parameters.

        Raises:
            TypeError: If ``gain`` is not numeric.
            ValueError: If ``gain`` is not positive.
        """
        super().__init__(**kwargs)
        self.gain = gain
        self._validate_gain(self.gain)

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply logarithmic enhancement.

        Args:
            image: Input image array.
            **kwargs: Optional ``gain`` override.

        Returns:
            Log-enhanced image array.
        """
        gain = kwargs.get("gain", self.gain)
        self._validate_gain(gain)

        original_dtype = image.dtype
        image_float = image.astype(np.float32)

        if np.issubdtype(original_dtype, np.integer):
            max_value = np.iinfo(original_dtype).max
            image_float = image_float / max_value
        else:
            image_float = np.clip(image_float, 0.0, 1.0)

        enhanced = np.log1p(gain * image_float) / np.log1p(gain)

        if np.issubdtype(original_dtype, np.integer):
            max_value = np.iinfo(original_dtype).max
            enhanced = np.rint(enhanced * max_value)

        return enhanced.astype(original_dtype, copy=False)

    def get_params(self) -> Dict[str, Any]:
        """Get enhancer parameters.

        Returns:
            Dictionary containing base parameters and ``gain``.
        """
        params = super().get_params()
        params.update({"gain": self.gain})
        return params

    def set_params(self, **params: Any) -> "Log":
        """Set enhancer parameters.

        Args:
            **params: Parameter names and values.

        Returns:
            The enhancer itself.
        """
        super().set_params(**params)

        if "gain" in params:
            self._validate_gain(self.gain)

        return self

    @staticmethod
    def _validate_gain(gain: float) -> None:
        """Validate gain value.

        Args:
            gain: Gain value.

        Raises:
            TypeError: If ``gain`` is not numeric.
            ValueError: If ``gain`` is not positive.
        """
        if not isinstance(gain, (int, float)):
            raise TypeError(f"gain must be int or float, got {type(gain)!r}.")

        if gain <= 0:
            raise ValueError(f"gain must be positive, got {gain!r}.")
