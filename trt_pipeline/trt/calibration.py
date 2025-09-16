"""INT8 calibration helpers for TensorRT 10.13.

This module houses a reusable entropy calibrator that operates on image
folders or NumPy tensors.  The implementation mirrors the preprocessing
pipeline used during runtime inference: RGB conversion, resize,
normalisation by ImageNet mean/std and CHW ordering.  TensorRT's Python
bindings require the calibrator to return device pointers in the same
order as requested by ``get_batch``; the implementation below stores
per-input metadata so multi-input networks (e.g. MixFormer template and
search branches) are handled transparently.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency in CI containers
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover - optional dependency in CI containers
    import pycuda.driver as cuda
    import pycuda.autoinit  # type: ignore  # noqa: F401  # ensure CUDA context
except Exception:  # pragma: no cover
    cuda = None

try:  # pragma: no cover
    import tensorrt as trt
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "TensorRT is required to use the calibration helpers"
    ) from exc

LOGGER = logging.getLogger(__name__)

ImageList = Sequence[str]
InputShape = Tuple[int, int, int]
InputMetadata = Mapping[str, InputShape]


@dataclass(slots=True)
class CalibrationSettings:
    """Runtime parameters for :class:`ImageEntropyCalibrator`."""

    data_dir: str
    cache_file: str
    batch_size: int = 1
    max_samples: Optional[int] = None
    mean: Sequence[float] = (0.485, 0.456, 0.406)
    std: Sequence[float] = (0.229, 0.224, 0.225)
    allowed_extensions: Sequence[str] = ("*.npy", "*.jpg", "*.jpeg", "*.png", "*.bmp")


class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """Entropy calibrator that mirrors the production preprocessing pipeline."""

    def __init__(
        self,
        settings: CalibrationSettings,
        input_metadata: InputMetadata,
    ) -> None:
        trt.IInt8EntropyCalibrator2.__init__(self)
        if cuda is None:
            raise RuntimeError("pycuda is required for INT8 calibration")

        self._settings = settings
        self._mean = np.asarray(settings.mean, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.asarray(settings.std, dtype=np.float32).reshape(3, 1, 1)
        self._input_shapes: Dict[str, InputShape] = {
            name: tuple(shape) for name, shape in input_metadata.items()
        }
        self._device_memory: Dict[str, cuda.DeviceAllocation] = {}
        self._files = self._gather_files(settings)
        self._next_index = 0

        bytes_per_sample = np.dtype(np.float32).itemsize
        for name, shape in self._input_shapes.items():
            volume = int(np.prod((settings.batch_size,) + shape))
            allocation = cuda.mem_alloc(volume * bytes_per_sample)
            self._device_memory[name] = allocation
            LOGGER.debug(
                "Allocated %.1f KiB for %s",
                volume * bytes_per_sample / 1024,
                name,
            )

    # ------------------------------------------------------------------
    # Required TensorRT calibrator methods
    # ------------------------------------------------------------------
    def get_batch_size(self) -> int:  # noqa: D401 - interface defined by TensorRT
        return self._settings.batch_size

    def get_batch(self, names: Sequence[str]) -> List[int] | None:  # noqa: D401
        if self._next_index >= len(self._files):
            return None

        path = self._files[self._next_index]
        self._next_index += 1

        device_ptrs: List[int] = []
        for name in names:
            if name not in self._input_shapes:
                raise KeyError(f"Input {name} not present in metadata")
            shape = self._input_shapes[name]
            device_mem = self._device_memory[name]
            host_batch = self._prepare_batch(path, shape)
            cuda.memcpy_htod(device_mem, host_batch)
            device_ptrs.append(int(device_mem))
        return device_ptrs

    def read_calibration_cache(self) -> bytes | None:  # pragma: no cover - file IO
        cache = Path(self._settings.cache_file)
        if cache.exists():
            LOGGER.info("Loading INT8 calibration cache from %s", cache)
            return cache.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:  # pragma: no cover - file IO
        cache_path = Path(self._settings.cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(cache)
        LOGGER.info("Wrote INT8 calibration cache to %s", cache_path)

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _prepare_batch(self, path: str, shape: InputShape) -> np.ndarray:
        """Load, resize and normalise data for a single calibration sample."""

        arr = self._load_sample(path, shape)
        arr = (arr - self._mean) / self._std
        arr = np.ascontiguousarray(arr.reshape((self._settings.batch_size,) + arr.shape))
        return arr

    def _load_sample(self, path: str, shape: InputShape) -> np.ndarray:
        if path.endswith(".npy"):
            array = np.load(path).astype(np.float32)
            if array.ndim == 4 and array.shape[0] == 1:
                array = array[0]
            if array.ndim != 3:
                raise RuntimeError(f"Unexpected array shape {array.shape} in {path}")
            if array.shape[0] == shape[0]:
                chw = array
            elif array.shape[-1] == shape[0]:
                chw = np.transpose(array, (2, 0, 1))
            else:
                raise RuntimeError(f"Could not infer channel dimension for {path}")
            chw = self._resize_if_needed(chw, shape)
            if chw.max() > 1.0:
                chw = chw / 255.0
            return chw

        if cv2 is None:
            raise RuntimeError("OpenCV is required to read calibration images")
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to load image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (shape[2], shape[1]), interpolation=cv2.INTER_LINEAR)
        chw = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))
        return chw

    def _resize_if_needed(self, array: np.ndarray, shape: InputShape) -> np.ndarray:
        if array.shape[1:] == shape[1:]:
            return array
        if cv2 is None:
            raise RuntimeError("OpenCV required to resize calibration samples")
        hwc = np.transpose(array, (1, 2, 0))
        resized = cv2.resize(hwc, (shape[2], shape[1]), interpolation=cv2.INTER_LINEAR)
        return np.transpose(resized, (2, 0, 1))

    def _gather_files(self, settings: CalibrationSettings) -> ImageList:
        files: List[str] = []
        for pattern in settings.allowed_extensions:
            files.extend(
                glob(os.path.join(settings.data_dir, "**", pattern), recursive=True)
            )
        files = sorted(files)
        if settings.max_samples is not None:
            files = files[: settings.max_samples]
        if not files:
            raise RuntimeError(f"No calibration data found under {settings.data_dir}")
        LOGGER.info("Using %d calibration samples", len(files))
        return files


__all__ = [
    "CalibrationSettings",
    "ImageEntropyCalibrator",
]
