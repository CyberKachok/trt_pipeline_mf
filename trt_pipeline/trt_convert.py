import argparse
import os
from glob import glob

import numpy as np
import tensorrt as trt

try:
    import pycuda.autoinit  # noqa: F401  # ensures CUDA context
    import pycuda.driver as cuda
except Exception:  # pragma: no cover - runtime environment may not provide CUDA
    cuda = None
    pycuda_autoinit = None

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV may be optional for unit tests
    cv2 = None


class UAV123Calibrator(trt.IInt8EntropyCalibrator2):
    """Entropy calibrator using preprocessed UAV123 frames.

    The calibrator searches a directory for image/``.npy`` files, loads
    them into GPU memory one-by-one and feeds them to TensorRT during INT8
    engine building.  A calibration cache is written so subsequent builds can
    reuse the generated table without re-running the dataset.
    """

    def __init__(self, data_dir: str, input_shape, cache_file: str):
        super().__init__()
        self.data_dir = data_dir
        self.cache_file = cache_file
        self.input_shape = tuple(input_shape)
        # gather files from UAV123; supports npy or common image formats
        patterns = ["*.npy", "*.jpg", "*.png", "*.jpeg", "*.bmp"]
        files = []
        for p in patterns:
            files.extend(glob(os.path.join(self.data_dir, "**", p), recursive=True))
        if not files:
            raise RuntimeError(f"No calibration files found in {data_dir}")
        self.files = sorted(files)
        self.index = 0
        self.batch_size = 1
        # Allocate one batch worth of device memory
        vol = int(np.prod((self.batch_size,) + self.input_shape))
        if cuda is None:
            raise RuntimeError("pycuda not available")
        self.device_input = cuda.mem_alloc(vol * np.dtype(np.float32).itemsize)

    # ----- TensorRT calibrator interface -----
    def get_batch_size(self):
        return self.batch_size

    def preprocess(self, path):
        if path.endswith(".npy"):
            arr = np.load(path).astype(np.float32)
        else:
            if cv2 is None:
                raise RuntimeError("OpenCV required to read calibration images")
            img = cv2.imread(path)
            if img is None:
                raise RuntimeError(f"Failed to read image {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))
            arr = img.astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))  # CHW
        return arr

    def get_batch(self, names):  # noqa: D401 - interface requirement
        if self.index >= len(self.files):
            return None
        data = self.preprocess(self.files[self.index])
        self.index += 1
        data = data.reshape((self.batch_size,) + self.input_shape)
        cuda.memcpy_htod(self.device_input, data)
        return [int(self.device_input)]

    def read_calibration_cache(self):  # pragma: no cover - file IO
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):  # pragma: no cover - file IO
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = False,
    int8: bool = False,
    calib_dir: str | None = None,
    workspace: int = 1 << 30,
    calib_cache: str | None = None,
):
    """Build a TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model.
        engine_path: Where to write serialized engine.
        fp16: Enable FP16 precision.
        int8: Enable INT8 precision. ``calib_dir`` must be provided.
        calib_dir: Directory containing calibration frames from UAV123.
        workspace: Workspace size in bytes.
        calib_cache: Optional path for the calibration cache.  Defaults to
            ``engine_path`` with ``.calib`` extension.
    Returns:
        Path to the serialized engine.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(logger) as builder, builder.create_network(flag) as network, \
            trt.OnnxParser(network, logger) as parser:
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX model")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        calibrator = None
        if int8:
            if calib_dir is None:
                raise ValueError("calib_dir must be provided when int8=True")
            config.set_flag(trt.BuilderFlag.INT8)
            input_shape = tuple(network.get_input(0).shape)[1:]
            cache = calib_cache or engine_path + ".calib"
            calibrator = UAV123Calibrator(calib_dir, input_shape, cache)
            config.int8_calibrator = calibrator

        serialized_engine = builder.build_serialized_network(network, config)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
    print(f"[INFO] TensorRT engine saved to {engine_path}")
    return engine_path


def main():
    ap = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine")
    ap.add_argument('--onnx', required=True, help='Path to ONNX model')
    ap.add_argument('--engine', required=True, help='Output path for TensorRT engine')
    ap.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    ap.add_argument('--int8', action='store_true', help='Enable INT8 precision with calibration')
    ap.add_argument('--calib_dir', help='Directory of calibration data (UAV123 images)')
    ap.add_argument('--calib_cache', help='Path to save/load INT8 calibration cache')
    ap.add_argument('--workspace', type=int, default=1 << 30, help='Workspace size in bytes')
    args = ap.parse_args()

    build_engine(
        args.onnx,
        args.engine,
        fp16=args.fp16,
        int8=args.int8,
        calib_dir=args.calib_dir,
        workspace=args.workspace,
        calib_cache=args.calib_cache,
    )


if __name__ == '__main__':
    main()
