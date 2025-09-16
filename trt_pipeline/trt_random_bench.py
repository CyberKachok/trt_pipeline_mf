#!/usr/bin/env python3
"""Convert arbitrary ONNX models to TensorRT and benchmark inference speed.

The script builds TensorRT engines for the requested precision modes (FP32,
FP16, INT8) and measures execution latency using random input tensors.  INT8
calibration uses random data batches, so the produced engine is only suitable
for speed testing, not for accuracy-sensitive applications.
"""

import argparse
import os
import time
import numpy as np
import tensorrt as trt

try:
    import pycuda.autoinit  # noqa: F401 ensures CUDA context
    import pycuda.driver as cuda
except Exception:  # pragma: no cover - environment may lack CUDA
    cuda = None
    pycuda_autoinit = None


class RandomCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator that feeds random data."""

    def __init__(self, input_shapes, cache_file, batches: int = 20):
        trt.IInt8EntropyCalibrator2.__init__(self)
        if cuda is None:
            raise RuntimeError("pycuda not available")
        self.input_shapes = [tuple(s) for s in input_shapes]
        self.cache_file = cache_file
        self.batch_size = 1
        self.batches = batches
        self.index = 0
        self.device_inputs = []
        for shape in self.input_shapes:
            vol = int(np.prod((self.batch_size,) + shape))
            self.device_inputs.append(cuda.mem_alloc(vol * 4))

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):  # noqa: D401
        if self.index >= self.batches:
            return None
        ptrs = []
        for i, _ in enumerate(names):
            shape = (self.batch_size,) + self.input_shapes[i]
            data = np.random.random_sample(shape).astype(np.float32)
            cuda.memcpy_htod(self.device_inputs[i], data)
            ptrs.append(int(self.device_inputs[i]))
        self.index += 1
        return ptrs

    def read_calibration_cache(self):  # pragma: no cover - file IO
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):  # pragma: no cover - file IO
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def parse_shape(shape_str: str):
    return tuple(int(v) for v in shape_str.lower().split("x"))


def build_engine(onnx_path: str, engine_path: str, mode: str, shape, workspace: int, batches: int):
    logger = trt.Logger(trt.Logger.WARNING)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(logger) as builder, builder.create_network(flag) as network, \
            trt.OnnxParser(network, logger) as parser:
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)

        input_tensor = network.get_input(0)
        if any(dim <= 0 for dim in input_tensor.shape):
            profile = builder.create_optimization_profile()
            profile.set_shape(input_tensor.name, shape, shape, shape)
            config.add_optimization_profile(profile)
        else:
            assert tuple(input_tensor.shape) == shape, "Input shape mismatch"

        calibrator = None
        if mode == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        if mode == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            input_shapes = [shape[1:]]
            cache = engine_path + ".calib"
            calibrator = RandomCalibrator(input_shapes, cache, batches)
            config.int8_calibrator = calibrator
        if mode == "fp16_int8":
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.INT8)
            input_shapes = [shape[1:]]
            cache = engine_path + ".calib"
            calibrator = RandomCalibrator(input_shapes, cache, batches)
            config.int8_calibrator = calibrator

        serialized = builder.build_serialized_network(network, config)
        with open(engine_path, "wb") as f:
            f.write(serialized)
    return engine_path


def benchmark_engine(engine_path: str, shape, warmup: int, iters: int):
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    context.set_binding_shape(0, shape)

    # allocate buffers for all bindings
    bindings = []
    host_inputs = []
    for i in range(engine.num_bindings):
        dtype = trt.nptype(engine.get_binding_dtype(i))
        vol = trt.volume(context.get_binding_shape(i))
        host_mem = cuda.pagelocked_empty(vol, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(i):
            host_inputs.append(host_mem)
            input_device = device_mem
    for host_mem in host_inputs:
        host_mem[:] = np.random.random_sample(host_mem.size).astype(host_mem.dtype)
    for host_mem, device_mem in zip(host_inputs, [input_device]):
        cuda.memcpy_htod(device_mem, host_mem)

    stream = cuda.Stream()
    for _ in range(warmup):
        context.execute_async_v2(bindings, stream.handle, None)
    stream.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        context.execute_async_v2(bindings, stream.handle, None)
    stream.synchronize()
    t1 = time.perf_counter()

    total = t1 - t0
    avg_ms = (total / iters) * 1e3
    fps = iters / total if total > 0 else float("inf")
    return avg_ms, fps


def main():
    ap = argparse.ArgumentParser(description="ONNX to TensorRT speed benchmark with random data")
    ap.add_argument("--onnx", required=True, help="Path to ONNX model")
    ap.add_argument("--shape", required=True, help="Input shape NCHW, e.g., 1x3x224x224")
    ap.add_argument("--modes", nargs="+", default=["fp32", "fp16"],
                    help="Precision modes: fp32 fp16 int8 fp16_int8")
    ap.add_argument("--outdir", default=".", help="Directory to store engines")
    ap.add_argument("--workspace", type=int, default=1 << 31, help="Workspace size in bytes")
    ap.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    ap.add_argument("--iters", type=int, default=300, help="Benchmark iterations")
    ap.add_argument("--batches", type=int, default=20, help="Random calibration batches for INT8")
    args = ap.parse_args()

    shape = parse_shape(args.shape)
    basename = os.path.splitext(os.path.basename(args.onnx))[0]
    os.makedirs(args.outdir, exist_ok=True)

    for mode in args.modes:
        engine_path = os.path.join(args.outdir, f"{basename}_{mode}.engine")
        build_engine(args.onnx, engine_path, mode, shape, args.workspace, args.batches)
        avg_ms, fps = benchmark_engine(engine_path, shape, args.warmup, args.iters)
        print(f"[{mode.upper()}] {avg_ms:.3f} ms | {fps:.2f} FPS")


if __name__ == "__main__":
    main()
