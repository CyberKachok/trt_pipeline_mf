"""Runtime helpers for executing TensorRT engines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np

try:  # pragma: no cover - optional dependency in CI
    import pycuda.autoinit  # type: ignore  # noqa: F401  # ensure CUDA context
    import pycuda.driver as cuda
except Exception as exc:  # pragma: no cover
    raise ImportError("PyCUDA is required to run TensorRT engines") from exc

try:  # pragma: no cover
    import tensorrt as trt
except Exception as exc:  # pragma: no cover
    raise ImportError("TensorRT is required to run TensorRT engines") from exc

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BindingMemory:
    """Host and device storage for a single TensorRT binding."""

    name: str
    dtype: np.dtype
    mode: trt.TensorIOMode
    host: np.ndarray | None = None
    device: cuda.DeviceAllocation | None = None
    shape: tuple[int, ...] = ()

    def resize(self, new_shape: Iterable[int]) -> None:
        target_shape = tuple(int(dim) for dim in new_shape)
        volume = int(np.prod(target_shape))
        if self.host is None or self.host.size != volume:
            self.host = cuda.pagelocked_empty(volume, self.dtype)
            self.device = cuda.mem_alloc(self.host.nbytes)
        self.shape = target_shape

    def as_array(self) -> np.ndarray:
        assert self.host is not None
        return self.host.reshape(self.shape)


class TrtRunner:
    """Wraps TensorRT runtime execution with convenient numpy I/O."""

    def __init__(self, engine_path: str, profile_index: int = 0) -> None:
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as f:
            serialized = f.read()
        self._engine = self._runtime.deserialize_cuda_engine(serialized)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize engine {engine_path}")

        self._context = self._engine.create_execution_context()
        if profile_index:
            if hasattr(self._context, "set_optimization_profile"):
                self._context.set_optimization_profile(profile_index)
            elif hasattr(self._context, "set_optimization_profile_async"):
                stream = cuda.Stream()
                self._context.set_optimization_profile_async(profile_index, stream.handle)
                stream.synchronize()
            else:
                raise RuntimeError("TensorRT context does not expose profile selection APIs")

        self._stream = cuda.Stream()
        self._bindings: Dict[str, BindingMemory] = {}
        self._binding_order: list[str] = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            mode = self._engine.get_tensor_mode(name)
            self._bindings[name] = BindingMemory(name, dtype, mode)
            self._binding_order.append(name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def infer(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute the engine with ``inputs`` and return numpy outputs."""

        for name, array in inputs.items():
            if name not in self._bindings:
                raise KeyError(f"Unknown input tensor {name}")
            binding = self._bindings[name]
            if binding.mode != trt.TensorIOMode.INPUT:
                raise ValueError(f"Tensor {name} is not an input binding")
            arr = np.ascontiguousarray(array.astype(binding.dtype, copy=False))
            self._context.set_input_shape(name, arr.shape)
            binding.resize(arr.shape)
            np.copyto(binding.as_array(), arr)
            assert binding.device is not None
            cuda.memcpy_htod_async(binding.device, binding.host, self._stream)

        # Allocate output buffers based on the shapes reported by the context
        for name, binding in self._bindings.items():
            if binding.mode == trt.TensorIOMode.OUTPUT:
                shape = self._context.get_tensor_shape(name)
                binding.resize(shape)
                assert binding.device is not None

        # Set buffer addresses for execution
        for name in self._binding_order:
            binding = self._bindings[name]
            assert binding.device is not None
            self._context.set_tensor_address(name, int(binding.device))

        if not self._context.execute_async_v3(stream_handle=self._stream.handle):
            raise RuntimeError("TensorRT execution failed")

        outputs: Dict[str, np.ndarray] = {}
        for name, binding in self._bindings.items():
            if binding.mode == trt.TensorIOMode.OUTPUT:
                assert binding.device is not None and binding.host is not None
                cuda.memcpy_dtoh_async(binding.host, binding.device, self._stream)
        self._stream.synchronize()
        for name, binding in self._bindings.items():
            if binding.mode == trt.TensorIOMode.OUTPUT:
                outputs[name] = binding.as_array().copy()
        return outputs

    def get_input_names(self) -> Iterable[str]:
        return [name for name, binding in self._bindings.items() if binding.mode == trt.TensorIOMode.INPUT]

    def get_output_names(self) -> Iterable[str]:
        return [name for name, binding in self._bindings.items() if binding.mode == trt.TensorIOMode.OUTPUT]


__all__ = ["TrtRunner", "BindingMemory"]
