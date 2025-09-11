# trt_pipeline_mf

This repository provides tools for converting MixFormer checkpoints into
TensorRT engines and running tracking benchmarks.  The conversion pipeline now
supports **FP32**, **FP16** and **INT8** precision modes.

## INT8 calibration

When building an INT8 engine, `trt_convert.py` uses a custom
`UAV123Calibrator` that reads preprocessed UAV123 frames.  Calibration results
are cached next to the produced engine (e.g. `model_int8.engine.calib`) so that
re-building an INT8 engine reuses the calibration table without re-running the
dataset.

Example:

```bash
python trt_convert.py \
  --onnx model.onnx \
  --engine model_int8.engine \
  --int8 --calib_dir /path/to/UAV123
```

The top-level `trt_pipeline.py` script will automatically search FP32, FP16 and
INT8 engines (using `--calib_data` if provided) and report the best performing
variant.
