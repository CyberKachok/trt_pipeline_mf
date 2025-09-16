# MixFormer TensorRT pipeline

This repository contains a refactored end-to-end toolchain for exporting
MixFormerV2 checkpoints to ONNX, compiling TensorRT 10.13 engines and benchmarking
tracking quality on the UAV123 dataset.  The pipeline targets NVIDIA Jetson Orin
NX 16 GB devices (Ampere GPU + 2×DLA) but the Python CLI also runs on desktop
TensorRT installations.

## Project layout

```
trt_pipeline/
├── cli.py                     # unified CLI (export-onnx / build-trt / bench)
├── export/onnx_export.py      # Torch → ONNX helpers
├── trt/
│   ├── build.py               # high-level TensorRT builder
│   ├── calibration.py         # Entropy calibrator with mean/std preprocessing
│   └── runner.py              # execution wrapper around execute_async_v3
├── tools/bench_uav123.py      # dataset benchmark & report generator
├── configs/
│   ├── mixformer_v2_gpu.yaml  # GPU/FP16 preset (static shapes)
│   └── mixformer_v2_int8_dla.yaml  # INT8 + DLA0 preset with calibration
├── model_convert.py           # legacy wrappers (call into new modules)
├── trt_convert.py             # legacy wrappers (call into new modules)
└── tracking_test_uav123V3.py  # evaluation loop reused by the benchmark tool
```

## Requirements

* TensorRT 10.13.3 (tested with CUDA 12.6 / JetPack 6.1)
* PyCUDA (for calibration and runtime buffer management)
* PyTorch 1.13+ for ONNX export
* OpenCV (image loading & resizing during calibration)
* PyYAML (loading pipeline configuration files)

## 1. Export MixFormer to ONNX

The `export.onnx_export` module takes care of creating dummy inputs with
explicit batch dimensions, normalising tensors and validating the resulting
model.  Invoke it through the CLI:

```bash
python -m trt_pipeline.cli \
  --pipeline-config trt_pipeline/configs/mixformer_v2_gpu.yaml \
  export-onnx \
  --tracker-config weight-cfg/mixformer_v2.yaml \
  --checkpoint model/mixformer_v2.pth \
  --output build/mixformer_v2.onnx
```

Useful flags:

* `--opset {18|19|20|21}` – select the ONNX opset (default 18).
* `--dynamic-batch` / `--static-batch` – toggle dynamic batch dimensions.
* `--dynamic-spatial` – mark spatial axes as dynamic when required.

The pipeline configuration exposes input/output tensor names and default
shapes so the exported graph matches the expectations of TensorRT 10.x.

## 2. Build TensorRT engines

`trt.build.TrtBuilder` hides the verbose TensorRT 10.13 API.  It accepts the
same YAML configuration used during export, plus additional CLI flags:

```bash
# FP16 engine targeting the GPU
python -m trt_pipeline.cli \
  --pipeline-config trt_pipeline/configs/mixformer_v2_gpu.yaml \
  build-trt \
  --onnx build/mixformer_v2.onnx \
  --engine engines/mixformer_v2_fp16.engine \
  --fp16 --workspace-gb 4

# INT8 engine on DLA0 with GPU fallback
python -m trt_pipeline.cli \
  --pipeline-config trt_pipeline/configs/mixformer_v2_int8_dla.yaml \
  build-trt \
  --onnx build/mixformer_v2.onnx \
  --engine engines/mixformer_v2_int8_dla.engine \
  --int8 --use-dla --dla-core 0 --allow-gpu-fallback \
  --calib-dir data/calibration/uav123 \
  --calib-cache cache/mixformer_v2_int8_dla.cache
```

Key options:

* `--workspace-gb` – memory pool for tactic search (default 2 GiB).
* `--fp16` / `--no-fp16` – toggle FP16 kernels when available.
* `--int8` – enable INT8 with an entropy calibrator.
* `--use-dla` / `--gpu` – target one of the Orin NX DLA cores.
* `--allow-gpu-fallback` – automatically route unsupported ops to the GPU.
* `--strongly-typed` – create the network with strongly typed bindings.
* `--tactic-sources`, `--timing-cache`, `--profiling-verbosity` – fine tune
  TensorRT tactic selection and profiling output.

### INT8 calibration

`trt.calibration.ImageEntropyCalibrator` mirrors the runtime preprocessing:
images are converted to RGB, resized to each input tensor's shape, scaled to
[0, 1], then normalised with ImageNet mean and standard deviation.  The
calibrator accepts either raw frames (JPEG/PNG/BMP) or `.npy` tensors.  A cache
file is written alongside the engine so subsequent builds reuse the quantisation
table.

*Recommended dataset:* at least 1000 frames sampled from UAV123 (diverse scenes,
lighting and targets).  Use `prepare_uav123_calib.py` from the previous answer or
feed the benchmark dataset directly.

## 3. Benchmark on UAV123

The new benchmarking tool loads TensorRT engines via `trt.runner.TrtRunner` and
reuses the existing evaluation loop:

```bash
python -m trt_pipeline.tools.bench_uav123 \
  --tracker-config weight-cfg/mixformer_v2.yaml \
  --data-root /datasets/UAV123/data \
  --anno-root /datasets/UAV123/anno \
  --matlab-config weight-cfg/UAV123_config.txt \
  --output-dir results/uav123 \
  --variants fp16=engines/mixformer_v2_fp16.engine \
             int8_dla0=engines/mixformer_v2_int8_dla.engine \
  --save-json results/uav123/summary.json \
  --save-csv results/uav123/summary.csv
```

Each variant prints its average IoU and FPS while the JSON/CSV reports contain
additional metrics (AUC IoU, precision, failure rate).  The script can also be
used with PyTorch checkpoints for baseline comparison.

## Legacy scripts

`model_convert.py` and `trt_convert.py` now delegate to the refactored modules
so existing automation continues to work.  New users should prefer the unified
`cli.py` entry point which exposes the full TensorRT 10.13 feature set.

## Tips for Jetson Orin NX optimisation

* Keep input shapes static (112×112 template, 224×224 search) to avoid runtime
  shape switching overhead and to allow TensorRT to pre-compute tactics.
* Increase the workspace budget when possible (`--workspace-gb 4`) – the Orin
  NX 16 GB module has enough RAM to explore more convolution tactics.
* DLA works best with INT8 engines.  Always enable GPU fallback unless you are
  certain the network fits DLA's operator set.
* Monitor FPS improvements: FP16 typically doubles throughput versus FP32, and
  INT8 provides another ~1.3× speed-up (up to ~4× overall compared to FP32).

## Generating calibration frames

If you need to pre-compute `.npy` tensors for calibration, adapt the helper
below:

```bash
python prepare_uav123_calib.py \
  --uav123 /datasets/UAV123/data \
  --out data/calibration/uav123_np \
  -n 1500
```

The calibrator also accepts the raw JPEG/PNG frames directly, so this step is
optional.

## Troubleshooting

* Ensure TensorRT/PyCUDA/OpenCV versions match the JetPack environment.
* When using INT8, verify that the calibration dataset mirrors the production
  preprocessing (normalisation, colour order, etc.).
* For dynamic ONNX models, provide explicit optimisation profiles in the YAML
  config; otherwise the builder raises `no optimization profile defined`.
* Set `--profiling-verbosity detailed` to troubleshoot unsupported layers or
  tactic selection issues.
