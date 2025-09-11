#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple TensorRT inference speed benchmark for MixFormer engine (TrtTrackerWrapper).
Usage examples:
  python trt_bench.py --cfg path/to/config.yaml --engine MixFormer_ep0805_removed_mlp_IOUScore20.engine
  # specify iterations and warmup
  python trt_bench.py --cfg cfg.yaml --engine model.engine --iters 500 --warmup 50
  # end-to-end (preprocess + HtoD/DtoH + infer) timing
  python trt_bench.py --cfg cfg.yaml --engine model.engine --mode e2e --iters 200
  # choose device and disable template updates to get stable timings
  python trt_bench.py --cfg cfg.yaml --engine model.engine --device 0 --mode e2e --disable-updates

Notes:
- "engine" mode measures the pure engine execution only (after inputs are already on device).
- "e2e" mode measures end-to-end tracker.track(...) including preprocessing and HtoD/DtoH copies.
"""

import argparse
import time
import sys
import numpy as np
import cv2

# ---------- robust import for your wrapper ----------
try:
    # common layout: project_root/model/trt_tracker_wrapper.py
    from model.trt_tracker_wrapper import TrtTrackerWrapper
except Exception:
    # fallback if the file is next to this script
    from trt_tracker_wrapper import TrtTrackerWrapper

# pycuda init helpers
import pycuda.driver as cuda

def init_cuda(device_index: int = 0):
    cuda.init()
    dev = cuda.Device(device_index)
    pctx = dev.retain_primary_context()
    # Make the retained primary context current on this thread
    pctx.push()
    return dev, pctx

def release_cuda(pctx):
    try:
        # Pop current context and decrease the refcount on the primary context
        pctx.pop()
        pctx.detach()
    except Exception as e:
        print(f"[WARN] Releasing CUDA primary context raised: {e}", file=sys.stderr)

def make_test_frame(w=1280, h=720):
    """Create a synthetic image (solid color with a simple rectangle)."""
    img = np.full((h, w, 3), 64, dtype=np.uint8)
    cv2.rectangle(img, (w//3, h//3), (w//3 + 160, h//3 + 120), (200, 200, 200), -1)
    return img

def parse_bbox(s: str, W: int, H: int):
    """Parse 'x y w h' or use 'center:<w>x<h>' to auto-center."""
    try:
        if s.startswith("center:"):
            dims = s.split("center:")[1]
            tw, th = dims.lower().split("x")
            tw, th = int(tw), int(th)
            x = max(0, (W - tw) // 2)
            y = max(0, (H - th) // 2)
            return [x, y, tw, th]
        parts = [float(v) for v in s.strip().split()]
        assert len(parts) == 4
        return parts
    except Exception:
        raise argparse.ArgumentTypeError("bbox format must be 'x y w h' or 'center:<w>x<h>'")

def benchmark_engine(wrapper: TrtTrackerWrapper, warmup: int, iters: int):
    """Measure pure engine execution latency (context.execute_async_v3 + DtoH sync)."""
    # Warmup
    for _ in range(warmup):
        wrapper.infer()

    t0 = time.perf_counter()
    for _ in range(iters):
        wrapper.infer()
    cuda.Context.synchronize()  # just in case
    t1 = time.perf_counter()

    total = t1 - t0
    avg_ms = (total / iters) * 1e3
    fps = iters / total if total > 0 else float('inf')
    return avg_ms, fps

def benchmark_e2e(wrapper: TrtTrackerWrapper, frame, warmup: int, iters: int, disable_updates: bool):
    """Measure end-to-end track(...) including preprocessing and transfers."""
    if disable_updates:
        # Prevent template refreshes during the loop for stable timing
        # (Assumes these attributes exist on your TrackerWrapper)
        try:
            wrapper.update_interval = 10**9
            wrapper.max_pred_score = 1e9  # never be exceeded
        except Exception:
            pass

    # Warmup
    for i in range(warmup):
        wrapper.track(frame, i)

    t0 = time.perf_counter()
    last_state = None
    for i in range(iters):
        last_state, _, _ = wrapper.track(frame, i + warmup)
    cuda.Context.synchronize()
    t1 = time.perf_counter()

    total = t1 - t0
    avg_ms = (total / iters) * 1e3
    fps = iters / total if total > 0 else float('inf')
    return avg_ms, fps, last_state

def main():
    ap = argparse.ArgumentParser(description="MixFormer TRT inference speed benchmark")
    ap.add_argument("--cfg", required=True, type=str, help="Path to MixFormer YAML config")
    ap.add_argument("--engine", required=True, type=str, help="Path to TensorRT engine (.engine)")
    ap.add_argument("--image", type=str, default="", help="Optional path to an image for initialization")
    ap.add_argument("--bbox", type=str, default="center:160x120",
                    help="Init bbox 'x y w h' or 'center:<w>x<h>' (default: center:160x120)")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index")
    ap.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    ap.add_argument("--iters", type=int, default=3000, help="Measured iterations")
    ap.add_argument("--mode", choices=["engine", "e2e"], default="engine",
                    help="engine: measure engine only; e2e: end-to-end track()")
    ap.add_argument("--disable-updates", action="store_true",
                    help="For e2e mode: disable template updates for stable timing")
    args = ap.parse_args()

    # Init CUDA primary context (so TrtTrackerWrapper can attach to it)
    dev, pctx = init_cuda(args.device)
    print(f"[INFO] CUDA device {args.device}: {dev.name()}")

    try:
        # Prepare frame
        if args.image:
            frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[WARN] Failed to read image: {args.image}. Falling back to synthetic frame.")
                frame = make_test_frame()
        else:
            frame = make_test_frame()
        H, W = frame.shape[:2]
        bbox = parse_bbox(args.bbox, W, H)

        # Build tracker wrapper and initialize (this uploads templates to device)
        wrapper = TrtTrackerWrapper(args.cfg, args.engine)
        wrapper.initialize(frame, bbox)

        # Do one track() to populate search input (engine input #2) before engine-only timing
        state0, score0, fps0 = wrapper.track(frame, 1)

        if args.mode == "engine":
            avg_ms, fps = benchmark_engine(wrapper, args.warmup, args.iters)
            print(f"[RESULT][ENGINE] avg latency: {avg_ms:.3f} ms | FPS: {fps:.2f}")
        else:
            avg_ms, fps, last_state = benchmark_e2e(wrapper, frame, args.warmup, args.iters, args.disable_updates)
            print(f"[RESULT][E2E]    avg latency: {avg_ms:.3f} ms | FPS: {fps:.2f}")
            try:
                x, y, w, h = last_state
                print(f"[INFO] last predicted bbox: ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
            except Exception:
                pass

    finally:
        # Release CUDA primary context
        release_cuda(pctx)

if __name__ == "__main__":
    main()
