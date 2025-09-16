"""Benchmark TensorRT engines on the UAV123 dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

if __package__ in {None, ""}:  # pragma: no cover - script execution fallback
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from trt_pipeline.tracking_test_uav123V3 import run_tracking_evaluation
else:
    from ..tracking_test_uav123V3 import run_tracking_evaluation


def parse_variants(args: argparse.Namespace) -> List[Tuple[str, str]]:
    if args.variants:
        variants: List[Tuple[str, str]] = []
        for item in args.variants:
            if "=" not in item:
                raise ValueError(f"Variant '{item}' must be in name=path format")
            name, path = item.split("=", 1)
            variants.append((name, path))
        return variants
    if args.weights:
        return [(Path(args.weights).stem, args.weights)]
    raise ValueError("Provide either --weights or at least one --variant name=path pair")


def benchmark_variant(
    name: str,
    weight_path: str,
    args: argparse.Namespace,
) -> Dict[str, float]:
    metrics, _, _ = run_tracking_evaluation(
        cfg_path=args.tracker_config,
        ckpt_path=weight_path,
        data_root=args.data_root,
        anno_root=args.anno_root,
        output_dir=args.output_dir,
        matlab_config_path=args.matlab_config,
        annotation_txt_path=args.annotation_file or "",
        visualize=args.visualize,
        debug=args.debug,
    )
    if metrics is None:
        raise RuntimeError(f"Tracking evaluation failed for {name}")
    summary = {
        "variant": name,
        "Avg_IoU": metrics.get("Avg_IoU", 0.0),
        "Avg_FPS": metrics.get("Avg_FPS", 0.0),
        "AUC_IoU": metrics.get("AUC_IoU", 0.0),
        "AUC_Precision": metrics.get("AUC_Precision", 0.0),
        "Failure_Rate": metrics.get("Failure_Rate", 0.0),
    }
    return summary


def save_table(path: Path, rows: Iterable[Dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MixFormer TensorRT engines on UAV123")
    parser.add_argument("--tracker-config", required=True, help="Path to tracker YAML configuration")
    parser.add_argument("--weights", help="Single engine/checkpoint to evaluate")
    parser.add_argument(
        "--variants",
        nargs="+",
        help="Optional name=path pairs for multiple engines (e.g. fp16=engine_fp16.engine)",
    )
    parser.add_argument("--data-root", required=True, help="UAV123 image root directory")
    parser.add_argument("--anno-root", required=True, help="UAV123 annotations root directory")
    parser.add_argument("--matlab-config", required=True, help="Path to UAV123 MATLAB configuration file")
    parser.add_argument("--annotation-file", help="Optional TXT file with additional annotations")
    parser.add_argument("--output-dir", required=True, help="Where to store per-sequence results")
    parser.add_argument("--save-json", help="Optional JSON file for aggregate metrics")
    parser.add_argument("--save-csv", help="Optional CSV file for aggregate metrics")
    parser.add_argument("--visualize", action="store_true", help="Enable OpenCV visualisation window")
    parser.add_argument("--debug", action="store_true", help="Enable verbose tracker logging")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    variants = parse_variants(args)
    summaries: List[Dict[str, float]] = []
    for name, path in variants:
        summary = benchmark_variant(name, path, args)
        summaries.append(summary)
        print(f"{name}: IoU={summary['Avg_IoU']:.3f} FPS={summary['Avg_FPS']:.1f}")

    if args.save_json:
        Path(args.save_json).write_text(json.dumps(summaries, indent=2))
    if args.save_csv:
        save_table(Path(args.save_csv), summaries)


if __name__ == "__main__":
    main()
