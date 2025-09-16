import argparse
import os
from pathlib import Path

from model_convert import convert_to_onnx
from trt_convert import build_engine
from tracking_test_uav123V3 import run_tracking_evaluation


def main():
    parser = argparse.ArgumentParser(description="Full conversion and evaluation pipeline")
    parser.add_argument("--config", required=True, help="Path to tracker config file")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--ckpt", help="Path to PyTorch checkpoint (.pth)")
    input_group.add_argument("--onnx", help="Path to an existing ONNX model")
    parser.add_argument("--data_root", required=True, help="Root directory of UAV123 dataset images")
    parser.add_argument("--anno_root", required=True, help="Root directory of UAV123 annotations")
    parser.add_argument("--output_dir", required=True, help="Directory to store evaluation results")
    parser.add_argument("--matlab_config", required=True, help="Path to UAV123 MATLAB config file")
    parser.add_argument("--workspace", type=int, default=1 << 30, help="TensorRT workspace size in bytes")
    parser.add_argument("--calib_data", default=None, help="Directory of UAV123 frames for INT8 calibration")
    args = parser.parse_args()

    if args.onnx:
        onnx_path = Path(args.onnx)
        if not onnx_path.is_file():
            raise FileNotFoundError(f"Provided ONNX model not found: {onnx_path}")
    else:
        onnx_path = convert_to_onnx(args.config, args.ckpt)

    model_id_path = args.ckpt if args.ckpt else str(onnx_path)

    builder_variants = [
        {'name': 'fp32', 'fp16': False},
        {'name': 'fp16', 'fp16': True},
        {'name': 'int8', 'fp16': False, 'int8': True, 'calib_dir': args.calib_data or args.data_root},
    ]

    results = []
    for variant in builder_variants:
        engine_path = os.path.splitext(model_id_path)[0] + f"_{variant['name']}.engine"
        try:
            build_engine(
                str(onnx_path),
                engine_path,
                fp16=variant.get('fp16', False),
                int8=variant.get('int8', False),
                calib_dir=variant.get('calib_dir'),
                workspace=args.workspace,
            )
        except Exception as e:
            print(f"[ERROR] Failed to build engine for {variant['name']}: {e}")
            continue

        metrics, _, _ = run_tracking_evaluation(
            args.config,
            engine_path,
            args.data_root,
            args.anno_root,
            args.output_dir,
            args.matlab_config,
            "",
            False,
            False,
        )
        score = metrics.get('Avg_IoU', 0) * metrics.get('Avg_FPS', 0)
        results.append({'name': variant['name'], 'engine': engine_path, 'metrics': metrics, 'score': score})
        print(f"[RESULT] {variant['name']}: IoU={metrics.get('Avg_IoU', 0):.3f} FPS={metrics.get('Avg_FPS', 0):.2f}")

    if results:
        best = max(results, key=lambda x: x['score'])
        print('[BEST] configuration:', best['name'])
        print('[BEST] engine:', best['engine'])
        print('[BEST] metrics:', best['metrics'])

        cfg_name = os.path.splitext(os.path.basename(args.config))[0]

        model_tag = (
            os.path.splitext(os.path.basename(args.ckpt))[0]
            if args.ckpt
            else onnx_path.stem
        )
        report_path = os.path.join(args.output_dir, f"{cfg_name}_{model_tag}_trt_report.txt")

        with open(report_path, 'w') as f:
            f.write("variant,Avg_IoU,Avg_FPS\n")
            for r in results:
                f.write(
                    f"{r['name']},{r['metrics'].get('Avg_IoU', 0):.4f},{r['metrics'].get('Avg_FPS', 0):.2f}\n"
                )
        print(f"[INFO] Report saved to {report_path}")
    else:
        print('No successful evaluations')


if __name__ == '__main__':
    main()
