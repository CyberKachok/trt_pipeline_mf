import argparse
import tensorrt as trt


def build_engine(onnx_path: str, engine_path: str, fp16: bool = False, workspace: int = 1 << 30):
    """Build a TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model.
        engine_path: Where to write serialized engine.
        fp16: Enable FP16 precision.
        workspace: Workspace size in bytes.
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
    ap.add_argument('--workspace', type=int, default=1 << 30, help='Workspace size in bytes')
    args = ap.parse_args()

    build_engine(args.onnx, args.engine, fp16=args.fp16, workspace=args.workspace)


if __name__ == '__main__':
    main()
