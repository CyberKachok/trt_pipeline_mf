import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Инициализация TensorRT билдера и парсера ONNX
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network,
                                                                                              TRT_LOGGER) as parser:
    # Загрузка ONNX-модели и парсинг в TensorRT
    with open("/home/ilya/PycharmProjects/Trackers/MixformerBench/weight-cfg/mixformer2_base.onnx", "rb") as model_file:
        parser.parse(model_file.read())

    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    builder_config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
    builder_config.set_flag(trt.BuilderFlag.FP16)

    serialized_network = builder.build_serialized_network(network, builder_config)
    with open("mixformer2_base.engine", "wb") as f:
        f.write(serialized_network)