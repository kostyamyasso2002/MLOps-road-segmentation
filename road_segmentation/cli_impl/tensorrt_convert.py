from pathlib import Path

import tensorrt as trt


def build_engine(onnx_path: Path, engine_path: Path, max_batch_size: int, use_fp16: bool):
    """
    Строит TensorRT-engine из ONNX-файла и сохраняет его на диск.

    Аргументы:
        onnx_path (str): путь к входному ONNX-файлу.
        engine_path (str): куда сохранить сериализованный TensorRT-движок (.trt).
        max_batch_size (int): максимальный размер батча для двигателя.
        use_fp16 (bool): если True, включить FP16-режим (при наличии поддержки).
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Проверяем существование ONNX-файла
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Настраиваем builder config
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    if use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("WARNING: Платформа не поддерживает FP16, соберём FP32-движок.")

    print(f"[TensorRT] Парсим ONNX-модель: {onnx_path}")
    with open(onnx_path, "rb") as model_file:
        model_data = model_file.read()
    if not parser.parse(model_data):
        print("ERROR: Не удалось распарсить ONNX-модель.")
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        raise RuntimeError("OnnxParser не смог распарсить модель.")

    # Строим движок
    print("[TensorRT] Строим движок...")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Не удалось построить TensorRT engine.")

    # Сериализуем и сохраняем движок
    print(f"[TensorRT] Сериализуем и сохраняем движок: {engine_path}")
    serialized_engine = engine.serialize()
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print("[TensorRT] Готово.")
