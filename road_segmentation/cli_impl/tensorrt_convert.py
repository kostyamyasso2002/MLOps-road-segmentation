import subprocess
import sys
from pathlib import Path

from road_segmentation.constants import constants


def tensorrt_convert(onnx_path: Path, trt_path: Path, trt_docker_version) -> None:
    real_onnx_path = onnx_path.resolve()
    real_trt_path = trt_path.resolve()

    if not real_onnx_path.is_file():
        print(f"Error: ONNX-file not found: {real_onnx_path}")
        sys.exit(1)

    trt_dir = real_trt_path.parent
    trt_dir.mkdir(parents=True, exist_ok=True)

    onnx_dir = real_onnx_path.parent
    onnx_fname = real_onnx_path.name
    trt_fname = real_trt_path.name

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{onnx_dir}:/workspace/input:ro",
        "-v",
        f"{trt_dir}:/workspace/output",
        f"nvcr.io/nvidia/tensorrt:{trt_docker_version}",
        "trtexec",
        f"--onnx=/workspace/input/{onnx_fname}",
        f"--saveEngine=/workspace/output/{trt_fname}",
        f"--minShapes=raw_image:{constants.TRT_MIN_SHAPE}",
        f"--optShapes=raw_image:{constants.TRT_OPT_SHAPE}",
        f"--maxShapes=raw_image:{constants.TRT_MAX_SHAPE}",
    ]

    print("Running Docker converting ONNX -> TRT:")
    print(" ".join(docker_cmd))

    try:
        subprocess.run(docker_cmd, check=True)
        print(f"\nTensorRT created: {real_trt_path}")
    except subprocess.CalledProcessError as e:
        print("\nError while executing trtexec inside Docker:", file=sys.stderr)
        sys.exit(e.returncode)
