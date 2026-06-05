#!/usr/bin/env python3
"""
基于 OpenVINO 后端的 RapidOCR —— ONNX Runtime 的直接替代方案。
RapidOCR with OpenVINO backend — drop-in replacement for ONNX Runtime.

本模块用 OpenVINO 推理替换 RapidOCR 的 OrtInferSession，
实现正确的预处理/后处理并借助 OpenVINO 加速。
This replaces RapidOCR's OrtInferSession with OpenVINO inference,
giving correct preprocessing/postprocessing with OpenVINO acceleration.

用法 / Usage:
    python rapidocr_openvino.py <image_path> [--device CPU|GPU] [--bench] [--runs N]
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import openvino as ov


class OpenVINOInferSession:
    """使用 OpenVINO 的 RapidOCR OrtInferSession 直接替代方案。
    Drop-in replacement for RapidOCR's OrtInferSession using OpenVINO."""

    def __init__(self, config: dict, device: str = "CPU", ov_config: Optional[dict] = None):
        self.device = device
        self.ov_config = ov_config or {}

        model_path = config.get("model_path", "")
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # 如需则将 ONNX 即时转换为 OpenVINO IR
        # Convert ONNX to OpenVINO IR on the fly if needed
        onnx_path = Path(model_path)
        if onnx_path.suffix == ".onnx":
            # 检查是否已存在 IR 版本
            # Check if IR version exists
            xml_path = onnx_path.with_suffix(".xml")
            if not xml_path.exists():
                print(f"  Converting {onnx_path.name} to OpenVINO IR...")
                core = ov.Core()
                model = core.read_model(str(onnx_path))
                ov.save_model(model, str(xml_path))
                print(f"  Saved to {xml_path}")
            model_path = str(xml_path)

        core = ov.Core()
        self.compiled_model = core.compile_model(str(model_path), device, self.ov_config)
        self.infer_request = self.compiled_model.create_infer_request()

        # 缓存输入/输出名称以保持兼容性
        # Cache input/output names for compatibility
        self._input_names = [inp.get_any_name() for inp in self.compiled_model.inputs]
        self._output_names = [out.get_any_name() for out in self.compiled_model.outputs]

    def __call__(self, input_content: np.ndarray) -> list:
        """执行推理 —— 匹配 OrtInferSession 接口。
        Run inference — matches OrtInferSession interface."""
        # 设置输入张量（确保 C 连续）
        # Set input tensor (ensure C-contiguous)
        arr = np.ascontiguousarray(input_content, dtype=np.float32)
        input_tensor = ov.Tensor(array=arr)
        self.infer_request.set_input_tensor(input_tensor)
        self.infer_request.infer()

        # 获取输出
        # Get outputs
        results = []
        for i in range(len(self.compiled_model.outputs)):
            output = self.infer_request.get_output_tensor(i)
            results.append(output.data.copy())

        return results

    def get_input_names(self):
        return self._input_names

    def get_output_names(self):
        return self._output_names

    def have_key(self, key: str = "character") -> bool:
        # OpenVINO 不具备 ONNX 那样的模型元数据
        # OpenVINO doesn't have model metadata like ONNX
        return False

    def get_character_list(self, key: str = "character"):
        return []


def create_rapidocr_with_openvino(device: str = "CPU"):
    """创建使用 OpenVINO 后端的 RapidOCR 实例。
    Create a RapidOCR instance with OpenVINO backend."""
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()

    # 自动定位 RapidOCR 自带的 ONNX 模型目录
    # Auto-locate the ONNX model directory bundled with RapidOCR
    import rapidocr_onnxruntime
    model_dir = Path(rapidocr_onnxruntime.__file__).parent / "models"

    det_path = str(model_dir / "ch_PP-OCRv3_det_infer.onnx")
    rec_path = str(model_dir / "ch_PP-OCRv3_rec_infer.onnx")
    cls_path = str(model_dir / "ch_ppocr_mobile_v2.0_cls_infer.onnx")

    ov_config = {}
    if device == "GPU":
        ov_config = {"INFERENCE_PRECISION_HINT": "f16"}

    # 替换推理会话
    # Replace the inference sessions
    det_config = {"model_path": det_path, "use_cuda": False}
    rec_config = {"model_path": rec_path, "use_cuda": False}
    cls_config = {"model_path": cls_path, "use_cuda": False}

    ocr.text_detector.infer = OpenVINOInferSession(det_config, device, ov_config)
    ocr.text_recognizer.session = OpenVINOInferSession(rec_config, device, ov_config)
    if ocr.use_angle_cls:
        ocr.text_cls.session = OpenVINOInferSession(cls_config, device, ov_config)

    return ocr


def benchmark(image_path: str, runs: int = 20):
    """在同一图像上对所有配置进行基准测试。
    Benchmark all configurations on the same image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read {image_path}")
        return

    print(f"\n{'#'*70}")
    print(f"# Benchmark: {runs} runs | Image: {img.shape[1]}x{img.shape[0]}")
    print(f"{'#'*70}")

    configs = [
        ("RapidOCR-ORT-CPU",  None,   False),
        ("OpenVINO-FP32-CPU", "CPU",  False),
        ("OpenVINO-FP16-GPU", "GPU",  False),
    ]

    results_all = {}

    for name, device, int8 in configs:
        print(f"\n── {name} ──")

        if device is None:
            # 基准：使用 ONNX Runtime 的 RapidOCR
            # Baseline: RapidOCR with ONNX Runtime
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR()
        else:
            ocr = create_rapidocr_with_openvino(device=device)

        # 预热
        # Warm up
        for _ in range(3):
            result, _ = ocr(img)

        # 基准测试
        # Benchmark
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            result, _ = ocr(img)
            times.append(time.perf_counter() - t0)

        avg = np.mean(times) * 1000
        med = np.median(times) * 1000
        p95 = np.percentile(times, 95) * 1000
        best = np.min(times) * 1000

        print(f"  Avg:   {avg:7.1f} ms")
        print(f"  Med:   {med:7.1f} ms")
        print(f"  P95:   {p95:7.1f} ms")
        print(f"  Best:  {best:7.1f} ms")

        # 显示前几条结果
        # Show first few results
        if result:
            print(f"  Text lines: {len(result)}")
            for i, r in enumerate(result[:3]):
                text = r[1]
                conf = r[2] if len(r) > 2 else "?"
                print(f"    [{i+1}] {text} (conf={conf})")
            if len(result) > 3:
                print(f"    ... and {len(result)-3} more")

        results_all[name] = {"avg": avg, "med": med, "best": best}

    # 汇总
    # Summary
    print(f"\n{'='*70}")
    baseline = results_all.get("RapidOCR-ORT-CPU", {}).get("med", 1)
    for name, stats in results_all.items():
        speedup = baseline / stats["med"] if stats["med"] > 0 else 0
        print(f"  {name:25s}  med={stats['med']:6.1f}ms  vs ORT: {speedup:.2f}x")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="RapidOCR with OpenVINO backend")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"],
                        help="OpenVINO device (default: CPU)")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark all configurations")
    parser.add_argument("--runs", type=int, default=20,
                        help="Number of benchmark runs (default: 20)")
    args = parser.parse_args()

    if args.bench:
        benchmark(args.image, runs=args.runs)
    else:
        ocr = create_rapidocr_with_openvino(device=args.device)
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: Cannot read {args.image}")
            return

        t0 = time.perf_counter()
        result, _ = ocr(img)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"\nDevice: {args.device}")
        print(f"Time: {elapsed:.1f} ms")
        print(f"Found {len(result)} text regions:")
        for i, r in enumerate(result, 1):
            text = r[1]
            conf = r[2] if len(r) > 2 else "?"
            print(f"  [{i}] {text} (conf={conf})")


if __name__ == "__main__":
    main()
