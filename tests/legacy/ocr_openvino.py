#!/usr/bin/env python3
"""
PaddleOCR (PP-OCRv3) inference with OpenVINO — CPU vs Intel GPU benchmark

Usage:
    python ocr_openvino.py <image_path> [--device CPU|GPU] [--bench] [--runs N]

Examples:
    python ocr_openvino.py test.png                     # CPU inference
    python ocr_openvino.py test.png --device GPU         # Intel iGPU inference
    python ocr_openvino.py test.png --bench              # Benchmark CPU vs GPU
    python ocr_openvino.py test.png --bench --runs 50    # More benchmark iterations
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


# ── Model paths ────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent
DET_MODEL = MODEL_DIR / "ch_PP-OCRv3_det_infer.xml"
CLS_MODEL = MODEL_DIR / "ch_ppocr_mobile_v2.0_cls_infer.xml"
REC_MODEL = MODEL_DIR / "ch_PP-OCRv3_rec_infer.xml"

# ── PP-OCRv3 char dictionary (Chinese + English) ───────────────────────
# Download the dict if not present
DICT_URL = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.6/ppocr/utils/ppocr_keys_v1.txt"
DICT_PATH = MODEL_DIR / "ppocr_keys_v1.txt"


def download_dict():
    if not DICT_PATH.exists():
        import urllib.request
        print(f"Downloading char dictionary to {DICT_PATH}...")
        urllib.request.urlretrieve(DICT_URL, str(DICT_PATH))
        print("Done.")


def load_char_dict():
    download_dict()
    with open(DICT_PATH, "r", encoding="utf-8") as f:
        chars = ["blank"] + [line.strip() for line in f if line.strip()]
    chars.append(" ")  # space
    return chars


# ── Preprocessing ──────────────────────────────────────────────────────

def preprocess_det(img, max_side_len=960):
    """Preprocess image for text detection model."""
    h, w = img.shape[:2]
    ratio = max_side_len / max(h, w) if max(h, w) > max_side_len else 1.0
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized = cv2.resize(img, (new_w, new_h))

    # Pad to multiple of 32
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))

    blob = padded.astype(np.float32)
    blob -= np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)
    blob /= np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # NCHW
    return blob, (new_h, new_w), (pad_h, pad_w)


def preprocess_cls(img_crop, img_size=(48, 192)):
    """Preprocess crop for direction classification."""
    resized = cv2.resize(img_crop, (img_size[1], img_size[0]))
    blob = resized.astype(np.float32)
    blob -= np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)
    blob /= np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob


def preprocess_rec(img_crop, img_height=48):
    """Preprocess crop for text recognition model."""
    h, w = img_crop.shape[:2]
    ratio = w / h
    new_w = max(32, min(int(img_height * ratio), 320))
    resized = cv2.resize(img_crop, (new_w, img_height))
    blob = resized.astype(np.float32)
    blob -= np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)
    blob /= np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob


# ── Postprocessing ─────────────────────────────────────────────────────

def postprocess_det(pred, orig_shape, resized_shape, pad_shape, thresh=0.3):
    """Extract text boxes from detection model output."""
    pred = pred[0, 0]  # [H, W]
    binary = (pred > thresh).astype(np.uint8) * 255

    # Resize back to original image size
    rh, rw = resized_shape
    binary = binary[:rh, :rw]
    binary = cv2.resize(binary, (orig_shape[1], orig_shape[0]))

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        # Filter by size
        w_box = max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2]))
        h_box = min(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2]))
        if h_box < 5 or w_box < 5:
            continue
        boxes.append(box)
    return boxes


def postprocess_cls(pred):
    """Return True if text needs to be rotated 180°."""
    # cls model: index 0 = normal, index 1 = rotated
    return pred[0, 1] > pred[0, 0]


def postprocess_rec(pred, char_dict):
    """Decode recognition model output to text string."""
    # CTC greedy decode
    pred = pred[0]  # [T, num_classes]
    indices = np.argmax(pred, axis=1)
    text = []
    prev_idx = -1
    for idx in indices:
        if idx != 0 and idx != prev_idx:
            if idx < len(char_dict):
                text.append(char_dict[idx])
        prev_idx = idx
    return "".join(text)


# ── Crop & rotate ──────────────────────────────────────────────────────

def crop_text_region(img, box):
    """Crop and perspective-correct a text region from the image."""
    rect = cv2.boundingRect(box)
    x, y, w, h = rect
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    crop = img[y:y+h, x:x+w]
    return crop


# ── Main OCR class ─────────────────────────────────────────────────────

class OpenVINOOCR:
    def __init__(self, device="CPU"):
        self.core = ov.Core()
        self.device = device
        self.char_dict = load_char_dict()

        # Compile models
        print(f"Loading models on {device}...")
        t0 = time.perf_counter()

        self.det_model = self.core.compile_model(str(DET_MODEL), device)
        self.det_infer = self.det_model.create_infer_request()

        self.cls_model = self.core.compile_model(str(CLS_MODEL), device)
        self.cls_infer = self.cls_model.create_infer_request()

        self.rec_model = self.core.compile_model(str(REC_MODEL), device)
        self.rec_infer = self.rec_model.create_infer_request()

        elapsed = time.perf_counter() - t0
        print(f"Models loaded in {elapsed:.2f}s")

    def detect(self, img):
        """Detect text regions. Returns list of bounding boxes."""
        orig_shape = img.shape[:2]
        blob, resized_shape, pad_shape = preprocess_det(img)
        self.det_infer.infer(blob)
        pred = self.det_infer.get_output_tensor().data
        boxes = postprocess_det(pred, orig_shape, resized_shape, pad_shape)
        return boxes

    def classify(self, crop):
        """Classify text direction. Returns True if needs 180° rotation."""
        blob = preprocess_cls(crop)
        self.cls_infer.infer(blob)
        pred = self.cls_infer.get_output_tensor().data
        return postprocess_cls(pred)

    def recognize(self, crop):
        """Recognize text in a crop. Returns text string."""
        blob = preprocess_rec(crop)
        self.rec_infer.infer(blob)
        pred = self.rec_infer.get_output_tensor().data
        return postprocess_rec(pred, self.char_dict)

    def run(self, img, verbose=True):
        """Full OCR pipeline: detect → classify → recognize."""
        t_start = time.perf_counter()

        # 1. Detect
        t0 = time.perf_counter()
        boxes = self.detect(img)
        t_det = time.perf_counter() - t0

        # 2. For each box: crop → classify → recognize
        results = []
        t_cls_total = 0
        t_rec_total = 0

        for box in boxes:
            crop = crop_text_region(img, box)
            if crop.size == 0:
                continue

            # Classify direction
            t0 = time.perf_counter()
            need_rotate = self.classify(crop)
            t_cls_total += time.perf_counter() - t0

            if need_rotate:
                crop = cv2.rotate(crop, cv2.ROTATE_180)

            # Recognize text
            t0 = time.perf_counter()
            text = self.recognize(crop)
            t_rec_total += time.perf_counter() - t0

            if text.strip():
                results.append(text.strip())

        t_total = time.perf_counter() - t_start

        if verbose:
            print(f"\n{'='*60}")
            print(f"Device: {self.device}")
            print(f"Detection:  {t_det*1000:7.1f} ms  ({len(boxes)} regions)")
            print(f"Classification: {t_cls_total*1000:7.1f} ms")
            print(f"Recognition: {t_rec_total*1000:7.1f} ms")
            print(f"Total:      {t_total*1000:7.1f} ms")
            print(f"{'='*60}")
            print("Recognized text:")
            for i, text in enumerate(results, 1):
                print(f"  [{i}] {text}")

        return results, {
            "device": self.device,
            "det_ms": t_det * 1000,
            "cls_ms": t_cls_total * 1000,
            "rec_ms": t_rec_total * 1000,
            "total_ms": t_total * 1000,
            "num_regions": len(boxes),
        }


def benchmark(img, runs=20):
    """Benchmark CPU vs GPU on the same image."""
    print(f"\n{'#'*60}")
    print(f"# Benchmark: {runs} runs")
    print(f"{'#'*60}")

    results = {}
    for device in ["CPU", "GPU"]:
        ocr = OpenVINOOCR(device=device)

        # Warm up
        for _ in range(3):
            ocr.run(img, verbose=False)

        # Timed runs
        times = []
        for i in range(runs):
            _, stats = ocr.run(img, verbose=False)
            times.append(stats["total_ms"])

        avg = np.mean(times)
        std = np.std(times)
        med = np.median(times)
        p95 = np.percentile(times, 95)
        best = np.min(times)

        print(f"\n── {device} ──")
        print(f"  Avg:   {avg:7.1f} ms  (±{std:.1f})")
        print(f"  Med:   {med:7.1f} ms")
        print(f"  P95:   {p95:7.1f} ms")
        print(f"  Best:  {best:7.1f} ms")

        results[device] = {
            "avg": avg, "med": med, "p95": p95, "best": best,
            "det_ms": stats["det_ms"], "cls_ms": stats["cls_ms"],
            "rec_ms": stats["rec_ms"],
        }

    # Summary
    if "CPU" in results and "GPU" in results:
        speedup = results["CPU"]["avg"] / results["GPU"]["avg"] if results["GPU"]["avg"] > 0 else 0
        print(f"\n{'='*60}")
        print(f"GPU speedup vs CPU: {speedup:.2f}x")
        if speedup < 1.0:
            print(f"  (GPU is {1/speedup:.2f}x slower — common for small models)")
        print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR with OpenVINO")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"],
                        help="OpenVINO device (default: CPU)")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark CPU vs GPU")
    parser.add_argument("--runs", type=int, default=20,
                        help="Number of benchmark runs (default: 20)")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Cannot read image: {args.image}")
        return

    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    if args.bench:
        benchmark(img, runs=args.runs)
    else:
        ocr = OpenVINOOCR(device=args.device)
        ocr.run(img)


if __name__ == "__main__":
    main()
