#!/usr/bin/env python3
"""
OBS MJPEG UDP 接收 + OpenVINO FP32 OCR + 血量解析
OBS MJPEG UDP receiver + OpenVINO FP32 OCR + HP parsing

OBS 设置 / OBS Setup:
  Settings → Output → Output Mode: Advanced
  Recording → Type: Custom Output (FFmpeg)
  Container Format:  mjpeg
  Video Encoder:     mjpeg
  Output URL:        udp://<本机IP>:1234
  Encoder Settings:  (留空即可，默认质量足够 / leave blank, default quality is sufficient)

本脚本监听 UDP，按 JPEG 帧边界 (FFD8...FFD9) 重组帧，解码后送入 OCR，
并通过 HPParser 去除 OCR 扰动，输出稳定的 current/max 血量。

This script listens on UDP, reassembles frames by JPEG boundary markers (FFD8...FFD9),
decodes them, feeds them into OCR, and uses HPParser to remove OCR noise,
outputting stable current/max HP values.
"""

import argparse
import os
import socket
import sys
import time

# 将 lib/ 加入模块搜索路径 // Add lib/ to module search path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import cv2
import numpy as np

W, H = 256, 64

# JPEG 帧边界标记 // JPEG frame boundary markers
JPEG_START = b'\xff\xd8'
JPEG_END = b'\xff\xd9'


class MJPEGReceiver:
    """从 UDP 流中提取完整 JPEG 帧。// Extract complete JPEG frames from a UDP stream."""

    def __init__(self, port=1234):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(10.0)
        self.buffer = b''
        self.port = port
        print(f"Listening on UDP :{port}  (MJPEG mode)")

    def recv_frame(self):
        """返回解码后的 numpy array，或 None。// Return decoded numpy array, or None."""
        while True:
            start = self.buffer.find(JPEG_START)
            if start != -1:
                end = self.buffer.find(JPEG_END, start)
                if end != -1:
                    jpeg_data = self.buffer[start:end + 2]
                    self.buffer = self.buffer[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        return frame
                    continue

            try:
                data, _ = self.sock.recvfrom(65535)
                self.buffer += data
                # 缓冲区过大时裁剪，防止内存溢出 // Trim buffer when too large to prevent OOM
                if len(self.buffer) > 10 * 1024 * 1024:
                    last_start = self.buffer.rfind(JPEG_START)
                    if last_start > 0:
                        self.buffer = self.buffer[last_start:]
                    else:
                        self.buffer = self.buffer[-1 * 1024 * 1024:]
            except socket.timeout:
                return None

    def close(self):
        self.sock.close()


def main():
    parser = argparse.ArgumentParser(description="OBS MJPEG → OCR → HP")
    parser.add_argument("--port", type=int, default=1234, help="UDP 端口 (默认 1234) // UDP port (default 1234)")
    parser.add_argument("--stats-interval", type=int, default=60, help="每 N 帧打印统计 // Print stats every N frames")
    args = parser.parse_args()

    # 初始化 OCR — FP32 CPU // Initialize OCR — FP32 CPU
    from rapidocr_openvino import create_rapidocr_with_openvino
    from rapidocr_onnxruntime.ch_ppocr_v3_det.utils import create_operators
    ocr = create_rapidocr_with_openvino(device="CPU")
    ocr.text_detector.preprocess_op = create_operators({
        'DetResizeForTest': {'limit_side_len': 64, 'limit_type': 'min'},
        'NormalizeImage': {'std': [0.229, 0.224, 0.225], 'mean': [0.485, 0.456, 0.406],
                           'scale': '1./255.', 'order': 'hwc'},
        'ToCHWImage': None,
        'KeepKeys': {'keep_keys': ['image', 'shape']},
    })
    print("  [配置] OpenVINO FP32 CPU, limit_side_len=64")

    # 初始化血量解析器 // Initialize HP parser
    from hp_parser import HPParser
    hp_parser = HPParser()

    receiver = MJPEGReceiver(port=args.port)
    print(f"\n等待 OBS 推流中... (Ctrl+C 退出)\n")

    frame_count = 0
    ocr_times = []
    recv_times = []
    t_start = time.perf_counter()

    try:
        while True:
            # 接收帧 // Receive frame
            t0 = time.perf_counter()
            frame = receiver.recv_frame()
            recv_ms = (time.perf_counter() - t0) * 1000

            if frame is None:
                print("\r  等待推流...", end="", flush=True)
                continue

            recv_times.append(recv_ms)

            # OCR 识别 // OCR recognition
            t0 = time.perf_counter()
            result, _ = ocr(frame)
            ocr_ms = (time.perf_counter() - t0) * 1000
            ocr_times.append(ocr_ms)

            frame_count += 1

            # 解析血量 // Parse HP
            ocr_texts = [r[1] for r in result] if result else []
            hp = hp_parser.parse(ocr_texts)

            # 显示结果 // Display results
            line = f"\r  Frame {frame_count}: ocr={ocr_ms:.1f}ms"
            if hp.valid:
                line += f"  HP: {hp.current}/{hp.max_hp} [{hp.confidence}]"
            else:
                line += f"  HP: --- raw={hp.raw!r}"
            print(line, end="", flush=True)

            # 定期统计 // Periodic stats
            if frame_count % args.stats_interval == 0:
                elapsed = time.perf_counter() - t_start
                fps = frame_count / elapsed
                avg_ocr = np.mean(ocr_times[-args.stats_interval:])
                p95_ocr = np.percentile(ocr_times[-args.stats_interval:], 95)
                print(f"\n  ── Stats: {fps:.1f} FPS | OCR avg={avg_ocr:.1f}ms p95={p95_ocr:.1f}ms ──")

    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
        if ocr_times:
            print(f"\n\n最终统计 ({frame_count} 帧):")
            print(f"  OCR avg:  {np.mean(ocr_times):.1f} ms")
            print(f"  OCR p95:  {np.percentile(ocr_times, 95):.1f} ms")
            print(f"  OCR best: {np.min(ocr_times):.1f} ms")
            total = time.perf_counter() - t_start
            print(f"  总 FPS:   {frame_count / total:.1f}")


if __name__ == "__main__":
    main()
