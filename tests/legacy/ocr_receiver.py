#!/usr/bin/env python3
"""
OBS rawvideo UDP 接收 + OpenVINO INT8 OCR

OBS 设置:
  Settings → Output → Output Mode: Advanced
  Recording → Type: Custom Output (FFmpeg)
  Container Format:  nut
  Video Encoder:     rawvideo
  Output URL:        udp://<本机IP>:1234
  Encoder Settings:  pixel_format=bgr24

本脚本监听 UDP 端口，逐帧接收裸像素并送入 OCR。
"""

import argparse
import socket
import time

import cv2
import numpy as np

# ── 帧参数 ────────────────────────────────────────────────────
W, H, CH = 256, 64, 3
FRAME_SIZE = W * H * CH  # 49152 bytes


# ── UDP 接收器 ─────────────────────────────────────────────────
class UDPReceiver:
    """接收 OBS rawvideo UDP 流。"""

    def __init__(self, port=1234, width=W, height=H, channels=CH):
        self.frame_size = width * height * channels
        self.shape = (height, width, channels)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(None)  # 无限等待，直到 OBS 推流
        print(f"Listening on UDP :{port}  (frame={self.frame_size} bytes, {width}x{height})")

    def recv_frame(self):
        """接收一帧。返回 numpy array 或 None。"""
        data, _ = self.sock.recvfrom(65535)
        if len(data) == self.frame_size:
            return np.frombuffer(data, dtype=np.uint8).reshape(self.shape).copy()
        return None  # nut 容器头等非帧数据，跳过

    def close(self):
        self.sock.close()


# ── 主循环 ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="OBS rawvideo → OCR")
    parser.add_argument("--port", type=int, default=1234, help="UDP 端口 (默认 1234)")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"], help="OpenVINO 设备")
    parser.add_argument("--no-int8", action="store_true", help="不用 INT8，用 FP32")
    parser.add_argument("--show", action="store_true", help="显示画面窗口 (调试用)")
    parser.add_argument("--stats-interval", type=int, default=60, help="每 N 帧打印统计")
    args = parser.parse_args()

    # 初始化 OCR（使用已验证的 RapidOCR + OpenVINO 后端）
    from rapidocr_openvino import create_rapidocr_with_openvino
    ocr = create_rapidocr_with_openvino(device=args.device, int8=not args.no_int8)

    # 关键优化：对小图 256x64，把 det 的 limit_side_len 从默认 736 降到 96
    # 否则 64px 短边会被放大到 736px，推理量暴增 60 倍
    for attr_name in ['text_detector']:
        det = getattr(ocr, attr_name, None)
        if det and hasattr(det, 'preprocess_op'):
            from rapidocr_onnxruntime.ch_ppocr_v3_det.utils import create_operators
            det.preprocess_op = create_operators({
                'DetResizeForTest': {'limit_side_len': 96, 'limit_type': 'min'},
                'NormalizeImage': {'std': [0.229, 0.224, 0.225], 'mean': [0.485, 0.456, 0.406],
                                   'scale': '1./255.', 'order': 'hwc'},
                'ToCHWImage': None,
                'KeepKeys': {'keep_keys': ['image', 'shape']},
            })
            print("  [优化] det limit_side_len: 736 → 96 (适配 256x64 小图)")

    receiver = UDPReceiver(port=args.port)
    print(f"\n等待 OBS 推流中... (Ctrl+C 退出)\n")

    frame_count = 0
    ocr_times = []
    t_start = time.perf_counter()

    try:
        while True:
            frame = receiver.recv_frame()
            if frame is None:
                continue

            # OCR
            t0 = time.perf_counter()
            result, _ = ocr(frame)
            ocr_ms = (time.perf_counter() - t0) * 1000
            ocr_times.append(ocr_ms)

            frame_count += 1

            # 显示结果
            texts = [r[1] for r in result] if result else []
            print(f"\r  Frame {frame_count}: {ocr_ms:5.1f}ms  |  {' | '.join(texts) if texts else '(无文本)'}",
                  end="", flush=True)

            # 调试窗口
            if args.show:
                cv2.imshow("OCR", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 定期统计
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
            total_time = time.perf_counter() - t_start
            print(f"  接收 FPS: {frame_count / total_time:.1f}")


if __name__ == "__main__":
    main()
