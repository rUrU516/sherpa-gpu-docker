#!/usr/bin/env python3
"""测试 FunASR nano GPU 推理"""

import os
import sys
import time
import wave
import numpy as np

# 添加 sherpa_onnx 路径
sys.path.insert(0, "/usr/local/lib/python3.10/site-packages")

import sherpa_onnx


def format_time(seconds):
    """格式化时间"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def read_wav(path):
    """读取 WAV 文件"""
    with wave.open(path, "rb") as f:
        samples = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        return samples.astype(np.float32) / 32768, f.getframerate()


def main():
    print("=" * 60)
    print("FunASR nano GPU 测试")
    print("=" * 60)

    model_dir = "/app/models/funasr-nano-int8"
    vad_path = "/app/models/vad/silero_vad.onnx"
    wav_path = os.environ.get("TEST_WAV", "/app/wavs/test.wav")

    # 检查文件
    for p in [model_dir, vad_path, wav_path]:
        if not os.path.exists(p):
            print(f"错误: 找不到 {p}")
            return

    # 加载模型
    print("\n[1] 加载 FunASR nano (GPU)...")
    t0 = time.time()
    
    recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
        encoder_adaptor=f"{model_dir}/encoder_adaptor.int8.onnx",
        llm=f"{model_dir}/llm.int8.onnx",
        embedding=f"{model_dir}/embedding.int8.onnx",
        tokenizer=f"{model_dir}/Qwen3-0.6B",
        num_threads=4,
        provider="cuda",  # 使用 GPU
        debug=True,
        system_prompt="你是一个专业健身教练",
        user_prompt="语音转写:",
        max_new_tokens=512,
        temperature=1e-6,
        top_p=0.8,
        seed=42,
        itn=True,
        hotwords="fiture,魔镜,史密斯",
        language="中文",
    )
    print(f"    加载完成: {time.time() - t0:.2f}s")

    # 加载 VAD
    print("\n[2] 加载 VAD...")
    t0 = time.time()
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = vad_path
    vad_config.silero_vad.threshold = 0.5
    vad_config.silero_vad.min_silence_duration = 1.0
    vad_config.silero_vad.max_speech_duration = 25.0
    vad_config.sample_rate = 16000
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=300)
    print(f"    加载完成: {time.time() - t0:.2f}s")

    # 读取音频
    print("\n[3] 读取音频...")
    samples, sr = read_wav(wav_path)
    duration = len(samples) / sr
    print(f"    时长: {duration:.1f}s")

    # VAD 分段
    print("\n[4] VAD 分段...")
    t0 = time.time()
    window = vad_config.silero_vad.window_size
    buf = samples.copy()
    while len(buf) > window:
        vad.accept_waveform(buf[:window])
        buf = buf[window:]
    vad.flush()

    segments = []
    while not vad.empty():
        seg = vad.front
        segments.append({
            "start": seg.start / sr,
            "duration": len(seg.samples) / sr,
            "samples": seg.samples
        })
        vad.pop()
    print(f"    找到 {len(segments)} 段, 耗时 {time.time() - t0:.2f}s")

    # ASR
    print("\n[5] ASR 推理...")
    results = []
    total_asr = 0

    for i, seg in enumerate(segments):
        t0 = time.time()
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, seg["samples"])
        recognizer.decode_stream(stream)
        elapsed = time.time() - t0
        total_asr += elapsed

        text = stream.result.text
        if text and text != "/sil":
            results.append({"start": seg["start"], "text": text})
            print(f"  [{format_time(seg['start'])}] {text[:60]}{'...' if len(text) > 60 else ''}")

    # 汇总
    print("\n" + "=" * 60)
    print(f"音频: {duration:.1f}s | ASR: {total_asr:.2f}s | RTF: {total_asr/duration:.3f}")
    print("=" * 60)

    # 保存
    with open("/app/output/transcript.txt", "w") as f:
        for r in results:
            f.write(f"[{format_time(r['start'])}] {r['text']}\n")
    print(f"\n保存到: /app/output/transcript.txt")


if __name__ == "__main__":
    main()