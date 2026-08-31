#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音深度学习特征提取（并行加速版 v2 - 分阶段断点续传）

断点续传策略:
  - Phase 2 (音频预处理): 每处理完一个 chunk (2000条) 写入缓存
  - Phase 3 (GPU情感推理): 每处理完一个 batch 写入缓存
  - 中断后重跑自动跳过已完成的部分

加速策略:
  1. 跳过无语音的对话（16万条中只处理~1.9万条有语音的）
  2. 多进程并行音频预处理（CPU 密集型 librosa I/O 并行化）
  3. GPU batch 情感推理（攒满 batch 一次推理）

提取特征（每次问诊聚合均值）:
  1. 平均语音时长     (avg_duration_sec)        —— 单条语音的秒数
  2. 频谱质心均值     (spectral_centroid_mean)   —— 声音亮度指标 (Hz)
  3. 语速             (speech_rate)              —— 能量活跃帧数 / 总帧数 (0~1)
  4. 声音温暖度       (warmth_score)             —— 基于 emotion2vec+ VA 值 (0~1)

依赖安装:
  pip install funasr modelscope librosa soundfile tqdm pandas numpy torch openpyxl

模型下载（提前执行）:
  python extract_voice_dl.py --download-models

运行:
  python extract_voice_dl.py --audio /course75/OHC/mp3_1 /course75/OHC/mp3_2 /course75/OHC/mp3_3 /course75/OHC/mp3_4 /course75/OHC/mp3_5
"""

import os
import sys
import re
import json
import time
import argparse
import tempfile
import warnings
from datetime import timedelta
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ===================== 配置 =====================
MODEL_DIR = "/mnt/models"
EMOTION2VEC_LOCAL = os.path.join(MODEL_DIR, "emotion2vec_plus_base")
EMOTION2VEC_MODEL_ID = "iic/emotion2vec_plus_base"

DATA_DIR = "/mnt/data_zhengli"
INPUT_FILE = "doctor_records_cleaned.xlsx"
OUTPUT_DETAIL = "doctor_records_with_features_voice_dl.xlsx"
OUTPUT_WEEKLY = "panel_weekly_features_voice_dl.xlsx"
CHECKPOINT_DIR_NAME = "voice_checkpoint"

EMOTION_BATCH_SIZE = 32
NUM_WORKERS = 6

# ===================== 情感 → VA → 温暖度映射 =====================
EMOTION_VA_MAP = {
    0: {"valence": 0.10, "arousal": 0.85},  # angry
    1: {"valence": 0.15, "arousal": 0.70},  # disgusted
    2: {"valence": 0.20, "arousal": 0.90},  # fearful
    3: {"valence": 0.90, "arousal": 0.75},  # happy
    4: {"valence": 0.50, "arousal": 0.50},  # neutral
    5: {"valence": 0.50, "arousal": 0.50},  # other
    6: {"valence": 0.25, "arousal": 0.30},  # sad
    7: {"valence": 0.60, "arousal": 0.80},  # surprised
    8: {"valence": 0.50, "arousal": 0.50},  # unknown
}
WARMTH_VALENCE_WEIGHT = 0.7
WARMTH_AROUSAL_WEIGHT = 0.3
SPEECH_RATE_ENERGY_THRESHOLD = 0.01


# ===================== 断点缓存管理 =====================
class CheckpointManager:
    """
    分阶段断点缓存管理器。
    使用一个文件夹存放多个 JSON 文件：
      - phase2_acoustic.json : {mp3_filename: {duration_sec, spectral_centroid, speech_rate}}
      - phase3_emotion.json  : {mp3_filename: {valence, arousal, warmth_score}}
      - phase4_dialogue.json : {row_index: {avg_duration_sec, ...}}  最终结果
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.acoustic_path = os.path.join(checkpoint_dir, "phase2_acoustic.json")
        self.emotion_path = os.path.join(checkpoint_dir, "phase3_emotion.json")
        self.dialogue_path = os.path.join(checkpoint_dir, "phase4_dialogue.json")

        self.acoustic_cache = self._load(self.acoustic_path)    # mp3_fname → acoustic features
        self.emotion_cache = self._load(self.emotion_path)      # mp3_fname → emotion features
        self.dialogue_cache = self._load(self.dialogue_path)    # row_idx_str → aggregated features

    @staticmethod
    def _load(path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @staticmethod
    def _save(path: str, data: dict):
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp_path, path)  # 原子替换，防止写到一半断电导致文件损坏

    def save_acoustic(self):
        self._save(self.acoustic_path, self.acoustic_cache)

    def save_emotion(self):
        self._save(self.emotion_path, self.emotion_cache)

    def save_dialogue(self):
        self._save(self.dialogue_path, self.dialogue_cache)

    def status(self):
        print(f"  断点缓存状态:")
        print(f"    Phase 2 声学特征: {len(self.acoustic_cache)} 条")
        print(f"    Phase 3 情感特征: {len(self.emotion_cache)} 条")
        print(f"    Phase 4 对话聚合: {len(self.dialogue_cache)} 条")


# ===================== 0. 模型下载 =====================
def download_models():
    """通过 ModelScope 下载 emotion2vec+ base 模型到本地"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"下载模型到: {MODEL_DIR}")
    print(f"\n[1/1] 下载 emotion2vec+ base 模型...")
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        try:
            from modelscope import snapshot_download
        except ImportError:
            print("  modelscope 未安装，正在安装...")
            os.system("pip install modelscope -q")
            from modelscope import snapshot_download
    try:
        cache_dir = snapshot_download(EMOTION2VEC_MODEL_ID, local_dir=EMOTION2VEC_LOCAL)
        print(f"  已保存至 {cache_dir}")
        print("\n模型下载完成！")
    except Exception as e:
        print(f"  下载失败: {e}")
        print(f"  手动下载: https://modelscope.cn/models/{EMOTION2VEC_MODEL_ID}")
        print(f"  上传到: {EMOTION2VEC_LOCAL}/")
        sys.exit(1)


# ===================== 1. 音频预处理（可在子进程中运行） =====================
def _preprocess_single_mp3(audio_path: str, target_sr: int = 16000) -> Optional[dict]:
    """
    子进程调用：读取 MP3 → 提取声学特征 + 写入临时 WAV。
    返回 {acoustic features, tmp_wav_path} 或 None。
    """
    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        y = np.clip(y, -1.0, 1.0)
        y, _ = librosa.effects.trim(y, top_db=20)

        if len(y) < sr * 0.1:
            return None

        duration_sec = len(y) / sr
        sc_frames = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid = float(np.mean(sc_frames))
        rms_frames = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
        active_frames = np.sum(rms_frames > SPEECH_RATE_ENERGY_THRESHOLD)
        speech_rate = float(active_frames / len(rms_frames)) if len(rms_frames) > 0 else 0.0

        # 写临时 WAV（给 emotion2vec 用）
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, y, target_sr, subtype="PCM_16")

        return {
            "duration_sec": round(float(duration_sec), 4),
            "spectral_centroid": round(spectral_centroid, 2),
            "speech_rate": round(speech_rate, 4),
            "tmp_wav": tmp.name,
        }
    except Exception:
        return None


def preprocess_batch_parallel(mp3_paths: List[str], num_workers: int = 6) -> List[Optional[dict]]:
    """多进程并行预处理一批 MP3 文件"""
    results = [None] * len(mp3_paths)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(_preprocess_single_mp3, path): idx
            for idx, path in enumerate(mp3_paths)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = None

    return results


# ===================== 2. 情感 → 温暖度 =====================
def scores_to_warmth(scores: list) -> dict:
    scores = np.array(scores, dtype=np.float64)
    total = scores.sum()
    if total < 1e-8:
        return {"valence": 0.5, "arousal": 0.5, "warmth_score": 0.5}
    valence, arousal = 0.0, 0.0
    for idx, prob in enumerate(scores):
        va = EMOTION_VA_MAP.get(idx, {"valence": 0.5, "arousal": 0.5})
        valence += prob * va["valence"]
        arousal += prob * va["arousal"]
    valence /= total
    arousal /= total
    arousal_term = 1.0 - abs(arousal - 0.5)
    warmth_score = WARMTH_VALENCE_WEIGHT * valence + WARMTH_AROUSAL_WEIGHT * arousal_term
    warmth_score = float(np.clip(warmth_score, 0.0, 1.0))
    return {"valence": round(valence, 4), "arousal": round(arousal, 4),
            "warmth_score": round(warmth_score, 4)}


# ===================== 3. 工具函数 =====================
def extract_filename_from_url(voice_url: str) -> Optional[str]:
    """
    URL: .../37UAAAB0cYtCv8YW-b3583091-44e2-4e4d-9859-2660fd16d15f.mp3
    → 去横杠取最后12位 → 2660fd16d15f.mp3
    """
    if not voice_url:
        return None
    match = re.search(r'/([^/]+)\.mp3', voice_url)
    if not match:
        return None
    full_name = match.group(1)
    no_dash = full_name.replace("-", "")
    last12 = no_dash[-12:]
    return last12.lower() + ".mp3"


def extract_voice_urls(dialogue_text: str) -> list:
    """从对话文本提取医生语音 URL"""
    if pd.isna(dialogue_text) or dialogue_text == "":
        return []
    return re.findall(r'\[医生\]:.*?\[语音[（(](.*?)[)）]\]', dialogue_text)


def build_audio_map(audio_folders: list) -> dict:
    """扫描文件夹，构建 {文件名小写: 完整路径}"""
    print("  扫描音频文件...")
    audio_map = {}
    for folder in audio_folders:
        if not os.path.exists(folder):
            print(f"    文件夹不存在: {folder}")
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(".mp3"):
                audio_map[fname.lower()] = os.path.join(folder, fname)
    print(f"  找到 {len(audio_map)} 个 MP3 文件")
    return audio_map


def assign_week(creation_time: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(creation_time):
        return pd.NaT
    days_since_monday = creation_time.weekday()
    week_monday = creation_time - timedelta(days=days_since_monday)
    return pd.Timestamp(week_monday.date())


# ===================== 4. 核心：分阶段流水线（含断点续传） =====================
def run_phase2_acoustic(
    all_mp3_paths: List[str],
    all_mp3_fnames: List[str],
    ckpt: CheckpointManager,
    num_workers: int = 6,
) -> Dict[str, dict]:
    """
    Phase 2: 多进程并行音频预处理。
    按 mp3 文件名缓存，每处理完一个 chunk 写入断点。
    返回: {mp3_fname: {duration_sec, spectral_centroid, speech_rate, tmp_wav}}
    """
    # 找出需要处理的（不在缓存中的）
    todo_indices = []
    for i, fname in enumerate(all_mp3_fnames):
        if fname not in ckpt.acoustic_cache:
            todo_indices.append(i)

    cached = len(all_mp3_fnames) - len(todo_indices)
    print(f"    总唯一 MP3: {len(all_mp3_fnames)}")
    print(f"    已有缓存: {cached}")
    print(f"    需要处理: {len(todo_indices)}")

    if not todo_indices:
        # 全部已缓存，但仍需为 Phase 3 生成临时 WAV 文件
        # 对缓存中的条目，Phase 3 会重新生成 WAV（见下方逻辑）
        return ckpt.acoustic_cache

    CHUNK_SIZE = 2000
    t0 = time.time()
    total = len(todo_indices)
    processed = 0

    for chunk_start in range(0, total, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total)
        chunk_indices = todo_indices[chunk_start:chunk_end]
        chunk_paths = [all_mp3_paths[i] for i in chunk_indices]
        chunk_fnames = [all_mp3_fnames[i] for i in chunk_indices]

        # 多进程并行处理这一批
        chunk_results = preprocess_batch_parallel(chunk_paths, num_workers=num_workers)

        # 写入缓存（只保存声学特征，不保存 tmp_wav 路径到磁盘——那是临时的）
        for fname, result in zip(chunk_fnames, chunk_results):
            if result is not None:
                ckpt.acoustic_cache[fname] = {
                    "duration_sec": result["duration_sec"],
                    "spectral_centroid": result["spectral_centroid"],
                    "speech_rate": result["speech_rate"],
                }
                # tmp_wav 暂存在内存中，不写入 JSON
            else:
                # 标记为处理过但失败
                ckpt.acoustic_cache[fname] = None

        processed += len(chunk_indices)

        # 每个 chunk 写入断点
        ckpt.save_acoustic()

        elapsed = time.time() - t0
        eta = elapsed / processed * (total - processed) if processed > 0 else 0
        print(f"    预处理进度: {processed}/{total} ({processed/total*100:.1f}%)  "
              f"耗时: {elapsed:.0f}s  ETA: {eta:.0f}s")

    preprocess_time = time.time() - t0
    valid_count = sum(1 for v in ckpt.acoustic_cache.values() if v is not None)
    print(f"    Phase 2 完成: {valid_count}/{len(all_mp3_fnames)} 成功  "
          f"耗时: {preprocess_time:.0f}s")

    return ckpt.acoustic_cache


def run_phase3_emotion(
    all_mp3_paths: List[str],
    all_mp3_fnames: List[str],
    acoustic_cache: dict,
    ckpt: CheckpointManager,
    model_dir: str,
    device: str = "cuda",
    emotion_batch_size: int = 32,
    num_workers: int = 6,
) -> Dict[str, dict]:
    """
    Phase 3: GPU 批量情感推理。
    需要重新生成临时 WAV（因为 Phase 2 缓存中不含 WAV 路径）。
    按 mp3 文件名缓存，每个 batch 写入断点。
    返回: {mp3_fname: {valence, arousal, warmth_score}}
    """
    # 找出需要情感推理的（声学特征有效 且 不在情感缓存中）
    todo_fnames = []
    todo_paths = []
    for i, fname in enumerate(all_mp3_fnames):
        ac = acoustic_cache.get(fname)
        if ac is not None and fname not in ckpt.emotion_cache:
            todo_fnames.append(fname)
            todo_paths.append(all_mp3_paths[i])

    cached = sum(1 for fname in all_mp3_fnames if fname in ckpt.emotion_cache)
    print(f"    总唯一 MP3: {len(all_mp3_fnames)}")
    print(f"    声学有效: {sum(1 for v in acoustic_cache.values() if v is not None)}")
    print(f"    已有情感缓存: {cached}")
    print(f"    需要情感推理: {len(todo_fnames)}")

    if not todo_fnames:
        return ckpt.emotion_cache

    # 加载模型
    from funasr import AutoModel
    model_id = model_dir if model_dir else EMOTION2VEC_MODEL_ID
    print(f"    加载模型: {model_id}")
    emotion_model = AutoModel(model=model_id, hub="ms", device=device)
    print(f"    模型就绪 | device={device}")

    # 分 chunk 处理：先生成临时 WAV → 批量推理 → 清理 WAV → 保存断点
    # WAV 生成也用多进程加速
    PREP_CHUNK = 2000  # 每次准备 2000 条 WAV，推理完再做下一批
    t0 = time.time()
    total = len(todo_fnames)
    processed = 0

    for chunk_start in range(0, total, PREP_CHUNK):
        chunk_end = min(chunk_start + PREP_CHUNK, total)
        chunk_fnames_batch = todo_fnames[chunk_start:chunk_end]
        chunk_paths_batch = todo_paths[chunk_start:chunk_end]

        # 多进程并行生成临时 WAV
        wav_results = preprocess_batch_parallel(chunk_paths_batch, num_workers=num_workers)

        # 收集有效的 WAV
        valid_items = []  # (fname, wav_path)
        for fname, result in zip(chunk_fnames_batch, wav_results):
            if result is not None and result.get("tmp_wav"):
                valid_items.append((fname, result["tmp_wav"]))
            else:
                # 无法生成 WAV，给默认值
                ckpt.emotion_cache[fname] = {"valence": 0.5, "arousal": 0.5, "warmth_score": 0.5}

        # 批量 GPU 推理
        for bi in range(0, len(valid_items), emotion_batch_size):
            batch_items = valid_items[bi:bi + emotion_batch_size]
            batch_fnames_b = [item[0] for item in batch_items]
            batch_wavs = [item[1] for item in batch_items]

            try:
                rec_results = emotion_model.generate(
                    batch_wavs, granularity="utterance", extract_embedding=False,
                )
                for r, fname in zip(rec_results, batch_fnames_b):
                    scores = r.get("scores", [])
                    if len(scores) == 9:
                        ckpt.emotion_cache[fname] = scores_to_warmth(scores)
                    else:
                        ckpt.emotion_cache[fname] = {"valence": 0.5, "arousal": 0.5,
                                                      "warmth_score": 0.5}
            except Exception:
                for fname in batch_fnames_b:
                    ckpt.emotion_cache[fname] = {"valence": 0.5, "arousal": 0.5,
                                                  "warmth_score": 0.5}

        # 清理这一批的临时 WAV 文件
        for fname, result in zip(chunk_fnames_batch, wav_results):
            if result is not None and result.get("tmp_wav"):
                try:
                    os.remove(result["tmp_wav"])
                except OSError:
                    pass

        processed += len(chunk_fnames_batch)

        # 每个 chunk 写入断点
        ckpt.save_emotion()

        elapsed = time.time() - t0
        eta = elapsed / processed * (total - processed) if processed > 0 else 0
        print(f"    情感推理进度: {processed}/{total} ({processed/total*100:.1f}%)  "
              f"耗时: {elapsed:.0f}s  ETA: {eta:.0f}s")

    # 释放模型
    del emotion_model
    torch.cuda.empty_cache()

    emotion_time = time.time() - t0
    print(f"    Phase 3 完成  耗时: {emotion_time:.0f}s")

    return ckpt.emotion_cache


def run_phase4_aggregate(
    voice_indices: List[int],
    dialogues: List[str],
    audio_map: dict,
    acoustic_cache: dict,
    emotion_cache: dict,
    ckpt: CheckpointManager,
) -> dict:
    """
    Phase 4: 按对话聚合特征。速度很快（纯内存操作），不需要细粒度断点。
    """
    new_count = 0
    for row_idx in voice_indices:
        cache_key = str(row_idx)
        if cache_key in ckpt.dialogue_cache:
            continue

        urls = extract_voice_urls(str(dialogues[row_idx]))
        durations, sc_vals, sr_vals, warmth_vals = [], [], [], []

        for url in urls:
            fname = extract_filename_from_url(url)
            if not fname:
                continue

            fname_lower = fname.lower()

            # 声学特征
            ac = acoustic_cache.get(fname_lower)
            if ac is not None:
                durations.append(ac["duration_sec"])
                sc_vals.append(ac["spectral_centroid"])
                sr_vals.append(ac["speech_rate"])

            # 情感特征
            emo = emotion_cache.get(fname_lower,
                                     {"valence": 0.5, "arousal": 0.5, "warmth_score": 0.5})
            if emo is not None:
                warmth_vals.append(emo["warmth_score"])

        def safe_mean(lst):
            return round(float(np.mean(lst)), 4) if lst else None

        ckpt.dialogue_cache[cache_key] = {
            "avg_duration_sec": safe_mean(durations),
            "spectral_centroid_mean": round(float(np.mean(sc_vals)), 2) if sc_vals else None,
            "speech_rate": safe_mean(sr_vals),
            "warmth_score": safe_mean(warmth_vals),
            "voice_count": len(urls),
        }
        new_count += 1

    ckpt.save_dialogue()
    print(f"    Phase 4 完成: 新聚合 {new_count} 条")

    return ckpt.dialogue_cache


# ===================== 5. 主流程 =====================
def main():
    parser = argparse.ArgumentParser(description="语音深度学习特征提取（并行加速版 v2）")
    parser.add_argument("--download-models", action="store_true",
                        help="仅下载模型到 /mnt/models")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--audio", nargs="+", default=None,
                        help="MP3 音频文件夹路径（可多个）")
    parser.add_argument("--batch-size", type=int, default=EMOTION_BATCH_SIZE,
                        help="emotion2vec batch size")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help="音频预处理并行进程数")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="断点缓存目录")
    args = parser.parse_args()

    if args.download_models:
        download_models()
        return

    path = args.data_dir
    if path and not path.endswith("/"):
        path += "/"

    input_file = path + INPUT_FILE
    output_detail = path + OUTPUT_DETAIL
    output_weekly = path + OUTPUT_WEEKLY
    checkpoint_dir = args.checkpoint_dir or (path + CHECKPOINT_DIR_NAME)

    audio_folders = args.audio
    if not audio_folders:
        candidates = [path + d for d in ["audio", "audios", "voice", "mp3"]]
        candidates += ["/mnt/audio", "/mnt/voice"]
        audio_folders = [d for d in candidates if os.path.isdir(d)]
        if not audio_folders:
            print("请用 --audio 指定音频文件夹")
            sys.exit(1)

    use_gpu = not args.no_gpu
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("语音深度学习特征提取（并行加速版 v2 - 分阶段断点续传）")
    print("=" * 70)
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  GPU: 否（CPU 模式）")
    print(f"  并行进程数: {args.workers}")
    print(f"  emotion batch size: {args.batch_size}")
    print(f"  音频目录: {audio_folders}")
    print(f"  断点目录: {checkpoint_dir}")

    # ---------- 读取数据 ----------
    print(f"\n读取数据: {input_file}")
    df = pd.read_excel(input_file)
    print(f"  原始数据: {df.shape[0]} 行, {df.shape[1]} 列")

    # ---------- 划分周 ----------
    print("\n划分问诊所属周...")
    df["周"] = df["创建时间"].apply(assign_week)
    print(f"  周数范围: {df['周'].min()} ~ {df['周'].max()}")
    print(f"  共 {df['周'].nunique()} 个周")

    # ---------- 识别有语音的对话 ----------
    dialogues = df["对话"].tolist()
    has_voice_mask = df["对话"].astype(str).str.contains(
        r'\[医生\]:.*?\[语音', na=False, regex=True
    )
    voice_indices = list(has_voice_mask[has_voice_mask].index)
    no_voice_indices = list(has_voice_mask[~has_voice_mask].index)

    print(f"\n语音数据概览:")
    print(f"  有语音的对话: {len(voice_indices)} 条 (仅处理这些)")
    print(f"  无语音的对话: {len(no_voice_indices)} 条 (直接跳过，特征设 NaN)")

    # ---------- 构建音频文件映射 ----------
    print("\n构建音频文件映射...")
    audio_map = build_audio_map(audio_folders)

    # ---------- 初始化断点管理器 ----------
    ckpt = CheckpointManager(checkpoint_dir)
    ckpt.status()

    # ---------- Phase 1: 提取 URL → 本地路径 ----------
    print("\n[Phase 1] 解析语音 URL → 本地文件映射...")
    all_mp3_fnames = []   # 去重后的 mp3 文件名列表
    all_mp3_paths = []    # 对应的完整路径
    fname_set = set()

    for row_idx in tqdm(voice_indices, desc="  解析URL", unit="条"):
        urls = extract_voice_urls(str(dialogues[row_idx]))
        for url in urls:
            fname = extract_filename_from_url(url)
            if fname and fname.lower() in audio_map and fname.lower() not in fname_set:
                fname_set.add(fname.lower())
                all_mp3_fnames.append(fname.lower())
                all_mp3_paths.append(audio_map[fname.lower()])

    print(f"  去重后唯一 MP3: {len(all_mp3_fnames)}")

    total_start = time.time()

    # ---------- Phase 2: 多进程并行音频预处理 ----------
    print(f"\n[Phase 2] 多进程并行音频预处理 (workers={args.workers})...")
    acoustic_cache = run_phase2_acoustic(
        all_mp3_paths, all_mp3_fnames, ckpt, num_workers=args.workers,
    )

    # ---------- Phase 3: GPU 批量情感推理 ----------
    print(f"\n[Phase 3] GPU 批量情感推理 (batch_size={args.batch_size})...")
    model_dir = EMOTION2VEC_LOCAL if os.path.exists(EMOTION2VEC_LOCAL) else None
    emotion_cache = run_phase3_emotion(
        all_mp3_paths, all_mp3_fnames, acoustic_cache, ckpt,
        model_dir=model_dir, device=device,
        emotion_batch_size=args.batch_size, num_workers=args.workers,
    )

    # ---------- Phase 4: 按对话聚合 ----------
    print(f"\n[Phase 4] 按对话聚合特征...")
    dialogue_cache = run_phase4_aggregate(
        voice_indices, dialogues, audio_map,
        acoustic_cache, emotion_cache, ckpt,
    )

    total_elapsed = time.time() - total_start
    print(f"\n  总处理耗时: {total_elapsed / 60:.1f} 分钟")

    # ---------- 组装特征到 DataFrame ----------
    print("\n组装特征...")
    _no_voice = {
        "avg_duration_sec": None, "spectral_centroid_mean": None,
        "speech_rate": None, "warmth_score": None, "voice_count": 0,
    }
    features_list = []
    for i in range(len(df)):
        cache_key = str(i)
        if cache_key in dialogue_cache:
            features_list.append(dialogue_cache[cache_key])
        else:
            features_list.append(_no_voice)

    rename_map = {
        "avg_duration_sec": "平均语音时长",
        "spectral_centroid_mean": "频谱质心均值",
        "speech_rate": "语速",
        "warmth_score": "声音温暖度",
        "voice_count": "语音数量",
    }
    features_df = pd.DataFrame(features_list).rename(columns=rename_map)
    df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    valid_voice = df["平均语音时长"].notna().sum()
    no_voice = df["平均语音时长"].isna().sum()
    print(f"  有语音的问诊: {valid_voice}/{len(df)} ({valid_voice / len(df) * 100:.1f}%)")
    print(f"  无语音的问诊: {no_voice}/{len(df)} ({no_voice / len(df) * 100:.1f}%) -> NaN")

    # ---------- 特征统计 ----------
    print("\n特征统计（明细级，仅有语音的记录）:")
    for col in ["平均语音时长", "频谱质心均值", "语速", "声音温暖度"]:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 0:
                print(f"  {col:15s}: N={len(valid):5d}  "
                      f"mean={valid.mean():.4f}  std={valid.std():.4f}  "
                      f"range=[{valid.min():.4f}, {valid.max():.4f}]")
            else:
                print(f"  {col:15s}: 无有效值")

    # ---------- 按 (id, 周) 聚合 ----------
    print(f"\n按 (id, 周) 聚合...")
    meta_agg = {}
    if "姓名" in df.columns:
        meta_agg["姓名"] = "first"

    feature_cols = ["平均语音时长", "频谱质心均值", "语速", "声音温暖度"]

    def nanmean_agg(x):
        valid = x.dropna()
        return valid.mean() if len(valid) > 0 else np.nan

    feature_agg = {col: nanmean_agg for col in feature_cols}
    agg_dict = {**meta_agg, "语音数量": "sum", **feature_agg, "对话": "count"}
    agg_dict_filtered = {k: v for k, v in agg_dict.items() if k in df.columns}

    panel_weekly = df.groupby(["id", "周"]).agg(agg_dict_filtered).reset_index()
    panel_weekly.rename(columns={"对话": "本周问诊次数"}, inplace=True)

    voice_count_per_group = (
        df[df["平均语音时长"].notna()]
        .groupby(["id", "周"]).size()
        .reset_index(name="有语音问诊次数")
    )
    panel_weekly = panel_weekly.merge(voice_count_per_group, on=["id", "周"], how="left")
    panel_weekly["有语音问诊次数"] = panel_weekly["有语音问诊次数"].fillna(0).astype(int)
    panel_weekly["语音使用率"] = (
        panel_weekly["有语音问诊次数"] / panel_weekly["本周问诊次数"]
    ).round(4)

    print(f"  聚合完成: {panel_weekly.shape[0]} 行")
    print(f"  唯一医生数: {panel_weekly['id'].nunique()}")

    print(f"\n聚合后特征统计:")
    for col in feature_cols + ["语音使用率", "本周问诊次数"]:
        if col in panel_weekly.columns:
            valid = panel_weekly[col].dropna()
            if col in feature_cols:
                valid = valid[valid > 0] if len(valid) > 0 else valid
            if len(valid) > 0:
                print(f"  {col:15s}: N={len(valid):5d}  mean={valid.mean():.4f}  "
                      f"median={valid.median():.4f}  std={valid.std():.4f}")

    no_voice_weeks = panel_weekly["平均语音时长"].isna().sum()
    has_voice_weeks = panel_weekly["平均语音时长"].notna().sum()
    print(f"\n  有语音的医生-周: {has_voice_weeks}  无语音的医生-周: {no_voice_weeks}")

    # ---------- 保存 ----------
    print(f"\n保存明细: {output_detail}")
    df.to_excel(output_detail, index=False)
    print(f"保存聚合: {output_weekly}")
    panel_weekly.to_excel(output_weekly, index=False)

    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("全部完成！")
    print(f"  总耗时: {(time.time() - total_start) / 60:.1f} 分钟")
    print(f"  明细文件: {output_detail}")
    print(f"  聚合文件: {output_weekly}")
    print("=" * 70)


if __name__ == "__main__":
    main()
