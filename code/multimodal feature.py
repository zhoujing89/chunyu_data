import pandas as pd
import math
# 问诊数据预处理
path = "/Users/Desktop/春雨数据整理/数据/"  #需修改

file1 = path + "doctor_records_2022_05_02.xls"
file2 = path + "doctor_records_2022_07_04.xlsx"

# Excel 最大行数限制（.xlsx）
MAX_ROWS = 1048575

sheet1 = "送心意明细"
sheet2 = "热门咨询明细"
sheet3 = "热门咨询明细-分页1"

# ========= 读取 =========
f1_s1 = pd.read_excel(file1, sheet_name=sheet1)
f1_s2 = pd.read_excel(file1, sheet_name=sheet2)
f1_s3 = pd.read_excel(file1, sheet_name=sheet3)

f2_s1 = pd.read_excel(file2, sheet_name=sheet1)
f2_s2 = pd.read_excel(file2, sheet_name=sheet2)
f2_s3 = pd.read_excel(file2, sheet_name=sheet3)


# ========= Sheet1：直接并集去重 =========
final_s1 = (
    pd.concat([f1_s1, f2_s1], ignore_index=True)
    .drop_duplicates()
)


# ========= Sheet2+3：各自先合并，再总体合并 =========
data1 = pd.concat([f1_s2, f1_s3], ignore_index=True)
data2 = pd.concat([f2_s2, f2_s3], ignore_index=True)

final_big = (
    pd.concat([data1, data2], ignore_index=True)
    .drop_duplicates()
)

# ========= 自动分页函数 =========
def split_dataframe(df, base_name):
    sheets = {}
    n_parts = math.ceil(len(df) / MAX_ROWS)

    for i in range(n_parts):
        start = i * MAX_ROWS
        end = (i + 1) * MAX_ROWS
        name = base_name if i == 0 else f"{base_name}-分页{i}"
        sheets[name] = df.iloc[start:end]

    return sheets


big_sheets = split_dataframe(final_big, sheet2)


# ========= 导出 =========
output_file = path + "doctor_records_merged.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    final_s1.to_excel(writer, sheet_name=sheet1, index=False)

    for name, df in big_sheets.items():
        df.to_excel(writer, sheet_name=name, index=False)


print("✅ 合并完成，输出文件：", output_file)
print("Sheet1行数：", len(final_s1))
print("大表总行数：", len(final_big))

import glob

file = path + "doctor_records_merged.xlsx"

# ========= 读取（自动合并分页sheet） =========
xls = pd.ExcelFile(file)

# 只读取 热门咨询明细 相关sheet
target_sheets = [s for s in xls.sheet_names if s.startswith("热门咨询明细")]

df_list = [pd.read_excel(file, sheet_name=s) for s in target_sheets]
df = pd.concat(df_list, ignore_index=True)

print("原始行数：", len(df))

# ========= 1. 时间过滤 =========以创建时间为准
df["创建时间"] = pd.to_datetime(df["创建时间"], errors="coerce")

df = df[
    (df["创建时间"] >= "2022-01-03") &
    (df["创建时间"] <= "2022-06-27")
]

print("时间过滤后：", len(df))


##################################
"""
医生问诊对话特征提取与按(id, 周)聚合脚本
"""

from typing import List, Dict
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta


# ============================================================
# 1. 对话解析函数
# ============================================================
def parse_dialogue(dialogue_text: str) -> List[Dict]:
    """解析对话文本，提取每条消息的结构化信息"""
    if pd.isna(dialogue_text) or dialogue_text == '':
        return []
    messages = []

    pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\[(患者|医生)\]:\s*(.*?)(?=\n\s*\[|$)'
    matches = re.finditer(pattern, dialogue_text, re.DOTALL)

    for match in matches:
        timestamp_str = match.group(1)
        sender = match.group(2)
        content = match.group(3).strip()

        try:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        except:
            timestamp = None

        # 图片检测
        has_image = bool(re.search(r'\[(\d+)张图片\]', content))
        image_count = 0
        if has_image:
            img_match = re.search(r'\[(\d+)张图片\]', content)
            if img_match:
                image_count = int(img_match.group(1))

        # 语音检测 —— 兼容中文括号（）和英文括号()
        has_voice = bool(re.search(r'\[语音[（(].*?[)）]\]', content))

        # 提取纯文本（去除图片、语音标记）
        text_content = re.sub(r'\[\d+张图片\]', '', content)
        text_content = re.sub(r'\[语音[（(].*?[)）]\]', '', text_content)
        text_content = text_content.strip()

        # 统计表情符号
        emoji_count = 0
        emoji_count += len(re.findall(r'[\U0001F600-\U0001F64F]', content))
        emoji_count += len(re.findall(r'[\U0001F300-\U0001F5FF]', content))
        emoji_count += len(re.findall(r'[\U0001F680-\U0001F6FF]', content))
        emoji_count += len(re.findall(r'[\U0001F900-\U0001F9FF]', content))

        messages.append({
            'timestamp': timestamp,  # 时间戳
            'sender': sender,  # 谁发的
            'content': content,  # 原始内容
            'text_content': text_content,  # 纯文本
            'text_length': len(text_content),
            'has_image': has_image,
            'image_count': image_count,
            'has_voice': has_voice,
            'emoji_count': emoji_count
        })

    return messages


# ============================================================
# 2. 医生特征提取函数
# ============================================================
def extract_doctor_features_from_dialogue(dialogue_text: str, creation_time=None) -> Dict:
    """从单条对话中提取医生相关特征

    修改说明:
    1. 医生总回复时长: 只在患者->医生的首次切换时计算一次回复时长，
       医生连续发送多条消息不重复计算。单位为秒。
    2. 新增: 医生首次回复距创建时间的差（秒）。
    3. 医生文本条数 / 医生文本平均字数: 仅统计有实际文本内容的消息
       （排除纯图片、纯语音等无文字内容的消息）。
    """
    messages = parse_dialogue(dialogue_text)

    if not messages:
        return {
            '医生文本条数': 0,
            '医生总字数': 0,
            '医生文本平均字数': 0,
            '医生回复次数': 0,  # 1
            '医生总回复时长': 0,  # 2
            '医生平均回复时长': 0,  # 3
            '医生首次回复距创建时间_秒': np.nan,
            '对话图片数': 0,
            '对话表情数': 0,  # 4
            '医生语音条数': 0,
            '医生最大连续语音数': 0,
        }

    doctor_messages = [m for m in messages if m['sender'] == '医生']

    # --- 修改点3: 仅统计有实际文本内容的医生消息 ---
    # 有实际文本内容 = text_length > 0（去除图片/语音标记后仍有文字）
    doctor_text_messages = [m for m in doctor_messages if m['text_length'] > 0]
    doctor_text_count = len(doctor_text_messages)
    doctor_total_chars = sum(m['text_length'] for m in doctor_text_messages)
    doctor_avg_chars = doctor_total_chars / doctor_text_count if doctor_text_count > 0 else 0

    # --- 修改点1: 医生回复时长，避免连续回复重复计算 ---
    # 逻辑: 遍历消息序列，当发生"患者 -> 医生"的角色切换时，
    #        用医生该条消息的时间 - 前面最近一条患者消息的时间 = 一次回复时长。
    #        医生连续发送的后续消息不再重复计算。
    reply_times = []
    last_patient_timestamp = None  # 记录最近一条患者消息的时间
    last_sender = None  # 记录上一条消息的发送者

    for msg in messages:
        if msg['sender'] == '患者' and msg['timestamp']:
            last_patient_timestamp = msg['timestamp']
        elif msg['sender'] == '医生' and msg['timestamp']:
            # 只在 "患者 -> 医生" 切换时计算（即上一条是患者）
            if last_sender == '患者' and last_patient_timestamp is not None:
                time_diff = (msg['timestamp'] - last_patient_timestamp).total_seconds()
                if time_diff >= 0:
                    reply_times.append(time_diff)
        last_sender = msg['sender']

    doctor_reply_count = len(reply_times)
    doctor_total_reply_time = sum(reply_times)  # 单位: 秒
    doctor_avg_reply_time = np.mean(reply_times) if reply_times else 0

    # --- 修改点2: 医生首次回复距创建时间的差 ---
    doctor_first_reply_from_creation = np.nan
    if creation_time is not None and not pd.isna(creation_time):
        # 找到医生的第一条消息
        for msg in messages:
            if msg['sender'] == '医生' and msg['timestamp']:
                time_diff = (msg['timestamp'] - creation_time).total_seconds()
                doctor_first_reply_from_creation = time_diff
                break

    total_images = sum(m['image_count'] for m in messages)
    total_emojis = sum(m['emoji_count'] for m in messages)
    doctor_voice_count = sum(1 for m in doctor_messages if m['has_voice'])

    # 最大连续语音数
    max_consecutive_voice = 0
    current_consecutive = 0
    for msg in messages:
        if msg['sender'] == '医生' and msg['has_voice']:
            current_consecutive += 1
            max_consecutive_voice = max(max_consecutive_voice, current_consecutive)
        else:
            current_consecutive = 0

    return {
        '医生文本条数': doctor_text_count,
        '医生总字数': doctor_total_chars,
        '医生文本平均字数': doctor_avg_chars,
        '医生回复次数': doctor_reply_count,
        '医生总回复时长': doctor_total_reply_time,
        '医生平均回复时长': doctor_avg_reply_time,
        '医生首次回复距创建时间_秒': doctor_first_reply_from_creation,
        '对话图片数': total_images,
        '对话表情数': total_emojis,
        '医生语音条数': doctor_voice_count,
        '医生最大连续语音数': max_consecutive_voice,
    }


# ============================================================
# 3. 周划分函数
# ============================================================
def assign_week(creation_time: pd.Timestamp) -> pd.Timestamp:
    """根据创建时间分配所属周的周一日期"""
    if pd.isna(creation_time):
        return pd.NaT
    days_since_monday = creation_time.weekday()
    week_monday = creation_time - timedelta(days=days_since_monday)
    return pd.Timestamp(week_monday.date())


# ---------- 读取数据 ----------
file = path + "doctor_records_cleaned.xlsx"

print("=" * 60)
print("📂 读取数据...")
df = pd.read_excel(file)
print(f"  原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
print(f"  字段: {list(df.columns)}")

# ---------- 划分周 ----------
print("\n📅 划分问诊所属周...")
df['周'] = df['创建时间'].apply(assign_week)
print(f"  周数范围: {df['周'].min()} ~ {df['周'].max()}")
print(f"  共 {df['周'].nunique()} 个周")

# ---------- 语音格式预检 ----------
print("\n🔎 语音格式预检...")
sample_with_voice = df[df['对话'].astype(str).str.contains('语音', na=False)].head(5)
print(f"  含'语音'关键字的记录数: {df['对话'].astype(str).str.contains('语音', na=False).sum()}")

# 检查括号类型
cn_paren_count = df['对话'].astype(str).str.contains(r'\[语音（', na=False).sum()
en_paren_count = df['对话'].astype(str).str.contains(r'\[语音\(', na=False).sum()
print(f"  中文括号（）格式: {cn_paren_count} 条")
print(f"  英文括号()格式: {en_paren_count} 条")

# ---------- 提取对话特征 ----------
print(f"\n🔍 提取对话特征 (共 {len(df)} 条记录)...")

features_list = []
total = len(df)
for i, (idx, row) in enumerate(df.iterrows()):
    if (i + 1) % 10000 == 0 or i == 0:
        print(f"  进度: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")
    features = extract_doctor_features_from_dialogue(
        row.get('对话', ''),
        creation_time=row.get('创建时间', None)
    )
    features_list.append(features)

features_df = pd.DataFrame(features_list, index=df.index)
df = pd.concat([df, features_df], axis=1)
print(f"  ✅ 特征提取完成，新增 {len(features_df.columns)} 个特征列")

# 语音特征验证
voice_records = df[df['医生语音条数'] > 0]
print(f"  📢 含医生语音的问诊记录: {len(voice_records)} 条")
if len(voice_records) > 0:
    print(f"     医生语音条数分布: mean={voice_records['医生语音条数'].mean():.2f}, "
          f"max={voice_records['医生语音条数'].max()}")

# 首次回复特征验证
valid_first_reply = df[df['医生首次回复距创建时间_秒'].notna()]
print(f"  ⏱️ 有医生首次回复时间的记录: {len(valid_first_reply)} 条")
if len(valid_first_reply) > 0:
    print(f"     首次回复时间(秒)分布: mean={valid_first_reply['医生首次回复距创建时间_秒'].mean():.2f}, "
          f"median={valid_first_reply['医生首次回复距创建时间_秒'].median():.2f}, "
          f"max={valid_first_reply['医生首次回复距创建时间_秒'].max():.2f}")

# ---------- 按 (id, 周) 聚合 ----------
print(f"\n📊 按 (id, 周) 聚合...")

agg_dict = {
    # 基本信息
    '姓名': 'first',

    # 对话特征 - 求均值
    '医生文本条数': 'mean',
    '医生文本平均字数': 'mean',
    '医生平均回复时长': 'mean',
    '医生首次回复距创建时间_秒': 'mean',
    '对话图片数': 'mean',
    '对话表情数': 'mean',
    '医生语音条数': 'mean',

    # 最大连续语音 - 取 max
    '医生最大连续语音数': 'max',

    # 本周问诊次数
    '对话': 'count',
}

# 只保留 df 中实际存在的列
agg_dict_filtered = {k: v for k, v in agg_dict.items() if k in df.columns}

panel_weekly = df.groupby(['id', '周']).agg(agg_dict_filtered).reset_index()

# 重命名
panel_weekly.rename(columns={
    '对话': '本周问诊次数',
    '医生文本条数': '医生发送文本平均条数',
    '医生文本平均字数': '医生文本平均长度_字数',
    '医生平均回复时长': '医生回复平均速度_秒',
    '医生首次回复距创建时间_秒': '医生首次回复距创建时间平均_秒',
    '对话图片数': '对话中含图片平均数量',
    '对话表情数': '表情符号平均数量',
    '医生语音条数': '医生发送语音平均条数',
    '医生最大连续语音数': '最多连续发送语音数',
}, inplace=True)

# ---------- 打印统计 ----------
print(f"  ✅ 聚合完成: {panel_weekly.shape[0]} 行 (医生-周 组合)")
print(f"  唯一医生数: {panel_weekly['id'].nunique()}")
print(f"  唯一周数: {panel_weekly['周'].nunique()}")

print(f"\n📈 聚合后特征统计:")
feature_cols = [
    '本周问诊次数',
    '医生发送文本平均条数',
    '医生文本平均长度_字数',
    '医生回复平均速度_秒',
    '医生首次回复距创建时间平均_秒',
    '对话中含图片平均数量',
    '表情符号平均数量',
    '医生发送语音平均条数',
    '最多连续发送语音数',
]

for col in feature_cols:
    if col in panel_weekly.columns:
        mean_val = panel_weekly[col].mean()
        median_val = panel_weekly[col].median()
        std_val = panel_weekly[col].std()
        print(f"  {col:35s}: 均值={mean_val:8.2f}, 中位数={median_val:8.2f}, 标准差={std_val:8.2f}")




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


############medical term, politeness_density
record = pd.read_csv(r'yuanshishuju.csv', encoding='utf-8-sig')
import re
#生成患者对话和医生对话
record['对话'] = record['对话'].fillna('')
record['医生对话'] = record['对话'].apply(lambda x: ' '.join(re.findall('\[医生\]: (.*?)\\n', x)))
record['患者对话'] = record['对话'].apply(lambda x: ' '.join(re.findall('\[患者\]: (.*?)\\n', x)))

import jieba
import pandas as pd
import numpy as np  # 导入 numpy 处理空值
from tqdm import tqdm
import os

# 允许 tqdm 在 pandas 中使用
tqdm.pandas()

# --- 1. 初始化数据 ---
# 假设 record 已经定义好
data = record

# --- 2. 加载词典 (逻辑保持不变) ---
medical_words_set = set()
try:
    with open("H:\\bishe\\refined_medical_dict.txt", encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip().split()[0]
            if len(word) > 1:
                medical_words_set.add(word)
    for w in medical_words_set:
        jieba.add_word(w)
    print(f"成功加载医学词典，共 {len(medical_words_set)} 个词")
except:
    print("⚠️ 未找到医学词典，医学密度计算可能不准！")

stop_words_set = set()
try:
    with open('H:\\bishe\\doc_info\\dict\\stopwords-master\\cn_stopwords.txt', encoding='utf-8') as f:
        for line in f.readlines():
            stop_words_set.add(line.strip())
except:
    stop_words_set = {'的', '了', '和', '是', '就', '都', '而', '及', '与'}
    print("⚠️ 未找到停用词表，使用内置基础停用词。")

# 词典定义
politeness_words = {
    '您', '请', '祝', '麻烦', '谢谢', '感谢', '不用谢', '不客气', '没事', '没关系',
    '早日康复', '放心', '安心', '不用担心', '抱歉', '不好意思', '好的', '收到', '亲',
    '家长', '朋友', '您好', '拜托', '辛苦了', '承蒙', '见谅', '问候', '保重', '恭祝',
    '劳驾', '谦谢', '关照', '赐教', '恭喜', '安康', '顺遂'
}



# ================= 3. 核心计算函数  =================

def analyze_doctor_text_v2(text_content):
    """
    如果无法计算，统一返回 (None, None, None, None)
    返回: (medical_density, medical_term, politeness_density, word_count)
    """
    # 检查是否为字符串且不为空
    if not isinstance(text_content, str) or len(text_content.strip()) == 0:
        return None, None, None, None

    try:
        # 分词
        words = jieba.lcut(text_content)

        total_valid_words = 0
        medical_cnt = 0
        politeness_cnt = 0

        for w in words:
            # 跳过停用词和标点
            if w in stop_words_set or len(w.strip()) == 0:
                continue

            total_valid_words += 1

            if w in medical_words_set:
                medical_cnt += 1
            if w in politeness_words:
                politeness_cnt += 1

        # 如果没有有效词，无法计算比例，返回 None
        if total_valid_words == 0:
            return None, None, None, None

        # 计算指标
        medical_density = medical_cnt / total_valid_words
        medical_term = 1 if medical_cnt != 0 else 0  # 是否出现医学词
        politeness_density = politeness_cnt / total_valid_words

        return medical_density, medical_term, politeness_density, total_valid_words

    except Exception as e:
        # 捕获任何意外错误并返回 None
        return None, None, None, None


# ================= 4. 执行计算并保存 =================

print("正在计算文本指标...")

# 使用 progress_apply 替代 apply 以显示进度条
results = data['医生对话'].progress_apply(lambda x: pd.Series(analyze_doctor_text_v2(x)))

# 重命名列名
results.columns = ['medical_density', 'medical_term', 'politeness_density', 'word_count']

# 将结果合并回原表
data = pd.concat([data, results], axis=1)


def nanmean_agg(x):
    valid = x.dropna()
    return valid.mean() if len(valid) > 0 else np.nan

# 按 (医生id, 周) 分组聚合
weekly_politeness = (
    data.groupby(['id', '周'], dropna=False)
        .agg(
            politeness_density_mean=('politeness_density', nanmean_agg),  # 礼貌密度周均值
            politeness_count_valid=('politeness_density', lambda x: x.notna().sum()),  # 有效记录数
            consult_count=('politeness_density', 'size'),                 # 本周问诊总次数
        )
        .reset_index()
)

# 计算礼貌密度有效率（有礼貌密度值 / 本周总问诊）
weekly_politeness['politeness_valid_ratio'] = (
    weekly_politeness['politeness_count_valid'] / weekly_politeness['consult_count']
).round(4)


#  单独保存汇总面板
weekly_politeness.to_csv('politeness_weekly_panel.csv', index=False, encoding='utf-8-sig')





