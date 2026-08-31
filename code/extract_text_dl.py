#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习文本特征提取：情感正向性 + 专业性得分
按 (id, 周) 聚合，保存明细和聚合两份文件

使用前先在服务器上运行:
    pip install torch transformers pandas openpyxl tqdm

模型下载（提前执行，避免运行时联网）:
    python extract_text_dl.py --download-models
"""

import os
import sys
import time
import argparse
import re
import warnings
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ===================== 关键：设置 HuggingFace 国内镜像 =====================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ===================== 配置 =====================
MODEL_DIR = "/mnt/models"  # 模型保存目录
SENTIMENT_MODEL = "uer/roberta-base-finetuned-dianping-chinese"
SENTIMENT_LOCAL = os.path.join(MODEL_DIR, "roberta-dianping-chinese")

# 数据路径
DATA_DIR = "/mnt/data_zhengli"  # 你的数据目录
INPUT_FILE = "doctor_records_cleaned.xlsx"
OUTPUT_DETAIL = "doctor_records_with_features_text_dl.xlsx"
OUTPUT_WEEKLY = "panel_weekly_features_text_dl.xlsx"

BATCH_SIZE = 64  # A16 15G 显存足够跑 batch=64


# ===================== 0. 模型下载 =====================
def download_models():
    """提前下载模型到 /mnt/models，避免运行时联网"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"下载模型到: {MODEL_DIR}")

    print(f"\n[1/1] 下载情感分析模型: {SENTIMENT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
    tokenizer.save_pretrained(SENTIMENT_LOCAL)
    model.save_pretrained(SENTIMENT_LOCAL)
    print(f"  ✓ 已保存至 {SENTIMENT_LOCAL}")

    print("\n✅ 所有模型下载完成！可以断网运行主流程。")


# ===================== 1. 对话解析 =====================
def extract_doctor_text(dialogue_text: str) -> str:
    """从对话中提取医生的纯文本内容（拼接）"""
    if pd.isna(dialogue_text) or dialogue_text == "":
        return ""

    pattern = r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\[医生\]:\s*(.*?)(?=\n\s*\[|$)"
    matches = re.finditer(pattern, dialogue_text, re.DOTALL)

    doctor_texts = []
    for m in matches:
        content = m.group(1).strip()
        # 去除图片、语音标记
        content = re.sub(r"\[\d+张图片\]", "", content)
        content = re.sub(r"\[语音[（(].*?[)）]\]", "", content)
        content = content.strip()
        if content:
            doctor_texts.append(content)

    return " ".join(doctor_texts)


# ===================== 2. 专业性得分（纯规则，不需要 GPU） =====================
MEDICAL_DICT: Dict[str, List[str]] = {
    "疾病": [
        "感冒", "发烧", "咳嗽", "湿疹", "皮炎", "过敏", "高血压", "糖尿病",
        "冠心病", "哮喘", "肺炎", "胃炎", "肠炎", "腹泻", "便秘", "头痛",
        "失眠", "焦虑", "抑郁", "肿瘤", "癌症", "炎症", "感染", "溃疡",
        "结石", "骨折", "扭伤", "疼痛", "肿胀", "出血", "贫血", "心律",
        "血压", "血糖", "胆固醇", "脂肪肝", "肝炎", "肾炎", "尿路感染",
        "心脏病", "脑血管", "中风", "痛风", "关节炎", "颈椎病", "腰椎",
        "甲状腺", "支气管炎", "鼻炎", "咽炎", "扁桃体", "中耳炎",
        "甲沟炎", "甲钩炎", "脓肿", "化脓", "红肿",
    ],
    "症状": [
        "疼", "痛", "酸", "胀", "痒", "麻", "晕", "吐", "泻", "咳",
        "喘", "肿", "红", "热", "冷", "乏", "累", "困", "虚", "弱",
        "恶心", "呕吐", "腹痛", "胸痛", "头晕", "头痛", "乏力", "盗汗",
        "心悸", "气短", "胸闷", "水肿", "黄疸", "出汗", "寒战", "发脓",
    ],
    "药物": [
        "片", "胶囊", "颗粒", "口服液", "注射", "滴剂", "软膏", "乳膏",
        "阿莫西林", "头孢", "青霉素", "布洛芬", "对乙酰氨基酚", "维生素",
        "钙片", "益生菌", "止痛药", "消炎药", "抗生素", "中药", "西药",
        "阿司匹林", "甲硝唑", "克林霉素", "红霉素", "氨溴索", "沐舒坦",
        "药膏",
    ],
    "检查治疗": [
        "检查", "化验", "血常规", "尿常规", "B超", "CT", "MRI", "X光",
        "心电图", "彩超", "核磁", "内镜", "活检", "穿刺", "测量", "监测",
        "肝功能", "肾功能", "血脂", "血糖", "甲功", "凝血", "电解质",
        "治疗", "手术", "住院", "输液", "服药", "用药", "涂抹", "敷贴",
        "理疗", "针灸", "按摩", "休息", "观察", "复查", "随访", "换药",
        "切开", "排脓", "修剪", "清创", "拔甲", "局麻",
    ],
    "科室": [
        "外科", "内科", "骨科", "皮肤科", "急诊", "普外科", "儿科",
        "妇科", "眼科", "耳鼻喉", "口腔科", "泌尿科", "心内科",
    ],
}


def compute_professionalism(text: str) -> float:
    """基于医疗实体密度计算专业性得分"""
    if not text:
        return 0.0
    entity_count = 0
    for entities in MEDICAL_DICT.values():
        for ent in entities:
            entity_count += text.count(ent)
    return entity_count / len(text) if len(text) > 0 else 0.0


# ===================== 3. 情感正向性（GPU 批量推理） =====================
class SentimentScorer:
    """情感正向性批量打分器"""

    def __init__(self, model_path: str, device: str = "cuda", batch_size: int = 64):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        print(f"  设备: {self.device}")
        print(f"  加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("  ✓ 情感分析模型就绪")

    @torch.no_grad()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        """
        对一批文本返回正向概率。
        uer/roberta-base-finetuned-dianping-chinese 是二分类 [negative, positive]。
        """
        positive_scores = np.full(len(texts), 0.5)  # 默认 0.5

        # 过滤空文本
        valid_idx = [i for i, t in enumerate(texts) if t and len(t) >= 2]
        if not valid_idx:
            return positive_scores

        valid_texts = [texts[i][:510] for i in valid_idx]  # 截断

        enc = self.tokenizer(
            valid_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**enc).logits  # [batch, 2]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [batch, 2]

        for j, idx in enumerate(valid_idx):
            positive_scores[idx] = float(probs[j, 1])  # positive 在 index=1

        return positive_scores

    def score_all(self, texts: List[str]) -> np.ndarray:
        """对全部文本打分，显示进度和预估剩余时间"""
        n = len(texts)
        scores = np.empty(n, dtype=np.float64)

        total_batches = (n + self.batch_size - 1) // self.batch_size
        start_time = time.time()

        pbar = tqdm(total=total_batches, desc="情感正向性推理", unit="batch")
        for i in range(0, n, self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_scores = self.score_batch(batch)
            scores[i : i + len(batch)] = batch_scores

            pbar.update(1)
            elapsed = time.time() - start_time
            done_batches = pbar.n
            if done_batches > 0:
                eta = elapsed / done_batches * (total_batches - done_batches)
                pbar.set_postfix({"ETA": f"{eta:.0f}s"})
        pbar.close()

        return scores


# ===================== 4. 周划分（与原脚本一致） =====================
def assign_week(creation_time: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(creation_time):
        return pd.NaT
    days_since_monday = creation_time.weekday()
    week_monday = creation_time - timedelta(days=days_since_monday)
    return pd.Timestamp(week_monday.date())


# ===================== 5. 主流程 =====================
def main():
    parser = argparse.ArgumentParser(description="深度学习文本特征提取")
    parser.add_argument("--download-models", action="store_true", help="仅下载模型到 /mnt/models")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="数据目录路径")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="GPU 推理 batch size")
    args = parser.parse_args()

    # ---------- 下载模式 ----------
    if args.download_models:
        download_models()
        return

    path = args.data_dir
    if path and not path.endswith("/"):
        path += "/"

    input_file = path + INPUT_FILE
    output_detail = path + OUTPUT_DETAIL
    output_weekly = path + OUTPUT_WEEKLY

    print("=" * 70)
    print("深度学习文本特征提取（情感正向性 + 专业性得分）")
    print("=" * 70)

    # ---------- 读取数据 ----------
    print(f"\n📂 读取数据: {input_file}")
    df = pd.read_excel(input_file)
    print(f"  原始数据: {df.shape[0]} 行, {df.shape[1]} 列")

    # ---------- 划分周 ----------
    print("\n📅 划分问诊所属周...")
    df["周"] = df["创建时间"].apply(assign_week)
    print(f"  周数范围: {df['周'].min()} ~ {df['周'].max()}")
    print(f"  共 {df['周'].nunique()} 个周")

    # ---------- 提取医生文本 ----------
    print("\n📝 提取医生文本...")
    t0 = time.time()
    df["医生文本"] = df["对话"].apply(extract_doctor_text)
    valid_count = (df["医生文本"].str.len() > 0).sum()
    print(f"  ✓ 有效文本: {valid_count}/{len(df)}  ({time.time()-t0:.1f}s)")

    # ---------- 专业性得分（CPU，快速） ----------
    print("\n🏥 计算专业性得分...")
    t0 = time.time()
    n = len(df)
    prof_scores = np.empty(n, dtype=np.float64)
    pbar = tqdm(range(n), desc="专业性得分", unit="条")
    for i in pbar:
        prof_scores[i] = compute_professionalism(df.iloc[i]["医生文本"])
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            pbar.set_postfix({"ETA": f"{eta:.0f}s"})
    pbar.close()
    df["专业性得分"] = prof_scores
    print(f"  ✓ 完成 ({time.time()-t0:.1f}s)")

    # ---------- 情感正向性（GPU 推理） ----------
    print("\n😊 提取情感正向性...")
    model_path = SENTIMENT_LOCAL if os.path.exists(SENTIMENT_LOCAL) else SENTIMENT_MODEL
    print(f"  模型路径: {model_path}")
    scorer = SentimentScorer(model_path, device="cuda", batch_size=args.batch_size)

    texts = df["医生文本"].tolist()
    df["情感正向性"] = scorer.score_all(texts)

    # 清理 GPU 显存
    del scorer
    torch.cuda.empty_cache()

    # ---------- 特征统计 ----------
    print("\n📈 特征统计（明细级）:")
    for col in ["情感正向性", "专业性得分"]:
        vals = df[col]
        print(f"  {col:12s}: mean={vals.mean():.4f}, median={vals.median():.4f}, "
              f"std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")

    # ---------- 按 (id, 周) 聚合 ----------
    print(f"\n📊 按 (id, 周) 聚合...")

    agg_dict = {
        "姓名": "first",
        "情感正向性": "mean",
        "专业性得分": "mean",
        "对话": "count",
    }
    agg_dict_filtered = {k: v for k, v in agg_dict.items() if k in df.columns}

    panel_weekly = df.groupby(["id", "周"]).agg(agg_dict_filtered).reset_index()
    panel_weekly.rename(columns={
        "对话": "本周问诊次数",
        "情感正向性": "情感正向性_周均值",
        "专业性得分": "专业性得分_周均值",
    }, inplace=True)

    print(f"  ✓ 聚合完成: {panel_weekly.shape[0]} 行 (医生-周 组合)")
    print(f"  唯一医生数: {panel_weekly['id'].nunique()}")
    print(f"  唯一周数: {panel_weekly['周'].nunique()}")

    print(f"\n📈 聚合后特征统计:")
    for col in ["情感正向性_周均值", "专业性得分_周均值", "本周问诊次数"]:
        if col in panel_weekly.columns:
            vals = panel_weekly[col]
            print(f"  {col:20s}: mean={vals.mean():.4f}, median={vals.median():.4f}, std={vals.std():.4f}")

    # ---------- 保存 ----------
    print(f"\n💾 保存明细: {output_detail}")
    # 保存时去掉中间列"医生文本"以节省空间，如需保留可注释下行
    df.drop(columns=["医生文本"], errors="ignore").to_excel(output_detail, index=False)

    print(f"💾 保存聚合: {output_weekly}")
    panel_weekly.to_excel(output_weekly, index=False)

    print("\n" + "=" * 70)
    print("✅ 全部完成！")
    print(f"  明细文件: {output_detail}")
    print(f"  聚合文件: {output_weekly}")
    print("=" * 70)


if __name__ == "__main__":
    main()
