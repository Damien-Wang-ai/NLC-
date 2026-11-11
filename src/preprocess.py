# -*- coding: utf-8 -*-
"""
preprocess.py —— 数据清洗、分词与划分【负责人：成员 A（数据与预处理）】

输入：data/raw/ced_dataset.jsonl（由 download_data.py 下载的 CED 中文谣言数据，
      字段：id / text（微博原文）/ label（0=非谣言，1=谣言））
输出：data/dataset.csv  与  results/dataset_stats.json

处理步骤：
    1. 读取原始数据并校验标签分布；
    2. 文本清洗（去 URL、@用户、#话题#、表情符号、多余标点）；
    3. jieba 分词，得到词序列文本（供 TF-IDF 与 TextCNN 使用）；
    4. 过滤过短样本、按清洗后文本去重；
    5. 按 7:1:2 分层划分为训练集/验证集/测试集（固定随机种子）。

运行方式：python src/preprocess.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
from common import (DATASET_CSV, LABEL_NAMES, RAW_DIR, RESULT_DIR, SEED,  # noqa: E402
                    clean_text, has_chinese, save_json, tokenize)

RAW_JSONL = RAW_DIR / "ced_dataset.jsonl"
MIN_CHARS = 6      # 清洗后少于 6 个汉字的样本视为噪声丢弃
MAX_CHARS = 300    # 超长微博截断，避免极端长度影响模型训练


def read_raw() -> pd.DataFrame:
    """读取 download_data.py 产出的原始数据文件。"""
    if not RAW_JSONL.exists():
        raise FileNotFoundError(
            f"未找到 {RAW_JSONL}，请先运行：python src/download_data.py"
        )
    records = []
    with open(RAW_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records).rename(columns={"text": "raw_text"})
    print(f"[读取] {RAW_JSONL.name}: {len(df)} 条")
    return df


def build_dataset() -> pd.DataFrame:
    """完整数据流程：清洗 -> 分词 -> 去重 -> 分层划分。"""
    df = read_raw()

    # 1) 标签与内容校验
    df["label"] = df["label"].astype(int)
    df["raw_text"] = df["raw_text"].astype(str).str.slice(0, MAX_CHARS)
    df = df[df["raw_text"].map(has_chinese)].copy()
    print(f"[中文校验] 保留含中文样本 {len(df)} 条，"
          f"标签分布 {df['label'].value_counts().to_dict()}")

    # 2) 文本清洗 + 分词
    df["clean_text"] = df["raw_text"].map(clean_text)
    df = df[df["clean_text"].str.len() >= MIN_CHARS].copy()
    df["text"] = df["clean_text"].map(tokenize)

    # 3) 去重（同一条谣言常被多次发布，需避免重复计入）
    before = len(df)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    print(f"[去重] {before} -> {len(df)} 条")

    # 4) 分层划分：训练 70% / 验证 10% / 测试 20%
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"])
    train_df, val_df = train_test_split(
        train_df, test_size=0.125, random_state=SEED, stratify=train_df["label"])
    train_df["split"], val_df["split"], test_df["split"] = "train", "val", "test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def main() -> None:
    df = build_dataset()
    df = df[["id", "raw_text", "clean_text", "text", "label", "split"]]
    df.to_csv(DATASET_CSV, index=False, encoding="utf-8-sig")

    stats = {
        "total": int(len(df)),
        "splits": {k: int(v) for k, v in df["split"].value_counts().items()},
        "labels": {LABEL_NAMES[k]: int(v) for k, v in df["label"].value_counts().items()},
        "avg_words": round(float(df["text"].str.split().str.len().mean()), 2),
        "max_words": int(df["text"].str.split().str.len().max()),
    }
    save_json(stats, RESULT_DIR / "dataset_stats.json")
    print(f"\n[完成] 数据集已保存至 {DATASET_CSV}")
    print(f"[统计] {stats}")


if __name__ == "__main__":
    main()
