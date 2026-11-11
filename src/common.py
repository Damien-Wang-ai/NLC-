# -*- coding: utf-8 -*-
"""
common.py —— 项目公共模块【负责人：成员 A（数据与预处理）】
职责：
    1. 统一维护项目路径常量与随机种子；
    2. 提供中文文本清洗与 jieba 分词接口（成员 B/C/D 共用）；
    3. 提供 JSON 读写与数据集读取接口。
说明：本模块被 traditional.py / textcnn.py / gui.py 共同引用，
     是四人协作的“接口约定”文件，修改前需通知其他成员。
"""
import json
import random
import re
from pathlib import Path

import jieba
import pandas as pd

# ---------------- 1. 路径与全局配置 ----------------
BASE_DIR = Path(__file__).resolve().parents[1]   # 项目根目录
DATA_DIR = BASE_DIR / "data"                     # 数据目录
RAW_DIR = DATA_DIR / "raw"                       # 原始数据目录
DATASET_CSV = DATA_DIR / "dataset.csv"           # 预处理后数据集
MODEL_DIR = BASE_DIR / "models"                  # 模型保存目录
RESULT_DIR = BASE_DIR / "results"                # 实验结果目录

for _d in (DATA_DIR, RAW_DIR, MODEL_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42          # 随机种子：保证四人实验结果可复现
MAX_LEN = 100      # TextCNN 单条文本最大词数
LABEL_NAMES = {0: "真实新闻", 1: "虚假新闻（谣言）"}

# ---------------- 2. 文本清洗与分词 ----------------
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")            # URL
_AT_RE = re.compile(r"@[\w\u4e00-\u9fa5\-_]{1,30}")          # @某人
_TOPIC_RE = re.compile(r"#[^#]{1,30}#")                      # #话题#
_EMOJI_RE = re.compile(r"\[[^\[\]]{1,10}\]")                 # [微笑] 之类表情
_KEEP_RE = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9]+")         # 仅保留汉字/字母/数字


def clean_text(text: str) -> str:
    """对单条微博/新闻文本做规范化清洗，返回清洗后的纯文本。"""
    text = str(text)
    text = _URL_RE.sub(" ", text)        # 去掉链接
    text = _AT_RE.sub(" ", text)         # 去掉 @用户
    text = _TOPIC_RE.sub(" ", text)      # 去掉 #话题# 的井号
    text = _EMOJI_RE.sub(" ", text)      # 去掉表情占位符
    text = _KEEP_RE.sub(" ", text)       # 去掉标点等无关符号
    return re.sub(r"\s+", " ", text).strip()


def has_chinese(text: str, threshold: int = 4) -> bool:
    """判断文本是否包含足够多的汉字，用于过滤掉数据集中的英文条目。"""
    return len(re.findall(r"[\u4e00-\u9fa5]", str(text))) >= threshold


def tokenize(text: str) -> str:
    """清洗 + jieba 分词，返回以空格分隔的词串（TF-IDF 与 TextCNN 统一使用该结果）。"""
    words = jieba.lcut(clean_text(text))
    # 过滤单字标点与空白，保留长度>=1的词
    return " ".join(w for w in words if w.strip())


# ---------------- 3. 通用 IO ----------------
def set_seed(seed: int = SEED) -> None:
    """固定随机种子，保证划分数据集、初始化模型时结果一致。"""
    random.seed(seed)


def save_json(obj, path) -> None:
    """把实验结果（字典/列表）写入 JSON 文件，统一使用 UTF-8 编码。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset() -> pd.DataFrame:
    """读取预处理后的数据集（列：text=分词结果，raw_text=原始文本，label=标签）。"""
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"未找到 {DATASET_CSV}，请先运行：python src/preprocess.py"
        )
    return pd.read_csv(DATASET_CSV)
