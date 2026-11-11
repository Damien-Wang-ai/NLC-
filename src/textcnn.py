# -*- coding: utf-8 -*-
"""
textcnn.py —— 深度学习模型 TextCNN【负责人：成员 C（深度学习模型）】

模型结构（对应说明书中的“模型结构图”）：
    输入文本(词序列, 长度100)
      -> 词嵌入层 Embedding(vocab, 128)
      -> 多尺度一维卷积 Conv1d(kernel = 2/3/4, 各100个卷积核) + ReLU
      -> 时间维最大池化 MaxPool1d -> 拼接为 300 维句向量
      -> Dropout(0.5) -> 全连接层 -> Softmax(2类：真实/虚假)

训练策略：Adam(lr=1e-3) + 交叉熵损失 + 早停（以验证集 F1 为准），
         最终在测试集上评估，并与成员 B 的传统模型对比。

运行方式：python src/textcnn.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent))
from common import MAX_LEN, MODEL_DIR, RESULT_DIR, SEED, load_dataset, save_json  # noqa: E402

VOCAB_PATH = MODEL_DIR / "textcnn_vocab.json"
WEIGHT_PATH = MODEL_DIR / "textcnn.pt"

# 超参数
EMBED_DIM = 128        # 词向量维度
NUM_FILTERS = 100      # 每种卷积核的数量
KERNEL_SIZES = (2, 3, 4)   # 卷积窗口大小（对应 2/3/4 元短语特征）
DROPOUT = 0.5
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
MIN_FREQ = 2           # 词频低于该值的词视为 <unk>
PAD, UNK = 0, 1        # 词表索引：0=<pad>, 1=<unk>

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
    """把分词后的文本转成固定长度的词索引序列。"""

    def __init__(self, texts, labels, word2idx):
        self.samples = []
        for text, label in zip(texts, labels):
            ids = [word2idx.get(w, UNK) for w in str(text).split()][:MAX_LEN]
            ids += [PAD] * (MAX_LEN - len(ids))     # 不足则补 <pad>
            self.samples.append((torch.tensor(ids, dtype=torch.long),
                                 torch.tensor(int(label), dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TextCNN(nn.Module):
    """TextCNN：词嵌入 + 多尺度卷积 + 最大池化 + 全连接分类。"""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_filters=NUM_FILTERS,
                 kernel_sizes=KERNEL_SIZES, num_classes=2, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)          # (B, embed, L)
        feats = [torch.relu(conv(emb)) for conv in self.convs]
        pooled = [torch.max(f, dim=2).values for f in feats]   # 时间维最大池化
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


def build_vocab(texts):
    """基于训练集统计词频，构建词表（含 <pad>/<unk>）。"""
    counter = Counter(w for text in texts for w in str(text).split())
    words = [w for w, c in counter.most_common() if c >= MIN_FREQ]
    word2idx = {"<pad>": PAD, "<unk>": UNK}
    for w in words:
        word2idx.setdefault(w, len(word2idx))
    print(f"词表大小：{len(word2idx)}（词频阈值 {MIN_FREQ}）")
    return word2idx


def train_model(model, train_loader, val_loader):
    """训练模型，以验证集 F1 作为早停与最优权重选择依据。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_f1, best_state, bad_epochs, patience = 0.0, None, 0, 3   # 连续 3 轮不提升则早停

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        # 每个 epoch 结束后在验证集上评估
        val_f1, val_acc = quick_eval(model, val_loader)
        print(f"Epoch {epoch:>2} | 训练损失 {total_loss / len(train_loader.dataset):.4f} "
              f"| 验证准确率 {val_acc:.4f} | 验证F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"验证集F1连续 {patience} 轮未提升，触发早停（Epoch {epoch}）")
                break

    model.load_state_dict(best_state)
    return best_f1


@torch.no_grad()
def quick_eval(model, loader):
    """快速评估：返回 (F1, 准确率)。"""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        logits = model(x.to(DEVICE))
        preds += logits.argmax(dim=1).cpu().tolist()
        labels += y.tolist()
    return (f1_score(labels, preds, zero_division=0),
            accuracy_score(labels, preds))


@torch.no_grad()
def evaluate(model, loader):
    """完整评估：输出与传统模型一致的评价指标。"""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds += model(x.to(DEVICE)).argmax(dim=1).cpu().tolist()
        labels += y.tolist()
    return {
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def main() -> None:
    torch.manual_seed(SEED)          # 固定随机种子，保证结果可复现
    df = load_dataset()
    train, val, test = (df[df["split"] == s] for s in ("train", "val", "test"))
    print(f"数据规模：训练 {len(train)} / 验证 {len(val)} / 测试 {len(test)} | 设备 {DEVICE}")

    word2idx = build_vocab(train["text"].tolist())
    loaders = {
        "train": DataLoader(TextDataset(train["text"], train["label"], word2idx),
                            batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(TextDataset(val["text"], val["label"], word2idx), batch_size=BATCH_SIZE),
        "test": DataLoader(TextDataset(test["text"], test["label"], word2idx), batch_size=BATCH_SIZE),
    }

    model = TextCNN(len(word2idx)).to(DEVICE)
    print(f"模型参数总量：{sum(p.numel() for p in model.parameters())}")

    t0 = time.time()
    best_f1 = train_model(model, loaders["train"], loaders["val"])
    seconds = round(time.time() - t0, 2)

    torch.save(model.state_dict(), WEIGHT_PATH)
    save_json({"word2idx": word2idx, "max_len": MAX_LEN, "embed_dim": EMBED_DIM,
               "num_filters": NUM_FILTERS, "kernel_sizes": list(KERNEL_SIZES)},
              VOCAB_PATH)

    metrics = {
        "model": "TextCNN",
        "key": "textcnn",
        "val": evaluate(model, loaders["val"]),
        "test": evaluate(model, loaders["test"]),
        "train_seconds": seconds,
        "config": {"embed_dim": EMBED_DIM, "num_filters": NUM_FILTERS,
                   "kernel_sizes": list(KERNEL_SIZES), "batch_size": BATCH_SIZE,
                   "lr": LR, "epochs": EPOCHS, "max_len": MAX_LEN, "device": str(DEVICE)},
        "best_val_f1": round(float(best_f1), 4),
    }
    print(f"\n验证集指标：{metrics['val']}")
    print(f"测试集指标：{metrics['test']}")
    save_json(metrics, RESULT_DIR / "textcnn_metrics.json")
    print(f"[完成] 模型保存至 {WEIGHT_PATH}，指标保存至 {RESULT_DIR / 'textcnn_metrics.json'}")


if __name__ == "__main__":
    main()
