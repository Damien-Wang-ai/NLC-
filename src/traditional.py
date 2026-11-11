# -*- coding: utf-8 -*-
"""
traditional.py —— 传统机器学习模型对比【负责人：成员 B（传统机器学习模型）】

模型方案：TF-IDF 文本特征 + 4 种经典分类器
    1. 逻辑回归 LogisticRegression（并对正则强度 C 做网格搜索调参）
    2. 多项式朴素贝叶斯 MultinomialNB
    3. 线性支持向量机 LinearSVC
    4. 随机森林 RandomForest

每个模型都输出：验证集/测试集上的 准确率、精确率、召回率、F1、混淆矩阵，
并与成员 C 的 TextCNN 在同一测试集上对比（见 run_all.py）。

运行方式：python src/traditional_models.py
"""
import sys
import time
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.append(str(Path(__file__).resolve().parent))
from common import MODEL_DIR, RESULT_DIR, SEED, load_dataset, save_json  # noqa: E402

# 模型键名 -> (中文显示名, 分类器)
MODEL_ZOO = {
    "lg": ("逻辑回归", LogisticRegression(max_iter=1000, random_state=SEED)),
    "nb": ("朴素贝叶斯", MultinomialNB()),
    "svm": ("线性SVM", CalibratedClassifierCV(LinearSVC(random_state=SEED), cv=3)),
    "rf": ("随机森林", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
}


def build_tfidf() -> TfidfVectorizer:
    """构建 TF-IDF 特征提取器：词级 1-2 元文法，限制特征维度以控制内存。"""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
    )


def evaluate(pipeline, texts, labels) -> dict:
    """统一计算各项评价指标，返回指标字典（含混淆矩阵）。"""
    pred = pipeline.predict(texts)
    return {
        "accuracy": round(float(accuracy_score(labels, pred)), 4),
        "precision": round(float(precision_score(labels, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, pred, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(labels, pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(labels, pred).tolist(),
    }


def train_one(key: str, x_train, y_train, x_val, y_val, x_test, y_test) -> dict:
    """训练单个传统模型：TF-IDF + 分类器，含逻辑回归网格搜索调参。"""
    name, clf = MODEL_ZOO[key]
    print(f"\n===== 训练模型：{name} =====")
    t0 = time.time()

    if key == "lg":  # 逻辑回归用 3 折网格搜索选择最优正则强度 C
        pipe = Pipeline([("tfidf", build_tfidf()), ("clf", clf)])
        grid = GridSearchCV(pipe, {"clf__C": [0.5, 1.0, 2.0, 5.0]},
                            scoring="f1", cv=3, n_jobs=-1)
        grid.fit(x_train, y_train)
        pipeline, best_params = grid.best_estimator_, {"C": grid.best_params_["clf__C"]}
        print(f"网格搜索最优参数：{best_params}，验证集F1={grid.best_score_:.4f}")
    else:
        pipeline = Pipeline([("tfidf", build_tfidf()), ("clf", clf)])
        pipeline.fit(x_train, y_train)
        best_params = {}

    metrics = {
        "model": name,
        "key": key,
        "params": best_params,
        "val": evaluate(pipeline, x_val, y_val),
        "test": evaluate(pipeline, x_test, y_test),
        "train_seconds": round(time.time() - t0, 2),
    }
    joblib.dump(pipeline, MODEL_DIR / f"{key}.joblib")   # 保存模型供 GUI 调用
    print(f"验证集指标：{metrics['val']}")
    print(f"测试集指标：{metrics['test']}")
    print(f"用时 {metrics['train_seconds']} 秒，模型已保存至 models/{key}.joblib")
    return metrics


def main() -> None:
    df = load_dataset()
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    print(f"数据规模：训练 {len(train)} / 验证 {len(val)} / 测试 {len(test)}")

    results = {}
    for key in MODEL_ZOO:
        results[key] = train_one(
            key,
            train["text"].tolist(), train["label"].tolist(),
            val["text"].tolist(), val["label"].tolist(),
            test["text"].tolist(), test["label"].tolist(),
        )

    # 汇总对比表（按测试集 F1 排序）
    ranking = sorted(
        [(v["model"], v["test"]["accuracy"], v["test"]["f1"]) for v in results.values()],
        key=lambda x: -x[2],
    )
    print("\n===== 传统模型测试集对比（按F1降序）=====")
    for name, acc, f1 in ranking:
        print(f"{name:<8} 准确率={acc:.4f}  F1={f1:.4f}")

    save_json(results, RESULT_DIR / "traditional_metrics.json")
    print(f"\n[完成] 指标已保存至 {RESULT_DIR / 'traditional_metrics.json'}")


if __name__ == "__main__":
    main()
