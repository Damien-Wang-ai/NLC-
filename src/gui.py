# -*- coding: utf-8 -*-
"""
gui.py —— 可视化交互界面 + 模型性能对比图【负责人：成员 D（界面与可视化）】

功能：
    1. 检测页：输入一条新闻/微博文本 -> 选择模型 -> 输出“真实/虚假”判定与置信度；
    2. 对比页：读取 results/*.json，绘制各模型在测试集上的 准确率/F1 对比柱状图；
    3. 支持“全部模型”一键对比，直观展示成员 B、C 所建模型的优劣。

运行方式：python src/gui.py   （需先运行 preprocess.py / traditional.py / textcnn.py）
"""
import os
import sys
from pathlib import Path

# 在导入 PyQt5 之前，显式指定 Qt 平台插件路径，
# 避免因环境变量缺失导致 “no Qt platform plugin” 报错。
_QT_PLUGIN_DIR = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages" / "PyQt5" / "Qt5" / "plugins"
if _QT_PLUGIN_DIR.exists():
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(_QT_PLUGIN_DIR))

import json

import joblib
import matplotlib
import torch

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from PyQt5.QtCore import Qt                          # noqa: E402
from PyQt5.QtWidgets import (QApplication, QComboBox, QGridLayout, QGroupBox,  # noqa: E402
                             QHBoxLayout, QLabel, QProgressBar, QPushButton,
                             QTabWidget, QTextEdit, QVBoxLayout, QWidget)

sys.path.append(str(Path(__file__).resolve().parent))
from common import (LABEL_NAMES, MAX_LEN, MODEL_DIR, RESULT_DIR, load_json,  # noqa: E402
                    tokenize)
from textcnn import TextCNN, VOCAB_PATH, WEIGHT_PATH  # noqa: E402

# 中文字体设置，避免图表中文乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SKLEARN_MODELS = {"lg": "逻辑回归", "nb": "朴素贝叶斯", "svm": "线性SVM", "rf": "随机森林"}


def load_sklearn_model(key):
    """加载成员 B 保存的 TF-IDF + 传统分类器模型。"""
    path = MODEL_DIR / f"{key}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"缺少模型文件 {path.name}，请先运行 traditional.py")
    return joblib.load(path)


class TextCNNInfer:
    """封装成员 C 的 TextCNN 模型的推理接口，供界面调用。"""

    def __init__(self):
        if not (VOCAB_PATH.exists() and WEIGHT_PATH.exists()):
            raise FileNotFoundError("缺少 TextCNN 模型文件，请先运行 textcnn.py")
        cfg = load_json(VOCAB_PATH)
        self.word2idx = cfg["word2idx"]
        self.max_len = cfg.get("max_len", MAX_LEN)
        self.model = TextCNN(len(self.word2idx))
        self.model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, text):
        """返回模型对“虚假”类的预测概率。"""
        ids = [self.word2idx.get(w, 1) for w in tokenize(text).split()][:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        logits = self.model(torch.tensor([ids], dtype=torch.long))
        return float(torch.softmax(logits, dim=1)[0][1])


class ResultRow(QWidget):
    """单条检测结果：模型名 + 判定结论 + 置信度进度条。"""

    def __init__(self, model_name):
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.addWidget(QLabel(f"{model_name}："), 0, 0)
        self.verdict = QLabel("—")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        layout.addWidget(self.verdict, 0, 1)
        layout.addWidget(self.bar, 0, 2)
        layout.setColumnStretch(2, 1)

    def update_result(self, prob_fake):
        """prob_fake 为“虚假”类概率，>0.5 判定为虚假新闻。"""
        label = 1 if prob_fake >= 0.5 else 0
        conf = prob_fake if label == 1 else 1 - prob_fake
        color = "#d9534f" if label == 1 else "#5cb85c"
        self.verdict.setText(f"<b style='color:{color}'>{LABEL_NAMES[label]}</b>"
                             f"（置信度 {conf * 100:.1f}%）")
        self.bar.setValue(int(conf * 100))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("虚假新闻（谣言）检测系统 —— 自然语言处理课程设计")
        self.resize(940, 640)
        self.predictors = {}          # 延迟加载的模型缓存
        self.rows = {}                # 模型名 -> ResultRow

        tabs = QTabWidget(self)
        tabs.addTab(self.build_detect_tab(), "文本检测")
        tabs.addTab(self.build_compare_tab(), "模型性能对比")
        root = QVBoxLayout(self)
        root.addWidget(tabs)

    # ---------------- 检测页 ----------------
    def build_detect_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("请输入待检测的新闻或微博文本："))
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("例如：网传某地明天将发生大地震，请转发周知……")
        self.input_box.setPlainText("专家表示，喝高度白酒可以有效杀灭体内新型冠状病毒。")
        layout.addWidget(self.input_box, 2)

        top = QHBoxLayout()
        top.addWidget(QLabel("选择模型："))
        self.model_box = QComboBox()
        self.model_box.addItem("全部模型（对比）", "all")
        for key, name in SKLEARN_MODELS.items():
            self.model_box.addItem(name, key)
        self.model_box.addItem("TextCNN（深度学习）", "textcnn")
        top.addWidget(self.model_box)
        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.clicked.connect(self.on_detect)
        top.addWidget(self.detect_btn)
        top.addStretch(1)
        layout.addLayout(top)

        self.result_group = QGroupBox("检测结果")
        self.result_layout = QVBoxLayout(self.result_group)
        self.hint = QLabel("点击“开始检测”查看结果")
        self.result_layout.addWidget(self.hint)
        layout.addWidget(self.result_group, 1)
        return page

    def get_predictor(self, key):
        """按需加载并缓存模型，避免界面启动时加载全部模型。"""
        if key not in self.predictors:
            self.predictors[key] = TextCNNInfer() if key == "textcnn" else load_sklearn_model(key)
        return self.predictors[key]

    def on_detect(self):
        text = self.input_box.toPlainText().strip()
        if not text:
            self.hint.setText("请先输入文本内容！")
            return
        keys = (list(SKLEARN_MODELS) + ["textcnn"]) if self.model_box.currentData() == "all" \
            else [self.model_box.currentData()]
        # 清空旧结果
        for row in self.rows.values():
            self.result_layout.removeWidget(row)
            row.deleteLater()
        self.rows = {}
        self.hint.setText(f"共检测 {len(keys)} 个模型" if len(keys) > 1 else "")

        tokenized = tokenize(text)
        for key in keys:
            name = SKLEARN_MODELS.get(key, "TextCNN")
            try:
                predictor = self.get_predictor(key)
                # 传统模型：Pipeline 内置 TF-IDF，输入需为分词后的空格串
                prob = predictor.predict_proba([tokenized])[0][1] if key != "textcnn" \
                    else predictor.predict_proba(text)
            except FileNotFoundError as err:
                self.hint.setText(str(err))
                continue
            row = ResultRow(name)
            row.update_result(prob)
            self.result_layout.addWidget(row)
            self.rows[name] = row

    # ---------------- 对比页 ----------------
    def build_compare_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.figure = plt.figure(figsize=(8, 4.6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        btn = QPushButton("刷新模型对比图")
        btn.clicked.connect(self.draw_compare)
        layout.addWidget(btn)
        self.draw_compare()
        return page

    def collect_metrics(self):
        """汇总成员 B（传统模型）与成员 C（TextCNN）保存的测试集指标。"""
        data = {}
        trad_path = RESULT_DIR / "traditional_metrics.json"
        if trad_path.exists():
            for item in json.loads(trad_path.read_text(encoding="utf-8")).values():
                data[item["model"]] = item["test"]
        cnn_path = RESULT_DIR / "textcnn_metrics.json"
        if cnn_path.exists():
            item = json.loads(cnn_path.read_text(encoding="utf-8"))
            data[item["model"]] = item["test"]
        return data

    def draw_compare(self):
        data = self.collect_metrics()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if not data:
            ax.text(0.5, 0.5, "暂无实验结果\n请先运行 traditional.py 与 textcnn.py",
                    ha="center", va="center", fontsize=13)
        else:
            names = list(data)
            acc = [data[n]["accuracy"] * 100 for n in names]
            f1 = [data[n]["f1"] * 100 for n in names]
            x = range(len(names))
            ax.bar([i - 0.2 for i in x], acc, width=0.4, label="准确率(%)", color="#4a90d9")
            ax.bar([i + 0.2 for i in x], f1, width=0.4, label="F1(%)", color="#f0ad4e")
            ax.set_xticks(list(x))
            ax.set_xticklabels(names, rotation=12)
            ax.set_ylim(0, 100)
            ax.set_title("各模型在测试集上的性能对比")
            ax.legend()
            for i, (a, f) in enumerate(zip(acc, f1)):
                ax.text(i - 0.2, a + 1, f"{a:.1f}", ha="center", fontsize=8)
                ax.text(i + 0.2, f + 1, f"{f:.1f}", ha="center", fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
