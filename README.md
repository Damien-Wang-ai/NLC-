# 虚假新闻（谣言）检测系统 —— 自然语言处理课程设计



## 一、项目简介

本系统以**中文谣言数据集 CED**（清华大学 THUNLP 团队公开数据，
Song et al., 2018）为基础，使用 **Python** 实现虚假新闻（谣言）检测：
输入一条新闻/微博文本，系统输出其"真实 / 虚假"的判定结果与置信度，
并可对比多种模型的性能。

- 传统机器学习：TF-IDF + 逻辑回归 / 朴素贝叶斯 / 线性SVM / 随机森林
- 深度学习：TextCNN（词嵌入 + 多尺度卷积 + 最大池化）
- 可视化交互界面：PyQt5 图形界面（文本检测 + 模型性能对比图）

## 二、四人分工说明

| 成员 | 负责模块 | 对应文件 | 主要工作内容 |
| --- | --- | --- | --- |
| 成员 A | 数据与预处理 | `src/common.py`、`src/download_data.py`、`src/preprocess.py`、`src/run_all.py` | 编写数据下载脚本（多线程 + 断点续传）获取 CED 中文谣言数据；文本清洗（去 URL、@用户、话题、表情）；jieba 分词；去重与长度过滤；按 7:1:2 分层划分训练/验证/测试集；输出 `data/dataset.csv` 与数据统计；维护公共接口供其他成员调用 |
| 成员 B | 传统机器学习模型 | `src/traditional.py` | 构建 TF-IDF（1-2 元文法）特征；实现逻辑回归（网格搜索调参 C）、朴素贝叶斯、线性SVM、随机森林四个分类器；在验证集/测试集上计算准确率、精确率、召回率、F1 与混淆矩阵；保存模型与 `results/traditional_metrics.json` |
| 成员 C | 深度学习模型 | `src/textcnn.py` | 实现 TextCNN（Embedding → Conv1d 多尺度 2/3/4 → ReLU → 全局最大池化 → Dropout → 全连接）；构建词表；Adam 优化 + 早停训练；保存 `models/textcnn.pt` 与 `results/textcnn_metrics.json`；绘制模型结构说明对应说明书插图 |
| 成员 D | 界面与可视化 | `src/gui.py` | 基于 PyQt5 实现交互界面：文本输入、模型选择（单选/全部对比）、真假判定与置信度进度条；读取实验结果绘制各模型准确率/F1 对比柱状图；负责界面的调试与展示 |

> 协作接口约定：数据集统一为 `data/dataset.csv`（列：`raw_text, clean_text, text, label, split`，
> 其中 `text` 为分词后的空格串，`label` 中 0=真实、1=虚假）；模型统一保存到 `models/`，
> 指标统一保存到 `results/*.json`（结构见下文），四人按此接口并行开发、互不阻塞。

## 三、目录结构

```
自然语言处理课设/
├── data/
│   ├── raw/                  # 原始数据集（CED 微博原文 + 汇总 jsonl）
│   └── dataset.csv           # 预处理后数据集（成员产出）
├── models/                   # 训练得到的模型文件
│   ├── lg.joblib / nb.joblib / svm.joblib / rf.joblib   # 成员
│   └── textcnn.pt / textcnn_vocab.json                  # 成员
├── results/                  # 实验结果
│   ├── dataset_stats.json            # 数据统计（成员）
│   ├── traditional_metrics.json      # 传统模型指标（成员）
│   └── textcnn_metrics.json          # TextCNN 指标（成员）
├── src/
│   ├── common.py             # 公共模块（路径、清洗、分词、IO）
│   ├── download_data.py      # 数据集下载          —— 成员
│   ├── preprocess.py         # 数据预处理          —— 成员
│   ├── traditional.py        # 传统机器学习模型     —— 成员
│   ├── textcnn.py            # TextCNN 深度学习模型 —— 成员
│   ├── gui.py                # PyQt5 可视化界面     —— 成员
│   └── run_all.py            # 一键训练全流程
└── README.md
```

## 四、运行方式

```powershell
# 1. 创建并激活虚拟环境（已创建可跳过）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install numpy pandas scikit-learn jieba pyarrow PyQt5 matplotlib torch

# 3. 一键完成：数据预处理 + 传统模型训练 + TextCNN 训练
python src/run_all.py

# 4. 打开可视化交互界面
python src/gui.py
```

分模块单独运行：

```powershell
python src/download_data.py       # 下载 CED 数据集（已缓存则跳过）
python src/preprocess.py          # 生成 data/dataset.csv
python src/traditional.py         # 训练 4 个传统模型
python src/textcnn.py             # 训练 TextCNN
```

## 五、数据准备

数据为清华大学 THUNLP 公开的中文谣言数据集 **CED**（Weibo 谣言/非谣言微博原文），
共 3386 条原始微博，清洗去重后 3223 条用于实验。
由于 GitHub 访问受限，`download_data.py` 自动通过公共加速代理（gh-proxy.com）
多线程抓取原始文件，并缓存到 `data/raw/ced_original/`，支持断点续传：

- 类别标签由 `CED_Dataset/rumor-repost/` 与 `non-rumor-repost/` 的文件名确定
  （1=谣言 1534 条，0=非谣言 1848 条）；
- 微博原文取自 `CED_Dataset/original-microblog/`。

下载完成后会自动汇总为 `data/raw/ced_dataset.jsonl`，供 `preprocess.py` 使用。

## 六、实验结果（测试集）

| 模型 | 准确率 | 精确率 | 召回率 | F1 |
| --- | --- | --- | --- | --- |
| 线性SVM | 0.8822 | 0.8659 | 0.8597 | 0.8628 |
| 逻辑回归 | 0.8806 | 0.8972 | 0.8165 | 0.8550 |
| 随机森林 | 0.8217 | 0.7900 | 0.7986 | 0.7943 |
| 朴素贝叶斯 | 0.8217 | 0.9657 | 0.6079 | 0.7461 |
| TextCNN | 0.8388 | 0.8884 | 0.7158 | 0.7928 |

说明：样本量较小（训练 2255 条）且词向量随机初始化，
TextCNN 未使用预训练词向量，因此略低于 TF-IDF + 线性SVM。
