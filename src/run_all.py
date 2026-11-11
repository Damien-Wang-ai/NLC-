# -*- coding: utf-8 -*-
"""
run_all.py —— 一键跑通全流程【负责人：成员 A（流程集成）】

依次执行：
    0. download_data.py 数据集下载（成员 A，已缓存则自动跳过）
    1. preprocess.py   数据预处理（成员 A）
    2. traditional.py  传统机器学习模型训练与对比（成员 B）
    3. textcnn.py      TextCNN 深度学习模型训练（成员 C）
训练完成后运行 gui.py 即可打开可视化界面（成员 D）。

运行方式：python src/run_all.py
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ["download_data.py", "preprocess.py", "traditional.py", "textcnn.py"]


def main() -> None:
    for step in STEPS:
        print(f"\n{'=' * 20} 执行 {step} {'=' * 20}")
        result = subprocess.run([sys.executable, str(SRC / step)], cwd=SRC.parent)
        if result.returncode != 0:
            print(f"[中断] {step} 执行失败（返回码 {result.returncode}）")
            sys.exit(result.returncode)
    print("\n全部流程执行完毕！运行 python src/gui.py 打开可视化界面。")


if __name__ == "__main__":
    main()
