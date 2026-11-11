# -*- coding: utf-8 -*-
"""
download_data.py —— 数据集下载脚本【负责人：成员 A（数据与预处理）】

数据来源：清华大学 THUNLP 公开的中文谣言数据集 CED（Song et al., 2018,
         “CED: Credible Early Detection of Social Media Rumors”），
         包含 1538 条谣言微博与 1849 条非谣言微博的原文。

由于 GitHub 在国内访问受限，脚本通过公共加速代理拉取原始文件：
    1. 通过 ungh.cc 获取仓库文件清单（用于确定每条微博的类别标签）；
    2. 多线程下载 original-microblog 目录下的微博原文（约 3387 条）；
    3. 汇总为 data/raw/ced_dataset.jsonl，格式：{"id":..,"text":..,"label":0/1}

运行方式：python src/download_data.py
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from common import RAW_DIR  # noqa: E402

REPO = "thunlp/Chinese_Rumor_Dataset"
BRANCH = "master"
PROXY = "https://gh-proxy.com/https://raw.githubusercontent.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}          # 代理要求带 UA
WORKERS = 32                                     # 并发线程数

ORIGIN_DIR = RAW_DIR / "ced_original"            # 微博原文缓存目录
LABEL_FILE = RAW_DIR / "ced_labels.json"         # id -> 标签
OUT_FILE = RAW_DIR / "ced_dataset.jsonl"         # 汇总数据集


def fetch(url: str, timeout: int = 40) -> bytes:
    """带重试的 HTTP GET，返回响应字节内容。"""
    last_err = None
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as err:                 # 网络抖动时重试
            last_err = err
    raise RuntimeError(f"下载失败：{url} -> {last_err}")


def build_label_map() -> dict:
    """根据 rumor-repost / non-rumor-repost 目录的文件名确定每条微博的标签。"""
    if LABEL_FILE.exists():
        print(f"[缓存] 直接读取标签文件 {LABEL_FILE}")
        return json.loads(LABEL_FILE.read_text(encoding="utf-8"))

    print("[1/3] 获取仓库文件清单 ...")
    data = json.loads(fetch(f"https://ungh.cc/repos/{REPO}/files/{BRANCH}"))
    labels = {}
    for item in data["files"]:
        path = item["path"]
        name = Path(path).name
        if name.startswith(".") or not name.endswith(".json"):
            continue
        if path.startswith("CED_Dataset/rumor-repost/"):
            labels[name] = 1            # 1 = 谣言
        elif path.startswith("CED_Dataset/non-rumor-repost/"):
            labels[name] = 0            # 0 = 非谣言
    LABEL_FILE.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    print(f"      标签总数 {len(labels)}（谣言 {sum(labels.values())}，"
          f"非谣言 {len(labels) - sum(labels.values())}）")
    return labels


def download_one(name: str) -> Path:
    """下载单条微博原文，已存在则跳过（支持断点续传）。"""
    out = ORIGIN_DIR / name
    if out.exists() and out.stat().st_size > 0:
        return out
    out.write_bytes(fetch(f"{PROXY}/{REPO}/{BRANCH}/CED_Dataset/original-microblog/{name}"))
    return out


def main() -> None:
    ORIGIN_DIR.mkdir(parents=True, exist_ok=True)
    labels = build_label_map()
    todo = sorted(labels)
    print(f"[2/3] 开始下载 {len(todo)} 条微博原文（并发 {WORKERS}）...")

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, name): name for name in todo}
        for fut in as_completed(futures):
            done += 1
            if fut.exception():
                print(f"      [失败] {futures[fut]}: {fut.exception()}")
            if done % 300 == 0:
                print(f"      进度 {done}/{len(todo)}")

    print("[3/3] 汇总数据集 ...")
    records = []
    for name, label in labels.items():
        fp = ORIGIN_DIR / name
        if not fp.exists():
            continue
        try:
            text = json.loads(fp.read_text(encoding="utf-8")).get("text") or ""
        except json.JSONDecodeError:
            continue
        if text.strip():
            records.append({"id": name[:-5], "text": text, "label": label})
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[完成] 共 {len(records)} 条，已保存至 {OUT_FILE}")


if __name__ == "__main__":
    main()
