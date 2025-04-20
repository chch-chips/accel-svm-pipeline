#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import csv
from collections import Counter

# 固定文件路径
TIMESTAMP_FILE = "4-17 2点/timestamp_marks.csv"         # 手动标记时间戳
ANNOTATION_DIR = "4-17 2点/Annotations"                # 目标检测标注 txt 文件夹
FEATURE_FILE    = "4-17 2点/feature_data.csv"          # SVM 特征数据文件
DELTA_MS        = 1000                                  # ±delta 匹配容差（毫秒）

# 类别映射（按序号）
LABEL_NAMES = [
    "Crack", "Manhole", "Net", "Pothole",
    "Patch-Crack", "Patch-Net", "Patch-Pothole",
    "other", "Other"
]


def load_ground_truth(ts_file):
    """读取手动标记的真实事件时间戳（毫秒）"""
    gts = []
    with open(ts_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            gts.append(int(row[0]))
    return sorted(gts)


def load_vision_detections(annotation_dir):
    """
    读取目标检测中各类别的检测时间，并统计各类别总次数
    返回：
      dets: 仅类别3的检测时间列表
      counts: Counter，各类别总计数
    """
    dets = []
    counts = Counter()
    for path in glob.glob(os.path.join(annotation_dir, '*.txt')):
        fname = os.path.basename(path)
        ts_str, _ = os.path.splitext(fname)
        try:
            ts = int(ts_str)
        except ValueError:
            continue
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                label = int(parts[0])
                counts[label] += 1
                if label == 3:  # 只收集 Pothole 的时间戳
                    dets.append(ts)
    return sorted(dets), counts


def load_svm_detections(feature_file):
    """读取并聚合 SVM 加速度检测中连续的 -1 窗口为单一事件"""
    raw_dets = []
    with open(feature_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(float(row['label'])) == -1:
                raw_dets.append(int(float(row['window_end'])))
    # 聚合连续的 -1 事件：若两次检测时间间隔 <= DELTA_MS 则视为同一次事件
    collapsed = []
    for t in sorted(raw_dets):
        if not collapsed or t - collapsed[-1] > DELTA_MS:
            collapsed.append(t)
    return collapsed


def evaluate(dets, gts, delta_ms):
    """
    计算 TP, FP, FN
    贪心匹配：每个检测事件匹配一个真实事件，若在 ±delta_ms 范围内则为 TP
    """
    matched = set()
    tp = 0
    for d in dets:
        candidates = [(abs(d - gt), i) for i, gt in enumerate(gts) if i not in matched]
        if not candidates:
            continue
        dist, idx = min(candidates, key=lambda x: x[0])
        if dist <= delta_ms:
            tp += 1
            matched.add(idx)
    fp = len(dets) - tp
    fn = len(gts) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return tp, fp, fn, precision, recall, f1


def main():
    # 加载数据
    gts         = load_ground_truth(TIMESTAMP_FILE)
    vision, vc  = load_vision_detections(ANNOTATION_DIR)
    svm         = load_svm_detections(FEATURE_FILE)
    combo       = sorted(set(vision) | set(svm))

    # 输出目标检测各类别统计
    print("=== 目标检测各类别统计 ===")
    total = sum(vc.values())
    for label, cnt in sorted(vc.items()):
        name = LABEL_NAMES[label] if 0 <= label < len(LABEL_NAMES) else f"label_{label}"
        print(f"{name:<15} ({label}): {cnt}")
    print(f"总检测框数: {total}\n")

    # 评估
    print("=== 1. 仅目标检测（Pothole） ===")
    tp, fp, fn, p, r, f1 = evaluate(vision, gts, DELTA_MS)
    print(f"TP={tp}, FP={fp}, FN={fn}, Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}\n")

    print("=== 2. 仅SVM加速度检测（聚合连续事件） ===")
    tp, fp, fn, p, r, f1 = evaluate(svm, gts, DELTA_MS)
    print(f"事件数={len(svm)}, TP={tp}, FP={fp}, FN={fn}, Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}\n")

    print("=== 3. 目标检测 + SVM 结合 ===")
    tp, fp, fn, p, r, f1 = evaluate(combo, gts, DELTA_MS)
    print(f"TP={tp}, FP={fp}, FN={fn}, Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")

if __name__ == '__main__':
    main()
