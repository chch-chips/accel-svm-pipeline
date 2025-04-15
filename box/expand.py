import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# 1. 读取已标注的窗口数据，假设文件 feature_data_labeled.csv 中已有部分窗口标记为 -1，其余为 1
df = pd.read_csv("2025-04-01/feature_data_labeled.csv")
# 我们的特征列，需确保顺序与训练时一致
feature_cols = [
    "meanx", "meany", "meanz",
    "sdx", "sdy", "sdz",
    "meanxg", "meanyg", "meanzg",
    "sdxg", "sdyg", "sdzg"
]

# 2. 提取当前标记为坑洼的窗口（正样本）以及正常窗口
pothole_df = df[df['label'] == -1]
normal_df = df[df['label'] == 1]

# 提取特征矩阵
pothole_features = pothole_df[feature_cols].values
normal_features = normal_df[feature_cols].values

# 3. 计算每个正常窗口与所有坑洼窗口之间的欧氏距离
# 结果矩阵 shape = (n_normal, n_pothole)
dist_matrix = cdist(normal_features, pothole_features, metric="euclidean")
# 对每个正常窗口，找出与最近坑洼窗口的最小距离
min_dists = dist_matrix.min(axis=1)

# 4. 选择一个阈值：这里的阈值需要你根据实际数据分布来确定，
# 例如可以画出所有正常窗口与坑洼窗口最小距离的直方图进行分析
plt.hist(min_dists, bins=50)
plt.xlabel("最小欧氏距离")
plt.ylabel("窗口数")
plt.title("正常窗口与坑洼窗口最小距离分布")
plt.show()

# 假设根据观察选定一个阈值（例如 0.5，单位与特征量纲有关）
threshold = 0.55

# 5. 找出哪些正常窗口的最小距离低于阈值
to_update_idx = normal_df[min_dists < threshold].index

# 对这些窗口，将 label 更新为 -1
df.loc[to_update_idx, "label"] = -1

# 6. 保存扩展标注后的数据
df.to_csv("feature_data_labeled_expanded.csv", index=False)
print(f"已更新 {len(to_update_idx)} 个窗口为坑洼事件。扩展后的数据保存到 feature_data_labeled_expanded.csv")
