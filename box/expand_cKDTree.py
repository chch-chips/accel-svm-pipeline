import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# 1. 读取已标注的窗口数据
df = pd.read_csv("2025-04-01/feature_data_labeled.csv")
feature_cols = [
    "meanx", "meany", "meanz",
    "sdx", "sdy", "sdz",
    "meanxg", "meanyg", "meanzg",
    "sdxg", "sdyg", "sdzg"
]

# 2. 分离已标注为坑洼 (-1) 和正常 (1) 的窗口
pothole_df = df[df['label'] == -1]
normal_df = df[df['label'] == 1]

pothole_features = pothole_df[feature_cols].values
normal_features = normal_df[feature_cols].values

# 3. 使用 cKDTree 快速查找每个正常窗口与最近坑洼窗口之间的欧氏距离
tree = cKDTree(pothole_features)
distances, _ = tree.query(normal_features, k=1)

# 4. 绘制直方图
plt.hist(distances, bins=50)
plt.xlabel("最小欧氏距离")
plt.ylabel("窗口数")
plt.title("正常窗口与坑洼窗口最小距离分布")
plt.show()

# 5. 设定阈值，比如 0.5（根据直方图调整）
threshold = 0.5

# 6. 找出那些正常窗口中最近距离小于阈值的窗口
to_update_idx = normal_df[distances < threshold].index

# 7. 将这些窗口的 label 更新为 -1
df.loc[to_update_idx, "label"] = -1

# 8. 保存扩展标注后的数据
df.to_csv("feature_data_labeled_expanded.csv", index=False)
print(f"已更新 {len(to_update_idx)} 个窗口为坑洼事件。扩展后的数据保存到 feature_data_labeled_expanded.csv")
