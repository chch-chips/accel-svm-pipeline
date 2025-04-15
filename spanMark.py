import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib
import matplotlib.pyplot as plt

# 设置支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------------------------
# 1. 从原始数据检测颠簸区间
# ---------------------------
# 读取原始加速度数据
raw_df = pd.read_csv("2025-04-09/raw_sensor_data.csv")
# 筛选加速度数据（假设 sensor_type 包含 "Accelerometer"）
acc_df = raw_df[raw_df['sensor_type'].str.contains("Accelerometer", case=False)].copy()

# 重置索引，确保索引连续
acc_df = acc_df.reset_index(drop=True)

# 计算加速度模值
acc_df['acc_mag'] = np.sqrt(acc_df['acc_x']**2 + acc_df['acc_y']**2 + acc_df['acc_z']**2)
# 平滑信号（窗口大小可调整）
acc_df['acc_mag_smooth'] = uniform_filter1d(acc_df['acc_mag'], size=5)

# 绘制平滑后的加速度模值（调试用）
plt.figure(figsize=(14,4))
plt.plot(acc_df['timestamp'], acc_df['acc_mag_smooth'], label="平滑后的加速度模值")
plt.xlabel("时间戳")
plt.ylabel("加速度模值")
plt.title("原始加速度数据平滑后的曲线")
plt.legend()
plt.show()

# 设定阈值，根据数据调整（示例阈值 12）
threshold = 12

# 标记颠簸点：判断 acc_mag_smooth 是否超过阈值
acc_df['is_bump'] = acc_df['acc_mag_smooth'] > threshold

# 提取连续的颠簸区间，记录起始和结束时间戳
bump_intervals = []
in_interval = False
start_time = None

# 使用重置后的索引，这样 idx 就是连续整数
for idx, row in acc_df.iterrows():
    if row['is_bump'] and not in_interval:
        in_interval = True
        start_time = row['timestamp']
    elif not row['is_bump'] and in_interval:
        # 使用 iloc 获取前一行的时间戳
        end_time = acc_df.iloc[idx-1]['timestamp']
        bump_intervals.append((start_time, end_time))
        in_interval = False

if in_interval:
    end_time = acc_df.iloc[-1]['timestamp']
    bump_intervals.append((start_time, end_time))

print("检测到的颠簸区间：")
for interval in bump_intervals:
    print(interval)

# ---------------------------
# 2. 根据颠簸区间标记窗口数据
# ---------------------------
# 读取窗口特征数据
feature_df = pd.read_csv("2025-04-09/feature_data.csv")
# 假设窗口数据包含 window_start, window_end 等字段
feature_df['label'] = 1  # 默认标记为正常

def mark_window(window_start, window_end, intervals):
    for (b_start, b_end) in intervals:
        # 如果窗口与颠簸区间有交集，则返回 -1
        if window_end >= b_start and window_start <= b_end:
            return -1
    return 1

feature_df['label'] = feature_df.apply(lambda row: mark_window(row['window_start'], row['window_end'], bump_intervals), axis=1)

# 输出最终标记为 -1 的窗口数量
num_pothole = (feature_df['label'] == -1).sum()
print("最终标记为坑洼 (-1) 的窗口数量：", num_pothole)

# 保存标记后的窗口数据
feature_df.to_csv("feature_data_labeled_span.csv", index=False)
print("已保存标记后的窗口数据到 feature_data_labeled_span.csv")
