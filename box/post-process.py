import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# 读取原始数据和筛选加速度数据（确保用 .copy() 避免警告）
raw_df = pd.read_csv("2025-04-01/raw_sensor_data.csv")
acc_df = raw_df[raw_df['sensor_type'].str.contains("Accelerometer", case=False)].copy()

# 计算加速度模值并平滑
acc_df['acc_mag'] = np.sqrt(acc_df['acc_x']**2 + acc_df['acc_y']**2 + acc_df['acc_z']**2)
acc_df['acc_mag_smooth'] = uniform_filter1d(acc_df['acc_mag'], size=5)

# 自动峰值检测
peaks, properties = find_peaks(acc_df['acc_mag_smooth'], height=12, distance=20)
print("检测到的峰值个数：", len(peaks))
peaks_timestamps = acc_df['timestamp'].iloc[peaks].values

# 加载手动标记数据
marks_df = pd.read_csv("2025-04-01/timestamp_marks.csv")
manual_marks = marks_df['timestamp'].values

# 设定最大允许的时间差（毫秒）
max_allowed_diff = 1000

# --- 向量化匹配 ---
# 计算手动标记与所有自动峰值的绝对时间差，生成一个 (n_manual, n_peaks) 的矩阵
diff_matrix = np.abs(manual_marks.reshape(-1, 1) - peaks_timestamps.reshape(1, -1))
# 对每个手动标记，找出最小差值对应的索引
min_indices = np.argmin(diff_matrix, axis=1)
min_diffs = diff_matrix[np.arange(diff_matrix.shape[0]), min_indices]
# 如果最小差值在阈值内，则校正为对应的自动峰值，否则标记为 NaN（误触）
corrected_marks = np.where(min_diffs <= max_allowed_diff, peaks_timestamps[min_indices], np.nan)
valid_flags = min_diffs <= max_allowed_diff

# 构造结果 DataFrame
results_df = pd.DataFrame({
    "manual_mark": manual_marks,
    "nearest_peak": peaks_timestamps[min_indices],
    "diff_ms": min_diffs,
    "corrected_mark": corrected_marks,
    "valid": valid_flags
})
results_df.to_csv("corrected_timestamp_marks.csv", index=False)
print("修正后的标记结果已保存到 corrected_timestamp_marks.csv")

# 如果绘图导致延时，也可以选择只绘制部分数据或关闭绘图
plt.figure(figsize=(14,6))
plt.plot(acc_df['timestamp'], acc_df['acc_mag_smooth'], label="加速度模值（平滑后）", color='blue')
plt.plot(acc_df['timestamp'].iloc[peaks], acc_df['acc_mag_smooth'].iloc[peaks], "rx", markersize=8, label="自动检测峰值")
for mark in manual_marks:
    plt.axvline(x=mark, color='green', linestyle='--', alpha=0.7)
plt.xlabel("时间戳")
plt.ylabel("加速度模值")
plt.title("加速度数据峰值检测与手动标记对比")
plt.legend()
plt.show()
