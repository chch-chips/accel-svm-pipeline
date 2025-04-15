import pandas as pd
import numpy as np

# 读取 feature_data.csv
feature_df = pd.read_csv("2025-04-01/feature_data.csv")
# 确保文件中包含 window_start 和 window_end 列（时间戳，单位需与你的 corrected_timestamp_marks.csv 保持一致）
# 如果不存在 label 列，则默认所有窗口为正常（1）
if 'label' not in feature_df.columns:
    feature_df['label'] = 1

# 读取经过校正的时间戳标记
corrected_df = pd.read_csv("2025-04-01/corrected_timestamp_marks.csv")
# 只保留有效的标记，即 valid==True 且 corrected_mark 非空
valid_marks = corrected_df[corrected_df['valid'] == True]['corrected_mark'].dropna().values
# 排序（如果还没有排序）
valid_marks = np.sort(valid_marks)

# 定义一个函数，根据窗口起始和结束时间，判断该窗口是否包含任意一个有效标记
def label_window(row, marks):
    # 如果 marks 中有任一时间戳处于窗口 [window_start, window_end] 内，则返回 -1；否则返回 1
    if np.any((marks >= row['window_start']) & (marks <= row['window_end'])):
        return -1
    else:
        return 1

# 应用该函数到每一行
feature_df['label'] = feature_df.apply(lambda row: label_window(row, valid_marks), axis=1)

# 保存更新后的 feature_data 到新的 CSV 文件中
feature_df.to_csv("feature_data_labeled.csv", index=False)
print("已将包含有效标记的窗口标记为 -1，并保存到 feature_data_labeled.csv")
