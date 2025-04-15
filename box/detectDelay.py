import pandas as pd

# 读取 feature_data.csv
df = pd.read_csv("../2025-04-07/feature_data.csv")

# 假设文件中有 window_start 和 window_end 列，单位为毫秒
# 计算窗口间隔（毫秒）
df['interval_ms'] = df['window_end'] - df['window_start']

# 如果间隔大于或等于2秒（2000毫秒），则标记为 True，否则为 False
df['long_interval'] = df['interval_ms'] >= 2000

# 保存结果到新文件中
df.to_csv("feature_data_marked.csv", index=False)
print("已将窗口间隔>=2秒的窗口标记，并保存到 feature_data_marked.csv")
