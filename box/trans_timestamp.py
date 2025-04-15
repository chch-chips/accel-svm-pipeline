import pandas as pd

# 读取原始数据
df = pd.read_csv("../2025-04-07/raw_sensor_data.csv")

# 假设字段名称为 "timestamp"，且单位为毫秒。如果单位不是毫秒，请调整 unit 参数
# 先将时间戳转换为 datetime 类型，然后格式化为字符串（只保留前三位微秒即为毫秒）
df["timestamp_converted"] = pd.to_datetime(df["timestamp"], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]

# 可选：将原来的时间戳列重命名或保留
# 保存到新的文件中
df.to_csv("2025-04-07/raw_sensor_data_converted.csv", index=False)

print("转换完成，文件已保存为 raw_sensor_data_converted.csv")
