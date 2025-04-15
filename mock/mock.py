import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# 参数设置
# ======================
# 采样参数：模拟10,000个采样点，采样率为100 Hz（每个采样点间隔 10 毫秒）
num_samples = 10000
sampling_rate = 100  # Hz
time_interval = 1000.0 / sampling_rate  # 毫秒
timestamps = np.arange(0, num_samples) * time_interval  # 单位：毫秒

# 基线加速度：假设手机静止时，z轴主要显示重力 9.81 m/s²，其它轴为0加上少量噪声
np.random.seed(42)  # 固定随机种子，便于结果重现
acc_x = np.random.normal(0, 0.2, num_samples)
acc_y = np.random.normal(0, 0.2, num_samples)
acc_z = 9.81 + np.random.normal(0, 0.2, num_samples)

# ======================
# 模拟坑洼事件
# ======================
# 我们定义若干个坑洼事件位置，和事件持续的采样点数（例如持续0.2秒，即20个采样点）
pothole_events = [
    {"start_idx": 2000, "duration": 20},
    {"start_idx": 5000, "duration": 20},
    {"start_idx": 8000, "duration": 20}
]

for event in pothole_events:
    idx = event["start_idx"]
    duration = event["duration"]
    # 模拟冲击：短时内给 z 轴增加一个突出的正冲击（例如从5加到0的衰减）
    # 这里我们用一个线性衰减函数来模拟
    impulse = np.linspace(5, 0, duration)
    acc_z[idx:idx+duration] += impulse
    # 同时，x 和 y 轴也可以有一些随机扰动，模拟侧向振动
    acc_x[idx:idx+duration] += np.random.normal(0, 1, duration)
    acc_y[idx:idx+duration] += np.random.normal(0, 1, duration)

# ======================
# 可选：绘制数据以检验效果
# ======================
plt.figure(figsize=(14,6))
plt.plot(timestamps, acc_z, label="acc_z")
plt.plot(timestamps, 9.81 + np.zeros_like(timestamps), 'k--', label="基线 9.81")
for event in pothole_events:
    plt.axvspan(event["start_idx"] * time_interval,
                (event["start_idx"] + event["duration"]) * time_interval,
                color='red', alpha=0.3, label='坑洼事件' if event == pothole_events[0] else "")
plt.xlabel("时间 (ms)")
plt.ylabel("加速度 (m/s²)")
plt.title("模拟加速度数据 - z轴示例")
plt.legend()
plt.show()

# ======================
# 保存数据到 CSV 文件
# ======================
# 生成 DataFrame，并添加一个 sensor_type 字段（例如标记为 "Accelerometer"）
df = pd.DataFrame({
    "timestamp": timestamps,
    "sensor_type": "Accelerometer",
    "acc_x": acc_x,
    "acc_y": acc_y,
    "acc_z": acc_z
})
df.to_csv("simulated_accelerometer_data.csv", index=False)
print("模拟数据已保存到 simulated_accelerometer_data.csv")
