import os
import pandas as pd


# 1. 时间戳转换为杭州时区时间格式的函数
def timestamp_to_shanghai_time(timestamp):
    """将时间戳转换为杭州时区时间（精确到毫秒）"""
    return pd.to_datetime(timestamp, unit='ms').tz_localize('UTC').tz_convert('Asia/Shanghai').strftime('%H-%M-%S.%f')[
           :-3]  # 替换冒号为破折号


# 2. 遍历指定文件夹，处理文件名为时间戳的 BMP 文件并重命名
def rename_bmp_files(folder_path):
    """遍历文件夹，将文件名中的时间戳转换为杭州时区时间格式"""
    for filename in os.listdir(folder_path):
        if filename.endswith(".bmp"):
            # 提取时间戳部分
            timestamp_str = filename.split('.')[0]
            if timestamp_str.isdigit():  # 确保是时间戳
                timestamp = int(timestamp_str)
                shanghai_time = timestamp_to_shanghai_time(timestamp)

                # 构造新文件名，避免冒号
                new_filename = f"{shanghai_time}.bmp"
                old_file_path = os.path.join(folder_path, filename)
                new_file_path = os.path.join(folder_path, new_filename)

                try:
                    # 重命名文件
                    os.rename(old_file_path, new_file_path)
                    print(f"文件 {filename} 已重命名为 {new_filename}")
                except OSError as e:
                    print(f"重命名失败：{e}")

# 测试函数
folder_path = '../4-19 17点/AnnotatedImages'  # 请替换为您的文件夹路径
rename_bmp_files(folder_path)
