import os
import datetime
from pytz import timezone

# 设置杭州时区
hangzhou_tz = timezone('Asia/Shanghai')

# 获取文件夹路径
folder_path = '../4-19 17点/RawImages'  # 请替换成你实际的文件夹路径

# 获取所有图片文件
for filename in os.listdir(folder_path):
    if filename.endswith('.bmp'):
        # 提取文件名中的时间部分
        try:
            time_str = filename.split('.')[0]  # 获取格式类似 '17-00-51.295' 或 '16-26-24.749' 的部分

            # 检查是否包含毫秒部分
            if '.' in time_str:
                # 如果有毫秒部分，确保它是六位数
                time_str = time_str[:8] + '.' + time_str[9:].ljust(6, '0')  # 填充毫秒部分至六位

                time_obj = datetime.datetime.strptime(time_str, "%H-%M-%S.%f")  # 解析带有六位毫秒的时间
            else:
                time_obj = datetime.datetime.strptime(time_str, "%H-%M-%S")  # 解析没有毫秒部分的时间

            # 为避免时区问题，设置为杭州时区
            time_obj = hangzhou_tz.localize(time_obj)

            # 转换为UTC时间
            utc_time = time_obj.astimezone(timezone('UTC'))

            # 获取时间戳
            timestamp = int(utc_time.timestamp())

            # 生成新的文件名
            new_filename = f"{timestamp}.bmp"

            # 拼接完整的文件路径
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)

            # 重命名文件
            os.rename(old_file, new_file)
            print(f"Renamed: {filename} -> {new_filename}")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
