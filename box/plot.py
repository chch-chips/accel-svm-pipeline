#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import folium


def plot_gps_on_map(csv_path: str, output_html: str = "gps_map.html"):
    """
    从 csv 读取 timestamp, lat, lon 三列，
    在地图上打点并连成轨迹，保存为 HTML。
    """
    # 1. 载入数据
    df = pd.read_csv(csv_path)

    # 2. 提取唯一的 GPS 记录
    gps_df = df[['timestamp', 'lat', 'lon']].drop_duplicates().reset_index(drop=True)

    # 3. 转换时间戳为 datetime（可选，用于弹窗显示）
    gps_df['time'] = pd.to_datetime(gps_df['timestamp'], unit='ms')

    # 4. 计算中心点
    center = [gps_df['lat'].mean(), gps_df['lon'].mean()]

    # 5. 创建地图
    m = folium.Map(location=center, zoom_start=14, control_scale=True)

    # 6. 打点并加时间弹窗
    for _, row in gps_df.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=row['time'].strftime("%Y-%m-%d %H:%M:%S"),
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    # 7. 连成轨迹
    folium.PolyLine(
        locations=gps_df[['lat', 'lon']].values.tolist(),
        color='red', weight=3, opacity=0.7
    ).add_to(m)

    # 8. 保存 HTML
    m.save(output_html)
    print(f"📍 已生成地图：{output_html}")


if __name__ == "__main__":
    # --- 路径写死在这里，脚本运行时不会读取任何命令行参数 ---
    csv_path = "../4-17 17点/raw_sensor_data.csv"  # 原始 CSV 文件位置
    output_html = "4-17 17点.html"  # 生成的地图文件名

    plot_gps_on_map(csv_path, output_html)
