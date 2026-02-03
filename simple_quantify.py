#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Western Blot Quantifier
ImageJ不要のシンプル版。画像処理のみで定量化。
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage


def parse_args():
    parser = argparse.ArgumentParser(description='Simple Western Blot Quantifier')
    parser.add_argument('-i', '--image', required=True, help='画像ファイル')
    parser.add_argument('-l', '--lanes', type=int, required=True, help='レーン数')
    parser.add_argument('-o', '--output', default=None, help='出力ディレクトリ')
    parser.add_argument('--band-height', type=int, default=15, help='バンド領域の高さ(px)')
    parser.add_argument('--sigma', type=float, default=2.0, help='平滑化の強度')
    return parser.parse_args()


def quantify(image_path, num_lanes, band_height=15, sigma=2.0):
    """画像を定量化"""
    # 画像読み込み
    img = Image.open(image_path).convert('L')
    img_array = 255 - np.array(img)  # 反転（暗いバンド→高い値）
    
    height, width = img_array.shape
    lane_width = width // num_lanes
    
    results = []
    
    for i in range(num_lanes):
        # レーン領域
        start_x = i * lane_width
        end_x = (i + 1) * lane_width
        lane_data = img_array[:, start_x:end_x]
        
        # プロファイル
        profile = np.mean(lane_data, axis=1)
        smoothed = ndimage.gaussian_filter1d(profile, sigma=sigma)
        
        # バンド位置（最大値）
        band_y = np.argmax(smoothed)
        band_y_start = max(0, band_y - band_height // 2)
        band_y_end = min(height, band_y + band_height // 2)
        
        # バンド領域
        band_region = img_array[band_y_start:band_y_end, start_x:end_x]
        
        # バックグラウンド
        bg = np.percentile(lane_data, 10)
        
        # 定量値
        corrected = np.maximum(band_region - bg, 0)
        volume = np.sum(corrected)
        
        results.append({
            'Lane': i + 1,
            'Band_Y': band_y,
            'Mean': np.mean(band_region),
            'Background': bg,
            'Volume': volume
        })
    
    df = pd.DataFrame(results)
    df['Relative_%'] = (df['Volume'] / df['Volume'].max() * 100).round(2)
    
    return df, img_array


def save_results(df, img_array, output_dir, prefix, num_lanes):
    """結果を保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV
    csv_path = os.path.join(output_dir, f'{prefix}_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"💾 {csv_path}")
    
    # グラフ
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 画像 + レーン分割
    height, width = img_array.shape
    lane_width = width // num_lanes
    
    axes[0].imshow(255 - img_array, cmap='gray', aspect='auto')
    for i in range(num_lanes + 1):
        axes[0].axvline(x=i * lane_width, color='red', linestyle='--', alpha=0.5)
    for i, row in df.iterrows():
        center_x = (i + 0.5) * lane_width
        axes[0].plot(center_x, row['Band_Y'], 'ro', markersize=8)
        axes[0].text(center_x, row['Band_Y'] - 5, str(row['Lane']), 
                    color='red', ha='center', fontweight='bold',
                    bbox=dict(facecolor='yellow', alpha=0.7, boxstyle='round'))
    axes[0].set_title('Western Blot with Lane Detection', fontweight='bold')
    
    # 棒グラフ
    colors = plt.cm.plasma(df['Relative_%'] / 100)
    axes[1].bar(df['Lane'], df['Relative_%'], color=colors, edgecolor='black')
    axes[1].set_title('Relative Band Intensity (%)', fontweight='bold')
    axes[1].set_xlabel('Lane')
    axes[1].set_ylabel('Relative %')
    axes[1].set_ylim(0, 110)
    axes[1].axhline(y=100, color='red', linestyle='--', alpha=0.5)
    
    for _, row in df.iterrows():
        axes[1].text(row['Lane'], row['Relative_%'] + 2, f"{row['Relative_%']:.1f}%",
                    ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{prefix}_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 {plot_path}")
    plt.close()


def main():
    args = parse_args()
    
    if args.output is None:
        args.output = os.path.dirname(os.path.abspath(args.image))
    
    prefix = Path(args.image).stem
    
    print("="*50)
    print("Simple Western Blot Quantifier")
    print("="*50)
    print(f"画像: {args.image}")
    print(f"レーン数: {args.lanes}")
    
    df, img_array = quantify(
        args.image, 
        args.lanes, 
        band_height=args.band_height,
        sigma=args.sigma
    )
    
    save_results(df, img_array, args.output, prefix, args.lanes)
    
    print("\n" + "="*50)
    print("結果")
    print("="*50)
    print(df[['Lane', 'Volume', 'Relative_%']].to_string(index=False))
    print("\n✅ 完了！")


if __name__ == "__main__":
    main()
