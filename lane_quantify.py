#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Western Blot Quantifier v4.0
レーンベース検出: 画像をレーンごとに分割し、各レーン内でバンドを検出

使い方:
    python3 lane_quantify.py -i image.png -l 12
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Western Blot Quantifier v4.0 (Lane-based)')
    parser.add_argument('-i', '--image', required=True, help='画像ファイル')
    parser.add_argument('-l', '--lanes', type=int, required=True, help='レーン数')
    parser.add_argument('-o', '--output', default=None, help='出力ディレクトリ')
    parser.add_argument('--exclude-last', action='store_true', help='最後のレーン（マーカー）を除外')
    return parser.parse_args()


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def detect_band_in_lane(lane_gray):
    """レーン内のバンドを検出して測定"""
    
    h, w = lane_gray.shape
    
    # ノイズ除去
    denoised = cv2.bilateralFilter(lane_gray, 5, 50, 50)
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    
    # Otsu's threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # モルフォロジー
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # バンドが検出されない場合、レーン全体を使用
        return None, binary
    
    # 最大の輪郭をバンドとする
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 面積が小さすぎる場合は無効
    if cv2.contourArea(largest_contour) < h * w * 0.01:
        return None, binary
    
    return largest_contour, binary


def measure_lane(lane_gray, contour=None):
    """レーンの強度を測定"""
    
    h, w = lane_gray.shape
    
    # 背景推定（レーンの上下端）
    bg_top = lane_gray[:max(1, int(h*0.1)), :].flatten()
    bg_bottom = lane_gray[int(h*0.9):, :].flatten()
    bg_intensity = np.median(np.concatenate([bg_top, bg_bottom]))
    
    if contour is not None:
        # 輪郭がある場合はマスクを使用
        mask = np.zeros(lane_gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        band_pixels = lane_gray[mask == 255]
        
        # バウンディングボックス
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # 重心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cy = int(M["m01"] / M["m00"])
        else:
            cy = y + bh // 2
    else:
        # 輪郭がない場合は中央領域を使用
        band_region = lane_gray[int(h*0.2):int(h*0.8), :]
        band_pixels = band_region.flatten()
        cy = h // 2
        y, bh = int(h*0.2), int(h*0.6)
    
    if len(band_pixels) == 0:
        return 0, 0, cy, 0
    
    # 強度計算（暗い = 高シグナル）
    inverted = 255 - band_pixels.astype(np.float64)
    bg_corrected_value = 255 - bg_intensity
    corrected = np.maximum(inverted - bg_corrected_value * 0.7, 0)
    
    volume = np.sum(corrected)
    mean_intensity = np.mean(corrected)
    
    return volume, mean_intensity, cy, len(band_pixels)


def process_image(img, gray, num_lanes, exclude_last=False):
    """画像をレーンごとに処理"""
    
    h, w = gray.shape
    lane_width = w // num_lanes
    
    results = []
    lane_data = []
    
    total_lanes = num_lanes - 1 if exclude_last else num_lanes
    
    for i in range(total_lanes):
        # レーン領域を切り出し
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        
        lane_gray = gray[:, x_start:x_end]
        lane_img = img[:, x_start:x_end]
        
        # バンド検出
        contour, binary = detect_band_in_lane(lane_gray)
        
        # 測定
        volume, mean_int, cy, area = measure_lane(lane_gray, contour)
        
        # 輪郭を元画像の座標系に変換
        if contour is not None:
            contour_global = contour.copy()
            contour_global[:, :, 0] += x_start
        else:
            contour_global = None
        
        results.append({
            'Lane': i + 1,
            'X_start': x_start,
            'X_end': x_end,
            'Y_center': cy,
            'Area': area,
            'Mean': round(mean_int, 2),
            'Volume': round(volume, 0)
        })
        
        lane_data.append({
            'contour': contour_global,
            'binary': binary,
            'x_start': x_start,
            'x_end': x_end
        })
    
    return results, lane_data


def create_visualization(img, gray, results, lane_data, output_path, num_lanes):
    """可視化"""
    
    df = pd.DataFrame(results)
    max_volume = df['Volume'].max()
    df['Relative_%'] = (df['Volume'] / max_volume * 100).round(2) if max_volume > 0 else 0
    
    h, w = gray.shape
    lane_width = w // num_lanes
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. 元画像 + レーン分割 + バンド検出
    overlay = img.copy()
    
    # レーン境界線
    for i in range(num_lanes + 1):
        x = i * lane_width
        cv2.line(overlay, (x, 0), (x, h), (255, 0, 0), 1)
    
    # バンド輪郭
    for i, ld in enumerate(lane_data):
        if ld['contour'] is not None:
            cv2.drawContours(overlay, [ld['contour']], -1, (0, 255, 0), 2)
        
        # レーン番号
        cx = (ld['x_start'] + ld['x_end']) // 2
        cv2.putText(overlay, str(i + 1), (cx - 10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    axes[0, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'Lane Detection ({len(lane_data)} lanes)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. 二値化結果（結合）
    binary_combined = np.zeros_like(gray)
    for ld in lane_data:
        x_start, x_end = ld['x_start'], ld['x_end']
        binary_combined[:, x_start:x_end] = ld['binary']
    
    axes[0, 1].imshow(binary_combined, cmap='gray')
    axes[0, 1].set_title('Binary Detection per Lane', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. Volume
    colors = plt.cm.viridis(df['Relative_%'] / 100)
    axes[1, 0].bar(df['Lane'], df['Volume'], color=colors, edgecolor='black')
    axes[1, 0].set_title('Band Volume', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Lane')
    axes[1, 0].set_ylabel('Volume')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Relative %
    bars = axes[1, 1].bar(df['Lane'], df['Relative_%'], color=colors, edgecolor='black')
    axes[1, 1].set_title('Relative Intensity (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Lane')
    axes[1, 1].set_ylabel('Relative %')
    axes[1, 1].set_ylim(0, 115)
    axes[1, 1].axhline(y=100, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for bar, rel in zip(bars, df['Relative_%']):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{rel:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Western Blot Quantifier v4.0 (Lane-based Detection)")
    print("=" * 60)
    print(f"画像: {args.image}")
    print(f"レーン数: {args.lanes}")
    if args.exclude_last:
        print("最後のレーン（マーカー）を除外")
    
    if args.output is None:
        args.output = os.path.dirname(os.path.abspath(args.image))
    os.makedirs(args.output, exist_ok=True)
    
    prefix = Path(args.image).stem
    
    img, gray = load_image(args.image)
    print(f"サイズ: {gray.shape[1]} x {gray.shape[0]}")
    
    print("\n🔬 レーンごとにバンド検出中...")
    results, lane_data = process_image(img, gray, args.lanes, args.exclude_last)
    print(f"   処理レーン数: {len(results)}")
    
    print("\n🎨 グラフ作成中...")
    plot_path = os.path.join(args.output, f'{prefix}_v4_plot.png')
    df = create_visualization(img, gray, results, lane_data, plot_path, args.lanes)
    print(f"   📊 {plot_path}")
    
    csv_path = os.path.join(args.output, f'{prefix}_v4_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   💾 {csv_path}")
    
    print("\n" + "=" * 60)
    print("結果")
    print("=" * 60)
    print(df[['Lane', 'Volume', 'Relative_%']].to_string(index=False))
    
    print("\n✅ 完了！")


if __name__ == "__main__":
    main()
