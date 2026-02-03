#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Western Blot Quantifier v2.0
OpenCV + Adaptive Thresholding でバンドを自動検出

使い方:
    python3 smart_quantify.py -i image.png
    python3 smart_quantify.py -i image.png -o results/
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Smart Western Blot Quantifier')
    parser.add_argument('-i', '--image', required=True, help='画像ファイル')
    parser.add_argument('-o', '--output', default=None, help='出力ディレクトリ')
    parser.add_argument('--min-area', type=int, default=500, help='最小バンド面積')
    parser.add_argument('--max-area', type=int, default=50000, help='最大バンド面積')
    parser.add_argument('--block-size', type=int, default=51, help='Adaptive threshold block size')
    parser.add_argument('--c-value', type=int, default=10, help='Adaptive threshold C value')
    parser.add_argument('--debug', action='store_true', help='デバッグ画像を出力')
    return parser.parse_args()


def load_image(image_path):
    """画像を読み込む"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def detect_bands(gray, block_size=51, c_value=10, min_area=500, max_area=50000):
    """Adaptive Thresholdingでバンドを検出"""
    
    # ガウシアンブラーでノイズ除去
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive Thresholding（暗い部分を検出）
    # ADAPTIVE_THRESH_GAUSSIAN_C: 近傍のガウシアン重み付き平均
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # 暗い部分を白に
        block_size, c_value
    )
    
    # モルフォロジー処理でノイズ除去・バンドを強調
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 輪郭検出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # フィルタリング: 面積とアスペクト比
    bands = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # バンドは横長（aspect_ratio > 1）であることが多い
            if aspect_ratio > 0.3:  # ある程度横に広がっている
                bands.append({
                    'contour': contour,
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'area': area,
                    'center_x': x + w // 2,
                    'center_y': y + h // 2,
                    'aspect_ratio': aspect_ratio
                })
    
    # X座標でソート（左から右）
    bands = sorted(bands, key=lambda b: b['center_x'])
    
    return bands, thresh


def measure_bands(gray, bands):
    """各バンドの強度を測定"""
    results = []
    
    # 背景強度を推定（画像全体の上位10%の輝度 = 薄い部分）
    bg_intensity = np.percentile(gray, 90)
    
    for i, band in enumerate(bands):
        # マスクを作成
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [band['contour']], -1, 255, -1)
        
        # バンド領域のピクセル値を取得
        band_pixels = gray[mask == 255]
        
        if len(band_pixels) == 0:
            continue
        
        # 強度を計算（暗いほど高い値に変換）
        # 反転: 255 - pixel_value
        inverted_pixels = 255 - band_pixels
        
        # バックグラウンド補正
        bg_corrected = 255 - bg_intensity
        corrected_pixels = np.maximum(inverted_pixels - bg_corrected, 0)
        
        # 積分強度（Volume）
        volume = np.sum(corrected_pixels)
        mean_intensity = np.mean(corrected_pixels)
        
        results.append({
            'Band': i + 1,
            'X': band['center_x'],
            'Y': band['center_y'],
            'Width': band['w'],
            'Height': band['h'],
            'Area': band['area'],
            'Mean_Intensity': round(mean_intensity, 2),
            'Volume': round(volume, 0),
            'contour': band['contour']
        })
    
    return results


def assign_lanes(results, num_lanes=None):
    """バンドをレーンにグループ化"""
    if not results:
        return results
    
    # X座標でクラスタリング
    x_coords = [r['X'] for r in results]
    
    if num_lanes is None:
        # 自動でレーン数を推定（バンド数をそのまま使用）
        num_lanes = len(results)
    
    # 簡易的なレーン割り当て（X座標順）
    for i, r in enumerate(results):
        r['Lane'] = i + 1
    
    return results


def create_visualization(img, gray, bands, results, thresh, output_path, debug=False):
    """結果を可視化"""
    
    # 検出結果のオーバーレイ
    overlay = img.copy()
    
    for i, band in enumerate(bands):
        # 輪郭を描画（緑）
        cv2.drawContours(overlay, [band['contour']], -1, (0, 255, 0), 2)
        
        # バンド番号
        cv2.putText(overlay, str(i + 1), 
                   (band['center_x'] - 10, band['center_y'] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # DataFrameを作成
    if results:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'contour'} for r in results])
        
        # 相対値を計算
        max_volume = df['Volume'].max()
        df['Relative_%'] = (df['Volume'] / max_volume * 100).round(2)
    else:
        df = pd.DataFrame()
    
    # 4パネルのFigure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 元画像 + 検出結果
    axes[0, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Detected Bands', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. 閾値処理結果
    axes[0, 1].imshow(thresh, cmap='gray')
    axes[0, 1].set_title('Adaptive Threshold Result', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. 棒グラフ（Volume）
    if not df.empty:
        colors = plt.cm.plasma(df['Relative_%'] / 100)
        axes[1, 0].bar(df['Band'], df['Volume'], color=colors, edgecolor='black')
        axes[1, 0].set_title('Band Volume (Integrated Intensity)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Band')
        axes[1, 0].set_ylabel('Volume')
        for _, row in df.iterrows():
            axes[1, 0].text(row['Band'], row['Volume'] * 1.02, f"{row['Volume']:.0f}",
                           ha='center', fontsize=8)
    
    # 4. 棒グラフ（Relative %）
    if not df.empty:
        axes[1, 1].bar(df['Band'], df['Relative_%'], color=colors, edgecolor='black')
        axes[1, 1].set_title('Relative Intensity (%)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Band')
        axes[1, 1].set_ylabel('Relative %')
        axes[1, 1].set_ylim(0, 110)
        axes[1, 1].axhline(y=100, color='red', linestyle='--', alpha=0.5)
        for _, row in df.iterrows():
            axes[1, 1].text(row['Band'], row['Relative_%'] + 2, f"{row['Relative_%']:.1f}%",
                           ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Smart Western Blot Quantifier v2.0")
    print("=" * 60)
    print(f"画像: {args.image}")
    
    # 出力ディレクトリ
    if args.output is None:
        args.output = os.path.dirname(os.path.abspath(args.image))
    os.makedirs(args.output, exist_ok=True)
    
    prefix = Path(args.image).stem
    
    # 画像読み込み
    img, gray = load_image(args.image)
    print(f"サイズ: {gray.shape[1]} x {gray.shape[0]}")
    
    # バンド検出
    print("\n🔬 バンド検出中...")
    bands, thresh = detect_bands(
        gray, 
        block_size=args.block_size,
        c_value=args.c_value,
        min_area=args.min_area,
        max_area=args.max_area
    )
    print(f"   検出されたバンド数: {len(bands)}")
    
    # 強度測定
    print("\n📊 強度測定中...")
    results = measure_bands(gray, bands)
    
    # レーン割り当て
    results = assign_lanes(results)
    
    # 可視化
    print("\n🎨 グラフ作成中...")
    plot_path = os.path.join(args.output, f'{prefix}_smart_plot.png')
    df = create_visualization(img, gray, bands, results, thresh, plot_path, args.debug)
    print(f"   📊 {plot_path}")
    
    # CSV保存
    if not df.empty:
        csv_path = os.path.join(args.output, f'{prefix}_smart_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   💾 {csv_path}")
        
        # 結果表示
        print("\n" + "=" * 60)
        print("結果")
        print("=" * 60)
        print(df[['Band', 'Volume', 'Relative_%']].to_string(index=False))
    else:
        print("\n⚠️ バンドが検出されませんでした")
        print("   --min-area を小さくするか、--block-size を調整してください")
    
    print("\n✅ 完了！")


if __name__ == "__main__":
    main()
