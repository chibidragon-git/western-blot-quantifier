#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Western Blot Quantifier v3.2
高精度バンド検出 + ノイズフィルタリング強化

使い方:
    python3 smart_quantify_v3.py -i image.png --lanes 12
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fclusterdata


def parse_args():
    parser = argparse.ArgumentParser(description='Smart Western Blot Quantifier v3.2')
    parser.add_argument('-i', '--image', required=True, help='画像ファイル')
    parser.add_argument('-o', '--output', default=None, help='出力ディレクトリ')
    parser.add_argument('--lanes', type=int, default=None, 
                       help='期待するレーン数（指定すると近接バンドをマージ）')
    parser.add_argument('--sensitivity', type=float, default=1.5, 
                       help='検出感度 (0.5=低, 1.0=標準, 2.0=高)')
    return parser.parse_args()


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def denoise_image(gray):
    """強力なノイズ除去"""
    # バイラテラルフィルタ（エッジを保持しながらノイズ除去）
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    # メディアンフィルタ（salt-and-pepperノイズ除去）
    denoised = cv2.medianBlur(denoised, 3)
    return denoised


def detect_bands_clean(gray, sensitivity=1.5):
    """クリーンなバンド検出"""
    
    # ノイズ除去
    denoised = denoise_image(gray)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # ガウシアンブラー
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Otsu's threshold（メイン）
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 追加: Adaptive threshold
    block_size = max(11, int(41 * sensitivity)) | 1
    c_value = max(3, int(10 / sensitivity))
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_value
    )
    
    # 両方のマスクのAND（より確実なバンドのみ）
    combined = cv2.bitwise_and(binary, adaptive)
    
    # モルフォロジー（ノイズ除去を強化）
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    
    # オープニング（小さいノイズ除去）
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
    # クロージング（バンド内の穴を埋める）
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    
    return cleaned


def filter_bands_by_properties(contours, gray, min_area_ratio=0.0008, max_area_ratio=0.2):
    """形状プロパティでバンドをフィルタリング"""
    
    img_area = gray.shape[0] * gray.shape[1]
    min_area = int(img_area * min_area_ratio)
    max_area = int(img_area * max_area_ratio)
    img_height = gray.shape[0]
    
    bands = []
    all_cy = []  # Y座標を収集してバンドラインを推定
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # アスペクト比（バンドは横長）
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.5:  # 縦長すぎるものは除外
            continue
        
        # Solidity（凸包に対する充填率）
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < 0.4:  # 形が不規則すぎるものは除外
            continue
        
        # 重心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        all_cy.append(cy)
        bands.append({
            'contour': contour,
            'x': x, 'y': y, 'w': w, 'h': h,
            'area': area, 'cx': cx, 'cy': cy,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity
        })
    
    # Y座標でフィルタリング（バンドは同じ高さにあるはず）
    if len(all_cy) >= 3:
        median_cy = np.median(all_cy)
        cy_std = np.std(all_cy)
        # 中央値から大きく外れるものを除外
        tolerance = max(cy_std * 2, img_height * 0.3)
        bands = [b for b in bands if abs(b['cy'] - median_cy) < tolerance]
    
    # X座標でソート
    bands = sorted(bands, key=lambda b: b['cx'])
    
    return bands


def merge_nearby_bands(bands, merge_distance):
    """近接バンドをマージ"""
    
    if len(bands) <= 1:
        return bands
    
    x_coords = np.array([[b['cx']] for b in bands])
    clusters = fclusterdata(x_coords, t=merge_distance, criterion='distance')
    
    merged_bands = []
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_bands = [bands[i] for i in cluster_indices]
        
        if len(cluster_bands) == 1:
            merged_bands.append(cluster_bands[0])
        else:
            # マージ
            all_contours = np.vstack([b['contour'] for b in cluster_bands])
            hull = cv2.convexHull(all_contours)
            x, y, w, h = cv2.boundingRect(hull)
            total_area = sum(b['area'] for b in cluster_bands)
            avg_cx = int(np.mean([b['cx'] for b in cluster_bands]))
            avg_cy = int(np.mean([b['cy'] for b in cluster_bands]))
            
            merged_bands.append({
                'contour': hull,
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': total_area, 'cx': avg_cx, 'cy': avg_cy,
                'merged': True
            })
    
    return sorted(merged_bands, key=lambda b: b['cx'])


def measure_bands(gray, bands):
    """バンド強度測定"""
    
    # 背景推定（画像の端）
    h, w = gray.shape
    bg_regions = [
        gray[:max(1, int(h*0.05)), :],  # 上端
        gray[min(h-1, int(h*0.95)):, :]  # 下端
    ]
    bg_intensity = np.median(np.concatenate([r.flatten() for r in bg_regions]))
    
    results = []
    for i, band in enumerate(bands):
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [band['contour']], -1, 255, -1)
        
        band_pixels = gray[mask == 255]
        if len(band_pixels) == 0:
            continue
        
        # 強度計算（暗い = 高シグナル）
        inverted = 255 - band_pixels.astype(np.float64)
        bg_corrected_value = 255 - bg_intensity
        corrected = np.maximum(inverted - bg_corrected_value * 0.7, 0)
        
        volume = np.sum(corrected)
        
        results.append({
            'Band': i + 1,
            'X': band['cx'],
            'Y': band['cy'],
            'Area': band['area'],
            'Volume': round(volume, 0),
            'contour': band['contour'],
            'merged': band.get('merged', False)
        })
    
    return results


def create_visualization(img, bands, results, binary_mask, output_path):
    """可視化"""
    
    if results:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'contour'} for r in results])
        max_volume = df['Volume'].max()
        df['Relative_%'] = (df['Volume'] / max_volume * 100).round(2)
    else:
        df = pd.DataFrame()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 検出結果
    overlay = img.copy()
    for i, band in enumerate(bands):
        color = (0, 255, 0) if not band.get('merged', False) else (255, 165, 0)
        cv2.drawContours(overlay, [band['contour']], -1, color, 2)
        cv2.putText(overlay, str(i + 1),
                   (band['cx'] - 10, band['cy'] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.circle(overlay, (band['cx'], band['cy']), 3, (255, 0, 0), -1)
    
    axes[0, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Detected Bands', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(binary_mask, cmap='gray')
    axes[0, 1].set_title('Binary Mask (Cleaned)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    if not df.empty:
        colors = plt.cm.viridis(df['Relative_%'] / 100)
        
        axes[1, 0].bar(df['Band'], df['Volume'], color=colors, edgecolor='black')
        axes[1, 0].set_title('Band Volume', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Band')
        axes[1, 0].set_ylabel('Volume')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        bars = axes[1, 1].bar(df['Band'], df['Relative_%'], color=colors, edgecolor='black')
        axes[1, 1].set_title('Relative Intensity (%)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Band')
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
    print("Smart Western Blot Quantifier v3.2 (Noise Filtered)")
    print("=" * 60)
    print(f"画像: {args.image}")
    
    if args.output is None:
        args.output = os.path.dirname(os.path.abspath(args.image))
    os.makedirs(args.output, exist_ok=True)
    
    prefix = Path(args.image).stem
    
    img, gray = load_image(args.image)
    print(f"サイズ: {gray.shape[1]} x {gray.shape[0]}")
    
    # クリーンな検出
    print("\n🔬 バンド検出中（ノイズフィルタリング強化）...")
    binary_mask = detect_bands_clean(gray, args.sensitivity)
    
    # 輪郭検出とフィルタリング
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bands = filter_bands_by_properties(contours, gray)
    print(f"   フィルタ後バンド数: {len(bands)}")
    
    # マージ
    if args.lanes and len(bands) > args.lanes:
        merge_distance = gray.shape[1] / args.lanes * 0.4
        bands = merge_nearby_bands(bands, merge_distance)
        print(f"   マージ後バンド数: {len(bands)}")
    
    # 測定
    print("\n📊 測定中...")
    results = measure_bands(gray, bands)
    
    # 可視化
    print("\n🎨 グラフ作成中...")
    plot_path = os.path.join(args.output, f'{prefix}_v3.2_plot.png')
    df = create_visualization(img, bands, results, binary_mask, plot_path)
    print(f"   📊 {plot_path}")
    
    if not df.empty:
        csv_path = os.path.join(args.output, f'{prefix}_v3.2_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   💾 {csv_path}")
        
        print("\n" + "=" * 60)
        print("結果")
        print("=" * 60)
        print(df[['Band', 'Volume', 'Relative_%']].to_string(index=False))
    
    print("\n✅ 完了！")


if __name__ == "__main__":
    main()
