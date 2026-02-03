#!/usr/bin/env python3
"""
Western Blot Quantifier - 改良版テスト
バンド領域の自動検出を改善
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d

# 正解データ
GROUND_TRUTH = {
    1: 100.0, 2: 88.5, 3: 27.7, 4: 31.5, 5: 40.8, 6: 82.6,
    7: 19.6, 8: 44.4, 9: 31.0, 10: 33.5, 11: 27.3, 12: 15.6
}

def load_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def measure_lane_v5(lane_gray, global_bg=None):
    """
    バージョン5: グローバル背景を使用
    画像全体の背景を使って補正
    """
    h, w = lane_gray.shape
    
    if global_bg is None:
        global_bg = 230  # 明るい部分のデフォルト
    
    # 反転して積分（シンプル）
    inverted = global_bg - lane_gray.astype(np.float64)
    inverted = np.maximum(inverted, 0)
    
    volume = np.sum(inverted)
    return volume


def measure_lane_v6(lane_gray, band_top, band_bottom):
    """
    バージョン6: 固定バンド領域を使用
    全レーンで同じY範囲を使う（バンドは同じ高さにあるはず）
    """
    h, w = lane_gray.shape
    
    # 背景推定（バンド領域外）
    if band_top > 5:
        bg_top = np.median(lane_gray[:band_top-2, :])
    else:
        bg_top = 230
    
    if band_bottom < h - 5:
        bg_bottom = np.median(lane_gray[band_bottom+2:, :])
    else:
        bg_bottom = 230
    
    bg = (bg_top + bg_bottom) / 2
    
    # バンド領域のみ積分
    band_region = lane_gray[band_top:band_bottom, :]
    inverted = bg - band_region.astype(np.float64)
    inverted = np.maximum(inverted, 0)
    
    volume = np.sum(inverted)
    return volume


def find_band_row(gray):
    """画像全体からバンドのY位置を検出"""
    h, w = gray.shape
    
    # 縦方向プロファイル（全体）
    profile = np.mean(gray, axis=1)
    smoothed = gaussian_filter1d(profile, sigma=2)
    
    # 最も暗い行を見つける
    min_row = np.argmin(smoothed)
    
    # バンド領域を推定（ピーク周辺）
    threshold = smoothed[min_row] + (smoothed.max() - smoothed[min_row]) * 0.5
    
    top = min_row
    while top > 0 and smoothed[top] < threshold:
        top -= 1
    
    bottom = min_row
    while bottom < h - 1 and smoothed[bottom] < threshold:
        bottom += 1
    
    # 余裕を持たせる
    margin = max(3, (bottom - top) // 2)
    top = max(0, top - margin)
    bottom = min(h - 1, bottom + margin)
    
    return top, bottom


def evaluate(results, name):
    df = pd.DataFrame(results)
    max_vol = df['Volume'].max()
    df['Relative_%'] = (df['Volume'] / max_vol * 100).round(1) if max_vol > 0 else 0
    
    errors = []
    for _, row in df.iterrows():
        lane = row['Lane']
        pred = row['Relative_%']
        true = GROUND_TRUTH[lane]
        errors.append(abs(pred - true))
    
    mae = np.mean(errors)
    print(f"\n=== {name} ===")
    print(f"MAE: {mae:.2f}%")
    
    return mae, df


def main():
    img, gray = load_image('test_image.png')
    h, w = gray.shape
    num_lanes = 12
    lane_width = w // num_lanes
    
    print(f"Image size: {w} x {h}")
    
    # グローバルバンド位置を検出
    band_top, band_bottom = find_band_row(gray)
    print(f"Detected band region: Y = {band_top} to {band_bottom}")
    
    # グローバル背景（画像の角）
    corners = [
        gray[:10, :10],
        gray[:10, -10:],
        gray[-10:, :10],
        gray[-10:, -10:]
    ]
    global_bg = np.median([np.median(c) for c in corners])
    print(f"Global background: {global_bg:.1f}")
    
    methods = []
    
    # v5: グローバル背景
    results_v5 = []
    for i in range(num_lanes):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        lane = gray[:, x_start:x_end]
        volume = measure_lane_v5(lane, global_bg)
        results_v5.append({'Lane': i + 1, 'Volume': volume})
    methods.append(('v5_global_bg', results_v5))
    
    # v6: 固定バンド領域
    results_v6 = []
    for i in range(num_lanes):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        lane = gray[:, x_start:x_end]
        volume = measure_lane_v6(lane, band_top, band_bottom)
        results_v6.append({'Lane': i + 1, 'Volume': volume})
    methods.append(('v6_fixed_band', results_v6))
    
    # v7: バンド領域のピクセル数（面積）ベース
    results_v7 = []
    threshold_v7 = global_bg - 40
    for i in range(num_lanes):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        lane = gray[band_top:band_bottom, x_start:x_end]
        # 閾値以下のピクセル数
        band_pixels = np.sum(lane < threshold_v7)
        results_v7.append({'Lane': i + 1, 'Volume': band_pixels})
    methods.append(('v7_pixel_count', results_v7))
    
    # v8: 強度の重み付き和
    results_v8 = []
    for i in range(num_lanes):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        lane = gray[band_top:band_bottom, x_start:x_end]
        
        # ローカル背景（レーン上下）
        local_bg = np.percentile(lane, 90)  # 上位10%を背景とみなす
        
        inverted = local_bg - lane.astype(np.float64)
        inverted = np.maximum(inverted, 0)
        volume = np.sum(inverted)
        results_v8.append({'Lane': i + 1, 'Volume': volume})
    methods.append(('v8_local_percentile', results_v8))
    
    best_mae = float('inf')
    best_method = None
    best_df = None
    
    for name, results in methods:
        mae, df = evaluate(results, name)
        if mae < best_mae:
            best_mae = mae
            best_method = name
            best_df = df
    
    print(f"\n{'='*60}")
    print(f"BEST: {best_method} (MAE: {best_mae:.2f}%)")
    print(f"{'='*60}")
    
    # 比較表
    print("\n詳細比較:")
    print(f"{'Lane':<6} {'Pred':<10} {'True':<10} {'Error':<10}")
    for _, row in best_df.iterrows():
        lane = row['Lane']
        pred = row['Relative_%']
        true = GROUND_TRUTH[lane]
        err = pred - true
        print(f"{lane:<6} {pred:<10.1f} {true:<10.1f} {err:+.1f}")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 元画像にバンド領域を表示
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.line(overlay, (0, band_top), (w, band_top), (0, 255, 0), 1)
    cv2.line(overlay, (0, band_bottom), (w, band_bottom), (0, 255, 0), 1)
    for i in range(num_lanes + 1):
        x = i * lane_width
        cv2.line(overlay, (x, 0), (x, h), (255, 0, 0), 1)
    
    axes[0, 0].imshow(overlay)
    axes[0, 0].set_title('Detected Band Region')
    axes[0, 0].axis('off')
    
    # 縦プロファイル
    profile = np.mean(gray, axis=1)
    axes[0, 1].plot(profile)
    axes[0, 1].axvspan(band_top, band_bottom, alpha=0.3, color='green')
    axes[0, 1].set_title('Vertical Profile')
    axes[0, 1].set_xlabel('Y position')
    axes[0, 1].set_ylabel('Intensity')
    
    # 予測 vs 正解
    lanes = best_df['Lane'].values
    pred = best_df['Relative_%'].values
    true = [GROUND_TRUTH[l] for l in lanes]
    
    x = np.arange(len(lanes))
    width = 0.35
    axes[1, 0].bar(x - width/2, pred, width, label='Predicted', alpha=0.8)
    axes[1, 0].bar(x + width/2, true, width, label='Ground Truth', alpha=0.8)
    axes[1, 0].set_xlabel('Lane')
    axes[1, 0].set_ylabel('Relative %')
    axes[1, 0].set_title('Predicted vs Ground Truth')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(lanes)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 散布図
    axes[1, 1].scatter(true, pred, s=100, alpha=0.7)
    for i, lane in enumerate(lanes):
        axes[1, 1].annotate(str(lane), (true[i], pred[i]), textcoords="offset points", xytext=(5,5))
    axes[1, 1].plot([0, 100], [0, 100], 'r--', alpha=0.5)
    axes[1, 1].set_xlabel('Ground Truth (%)')
    axes[1, 1].set_ylabel('Predicted (%)')
    axes[1, 1].set_title('Correlation')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pdca_result.png', dpi=150)
    print("\nSaved: pdca_result.png")


if __name__ == "__main__":
    main()
