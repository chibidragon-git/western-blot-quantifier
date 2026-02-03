#!/usr/bin/env python3
"""
Western Blot Quantifier - テスト・調整スクリプト
正解データに近づくようにパラメータを調整
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
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

def measure_lane_v1(lane_gray):
    """バージョン1: シンプルな積分"""
    # 背景推定
    h, w = lane_gray.shape
    bg = np.median(lane_gray)
    
    # 反転して積分
    inverted = 255 - lane_gray.astype(np.float64)
    bg_inv = 255 - bg
    corrected = np.maximum(inverted - bg_inv * 0.5, 0)
    
    volume = np.sum(corrected)
    return volume

def measure_lane_v2(lane_gray):
    """バージョン2: バンド領域のみ積分（プロファイルベース）"""
    h, w = lane_gray.shape
    
    # 縦方向プロファイル
    profile = np.mean(lane_gray, axis=1)
    smoothed = gaussian_filter1d(profile, sigma=2)
    
    # 背景
    bg = (np.mean(smoothed[:max(1,int(h*0.1))]) + np.mean(smoothed[int(h*0.9):])) / 2
    
    # 反転
    inverted = 255 - smoothed
    baseline = 255 - bg
    corrected = np.maximum(inverted - baseline * 0.7, 0)
    
    # ピーク検出
    if np.max(corrected) < 3:
        return 0
    
    peaks, _ = find_peaks(corrected, height=np.max(corrected)*0.2, distance=5)
    
    if len(peaks) == 0:
        peak_pos = np.argmax(corrected)
    else:
        peak_pos = peaks[np.argmax(corrected[peaks])]
    
    # バンド領域
    peak_height = corrected[peak_pos]
    threshold = peak_height * 0.2
    
    left = peak_pos
    while left > 0 and corrected[left] > threshold:
        left -= 1
    
    right = peak_pos
    while right < len(corrected) - 1 and corrected[right] > threshold:
        right += 1
    
    # その領域の元画像から積分
    band_region = lane_gray[left:right+1, :]
    inverted_region = 255 - band_region.astype(np.float64)
    bg_inv = 255 - bg
    corrected_region = np.maximum(inverted_region - bg_inv * 0.7, 0)
    
    volume = np.sum(corrected_region)
    return volume

def measure_lane_v3(lane_gray):
    """バージョン3: ローカル背景補正"""
    h, w = lane_gray.shape
    
    # 縦方向プロファイル
    profile = np.mean(lane_gray, axis=1)
    smoothed = gaussian_filter1d(profile, sigma=2)
    
    # ローリングボール的な背景推定
    from scipy.ndimage import minimum_filter1d, maximum_filter1d
    bg_profile = maximum_filter1d(minimum_filter1d(smoothed, size=h//3), size=h//3)
    
    # 反転して背景補正
    inverted = 255 - smoothed
    bg_inv = 255 - bg_profile
    corrected = np.maximum(inverted - bg_inv, 0)
    
    # 積分（全体）
    volume = np.sum(corrected) * w
    return volume

def measure_lane_v4(lane_gray):
    """バージョン4: 閾値ベース（バンド部分のみ）"""
    h, w = lane_gray.shape
    
    # 背景推定（上下10%の中央値）
    bg_top = lane_gray[:max(1,int(h*0.15)), :]
    bg_bottom = lane_gray[int(h*0.85):, :]
    bg = np.median(np.concatenate([bg_top.flatten(), bg_bottom.flatten()]))
    
    # 閾値（背景より暗い部分がバンド）
    threshold = bg - 20  # 背景より20暗い
    
    # バンドマスク
    band_mask = lane_gray < threshold
    
    if np.sum(band_mask) == 0:
        # バンドなし
        return 0
    
    # バンド領域のみ積分
    band_pixels = lane_gray[band_mask]
    inverted = 255 - band_pixels.astype(np.float64)
    bg_inv = 255 - bg
    corrected = np.maximum(inverted - bg_inv * 0.5, 0)
    
    volume = np.sum(corrected)
    return volume

def evaluate(results, name):
    """正解との誤差を評価"""
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
    print(df[['Lane', 'Volume', 'Relative_%']].to_string(index=False))
    
    return mae, df

def main():
    img, gray = load_image('test_image.png')
    h, w = gray.shape
    num_lanes = 12
    lane_width = w // num_lanes
    
    print(f"Image size: {w} x {h}")
    print(f"Lane width: {lane_width}")
    
    methods = [
        ('v1_simple', measure_lane_v1),
        ('v2_profile', measure_lane_v2),
        ('v3_rolling', measure_lane_v3),
        ('v4_threshold', measure_lane_v4),
    ]
    
    best_mae = float('inf')
    best_method = None
    best_df = None
    
    for name, measure_func in methods:
        results = []
        for i in range(num_lanes):
            x_start = i * lane_width
            x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
            lane_gray = gray[:, x_start:x_end]
            
            volume = measure_func(lane_gray)
            results.append({'Lane': i + 1, 'Volume': volume})
        
        mae, df = evaluate(results, name)
        
        if mae < best_mae:
            best_mae = mae
            best_method = name
            best_df = df
    
    print(f"\n{'='*50}")
    print(f"BEST: {best_method} (MAE: {best_mae:.2f}%)")
    print(f"{'='*50}")
    
    # 正解との比較
    print("\n比較:")
    print(f"{'Lane':<6} {'Pred':<10} {'True':<10} {'Error':<10}")
    for _, row in best_df.iterrows():
        lane = row['Lane']
        pred = row['Relative_%']
        true = GROUND_TRUTH[lane]
        err = pred - true
        print(f"{lane:<6} {pred:<10.1f} {true:<10.1f} {err:+.1f}")

if __name__ == "__main__":
    main()
