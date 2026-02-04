#!/usr/bin/env python3
"""
Western Blot Quantifier - 徹底的PDCA v2
"""

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# 正解データ
GROUND_TRUTH = {
    1: 100.0, 2: 88.5, 3: 27.7, 4: 31.5, 5: 40.8, 6: 82.6,
    7: 19.6, 8: 44.4, 9: 31.0, 10: 33.5, 11: 27.3, 12: 15.6
}

def load_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def evaluate(results, name, verbose=False):
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
    if verbose:
        print(f"\n=== {name} (MAE: {mae:.2f}%) ===")
        for _, row in df.iterrows():
            lane = row['Lane']
            pred = row['Relative_%']
            true = GROUND_TRUTH[lane]
            err = pred - true
            print(f"  Lane {int(lane)}: {pred:.1f}% vs {true:.1f}% ({err:+.1f})")
    
    return mae, df

def main():
    img, gray = load_image('test_image2.png')
    h, w = gray.shape
    num_lanes = 12
    lane_width = w // num_lanes
    
    print(f"Image: {w}x{h}")
    
    best_mae = float('inf')
    best_name = None
    best_params = None
    
    # パラメータグリッドサーチ
    for bg_percentile in [85, 90, 95]:
        for band_threshold in [0.3, 0.4, 0.5]:
            for sigma in [1.0, 1.5, 2.0, 3.0]:
                for margin_ratio in [0.3, 0.5, 0.7]:
                    # バンド領域検出
                    profile = np.mean(gray, axis=1)
                    smoothed = gaussian_filter1d(profile, sigma=sigma)
                    bg_val = np.percentile(smoothed, bg_percentile)
                    inverted = np.maximum(bg_val - smoothed, 0)
                    
                    if inverted.max() < 1:
                        continue
                    
                    peak = np.argmax(inverted)
                    thresh = inverted[peak] * band_threshold
                    
                    top = peak
                    while top > 0 and inverted[top] > thresh:
                        top -= 1
                    bottom = peak
                    while bottom < h - 1 and inverted[bottom] > thresh:
                        bottom += 1
                    
                    margin = int((bottom - top) * margin_ratio)
                    top = max(0, top - margin)
                    bottom = min(h - 1, bottom + margin)
                    
                    # 各レーン測定
                    results = []
                    for i in range(num_lanes):
                        x_start = i * lane_width
                        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
                        lane = gray[top:bottom+1, x_start:x_end]
                        
                        local_bg = np.percentile(lane, bg_percentile)
                        inv = np.maximum(local_bg - lane.astype(np.float64), 0)
                        vol = np.sum(inv)
                        results.append({'Lane': i + 1, 'Volume': vol})
                    
                    mae, _ = evaluate(results, "test", verbose=False)
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_name = f"bg{bg_percentile}_th{band_threshold}_s{sigma}_m{margin_ratio}"
                        best_params = {
                            'bg_percentile': bg_percentile,
                            'band_threshold': band_threshold,
                            'sigma': sigma,
                            'margin_ratio': margin_ratio,
                            'top': top,
                            'bottom': bottom
                        }
    
    print(f"\n{'='*60}")
    print(f"BEST: {best_name}")
    print(f"MAE: {best_mae:.2f}%")
    print(f"Params: {best_params}")
    print(f"{'='*60}")
    
    # 最適パラメータで再評価
    p = best_params
    profile = np.mean(gray, axis=1)
    smoothed = gaussian_filter1d(profile, sigma=p['sigma'])
    bg_val = np.percentile(smoothed, p['bg_percentile'])
    inverted = np.maximum(bg_val - smoothed, 0)
    peak = np.argmax(inverted)
    thresh = inverted[peak] * p['band_threshold']
    top = peak
    while top > 0 and inverted[top] > thresh: top -= 1
    bottom = peak
    while bottom < h - 1 and inverted[bottom] > thresh: bottom += 1
    margin = int((bottom - top) * p['margin_ratio'])
    top = max(0, top - margin)
    bottom = min(h - 1, bottom + margin)
    
    results = []
    for i in range(num_lanes):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        lane = gray[top:bottom+1, x_start:x_end]
        local_bg = np.percentile(lane, p['bg_percentile'])
        inv = np.maximum(local_bg - lane.astype(np.float64), 0)
        vol = np.sum(inv)
        results.append({'Lane': i + 1, 'Volume': vol})
    
    evaluate(results, "BEST", verbose=True)
    
    # さらに改善: 各レーン独自のバンド位置検出
    print("\n\n=== Per-Lane Peak Detection ===")
    results_per_lane = []
    for i in range(num_lanes):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        lane_full = gray[:, x_start:x_end]
        
        prof = gaussian_filter1d(np.mean(lane_full, axis=1), sigma=2)
        bg = np.percentile(prof, 90)
        inv = np.maximum(bg - prof, 0)
        
        if inv.max() > 1:
            pk = np.argmax(inv)
            t = pk
            while t > 0 and inv[t] > inv[pk]*0.3: t -= 1
            b = pk
            while b < h-1 and inv[b] > inv[pk]*0.3: b += 1
            t = max(0, t - 5)
            b = min(h-1, b + 5)
            
            roi = lane_full[t:b+1, :]
            roi_bg = np.percentile(roi, 90)
            vol = np.sum(np.maximum(roi_bg - roi.astype(np.float64), 0))
        else:
            vol = 0
        
        results_per_lane.append({'Lane': i + 1, 'Volume': vol})
    
    mae_per, _ = evaluate(results_per_lane, "Per-Lane Peak", verbose=True)
    
    # ハイブリッド: グローバルROI + Per-Laneバックグラウンド
    print("\n\n=== Hybrid (Global ROI + Per-Lane BG) ===")
    results_hybrid = []
    # グローバルROI範囲を全レーンのピークから決定
    all_peaks = []
    for i in range(num_lanes):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        lane = gray[:, x_start:x_end]
        prof = gaussian_filter1d(np.mean(lane, axis=1), sigma=2)
        bg = np.percentile(prof, 90)
        inv = np.maximum(bg - prof, 0)
        if inv.max() > 1:
            all_peaks.append(np.argmax(inv))
    
    if all_peaks:
        median_peak = int(np.median(all_peaks))
        roi_half = 30  # 固定幅
        global_top = max(0, median_peak - roi_half)
        global_bottom = min(h - 1, median_peak + roi_half)
    else:
        global_top, global_bottom = 0, h - 1
    
    for i in range(num_lanes):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        lane = gray[global_top:global_bottom+1, x_start:x_end]
        local_bg = np.percentile(lane, 92)
        inv = np.maximum(local_bg - lane.astype(np.float64), 0)
        vol = np.sum(inv)
        results_hybrid.append({'Lane': i + 1, 'Volume': vol})
    
    mae_hybrid, _ = evaluate(results_hybrid, "Hybrid", verbose=True)


if __name__ == "__main__":
    main()
