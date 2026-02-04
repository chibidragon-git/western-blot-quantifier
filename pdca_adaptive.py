#!/usr/bin/env python3
"""
Per-Lane Adaptive ROI - 各レーンで独自にバンド位置を検出
"""

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

GROUND_TRUTH = {
    1: 100.0, 2: 88.5, 3: 27.7, 4: 31.5, 5: 40.8, 6: 82.6,
    7: 19.6, 8: 44.4, 9: 31.0, 10: 33.5, 11: 27.3, 12: 15.6
}

def evaluate(results, name, verbose=True):
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
            lane = int(row['Lane'])
            pred = row['Relative_%']
            true = GROUND_TRUTH[lane]
            err = pred - true
            marker = "***" if abs(err) > 15 else ""
            print(f"  Lane {lane:2d}: {pred:5.1f}% vs {true:5.1f}% ({err:+6.1f}) {marker}")
    
    return mae, df

def main():
    img = cv2.imread('test_image2.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    num_lanes = 12
    lane_width = w // num_lanes
    
    print(f"Image: {w}x{h}")
    
    # グリッドサーチ
    best_mae = float('inf')
    best_params = None
    
    for roi_half in [15, 20, 25, 30, 35]:
        for bg_pct in [88, 90, 92, 95]:
            for peak_thresh in [0.2, 0.3, 0.4]:
                results = []
                
                for i in range(num_lanes):
                    x_start = i * lane_width
                    x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
                    lane = gray[:, x_start:x_end]
                    
                    # プロファイル
                    prof = gaussian_filter1d(np.mean(lane, axis=1), sigma=2)
                    bg = np.percentile(prof, bg_pct)
                    inv = np.maximum(bg - prof, 0)
                    
                    if inv.max() < 3:
                        results.append({'Lane': i + 1, 'Volume': 0})
                        continue
                    
                    # ピーク検出
                    pk = np.argmax(inv)
                    
                    # ROI範囲（ピーク中心の固定幅）
                    t = max(0, pk - roi_half)
                    b = min(h - 1, pk + roi_half)
                    
                    # ROI内積分
                    roi = lane[t:b+1, :]
                    roi_bg = np.percentile(roi, bg_pct)
                    vol = np.sum(np.maximum(roi_bg - roi.astype(np.float64), 0))
                    
                    results.append({'Lane': i + 1, 'Volume': vol})
                
                mae, _ = evaluate(results, "test", verbose=False)
                
                if mae < best_mae:
                    best_mae = mae
                    best_params = {
                        'roi_half': roi_half,
                        'bg_pct': bg_pct,
                        'peak_thresh': peak_thresh
                    }
    
    print(f"\nBEST Params: {best_params}")
    print(f"BEST MAE: {best_mae:.2f}%")
    
    # 最適パラメータで再実行
    p = best_params
    results = []
    roi_info = []
    
    for i in range(num_lanes):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        lane = gray[:, x_start:x_end]
        
        prof = gaussian_filter1d(np.mean(lane, axis=1), sigma=2)
        bg = np.percentile(prof, p['bg_pct'])
        inv = np.maximum(bg - prof, 0)
        
        if inv.max() < 3:
            results.append({'Lane': i + 1, 'Volume': 0})
            roi_info.append((0, h-1))
            continue
        
        pk = np.argmax(inv)
        t = max(0, pk - p['roi_half'])
        b = min(h - 1, pk + p['roi_half'])
        
        roi = lane[t:b+1, :]
        roi_bg = np.percentile(roi, p['bg_pct'])
        vol = np.sum(np.maximum(roi_bg - roi.astype(np.float64), 0))
        
        results.append({'Lane': i + 1, 'Volume': vol})
        roi_info.append((t, b))
    
    mae, df = evaluate(results, "Per-Lane Adaptive ROI", verbose=True)
    
    print("\nROI positions per lane:")
    for i, (t, b) in enumerate(roi_info):
        print(f"  Lane {i+1}: Y={t} to {b}")

if __name__ == "__main__":
    main()
