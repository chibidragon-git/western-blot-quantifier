#!/usr/bin/env python3
"""
レーン6と11を詳しく分析
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def main():
    img = cv2.imread('test_image2.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    num_lanes = 12
    lane_width = w // num_lanes
    
    print(f"Image: {w}x{h}")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i in range(num_lanes):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        lane = gray[:, x_start:x_end]
        
        # プロファイル
        prof = np.mean(lane, axis=1)
        
        ax = axes[i // 4, i % 4]
        
        # レーン画像
        ax2 = ax.twinx()
        ax.imshow(lane.T, cmap='gray', aspect='auto', extent=[0, h, 0, lane.shape[1]])
        ax.set_ylabel(f'Lane {i+1}')
        
        # プロファイルをオーバーレイ
        ax2.plot(np.arange(h), 255 - prof, 'r-', linewidth=2, alpha=0.7)
        ax2.set_ylim(0, 80)
        
        # 統計
        bg = np.percentile(prof, 90)
        inv = np.maximum(bg - prof, 0)
        pk = np.argmax(inv)
        
        ax.axhline(y=lane.shape[1]/2, color='g', linestyle='--', alpha=0.3)
        ax.set_title(f'Lane {i+1}: peak@{pk}, max_inv={inv[pk]:.1f}')
    
    plt.tight_layout()
    plt.savefig('lane_profiles.png', dpi=150)
    print("Saved lane_profiles.png")
    
    # Lane 6と1を詳しく比較
    print("\n=== Lane 1 vs Lane 6 詳細 ===")
    for lane_idx in [0, 5]:  # 0-indexed
        x_start = lane_idx * lane_width
        x_end = (lane_idx + 1) * lane_width if lane_idx < num_lanes - 1 else w
        lane = gray[:, x_start:x_end]
        
        prof = np.mean(lane, axis=1)
        bg = np.percentile(prof, 90)
        inv = np.maximum(bg - prof, 0)
        
        print(f"\nLane {lane_idx + 1}:")
        print(f"  Size: {lane.shape}")
        print(f"  Min/Max: {lane.min()}/{lane.max()}")
        print(f"  BG (90th): {bg:.1f}")
        print(f"  Inverted max: {inv.max():.1f}")
        print(f"  Peak position: {np.argmax(inv)}")
        print(f"  Sum of inverted: {inv.sum():.1f}")
        
        # ROI範囲
        pk = np.argmax(inv)
        thresh = inv[pk] * 0.3
        t = pk
        while t > 0 and inv[t] > thresh: t -= 1
        b = pk
        while b < h-1 and inv[b] > thresh: b += 1
        print(f"  ROI range (30%): {t} to {b}")
        
        # ROI内の積分
        roi = lane[t:b+1, :]
        roi_bg = np.percentile(roi, 90)
        roi_inv = np.maximum(roi_bg - roi.astype(np.float64), 0)
        print(f"  ROI volume: {roi_inv.sum():.1f}")

if __name__ == "__main__":
    main()
