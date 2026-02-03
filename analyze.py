#!/usr/bin/env python3
"""
画像を詳しく分析
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def main():
    img = cv2.imread('test_image.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    print(f"Image size: {w} x {h}")
    print(f"Intensity range: {gray.min()} - {gray.max()}")
    print(f"Mean: {gray.mean():.1f}, Std: {gray.std():.1f}")
    
    num_lanes = 12
    lane_width = w // num_lanes
    
    # 各レーンの統計
    print("\nLane analysis:")
    print(f"{'Lane':<6} {'Min':<6} {'Max':<6} {'Mean':<8} {'Dark pixels':<12}")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i in range(num_lanes):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        lane = gray[:, x_start:x_end]
        
        # 背景推定
        bg = np.median(lane)
        dark_threshold = bg - 30
        dark_pixels = np.sum(lane < dark_threshold)
        
        print(f"{i+1:<6} {lane.min():<6} {lane.max():<6} {lane.mean():<8.1f} {dark_pixels:<12}")
        
        # プロファイル
        profile = np.mean(lane, axis=1)
        
        ax = axes[i // 4, i % 4]
        ax.imshow(lane, cmap='gray', aspect='auto')
        ax.set_title(f'Lane {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('lane_analysis.png', dpi=150)
    print("\nSaved: lane_analysis.png")
    
    # 横方向プロファイル（バンドの位置を確認）
    horizontal_profile = np.mean(gray, axis=0)
    
    plt.figure(figsize=(14, 4))
    plt.plot(horizontal_profile)
    plt.title('Horizontal Profile (Average intensity per column)')
    plt.xlabel('X position')
    plt.ylabel('Intensity')
    
    # レーン境界を追加
    for i in range(num_lanes + 1):
        x = i * lane_width
        plt.axvline(x, color='r', linestyle='--', alpha=0.5)
        if i < num_lanes:
            plt.text(x + lane_width/2, horizontal_profile.max(), str(i+1), ha='center')
    
    plt.tight_layout()
    plt.savefig('horizontal_profile.png', dpi=150)
    print("Saved: horizontal_profile.png")

if __name__ == "__main__":
    main()
