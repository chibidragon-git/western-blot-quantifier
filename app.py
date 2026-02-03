#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Western Blot Quantifier v3.0 - Web App
プロファイルベース測定（ImageJ方式）
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def load_image(uploaded_file):
    """アップロードされた画像を読み込む"""
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        gray = img_array
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    return img_bgr, gray


def find_band_region(lane_profile, min_height_ratio=0.1):
    """プロファイルからバンド領域を検出"""
    # スムージング
    smoothed = gaussian_filter1d(lane_profile, sigma=3)
    
    # 背景補正（ローリングボール的な処理）
    # 上下10%を背景とみなす
    n = len(smoothed)
    bg_top = np.mean(smoothed[:max(1, int(n*0.1))])
    bg_bottom = np.mean(smoothed[int(n*0.9):])
    bg = (bg_top + bg_bottom) / 2
    
    # 反転（暗い=高シグナル）
    inverted = 255 - smoothed
    baseline = 255 - bg
    corrected = np.maximum(inverted - baseline * 0.8, 0)
    
    # ピーク検出
    max_val = np.max(corrected)
    if max_val < 5:  # シグナルなし
        return None, None, corrected
    
    min_height = max_val * min_height_ratio
    peaks, properties = find_peaks(corrected, height=min_height, distance=10)
    
    if len(peaks) == 0:
        # ピークがなければ最大値の位置を使用
        peak_pos = np.argmax(corrected)
    else:
        # 最も高いピーク
        peak_pos = peaks[np.argmax(properties['peak_heights'])]
    
    # ピーク周辺のバンド領域を決定（半値幅ベース）
    peak_height = corrected[peak_pos]
    half_height = peak_height / 2
    
    # 左端を探す
    left = peak_pos
    while left > 0 and corrected[left] > half_height * 0.3:
        left -= 1
    
    # 右端を探す
    right = peak_pos
    while right < len(corrected) - 1 and corrected[right] > half_height * 0.3:
        right += 1
    
    # 少し余裕を持たせる
    margin = max(5, (right - left) // 4)
    left = max(0, left - margin)
    right = min(len(corrected) - 1, right + margin)
    
    return left, right, corrected


def measure_lane_profile(lane_gray):
    """レーンをプロファイルベースで測定"""
    h, w = lane_gray.shape
    
    # 縦方向プロファイル（各行の平均強度）
    profile = np.mean(lane_gray, axis=1)
    
    # バンド領域検出
    top, bottom, corrected_profile = find_band_region(profile)
    
    if top is None:
        # バンドなし - 中央領域で計算
        top = int(h * 0.3)
        bottom = int(h * 0.7)
    
    # Volume計算（補正済みプロファイルの積分）
    volume = np.sum(corrected_profile[top:bottom+1]) * w
    
    # 平均強度
    mean_intensity = np.mean(corrected_profile[top:bottom+1])
    
    # バンド中心
    if np.sum(corrected_profile[top:bottom+1]) > 0:
        weights = corrected_profile[top:bottom+1]
        center_y = top + np.sum(np.arange(len(weights)) * weights) / np.sum(weights)
    else:
        center_y = (top + bottom) / 2
    
    return {
        'volume': volume,
        'mean': mean_intensity,
        'top': top,
        'bottom': bottom,
        'center_y': int(center_y),
        'profile': corrected_profile
    }


def process_image(img, gray, num_lanes, exclude_last=False):
    """画像を処理"""
    h, w = gray.shape
    lane_width = w // num_lanes
    
    results = []
    lane_data = []
    
    total_lanes = num_lanes - 1 if exclude_last else num_lanes
    
    for i in range(total_lanes):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        
        lane_gray = gray[:, x_start:x_end]
        
        # プロファイルベース測定
        measurement = measure_lane_profile(lane_gray)
        
        results.append({
            'Lane': i + 1,
            'Volume': round(measurement['volume'], 0),
            'Mean': round(measurement['mean'], 2),
        })
        
        lane_data.append({
            'x_start': x_start,
            'x_end': x_end,
            'top': measurement['top'],
            'bottom': measurement['bottom'],
            'center_y': measurement['center_y'],
            'profile': measurement['profile']
        })
    
    return results, lane_data


def create_overlay(img, gray, lane_data, num_lanes):
    """検出結果のオーバーレイを作成"""
    h, w = gray.shape
    lane_width = w // num_lanes
    
    overlay = img.copy()
    
    # レーン境界線
    for i in range(num_lanes + 1):
        x = i * lane_width
        cv2.line(overlay, (x, 0), (x, h), (255, 0, 0), 1)
    
    # バンド領域（矩形ROI）
    for i, ld in enumerate(lane_data):
        # ROI矩形
        pt1 = (ld['x_start'] + 2, ld['top'])
        pt2 = (ld['x_end'] - 2, ld['bottom'])
        cv2.rectangle(overlay, pt1, pt2, (0, 255, 0), 2)
        
        # レーン番号
        cx = (ld['x_start'] + ld['x_end']) // 2
        cv2.putText(overlay, str(i + 1), (cx - 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return overlay


def create_plot(df, lane_data):
    """棒グラフとプロファイルを作成"""
    n_lanes = len(lane_data)
    
    fig = plt.figure(figsize=(14, 8))
    
    # 上段: 棒グラフ
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    
    colors = plt.cm.viridis(df['Relative_%'] / 100)
    
    # Volume
    ax1.bar(df['Lane'], df['Volume'], color=colors, edgecolor='black')
    ax1.set_title('Band Volume', fontweight='bold')
    ax1.set_xlabel('Lane')
    ax1.set_ylabel('Volume')
    ax1.grid(axis='y', alpha=0.3)
    
    # Relative %
    bars = ax2.bar(df['Lane'], df['Relative_%'], color=colors, edgecolor='black')
    ax2.set_title('Relative Intensity (%)', fontweight='bold')
    ax2.set_xlabel('Lane')
    ax2.set_ylabel('Relative %')
    ax2.set_ylim(0, 115)
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, rel in zip(bars, df['Relative_%']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rel:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 下段: プロファイル
    ax3 = fig.add_subplot(2, 1, 2)
    
    for i, ld in enumerate(lane_data):
        profile = ld['profile']
        x = np.arange(len(profile))
        ax3.plot(x, profile, label=f'Lane {i+1}', alpha=0.7)
        # バンド領域をハイライト
        ax3.axvspan(ld['top'], ld['bottom'], alpha=0.1)
    
    ax3.set_title('Lane Profiles (Corrected)', fontweight='bold')
    ax3.set_xlabel('Position (pixels)')
    ax3.set_ylabel('Intensity')
    ax3.legend(loc='upper right', ncol=min(6, n_lanes), fontsize=8)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Streamlit App
# =============================================================================

st.set_page_config(
    page_title="Western Blot Quantifier",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Western Blot Quantifier")
st.markdown("プロファイルベース測定（ImageJ方式）")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    
    num_lanes = st.number_input("レーン数", min_value=1, max_value=30, value=12)
    exclude_last = st.checkbox("最後のレーン（マーカー）を除外")
    
    st.markdown("---")
    st.markdown("### 📖 使い方")
    st.markdown("""
    1. 画像をアップロード
    2. レーン数を設定
    3. 「定量化」ボタンをクリック
    """)
    
    st.markdown("---")
    st.markdown("### 📎 リンク")
    st.markdown("[GitHub](https://github.com/chibidragon-git/western-blot-quantifier)")

# メインエリア
uploaded_file = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])

if uploaded_file is not None:
    img, gray = load_image(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 元画像")
        st.image(uploaded_file, use_container_width=True)
    
    if st.button("🔬 定量化を実行", type="primary", use_container_width=True):
        with st.spinner("処理中..."):
            results, lane_data = process_image(img, gray, num_lanes, exclude_last)
            
            df = pd.DataFrame(results)
            max_volume = df['Volume'].max()
            df['Relative_%'] = (df['Volume'] / max_volume * 100).round(2) if max_volume > 0 else 0
            
            overlay = create_overlay(img, gray, lane_data, num_lanes)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.subheader("🎯 検出結果")
            st.image(overlay_rgb, use_container_width=True)
        
        st.markdown("---")
        
        # グラフ
        st.subheader("📊 定量結果")
        fig = create_plot(df, lane_data)
        st.pyplot(fig)
        
        # データテーブル
        st.subheader("📋 データ")
        st.dataframe(df, use_container_width=True)
        
        # CSVダウンロード
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 CSVをダウンロード",
            data=csv,
            file_name="quantification_results.csv",
            mime="text/csv"
        )

else:
    st.info("👆 画像をアップロードしてください")
    
    st.markdown("---")
    st.markdown("### ✨ 特徴")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📈 プロファイルベース")
        st.markdown("ImageJと同じ方式で安定した測定")
    
    with col2:
        st.markdown("#### 📊 即座に結果")
        st.markdown("グラフとCSVで定量結果を出力")
    
    with col3:
        st.markdown("#### 🔒 プライバシー")
        st.markdown("データはサーバーに保存されません")
