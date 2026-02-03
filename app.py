#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Western Blot Quantifier - Web App
Streamlitで動くWebアプリ版
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64


def load_image(uploaded_file):
    """アップロードされた画像を読み込む"""
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # グレースケールに変換
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        gray = img_array
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    return img_bgr, gray


def detect_band_in_lane(lane_gray):
    """レーン内のバンドを検出"""
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
        return None, binary
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < h * w * 0.01:
        return None, binary
    
    return largest_contour, binary


def measure_lane(lane_gray, contour=None):
    """レーンの強度を測定"""
    h, w = lane_gray.shape
    
    # 背景推定
    bg_top = lane_gray[:max(1, int(h*0.1)), :].flatten()
    bg_bottom = lane_gray[int(h*0.9):, :].flatten()
    bg_intensity = np.median(np.concatenate([bg_top, bg_bottom]))
    
    if contour is not None:
        mask = np.zeros(lane_gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        band_pixels = lane_gray[mask == 255]
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, bw, bh = cv2.boundingRect(contour)
            cy = y + bh // 2
    else:
        band_region = lane_gray[int(h*0.2):int(h*0.8), :]
        band_pixels = band_region.flatten()
        cy = h // 2
    
    if len(band_pixels) == 0:
        return 0, 0, cy, 0
    
    inverted = 255 - band_pixels.astype(np.float64)
    bg_corrected_value = 255 - bg_intensity
    corrected = np.maximum(inverted - bg_corrected_value * 0.7, 0)
    
    volume = np.sum(corrected)
    mean_intensity = np.mean(corrected)
    
    return volume, mean_intensity, cy, len(band_pixels)


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
        
        contour, binary = detect_band_in_lane(lane_gray)
        volume, mean_int, cy, area = measure_lane(lane_gray, contour)
        
        if contour is not None:
            contour_global = contour.copy()
            contour_global[:, :, 0] += x_start
        else:
            contour_global = None
        
        results.append({
            'Lane': i + 1,
            'Volume': round(volume, 0),
            'Mean': round(mean_int, 2)
        })
        
        lane_data.append({
            'contour': contour_global,
            'binary': binary,
            'x_start': x_start,
            'x_end': x_end
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
    
    # バンド輪郭
    for i, ld in enumerate(lane_data):
        if ld['contour'] is not None:
            cv2.drawContours(overlay, [ld['contour']], -1, (0, 255, 0), 2)
        
        cx = (ld['x_start'] + ld['x_end']) // 2
        cv2.putText(overlay, str(i + 1), (cx - 10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return overlay


def create_plot(df):
    """棒グラフを作成"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = plt.cm.viridis(df['Relative_%'] / 100)
    
    # Volume
    axes[0].bar(df['Lane'], df['Volume'], color=colors, edgecolor='black')
    axes[0].set_title('Band Volume', fontweight='bold')
    axes[0].set_xlabel('Lane')
    axes[0].set_ylabel('Volume')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Relative %
    bars = axes[1].bar(df['Lane'], df['Relative_%'], color=colors, edgecolor='black')
    axes[1].set_title('Relative Intensity (%)', fontweight='bold')
    axes[1].set_xlabel('Lane')
    axes[1].set_ylabel('Relative %')
    axes[1].set_ylim(0, 115)
    axes[1].axhline(y=100, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar, rel in zip(bars, df['Relative_%']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rel:.1f}%', ha='center', va='bottom', fontsize=8)
    
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
st.markdown("ウェスタンブロットのバンドを自動定量化")

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
    # 画像を表示
    img, gray = load_image(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 元画像")
        st.image(uploaded_file, use_container_width=True)
    
    # 定量化ボタン
    if st.button("🔬 定量化を実行", type="primary", use_container_width=True):
        with st.spinner("処理中..."):
            # 処理
            results, lane_data = process_image(img, gray, num_lanes, exclude_last)
            
            # DataFrame
            df = pd.DataFrame(results)
            max_volume = df['Volume'].max()
            df['Relative_%'] = (df['Volume'] / max_volume * 100).round(2) if max_volume > 0 else 0
            
            # オーバーレイ
            overlay = create_overlay(img, gray, lane_data, num_lanes)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.subheader("🎯 検出結果")
            st.image(overlay_rgb, use_container_width=True)
        
        st.markdown("---")
        
        # グラフ
        st.subheader("📊 定量結果")
        fig = create_plot(df)
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
    
    # デモ用の説明
    st.markdown("---")
    st.markdown("### ✨ 特徴")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔬 高精度検出")
        st.markdown("OpenCVベースの画像処理でバンドを自動検出")
    
    with col2:
        st.markdown("#### 📊 即座に結果")
        st.markdown("グラフとCSVで定量結果を出力")
    
    with col3:
        st.markdown("#### 🔒 プライバシー")
        st.markdown("データは送信されません（ローカル処理）")
