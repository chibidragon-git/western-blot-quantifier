#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Western Blot Quantifier v2.0 - Web App
レーンベース検出 + 強化ノイズフィルタリング
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter1d


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


def denoise_lane(lane_gray):
    """強力なノイズ除去"""
    # バイラテラルフィルタ（エッジを保持しながらノイズ除去）
    denoised = cv2.bilateralFilter(lane_gray, 9, 75, 75)
    # メディアンフィルタ（salt-and-pepperノイズ除去）
    denoised = cv2.medianBlur(denoised, 3)
    return denoised


def detect_band_in_lane(lane_gray, sensitivity=1.5):
    """レーン内のバンドを検出（v3.2ベースの強化版）"""
    h, w = lane_gray.shape
    
    # 強力なノイズ除去
    denoised = denoise_lane(lane_gray)
    
    # CLAHE（コントラスト強調）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # ガウシアンブラー
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Otsu's threshold（メイン）
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Adaptive threshold（補助）
    block_size = max(11, int(41 * sensitivity)) | 1
    c_value = max(3, int(10 / sensitivity))
    binary_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_value
    )
    
    # 両方のAND（確実なバンドのみ）
    binary = cv2.bitwise_and(binary_otsu, binary_adaptive)
    
    # モルフォロジー（ノイズ除去強化）
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # オープニング（小さいノイズ除去）
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    # クロージング（バンド内の穴を埋める）
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, binary
    
    # 形状フィルタリング
    valid_contours = []
    min_area = h * w * 0.005  # 最小面積
    max_area = h * w * 0.8    # 最大面積
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        x, y, cw, ch = cv2.boundingRect(contour)
        aspect_ratio = cw / ch if ch > 0 else 0
        
        # バンドは横長〜正方形（縦長すぎは除外）
        if aspect_ratio < 0.3:
            continue
        
        # Solidity（凸包充填率）
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < 0.3:
            continue
        
        valid_contours.append((contour, area))
    
    if not valid_contours:
        return None, binary
    
    # 最大の有効輪郭を選択
    largest_contour = max(valid_contours, key=lambda x: x[1])[0]
    
    return largest_contour, binary


def measure_lane(lane_gray, contour=None):
    """レーンの強度を測定"""
    h, w = lane_gray.shape
    
    # 背景推定（上端と下端）
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
        
        area = cv2.contourArea(contour)
    else:
        # バンドが検出されなかった場合、中央領域を使用
        band_region = lane_gray[int(h*0.2):int(h*0.8), :]
        band_pixels = band_region.flatten()
        cy = h // 2
        area = 0
    
    if len(band_pixels) == 0:
        return 0, 0, cy, 0
    
    # 強度計算（暗い = 高シグナル）
    inverted = 255 - band_pixels.astype(np.float64)
    bg_corrected_value = 255 - bg_intensity
    corrected = np.maximum(inverted - bg_corrected_value * 0.7, 0)
    
    volume = np.sum(corrected)
    mean_intensity = np.mean(corrected)
    
    return volume, mean_intensity, cy, area


def process_image(img, gray, num_lanes, exclude_last=False, sensitivity=1.5):
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
        
        contour, binary = detect_band_in_lane(lane_gray, sensitivity)
        volume, mean_int, cy, area = measure_lane(lane_gray, contour)
        
        if contour is not None:
            contour_global = contour.copy()
            contour_global[:, :, 0] += x_start
        else:
            contour_global = None
        
        results.append({
            'Lane': i + 1,
            'Volume': round(volume, 0),
            'Mean': round(mean_int, 2),
            'Area': area
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
st.markdown("ウェスタンブロットのバンドを自動定量化（レーンベース検出）")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    
    num_lanes = st.number_input("レーン数", min_value=1, max_value=30, value=12)
    exclude_last = st.checkbox("最後のレーン（マーカー）を除外")
    
    st.markdown("---")
    
    st.markdown("### 🔧 詳細設定")
    sensitivity = st.slider("検出感度", min_value=0.5, max_value=3.0, value=1.5, step=0.1,
                           help="高い値 = より多くのバンドを検出（ノイズも増える可能性）")
    
    st.markdown("---")
    st.markdown("### 📖 使い方")
    st.markdown("""
    1. 画像をアップロード
    2. レーン数を設定
    3. 必要に応じて感度調整
    4. 「定量化」ボタンをクリック
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
            results, lane_data = process_image(img, gray, num_lanes, exclude_last, sensitivity)
            
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
        st.dataframe(df[['Lane', 'Volume', 'Mean', 'Relative_%']], use_container_width=True)
        
        # CSVダウンロード
        csv = df[['Lane', 'Volume', 'Mean', 'Area', 'Relative_%']].to_csv(index=False).encode('utf-8-sig')
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
        st.markdown("#### 🔬 レーンベース検出")
        st.markdown("各レーン内で個別にバンドを検出、ノイズに強い")
    
    with col2:
        st.markdown("#### 📊 即座に結果")
        st.markdown("グラフとCSVで定量結果を出力")
    
    with col3:
        st.markdown("#### 🔒 プライバシー")
        st.markdown("データはサーバーに保存されません")
