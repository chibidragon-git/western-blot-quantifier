#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Western Blot Quantifier v5.0 - Web App
PDCA最適化済み: 固定バンド領域 + ローカル背景補正 (v8方式)
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
    
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        gray = img_array
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    return img_bgr, gray


def find_band_region(gray):
    """画像全体からバンド領域のY範囲を検出"""
    h, w = gray.shape
    
    # 縦方向プロファイル
    profile = np.mean(gray, axis=1)
    # スムージングを適切に
    smoothed = gaussian_filter1d(profile, sigma=1.5)
    
    # 背景（明るい部分）
    bg_val = np.percentile(smoothed, 90)
    
    # 反転（暗い=高シグナル）
    inverted = np.maximum(bg_val - smoothed, 0)
    
    if inverted.max() < 1:
        return 0, h - 1
    
    # 最も暗い行（バンドの中心）
    min_row = np.argmax(inverted)
    
    # バンド領域の閾値（ピークの40%）
    threshold = inverted[min_row] * 0.4
    
    # 上端を探す
    top = min_row
    while top > 0 and inverted[top] > threshold:
        top -= 1
    
    # 下端を探す
    bottom = min_row
    while bottom < h - 1 and inverted[bottom] > threshold:
        bottom += 1
    
    # 余裕を持たせる（バンド全体の50%分）
    margin = (bottom - top) // 2
    top = max(0, top - margin)
    bottom = min(h - 1, bottom + margin)
    
    return top, bottom


def measure_lane(lane_gray, band_top, band_bottom):
    """レーンの強度を測定 (PDCA v8方式)"""
    # バンド領域を切り出し
    band_region = lane_gray[band_top:band_bottom+1, :]
    
    # ローカル背景（上位10%パーセンタイル = 最も明るい部分）
    # これによりレーンごとの背景ムラを吸収
    local_bg = np.percentile(band_region, 90)
    
    # 反転して積分
    inverted = local_bg - band_region.astype(np.float64)
    inverted = np.maximum(inverted, 0)
    
    volume = np.sum(inverted)
    mean_intensity = np.mean(inverted)
    
    return volume, mean_intensity


def process_image(img, gray, num_lanes, exclude_last=False):
    """画像を処理"""
    h, w = gray.shape
    lane_width = w // num_lanes
    
    # バンド領域を検出
    band_top, band_bottom = find_band_region(gray)
    
    results = []
    lane_data = []
    
    total_lanes = num_lanes - 1 if exclude_last else num_lanes
    
    for i in range(total_lanes):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        
        lane_gray = gray[:, x_start:x_end]
        volume, mean_int = measure_lane(lane_gray, band_top, band_bottom)
        
        results.append({
            'Lane': i + 1,
            'Volume': round(volume, 0),
            'Mean': round(mean_int, 2),
        })
        
        lane_data.append({
            'x_start': x_start,
            'x_end': x_end,
        })
    
    return results, lane_data, band_top, band_bottom


def create_overlay(img, gray, lane_data, num_lanes, band_top, band_bottom):
    """検出結果のオーバーレイを作成"""
    h, w = gray.shape
    lane_width = w // num_lanes
    
    overlay = img.copy()
    
    # 背景を少し暗く
    overlay = cv2.addWeighted(overlay, 0.7, np.zeros(overlay.shape, overlay.dtype), 0, 0)
    
    # バンド領域の横線
    cv2.line(overlay, (0, band_top), (w, band_top), (0, 255, 0), 1)
    cv2.line(overlay, (0, band_bottom), (w, band_bottom), (0, 255, 0), 1)
    
    # レーン境界線
    for i in range(num_lanes + 1):
        x = i * lane_width
        cv2.line(overlay, (x, 0), (x, h), (255, 100, 100), 1)
    
    # レーン番号とROI
    for i, ld in enumerate(lane_data):
        # ROI矩形
        pt1 = (ld['x_start'] + 2, band_top)
        pt2 = (ld['x_end'] - 2, band_bottom)
        cv2.rectangle(overlay, pt1, pt2, (0, 255, 0), 2)
        
        cx = (ld['x_start'] + ld['x_end']) // 2
        cv2.putText(overlay, str(i + 1), (cx - 10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return overlay


def create_plot(df):
    """棒グラフを作成"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = plt.cm.viridis(df['Relative_%'] / 100)
    
    # Volume
    axes[0].bar(df['Lane'], df['Volume'], color=colors, edgecolor='black')
    axes[0].set_title('Band Volume (Integrated Intensity)', fontweight='bold')
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
                    f'{rel:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
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

st.title("🧬 Western Blot Quantifier v5.0")
st.markdown("PDCA最適化済み: 高精度バンド検出アルゴリズム")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    
    num_lanes = st.number_input("レーン数", min_value=1, max_value=30, value=12)
    exclude_last = st.checkbox("最後のレーン（マーカー）を除外", value=False)
    
    st.markdown("---")
    st.markdown("### 📖 使い方")
    st.markdown("""
    1. 画像をアップロード
    2. レーン数を設定
    3. 「定量化」ボタンをクリック
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ 最適化済みアルゴリズム")
    st.markdown("""
    - **プロファイルベースY範囲検出**: 縦方向の強度分布から最適なバンド領域を自動決定。
    - **Local Background Subtraction**: 各レーン内で背景を動的に推定し、シグナルのみを抽出。
    - **Integrated Intensity**: ROI内の全ピクセル強度を積分し、微細な差も正確にキャッチ。
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
            results, lane_data, band_top, band_bottom = process_image(
                img, gray, num_lanes, exclude_last
            )
            
            df = pd.DataFrame(results)
            max_volume = df['Volume'].max()
            df['Relative_%'] = (df['Volume'] / max_volume * 100).round(2) if max_volume > 0 else 0
            
            overlay = create_overlay(img, gray, lane_data, num_lanes, band_top, band_bottom)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.subheader("🎯 検出結果")
            st.image(overlay_rgb, use_container_width=True)
            st.caption(f"自動検出バンド領域: Y = {band_top} ~ {band_bottom}")
        
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
    
    st.markdown("---")
    st.markdown("### ✨ 特徴")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎯 科学的アプローチ")
        st.markdown("ImageJに近いプロファイル積分方式を採用")
    
    with col2:
        st.markdown("#### 📊 視認性の高い結果")
        st.markdown("ヒートマップカラーのグラフで一目瞭然")
    
    with col3:
        st.markdown("#### 🔒 完全ローカル処理")
        st.markdown("ブラウザ上で動作し、データはサーバーに保存されません")
