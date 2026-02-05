#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Western Blot Quantifier v4.2 - Web App
輪郭検出ベースのバンド認識
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


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


def detect_bands(gray, min_area=100, threshold=20):
    """輪郭検出でバンドを認識"""
    h, w = gray.shape
    
    # 背景を推定
    bg = np.percentile(gray, 90)
    
    # 反転して二値化
    inverted = np.maximum(0, bg - gray.astype(np.float64)).astype(np.uint8)
    _, binary = cv2.threshold(inverted, threshold, 255, cv2.THRESH_BINARY)
    
    # モルフォロジー処理でノイズ除去
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # バンド情報を抽出
    bands = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > min_area:
            # バンド領域の強度を計算
            band_region = gray[y:y+ch, x:x+cw]
            local_bg = np.percentile(band_region, 90)
            inv_region = np.maximum(0, local_bg - band_region.astype(np.float64))
            volume = np.sum(inv_region)
            mean_intensity = np.mean(inv_region)
            
            bands.append({
                'x': x,
                'y': y,
                'width': cw,
                'height': ch,
                'area': area,
                'volume': volume,
                'mean': mean_intensity,
                'contour': cnt
            })
    
    # X座標でソート（左から右）
    bands.sort(key=lambda b: b['x'])
    
    return bands, binary


def process_image(img, gray, min_area=100, threshold=20):
    """画像を処理"""
    bands, binary = detect_bands(gray, min_area, threshold)
    
    results = []
    for i, band in enumerate(bands):
        results.append({
            'Lane': i + 1,
            'X': band['x'],
            'Y': band['y'],
            'Width': band['width'],
            'Height': band['height'],
            'Volume': round(band['volume'], 0),
            'Mean': round(band['mean'], 2),
        })
    
    return results, bands, binary


def create_overlay(img, bands):
    """検出結果のオーバーレイを作成"""
    overlay = img.copy()
    
    for i, band in enumerate(bands):
        x, y, w, h = band['x'], band['y'], band['width'], band['height']
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(overlay, str(i + 1), (x + w // 2 - 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return overlay


def create_plot(df):
    """棒グラフを作成"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = plt.cm.viridis(df['Relative_%'] / 100)
    
    axes[0].bar(df['Lane'], df['Volume'], color=colors, edgecolor='black')
    axes[0].set_title('Band Volume', fontweight='bold')
    axes[0].set_xlabel('Lane')
    axes[0].set_ylabel('Volume')
    axes[0].grid(axis='y', alpha=0.3)
    
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

st.title("🧬 Western Blot Quantifier v4.2")
st.markdown("輪郭検出ベースのバンド自動認識")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    
    threshold = st.slider("検出閾値", min_value=5, max_value=50, value=20,
                          help="バンドと背景を分ける閾値")
    min_area = st.slider("最小面積", min_value=50, max_value=500, value=100,
                         help="ノイズ除去のための最小バンド面積")
    
    st.markdown("---")
    st.markdown("### 📎 リンク")
    st.markdown("[GitHub](https://github.com/chibidragon-git/western-blot-quantifier)")

# メインエリア
uploaded_file = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])

if uploaded_file is not None:
    img, gray = load_image(uploaded_file)
    h, w = gray.shape
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 元画像")
        st.image(uploaded_file, use_container_width=True)
        st.caption(f"サイズ: {w} x {h}")
    
    if st.button("🔬 定量化を実行", type="primary", use_container_width=True):
        with st.spinner("処理中..."):
            results, bands, binary = process_image(img, gray, min_area, threshold)
            
            if len(results) == 0:
                st.error("バンドが検出されませんでした。閾値を調整してください。")
            else:
                df = pd.DataFrame(results)
                max_volume = df['Volume'].max()
                df['Relative_%'] = (df['Volume'] / max_volume * 100).round(2) if max_volume > 0 else 0
                
                overlay = create_overlay(img, bands)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
                with col2:
                    st.subheader("🎯 検出結果")
                    st.image(overlay_rgb, use_container_width=True)
                    st.caption(f"{len(bands)}個のバンドを検出")
                
                st.markdown("---")
                
                # 二値化画像を表示
                with st.expander("🔍 二値化画像を表示"):
                    st.image(binary, use_container_width=True, caption="二値化結果")
                
                st.subheader("📊 定量結果")
                fig = create_plot(df)
                st.pyplot(fig)
                
                st.subheader("📋 データ")
                st.dataframe(df, use_container_width=True)
                
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
    st.markdown("### ✨ v4.2 の特徴")
    st.markdown("""
    - **輪郭検出**: バンドの形を自動認識
    - **ノイズ除去**: モルフォロジー処理
    - **パラメータ調整**: 閾値と最小面積を調整可能
    """)
