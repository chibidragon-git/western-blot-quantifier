#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Western Blot Quantifier v4.3 - Web App
スマートハイブリッド方式：濃いバンドは高閾値、薄いバンドは低閾値で検出
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


def detect_bands_smart(gray, low_thresh=10, high_thresh=20, weak_threshold=130, min_area=100):
    """スマートハイブリッド方式でバンドを検出"""
    h, w = gray.shape
    
    bg = np.percentile(gray, 90)
    inverted = np.maximum(0, bg - gray.astype(np.float64)).astype(np.uint8)
    
    kernel = np.ones((3, 3), np.uint8)
    
    # 低閾値で全バンド検出
    _, binary_low = cv2.threshold(inverted, low_thresh, 255, cv2.THRESH_BINARY)
    binary_low = cv2.morphologyEx(binary_low, cv2.MORPH_OPEN, kernel)
    binary_low = cv2.morphologyEx(binary_low, cv2.MORPH_CLOSE, kernel)
    contours_low, _ = cv2.findContours(binary_low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 高閾値で検出
    _, binary_high = cv2.threshold(inverted, high_thresh, 255, cv2.THRESH_BINARY)
    binary_high = cv2.morphologyEx(binary_high, cv2.MORPH_OPEN, kernel)
    binary_high = cv2.morphologyEx(binary_high, cv2.MORPH_CLOSE, kernel)
    contours_high, _ = cv2.findContours(binary_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 高閾値のバンド情報をdict化
    high_bands = {}
    for cnt in contours_high:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > min_area:
            high_bands[x] = (x, y, cw, ch, area, cnt)
    
    # 各バンドを処理
    bands = []
    
    for cnt in contours_low:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > min_area:
            band_region = inverted[y:y+ch, x:x+cw]
            max_val = band_region.max()
            
            # 強度に基づいて判定
            is_weak = max_val < weak_threshold
            
            if is_weak:
                # 薄いバンド → 低閾値の結果を使用
                local_bg = np.percentile(band_region, 90)
                inv_region = np.maximum(0, local_bg - band_region.astype(np.float64))
                volume = np.sum(inv_region)
                mean_intensity = np.mean(inv_region)
                bands.append({
                    'x': x, 'y': y, 'width': cw, 'height': ch,
                    'area': area, 'volume': volume, 'mean': mean_intensity,
                    'strength': 'weak', 'contour': cnt
                })
            else:
                # 濃いバンド → 高閾値の結果を探す
                found = False
                for hx, (hx2, hy, hw, hh, ha, hcnt) in high_bands.items():
                    if abs(x - hx) < 30:
                        hband_region = inverted[hy:hy+hh, hx2:hx2+hw]
                        local_bg = np.percentile(hband_region, 90)
                        inv_region = np.maximum(0, local_bg - hband_region.astype(np.float64))
                        volume = np.sum(inv_region)
                        mean_intensity = np.mean(inv_region)
                        bands.append({
                            'x': hx2, 'y': hy, 'width': hw, 'height': hh,
                            'area': ha, 'volume': volume, 'mean': mean_intensity,
                            'strength': 'strong', 'contour': hcnt
                        })
                        found = True
                        break
                if not found:
                    local_bg = np.percentile(band_region, 90)
                    inv_region = np.maximum(0, local_bg - band_region.astype(np.float64))
                    volume = np.sum(inv_region)
                    mean_intensity = np.mean(inv_region)
                    bands.append({
                        'x': x, 'y': y, 'width': cw, 'height': ch,
                        'area': area, 'volume': volume, 'mean': mean_intensity,
                        'strength': 'weak', 'contour': cnt
                    })
    
    # X座標でソート
    bands.sort(key=lambda b: b['x'])
    
    return bands


def create_overlay(img, bands):
    """検出結果のオーバーレイを作成"""
    overlay = img.copy()
    
    for i, band in enumerate(bands):
        x, y, w, h = band['x'], band['y'], band['width'], band['height']
        # 濃いバンド=緑、薄いバンド=黄色
        color = (0, 255, 0) if band['strength'] == 'strong' else (0, 255, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(overlay, str(i + 1), (x + w // 2 - 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
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

st.title("🧬 Western Blot Quantifier v4.3")
st.markdown("スマートハイブリッド方式：濃いバンドは高閾値、薄いバンドは低閾値で自動検出")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    
    st.subheader("閾値設定")
    low_thresh = st.slider("低閾値（薄いバンド用）", min_value=5, max_value=30, value=10,
                           help="薄いバンドを検出する際の閾値")
    high_thresh = st.slider("高閾値（濃いバンド用）", min_value=15, max_value=50, value=20,
                            help="濃いバンドを検出する際の閾値")
    weak_threshold = st.slider("薄いバンド判定閾値", min_value=50, max_value=200, value=130,
                               help="この値以下の強度のバンドを薄いバンドとして判定")
    
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
            bands = detect_bands_smart(gray, low_thresh, high_thresh, weak_threshold, min_area)
            
            if len(bands) == 0:
                st.error("バンドが検出されませんでした。閾値を調整してください。")
            else:
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
                        'Type': '薄' if band['strength'] == 'weak' else '濃',
                    })
                
                df = pd.DataFrame(results)
                max_volume = df['Volume'].max()
                df['Relative_%'] = (df['Volume'] / max_volume * 100).round(2) if max_volume > 0 else 0
                
                overlay = create_overlay(img, bands)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
                with col2:
                    st.subheader("🎯 検出結果")
                    st.image(overlay_rgb, use_container_width=True)
                    weak_count = sum(1 for b in bands if b['strength'] == 'weak')
                    strong_count = len(bands) - weak_count
                    st.caption(f"{len(bands)}個のバンドを検出（濃:{strong_count}、薄:{weak_count}）")
                
                st.markdown("---")
                
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
    st.markdown("### ✨ v4.3 の特徴")
    st.markdown("""
    - **スマートハイブリッド方式**: バンドの強度に応じて自動で閾値を切り替え
    - **濃いバンド**: 高閾値で精密に検出（緑色で表示）
    - **薄いバンド**: 低閾値で広めに検出（黄色で表示）
    - **パラメータ調整**: サイドバーで閾値を細かく調整可能
    """)
