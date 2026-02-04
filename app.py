#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Western Blot Quantifier v6.0 - Web App
各レーン独自のバンド検出 + 手動ROI調整オプション
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


def detect_band_per_lane(lane_gray, roi_half=25, bg_pct=90):
    """各レーンでバンド位置を検出"""
    h, w = lane_gray.shape
    
    # 縦方向プロファイル
    prof = gaussian_filter1d(np.mean(lane_gray, axis=1), sigma=2)
    bg = np.percentile(prof, bg_pct)
    inv = np.maximum(bg - prof, 0)
    
    if inv.max() < 3:
        return h // 4, 3 * h // 4, 0, 0
    
    # ピーク位置
    pk = np.argmax(inv)
    
    # ROI範囲
    top = max(0, pk - roi_half)
    bottom = min(h - 1, pk + roi_half)
    
    # 積分
    roi = lane_gray[top:bottom+1, :]
    roi_bg = np.percentile(roi, bg_pct)
    inv_roi = np.maximum(roi_bg - roi.astype(np.float64), 0)
    volume = np.sum(inv_roi)
    mean_intensity = np.mean(inv_roi)
    
    return top, bottom, volume, mean_intensity


def find_global_band_region(gray, threshold_ratio=0.3, margin_ratio=0.5):
    """グローバルなバンド領域を検出（フォールバック用）"""
    h, w = gray.shape
    
    profile = np.mean(gray, axis=1)
    smoothed = gaussian_filter1d(profile, sigma=1.5)
    bg_val = np.percentile(smoothed, 90)
    inverted = np.maximum(bg_val - smoothed, 0)
    
    if inverted.max() < 1:
        return 0, h - 1
    
    peak = np.argmax(inverted)
    thresh = inverted[peak] * threshold_ratio
    
    top = peak
    while top > 0 and inverted[top] > thresh:
        top -= 1
    bottom = peak
    while bottom < h - 1 and inverted[bottom] > thresh:
        bottom += 1
    
    margin = int((bottom - top) * margin_ratio)
    top = max(0, top - margin)
    bottom = min(h - 1, bottom + margin)
    
    return top, bottom


def process_image(img, gray, num_lanes, exclude_last=False, 
                  mode='per_lane', roi_half=25, bg_pct=90,
                  manual_top=None, manual_bottom=None):
    """画像を処理"""
    h, w = gray.shape
    lane_width = w // num_lanes
    
    results = []
    lane_data = []
    
    total_lanes = num_lanes - 1 if exclude_last else num_lanes
    
    # モードに応じて処理
    if mode == 'manual' and manual_top is not None and manual_bottom is not None:
        # 手動ROI
        global_top, global_bottom = manual_top, manual_bottom
        use_global = True
    elif mode == 'global':
        # グローバル自動検出
        global_top, global_bottom = find_global_band_region(gray)
        use_global = True
    else:
        # 各レーン独自
        use_global = False
        global_top, global_bottom = 0, h - 1
    
    for i in range(total_lanes):
        x_start = i * lane_width
        x_end = (i + 1) * lane_width if i < num_lanes - 1 else w
        
        lane_gray = gray[:, x_start:x_end]
        
        if use_global:
            # グローバルROI
            roi = lane_gray[global_top:global_bottom+1, :]
            roi_bg = np.percentile(roi, bg_pct)
            inv = np.maximum(roi_bg - roi.astype(np.float64), 0)
            volume = np.sum(inv)
            mean_int = np.mean(inv)
            lane_top, lane_bottom = global_top, global_bottom
        else:
            # レーン独自
            lane_top, lane_bottom, volume, mean_int = detect_band_per_lane(
                lane_gray, roi_half, bg_pct
            )
        
        results.append({
            'Lane': i + 1,
            'Volume': round(volume, 0),
            'Mean': round(mean_int, 2),
        })
        
        lane_data.append({
            'x_start': x_start,
            'x_end': x_end,
            'top': lane_top,
            'bottom': lane_bottom,
        })
    
    return results, lane_data, global_top, global_bottom


def create_overlay(img, gray, lane_data, num_lanes, global_top, global_bottom, use_global=True):
    """検出結果のオーバーレイを作成"""
    h, w = gray.shape
    lane_width = w // num_lanes
    
    overlay = img.copy()
    
    # グローバルROI線（使用時のみ）
    if use_global:
        cv2.line(overlay, (0, global_top), (w, global_top), (0, 255, 0), 1)
        cv2.line(overlay, (0, global_bottom), (w, global_bottom), (0, 255, 0), 1)
    
    # レーン境界線
    for i in range(num_lanes + 1):
        x = i * lane_width
        cv2.line(overlay, (x, 0), (x, h), (255, 100, 100), 1)
    
    # レーン番号とROI
    for i, ld in enumerate(lane_data):
        pt1 = (ld['x_start'] + 2, ld['top'])
        pt2 = (ld['x_end'] - 2, ld['bottom'])
        cv2.rectangle(overlay, pt1, pt2, (0, 255, 0), 2)
        
        cx = (ld['x_start'] + ld['x_end']) // 2
        cv2.putText(overlay, str(i + 1), (cx - 10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
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

st.title("🧬 Western Blot Quantifier v6.0")
st.markdown("レーンごとのバンド検出 + 手動ROI調整対応")

# サイドバー
with st.sidebar:
    st.header("⚙️ 基本設定")
    
    num_lanes = st.number_input("レーン数", min_value=1, max_value=30, value=12)
    exclude_last = st.checkbox("最後のレーン（マーカー）を除外", value=False)
    
    st.markdown("---")
    st.header("🎯 ROI検出モード")
    
    mode = st.radio(
        "検出モード",
        options=['global', 'per_lane', 'manual'],
        format_func=lambda x: {
            'global': '🌐 グローバル（全レーン共通）',
            'per_lane': '🔍 レーンごと（個別検出）',
            'manual': '✋ 手動設定'
        }[x],
        index=0
    )
    
    manual_top = None
    manual_bottom = None
    
    if mode == 'manual':
        st.markdown("### 手動ROI設定")
        manual_top = st.slider("ROI上端 (Y)", 0, 200, 20)
        manual_bottom = st.slider("ROI下端 (Y)", 0, 200, 80)
    
    st.markdown("---")
    st.header("🔧 詳細パラメータ")
    
    roi_half = st.slider("ROI半径（per_laneモード用）", 10, 50, 25)
    bg_pct = st.slider("背景パーセンタイル", 80, 98, 90)
    
    st.markdown("---")
    st.markdown("### 📎 リンク")
    st.markdown("[GitHub](https://github.com/chibidragon-git/western-blot-quantifier)")

# メインエリア
uploaded_file = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])

if uploaded_file is not None:
    img, gray = load_image(uploaded_file)
    h, w = gray.shape
    
    # 手動モードの場合、スライダーの最大値を更新
    if mode == 'manual':
        with st.sidebar:
            manual_top = st.slider("ROI上端 (Y)", 0, h, min(manual_top or 20, h), key="top2")
            manual_bottom = st.slider("ROI下端 (Y)", 0, h, min(manual_bottom or 80, h), key="bottom2")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 元画像")
        st.image(uploaded_file, use_container_width=True)
        st.caption(f"サイズ: {w} x {h}")
    
    if st.button("🔬 定量化を実行", type="primary", use_container_width=True):
        with st.spinner("処理中..."):
            results, lane_data, global_top, global_bottom = process_image(
                img, gray, num_lanes, exclude_last,
                mode=mode, roi_half=roi_half, bg_pct=bg_pct,
                manual_top=manual_top, manual_bottom=manual_bottom
            )
            
            df = pd.DataFrame(results)
            max_volume = df['Volume'].max()
            df['Relative_%'] = (df['Volume'] / max_volume * 100).round(2) if max_volume > 0 else 0
            
            use_global = mode in ['global', 'manual']
            overlay = create_overlay(img, gray, lane_data, num_lanes, global_top, global_bottom, use_global)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.subheader("🎯 検出結果")
            st.image(overlay_rgb, use_container_width=True)
            if use_global:
                st.caption(f"ROI: Y = {global_top} ~ {global_bottom}")
            else:
                st.caption("各レーンで個別にROIを検出")
        
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
    st.markdown("### ✨ v6.0 の新機能")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔍 レーンごと検出")
        st.markdown("各レーンで独自にバンド位置を検出（スマイリング対応）")
    
    with col2:
        st.markdown("#### ✋ 手動ROI調整")
        st.markdown("スライダーでROI範囲を手動設定可能")
    
    with col3:
        st.markdown("#### 🔧 パラメータ調整")
        st.markdown("背景補正やROIサイズを細かく調整")
