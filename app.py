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

# カスタムCSS
def apply_custom_css():
    st.markdown("""
    <style>
    /* メインコンテナ */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* ヘッダー */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #a0aec0;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* カード */
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .card-title {
        color: #e2e8f0;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* サイドバー */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2f 0%, #2d2d44 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* ボタン */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* ファイルアップローダー */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        border: 2px dashed rgba(102, 126, 234, 0.5);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* データフレーム */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
    }
    
    /* スライダー */
    .stSlider > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* メトリクス */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stMetricValue"] {
        color: #667eea;
        font-weight: 700;
    }
    
    /* 成功/エラーメッセージ */
    .stSuccess, .stInfo {
        background: rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* 特徴カード */
    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    
    .feature-title {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .feature-desc {
        color: #a0aec0;
        font-size: 0.9rem;
    }
    
    /* 結果バッジ */
    .result-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-strong {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .badge-weak {
        background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


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
    
    bands.sort(key=lambda b: b['x'])
    return bands


def create_overlay(img, bands):
    """検出結果のオーバーレイを作成"""
    overlay = img.copy()
    
    for i, band in enumerate(bands):
        x, y, w, h = band['x'], band['y'], band['width'], band['height']
        color = (0, 255, 0) if band['strength'] == 'strong' else (0, 255, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(overlay, str(i + 1), (x + w // 2 - 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return overlay


def create_plot(df):
    """スタイリッシュな棒グラフを作成"""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    
    for ax in axes:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#a0aec0')
        ax.spines['bottom'].set_color('#4a5568')
        ax.spines['left'].set_color('#4a5568')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # グラデーションカラー
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(df)))
    
    # Volume グラフ
    bars1 = axes[0].bar(df['Lane'], df['Volume'], color=colors, edgecolor='none', width=0.7)
    axes[0].set_title('Band Volume', fontweight='bold', color='#e2e8f0', fontsize=14, pad=15)
    axes[0].set_xlabel('Lane', color='#a0aec0', fontsize=11)
    axes[0].set_ylabel('Volume', color='#a0aec0', fontsize=11)
    axes[0].grid(axis='y', alpha=0.2, color='#4a5568')
    
    # Relative グラフ
    bars2 = axes[1].bar(df['Lane'], df['Relative_%'], color=colors, edgecolor='none', width=0.7)
    axes[1].set_title('Relative Intensity (%)', fontweight='bold', color='#e2e8f0', fontsize=14, pad=15)
    axes[1].set_xlabel('Lane', color='#a0aec0', fontsize=11)
    axes[1].set_ylabel('Relative %', color='#a0aec0', fontsize=11)
    axes[1].set_ylim(0, 120)
    axes[1].axhline(y=100, color='#667eea', linestyle='--', alpha=0.7, linewidth=2)
    axes[1].grid(axis='y', alpha=0.2, color='#4a5568')
    
    for bar, rel in zip(bars2, df['Relative_%']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rel:.1f}%', ha='center', va='bottom', fontsize=9, 
                    fontweight='bold', color='#e2e8f0')
    
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

apply_custom_css()

# ヘッダー
st.markdown('<h1 class="main-header">🧬 Western Blot Quantifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Smart Hybrid Detection • Automatic Band Recognition</p>', unsafe_allow_html=True)

# サイドバー
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    st.markdown("### 🎚️ Threshold")
    low_thresh = st.slider("Low (weak bands)", min_value=5, max_value=30, value=10)
    high_thresh = st.slider("High (strong bands)", min_value=15, max_value=50, value=20)
    weak_threshold = st.slider("Weak band cutoff", min_value=50, max_value=200, value=130)
    
    st.markdown("### 🔧 Filter")
    min_area = st.slider("Min area", min_value=50, max_value=500, value=100)
    
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[📦 GitHub](https://github.com/chibidragon-git/western-blot-quantifier)")
    st.markdown("---")
    st.markdown("**v4.3** • Smart Hybrid")

# メインエリア
uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], 
                                  label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #a0aec0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📤</div>
        <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">Drop your Western Blot image here</div>
        <div style="font-size: 0.9rem; color: #718096;">Supports PNG, JPG, TIFF</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎯 Smart Detection</div>
            <div class="feature-desc">Automatically adjusts threshold based on band intensity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">⚡ Hybrid Mode</div>
            <div class="feature-desc">Strong bands: tight ROI • Weak bands: wider ROI</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📊 Full Analysis</div>
            <div class="feature-desc">Volume, relative intensity, and CSV export</div>
        </div>
        """, unsafe_allow_html=True)

else:
    img, gray = load_image(uploaded_file)
    h, w = gray.shape
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card-title">📷 Original Image</div>', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)
        st.caption(f"Size: {w} × {h} px")
    
    with col2:
        st.markdown('<div class="card-title">🎯 Detection Result</div>', unsafe_allow_html=True)
        result_placeholder = st.empty()
        result_placeholder.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; 
                    height: 200px; color: #718096; font-style: italic;">
            Click "Analyze" to detect bands
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🔬 Analyze", type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            bands = detect_bands_smart(gray, low_thresh, high_thresh, weak_threshold, min_area)
            
            if len(bands) == 0:
                st.error("❌ No bands detected. Try adjusting the thresholds.")
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
                        'Type': '🟡 Weak' if band['strength'] == 'weak' else '🟢 Strong',
                    })
                
                df = pd.DataFrame(results)
                max_volume = df['Volume'].max()
                df['Relative_%'] = (df['Volume'] / max_volume * 100).round(2) if max_volume > 0 else 0
                
                overlay = create_overlay(img, bands)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
                with col2:
                    result_placeholder.image(overlay_rgb, use_container_width=True)
                    weak_count = sum(1 for b in bands if b['strength'] == 'weak')
                    strong_count = len(bands) - weak_count
                    
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 0.5rem;">
                        <span class="result-badge badge-strong">🟢 Strong: {strong_count}</span>
                        <span class="result-badge badge-weak">🟡 Weak: {weak_count}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown('<div class="card-title">📊 Quantification Results</div>', unsafe_allow_html=True)
                fig = create_plot(df)
                st.pyplot(fig)
                
                st.markdown("---")
                
                st.markdown('<div class="card-title">📋 Data Table</div>', unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="quantification_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
