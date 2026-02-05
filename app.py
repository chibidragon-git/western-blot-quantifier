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

# カスタムCSS（ダークテーマ + 白文字）
def apply_custom_css():
    st.markdown("""
    <style>
    /* ダークテーマ */
    .stApp {
        background: #0f172a;
        color: white !important;
    }
    
    /* 全ての文字を白に */
    .stApp, .stApp * {
        color: white !important;
    }
    
    /* ヘッダー */
    .main-header {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #94a3b8 !important;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* カード */
    .card-title {
        color: white !important;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* サイドバー */
    [data-testid="stSidebar"] {
        background: #1e293b;
    }
    
    [data-testid="stSidebar"], [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* ボタン */
    .stButton > button {
        background: linear-gradient(90deg, #0ea5e9 0%, #10b981 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.4);
    }
    
    /* ファイルアップローダー */
    [data-testid="stFileUploader"] {
        background: #1e293b;
        border-radius: 16px;
        padding: 2rem;
        border: 2px dashed #0ea5e9;
    }
    
    [data-testid="stFileUploader"], [data-testid="stFileUploader"] * {
        color: white !important;
    }
    
    /* 特徴カード */
    .feature-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #0ea5e9;
    }
    
    .feature-title {
        color: white !important;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .feature-desc {
        color: #94a3b8 !important;
        font-size: 0.9rem;
    }
    
    /* データフレーム */
    .stDataFrame, .stDataFrame * {
        color: white !important;
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
        background: #10b981;
        color: white !important;
    }
    
    .badge-weak {
        background: #f59e0b;
        color: white !important;
    }
    
    /* スライダー */
    .stSlider label, .stSlider * {
        color: white !important;
    }
    
    /* スライダーラベル強制 */
    [data-testid="stSlider"] label {
        color: white !important;
    }
    
    [data-testid="stSlider"] p {
        color: white !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        color: white !important;
    }
    
    /* Streamlitのデフォルトテキスト */
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: white !important;
    }
    
    p, span, label, div {
        color: white !important;
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
    """棒グラフを作成（ダークテーマ）"""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f172a')
    
    for ax in axes:
        ax.set_facecolor('#0f172a')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#475569')
        ax.spines['left'].set_color('#475569')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # グラデーションカラー
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(df)))
    
    # Volume グラフ
    axes[0].bar(df['レーン'], df['Volume'], color=colors, edgecolor='none', width=0.7)
    axes[0].set_title('Band Volume', fontweight='bold', color='white', fontsize=14, pad=15)
    axes[0].set_xlabel('レーン', color='white', fontsize=11)
    axes[0].set_ylabel('Volume', color='white', fontsize=11)
    axes[0].grid(axis='y', alpha=0.3, color='#475569')
    
    # Relative グラフ
    bars2 = axes[1].bar(df['レーン'], df['相対値_%'], color=colors, edgecolor='none', width=0.7)
    axes[1].set_title('相対強度 (%)', fontweight='bold', color='white', fontsize=14, pad=15)
    axes[1].set_xlabel('レーン', color='white', fontsize=11)
    axes[1].set_ylabel('相対値 %', color='white', fontsize=11)
    axes[1].set_ylim(0, 120)
    axes[1].axhline(y=100, color='#0ea5e9', linestyle='--', alpha=0.7, linewidth=2)
    axes[1].grid(axis='y', alpha=0.3, color='#475569')
    
    for bar, rel in zip(bars2, df['相対値_%']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rel:.1f}%', ha='center', va='bottom', fontsize=9, 
                    fontweight='bold', color='white')
    
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
st.markdown('<h1 class="main-header">🧬 Western Blot 定量ツール</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">スマートハイブリッド検出 • バンド自動認識</p>', unsafe_allow_html=True)

# サイドバー
with st.sidebar:
    st.markdown("## ⚙️ 設定")
    
    st.markdown("### 🎚️ 閾値")
    low_thresh = st.slider("低閾値（薄いバンド用）", min_value=5, max_value=30, value=10,
                           help="薄いバンドを検出するときの閾値。小さいほど薄いバンドも検出")
    high_thresh = st.slider("高閾値（濃いバンド用）", min_value=15, max_value=50, value=20,
                            help="濃いバンドを検出するときの閾値。大きいほどタイトに検出")
    weak_threshold = st.slider("薄いバンド判定値", min_value=50, max_value=200, value=130,
                               help="この値以下の強度のバンドを「薄いバンド」と判定")
    
    st.markdown("### 🔧 フィルター")
    min_area = st.slider("最小面積", min_value=50, max_value=500, value=100,
                         help="ノイズ除去。この面積以下の検出は除外")
    
    st.markdown("---")
    
    # 使い方
    with st.expander("📖 使い方"):
        st.markdown("""
        **1. 画像をアップロード**
        - Western Blotの画像をドラッグ&ドロップ
        - PNG, JPG, TIFF対応
        
        **2. 「解析」をクリック**
        - バンドを自動検出
        - 緑枠 = 濃いバンド
        - 黄枠 = 薄いバンド
        
        **3. 結果を確認**
        - グラフで相対強度を確認
        - CSVでデータをダウンロード
        
        **💡 うまく検出されない場合**
        - 薄いバンドが小さい → 低閾値を下げる
        - 濃いバンドが大きすぎ → 高閾値を上げる
        - ノイズが多い → 最小面積を上げる
        """)
    
    st.markdown("---")
    st.markdown("### 🔗 リンク")
    st.markdown("[📦 GitHub](https://github.com/chibidragon-git/western-blot-quantifier)")
    st.markdown("---")
    st.markdown("**v4.3** • スマートハイブリッド")

# メインエリア
uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], 
                                  label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📤</div>
        <div style="font-size: 1.2rem; margin-bottom: 0.5rem; color: white;">Western Blot画像をここにドロップ</div>
        <div style="font-size: 0.9rem; color: #94a3b8;">PNG, JPG, TIFF対応</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎯 スマート検出</div>
            <div class="feature-desc">バンドの濃さに応じて自動で閾値を調整</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">⚡ ハイブリッドモード</div>
            <div class="feature-desc">濃いバンド：タイトROI • 薄いバンド：広めROI</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📊 フル解析</div>
            <div class="feature-desc">Volume、相対強度、CSVエクスポート</div>
        </div>
        """, unsafe_allow_html=True)

else:
    img, gray = load_image(uploaded_file)
    h, w = gray.shape
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card-title">📷 元画像</div>', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)
        st.caption(f"サイズ: {w} × {h} px")
    
    with col2:
        st.markdown('<div class="card-title">🎯 検出結果</div>', unsafe_allow_html=True)
        result_placeholder = st.empty()
        result_placeholder.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; 
                    height: 200px; color: #94a3b8; font-style: italic;">
            「解析」をクリックしてバンドを検出
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🔬 解析", type="primary", use_container_width=True):
        with st.spinner("処理中..."):
            bands = detect_bands_smart(gray, low_thresh, high_thresh, weak_threshold, min_area)
            
            if len(bands) == 0:
                st.error("❌ バンドが検出されませんでした。閾値を調整してください。")
            else:
                results = []
                for i, band in enumerate(bands):
                    results.append({
                        'レーン': i + 1,
                        'X': band['x'],
                        'Y': band['y'],
                        '幅': band['width'],
                        '高さ': band['height'],
                        'Volume': round(band['volume'], 0),
                        '平均強度': round(band['mean'], 2),
                        'タイプ': '🟡 薄' if band['strength'] == 'weak' else '🟢 濃',
                    })
                
                df = pd.DataFrame(results)
                max_volume = df['Volume'].max()
                df['相対値_%'] = (df['Volume'] / max_volume * 100).round(2) if max_volume > 0 else 0
                
                overlay = create_overlay(img, bands)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
                with col2:
                    result_placeholder.image(overlay_rgb, use_container_width=True)
                    weak_count = sum(1 for b in bands if b['strength'] == 'weak')
                    strong_count = len(bands) - weak_count
                    
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 0.5rem;">
                        <span class="result-badge badge-strong">🟢 濃: {strong_count}</span>
                        <span class="result-badge badge-weak">🟡 薄: {weak_count}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown('<div class="card-title">📊 定量結果</div>', unsafe_allow_html=True)
                fig = create_plot(df)
                st.pyplot(fig)
                
                st.markdown("---")
                
                st.markdown('<div class="card-title">📋 データ</div>', unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 CSVダウンロード",
                    data=csv,
                    file_name="quantification_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
