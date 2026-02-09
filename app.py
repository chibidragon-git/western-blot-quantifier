#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Western Blot Quantifier v4.4 - Web App
水平プロファイル・ピーク検出方式：バンド位置を正確に特定
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal, ndimage

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
    
    /* サイドバー開閉ボタン */
    [data-testid="collapsedControl"] {
        color: white !important;
        background: #1e293b !important;
    }
    
    button[kind="headerNoPadding"] {
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
        background: white;
        border-radius: 16px;
        padding: 2rem;
        border: 2px dashed #0ea5e9;
    }
    
    [data-testid="stFileUploader"] * {
        color: #1e293b !important;
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


def detect_bands_peak(gray, sensitivity=0.3, min_band_width=10, merge_distance=15):
    """
    水平プロファイル・ピーク検出方式
    
    1. 画像を反転（バンドが暗い→明るいピークに）
    2. 水平方向に投影（各列の平均強度）
    3. ピーク検出でバンドのX位置を特定
    4. 各ピーク周辺で垂直プロファイルからY範囲を決定
    """
    h, w = gray.shape
    
    # 背景推定と反転
    bg = np.percentile(gray, 90)
    inverted = np.maximum(0, bg - gray.astype(np.float64))
    
    # ノイズ除去
    inverted_smooth = cv2.GaussianBlur(inverted.astype(np.float32), (5, 5), 0)
    
    # 水平プロファイル：各列の平均強度
    h_profile = np.mean(inverted_smooth, axis=0)
    
    # プロファイルをスムージング
    if len(h_profile) > 20:
        window = min(15, len(h_profile) // 4)
        if window % 2 == 0:
            window += 1
        if window >= 3:
            h_profile_smooth = signal.savgol_filter(h_profile, window, 2)
        else:
            h_profile_smooth = h_profile
    else:
        h_profile_smooth = h_profile
    
    # ピーク検出
    max_val = np.max(h_profile_smooth)
    if max_val == 0:
        return []
    
    prominence = max_val * sensitivity
    peaks, properties = signal.find_peaks(
        h_profile_smooth,
        prominence=prominence,
        width=min_band_width // 2,
        distance=min_band_width
    )
    
    if len(peaks) == 0:
        return []
    
    # 各ピークからバンドのバウンディングボックスを決定
    bands = []
    
    for peak_x in peaks:
        # ピーク周辺の幅を決定（半値幅ベース）
        peak_height = h_profile_smooth[peak_x]
        half_height = peak_height * 0.3
        
        # 左端を探す
        left = peak_x
        while left > 0 and h_profile_smooth[left] > half_height:
            left -= 1
        
        # 右端を探す
        right = peak_x
        while right < w - 1 and h_profile_smooth[right] > half_height:
            right += 1
        
        band_width = right - left
        if band_width < min_band_width:
            # 最小幅を確保
            center = (left + right) // 2
            left = max(0, center - min_band_width // 2)
            right = min(w - 1, center + min_band_width // 2)
            band_width = right - left
        
        # バンド領域の垂直プロファイルからY範囲を決定
        band_column = inverted_smooth[:, left:right]
        v_profile = np.mean(band_column, axis=1)
        
        # 垂直方向のピーク検出
        v_max = np.max(v_profile)
        if v_max == 0:
            continue
        
        v_half = v_max * 0.2
        
        # 上端を探す
        top = np.argmax(v_profile > v_half)
        # 下端を探す
        bottom = h - 1 - np.argmax(v_profile[::-1] > v_half)
        
        # マージンを追加
        margin_y = max(3, (bottom - top) // 8)
        margin_x = max(2, band_width // 8)
        top = max(0, top - margin_y)
        bottom = min(h - 1, bottom + margin_y)
        left = max(0, left - margin_x)
        right = min(w - 1, right + margin_x)
        
        band_width = right - left
        band_height = bottom - top
        
        if band_width < 5 or band_height < 5:
            continue
        
        # 定量計算
        band_region = inverted[top:bottom, left:right]
        volume = np.sum(band_region)
        mean_intensity = np.mean(band_region)
        max_intensity = np.max(band_region)
        
        bands.append({
            'x': left,
            'y': top,
            'width': band_width,
            'height': band_height,
            'peak_x': peak_x,
            'volume': volume,
            'mean': mean_intensity,
            'max_intensity': max_intensity,
            'strength': 'strong' if max_intensity > 30 else 'weak',
        })
    
    # 近すぎるバンドをマージ
    merged = []
    used = set()
    for i, b1 in enumerate(bands):
        if i in used:
            continue
        group = [b1]
        for j, b2 in enumerate(bands):
            if j <= i or j in used:
                continue
            if abs(b1['peak_x'] - b2['peak_x']) < merge_distance:
                group.append(b2)
                used.add(j)
        
        # グループの中で最もvolumeが大きいものを採用
        best = max(group, key=lambda b: b['volume'])
        # ただし範囲は全グループを包含
        x_min = min(b['x'] for b in group)
        y_min = min(b['y'] for b in group)
        x_max = max(b['x'] + b['width'] for b in group)
        y_max = max(b['y'] + b['height'] for b in group)
        
        best['x'] = x_min
        best['y'] = y_min
        best['width'] = x_max - x_min
        best['height'] = y_max - y_min
        
        # 再計算
        band_region = inverted[y_min:y_max, x_min:x_max]
        best['volume'] = np.sum(band_region)
        best['mean'] = np.mean(band_region)
        
        merged.append(best)
        used.add(i)
    
    merged.sort(key=lambda b: b['x'])
    return merged


def create_overlay(img, bands):
    """検出結果のオーバーレイを作成"""
    overlay = img.copy()
    
    for i, band in enumerate(bands):
        x, y, w, h = band['x'], band['y'], band['width'], band['height']
        color = (0, 255, 0) if band['strength'] == 'strong' else (0, 255, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        
        # ラベル位置の調整
        label_y = max(15, y - 5)
        cv2.putText(overlay, str(i + 1), (x + w // 2 - 5, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return overlay


def create_profile_plot(gray, bands):
    """水平プロファイルとピーク位置を可視化"""
    h, w = gray.shape
    bg = np.percentile(gray, 90)
    inverted = np.maximum(0, bg - gray.astype(np.float64))
    inverted_smooth = cv2.GaussianBlur(inverted.astype(np.float32), (5, 5), 0)
    h_profile = np.mean(inverted_smooth, axis=0)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 3))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    ax.plot(h_profile, color='#0ea5e9', linewidth=1.5, alpha=0.8)
    ax.fill_between(range(len(h_profile)), h_profile, alpha=0.2, color='#0ea5e9')
    
    for i, band in enumerate(bands):
        peak_x = band['peak_x']
        ax.axvline(x=peak_x, color='#10b981', linestyle='--', alpha=0.5)
        ax.annotate(str(i+1), (peak_x, h_profile[peak_x]), 
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=9, fontweight='bold', color='#10b981')
        
        # バンド範囲をハイライト
        ax.axvspan(band['x'], band['x'] + band['width'], alpha=0.1, color='#10b981')
    
    ax.set_title('水平プロファイル & ピーク検出', fontweight='bold', color='white', fontsize=12)
    ax.set_xlabel('X position (px)', color='white')
    ax.set_ylabel('Signal', color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, color='#475569')
    
    plt.tight_layout()
    return fig


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
st.markdown('<p class="sub-header">ピーク検出方式 • バンド自動認識 • レーン数指定不要</p>', unsafe_allow_html=True)

# サイドバー
with st.sidebar:
    st.markdown("## ⚙️ 設定")
    
    st.markdown("### 🎚️ ピーク検出")
    sensitivity = st.slider("検出感度", min_value=0.05, max_value=0.8, value=0.3, step=0.05,
                            help="小さいほど薄いバンドも検出。大きいほど明確なバンドのみ")
    min_band_width = st.slider("最小バンド幅 (px)", min_value=5, max_value=50, value=10,
                               help="これより狭いピークは無視")
    merge_distance = st.slider("マージ距離 (px)", min_value=5, max_value=50, value=15,
                               help="この距離以内のピークは1つのバンドとして統合")
    
    st.markdown("---")
    
    show_profile = st.checkbox("📈 プロファイル表示", value=True,
                               help="水平プロファイルとピーク位置を表示")
    
    st.markdown("---")
    
    # 使い方
    with st.expander("📖 使い方"):
        st.markdown("""
        **1. 画像をアップロード**
        - Western Blotの画像をドラッグ&ドロップ
        - PNG, JPG, TIFF対応
        
        **2. 「解析」をクリック**
        - ピーク検出でバンド位置を自動特定
        - 緑枠 = 濃いバンド
        - 黄枠 = 薄いバンド
        
        **3. 結果を確認**
        - プロファイルでピーク位置を確認
        - グラフで相対強度を確認
        - CSVでデータをダウンロード
        
        **💡 うまく検出されない場合**
        - バンドが少ない → 感度を下げる (0.1-0.2)
        - ノイズで誤検出 → 感度を上げる (0.4-0.6)
        - バンドが分離しない → マージ距離を下げる
        """)
    
    st.markdown("---")
    st.markdown("### 🔗 リンク")
    st.markdown("[📦 GitHub](https://github.com/chibidragon-git/western-blot-quantifier)")
    st.markdown("---")
    st.markdown("**v4.4** • ピーク検出方式")

# メインエリア
uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], 
                                  label_visibility="collapsed")

if uploaded_file is None:
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📤</div>
        <div style="font-size: 1.2rem; margin-bottom: 0.5rem; color: white;">Western Blot画像をここにドロップ</div>
        <div style="font-size: 0.9rem; color: #94a3b8;">PNG, JPG, TIFF対応 • レーン数の指定は不要</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎯 ピーク検出</div>
            <div class="feature-desc">水平プロファイルからバンド位置を正確に特定</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">⚡ 全自動</div>
            <div class="feature-desc">レーン数の指定不要。バンドを自動認識</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📊 フル解析</div>
            <div class="feature-desc">Volume、相対強度、プロファイル可視化、CSV出力</div>
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
        with st.spinner("ピーク検出中..."):
            bands = detect_bands_peak(gray, sensitivity, min_band_width, merge_distance)
            
            if len(bands) == 0:
                st.error("❌ バンドが検出されませんでした。感度を下げてみてください。")
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
                        <span class="result-badge" style="background: #6366f1;">📊 計: {len(bands)}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # プロファイル表示
                if show_profile:
                    st.markdown('<div class="card-title">📈 水平プロファイル</div>', unsafe_allow_html=True)
                    profile_fig = create_profile_plot(gray, bands)
                    st.pyplot(profile_fig)
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
