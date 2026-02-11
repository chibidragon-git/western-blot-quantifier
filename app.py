#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Western Blot Quantifier v5.0 - Web App
Multi-Threshold Erosion + Profile-based Y/X Fitting
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal


# カスタムCSS（ダークテーマ + 白文字）
def apply_custom_css():
    st.markdown("""
    <style>
    .stApp { background: #0f172a; color: white !important; }
    .stApp, .stApp * { color: white !important; }
    .main-header { color: white !important; font-size: 2.5rem; font-weight: 800; text-align: center; padding: 1rem 0; margin-bottom: 0.5rem; }
    .sub-header { color: #94a3b8 !important; text-align: center; font-size: 1rem; margin-bottom: 2rem; }
    .card-title { color: white !important; font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem; }
    [data-testid="stSidebar"] { background: #1e293b; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="collapsedControl"] { color: white !important; background: #1e293b !important; }
    button[kind="headerNoPadding"] { color: white !important; }
    .stButton > button { background: linear-gradient(90deg, #0ea5e9 0%, #10b981 100%); color: white !important; border: none; border-radius: 12px; padding: 0.75rem 2rem; font-weight: 600; font-size: 1rem; box-shadow: 0 4px 15px rgba(14, 165, 233, 0.4); }
    [data-testid="stFileUploader"] { background: white; border-radius: 16px; padding: 2rem; border: 2px dashed #0ea5e9; }
    [data-testid="stFileUploader"] * { color: #1e293b !important; }
    .feature-card { background: #1e293b; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid #0ea5e9; }
    .feature-title { color: white !important; font-weight: 600; margin-bottom: 0.3rem; }
    .feature-desc { color: #94a3b8 !important; font-size: 0.9rem; }
    .stDataFrame, .stDataFrame * { color: white !important; }
    .result-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin: 0.25rem; }
    .badge-strong { background: #10b981; color: white !important; }
    .badge-weak { background: #f59e0b; color: white !important; }
    .stSlider label, .stSlider * { color: white !important; }
    [data-testid="stSlider"] label { color: white !important; }
    [data-testid="stSlider"] p { color: white !important; }
    .stSlider [data-baseweb="slider"] { color: white !important; }
    .stMarkdown, .stMarkdown p, .stMarkdown span { color: white !important; }
    p, span, label, div { color: white !important; }
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


def detect_band_rows(gray):
    """垂直プロファイルからバンド行を検出"""
    bg = np.percentile(gray, 95)
    inverted = np.maximum(0, bg - gray.astype(np.float64))
    inverted_smooth = cv2.GaussianBlur(inverted.astype(np.float32), (5, 5), 0)
    v_profile = np.mean(inverted_smooth, axis=1)
    
    peaks, props = signal.find_peaks(v_profile, prominence=np.max(v_profile) * 0.12, distance=15)
    
    if len(peaks) == 0:
        peaks = np.array([np.argmax(v_profile)])
    
    rows = []
    for p in peaks:
        rows.append({
            'y': int(p),
            'intensity': float(v_profile[p]),
            'label': f"行 {len(rows)+1} (y={p})"
        })
    
    return rows


def detect_bands_v7(gray, target_y, merge_threshold=0.68):
    """
    v7: Multi-Threshold Erosion + Profile-based Y/X Fitting
    
    1. Multi-threshold vertical erosion → band X centers
    2. Cluster X centers → robust lane positions
    3. Width consistency check (iterative merge)
    4. Per-band Y fitting via vertical profile
    5. Per-band X trimming
    6. Height capping
    """
    h, w = gray.shape
    
    bg = np.percentile(gray, 95)
    inverted = np.maximum(0, bg - gray.astype(np.float64)).astype(np.uint8)
    inverted_f = inverted.astype(np.float32)
    inverted_smooth = cv2.GaussianBlur(inverted_f, (5, 5), 0)
    blurred = cv2.GaussianBlur(inverted, (3, 3), 0)
    max_val = int(np.max(blurred))
    
    if max_val < 5:
        return []
    
    y_search_top = max(0, target_y - 18)
    y_search_bot = min(h, target_y + 18)
    
    # Step 1: Multi-threshold erosion for X positions
    all_cx = []
    for thresh_pct in range(20, 70, 3):
        thresh_val = max_val * thresh_pct // 100
        if thresh_val < 1:
            continue
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
        
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        eroded = cv2.erode(binary, v_kernel, iterations=1)
        
        mask = np.zeros_like(eroded)
        mask[y_search_top:y_search_bot, :] = eroded[y_search_top:y_search_bot, :]
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 5 or bh < 2 or bw > w * 0.25:
                continue
            all_cx.append(x + bw / 2)
    
    if not all_cx:
        return []
    
    # Step 2: Cluster X centers
    all_cx.sort()
    cluster_dist = max(8, w // 40)
    
    clusters = []
    current = [all_cx[0]]
    for cx in all_cx[1:]:
        if cx - np.mean(current) < cluster_dist:
            current.append(cx)
        else:
            clusters.append(current)
            current = [cx]
    clusters.append(current)
    
    min_votes = max(2, len(all_cx) // 40)
    band_centers = []
    for cl in clusters:
        if len(cl) >= min_votes:
            band_centers.append(int(np.median(cl)))
    
    band_centers.sort()
    
    if not band_centers:
        return []
    
    # Step 3: Create boundaries from centers
    boundaries = []
    for i in range(len(band_centers)):
        if i == 0:
            left = max(0, band_centers[i] - (band_centers[1] - band_centers[i]) // 2) if len(band_centers) > 1 else 0
        else:
            left = (band_centers[i-1] + band_centers[i]) // 2
        
        if i == len(band_centers) - 1:
            right = min(w, band_centers[i] + (band_centers[i] - band_centers[i-1]) // 2) if len(band_centers) > 1 else w
        else:
            right = (band_centers[i] + band_centers[i+1]) // 2
        
        boundaries.append((left, right))
    
    # Step 4: Iterative width merge
    final_boundaries = list(boundaries)
    for iteration in range(5):
        widths = [r - l for l, r in final_boundaries]
        if not widths:
            break
        median_width = np.median(widths)
        
        merged_any = False
        new_boundaries = []
        i = 0
        while i < len(final_boundaries):
            l, r = final_boundaries[i]
            bw = r - l
            if bw < median_width * merge_threshold:
                if i + 1 < len(final_boundaries):
                    l_next, r_next = final_boundaries[i + 1]
                    bw_next = r_next - l_next
                    if bw_next < median_width * merge_threshold:
                        new_boundaries.append((l, r_next))
                        merged_any = True
                        i += 2
                        continue
                    else:
                        new_boundaries.append((l, r_next))
                        merged_any = True
                        i += 2
                        continue
                elif i > 0 and new_boundaries:
                    prev_l, prev_r = new_boundaries[-1]
                    new_boundaries[-1] = (prev_l, r)
                    merged_any = True
                    i += 1
                    continue
            new_boundaries.append((l, r))
            i += 1
        
        final_boundaries = new_boundaries
        if not merged_any:
            break
    
    # Step 5: Per-band Y/X fitting
    max_row_signal = np.max(inverted_smooth[y_search_top:y_search_bot, :].mean(axis=1))
    
    raw_bands = []
    for xs, xe in final_boundaries:
        strip = inverted_smooth[y_search_top:y_search_bot, xs:xe]
        v_prof = np.mean(strip, axis=1)
        peak_in_range = np.argmax(v_prof)
        peak_val = v_prof[peak_in_range]
        
        if peak_val < max_row_signal * 0.08:
            continue
        
        thresh_y = peak_val * 0.15
        y_s = peak_in_range
        while y_s > 0 and v_prof[y_s] > thresh_y:
            y_s -= 1
        y_e = peak_in_range
        while y_e < len(v_prof) - 1 and v_prof[y_e] > thresh_y:
            y_e += 1
        
        y_start = y_search_top + y_s
        y_end = y_search_top + y_e + 1
        
        band_region = inverted_smooth[y_start:y_end, xs:xe]
        h_prof = np.mean(band_region, axis=0)
        h_max = np.max(h_prof)
        if h_max < 1:
            continue
        trim_t = h_max * 0.15
        
        x_s = 0
        while x_s < len(h_prof) - 1 and h_prof[x_s] < trim_t:
            x_s += 1
        x_e = len(h_prof) - 1
        while x_e > 0 and h_prof[x_e] < trim_t:
            x_e -= 1
        
        ax = xs + x_s
        axe = xs + x_e + 1
        if axe - ax < 3:
            continue
        
        raw_bands.append({
            'x': ax, 'y': y_start, 'w': axe - ax, 'h': y_end - y_start,
            'peak_val': peak_val, 'peak_x': (ax + axe) // 2
        })
    
    # Step 6: Height capping
    if not raw_bands:
        return []
    
    heights = [b['h'] for b in raw_bands]
    median_h = np.median(heights)
    max_h = int(median_h * 1.3)
    
    bands = []
    for rb in raw_bands:
        y_start, y_end = rb['y'], rb['y'] + rb['h']
        if rb['h'] > max_h:
            cy = (y_start + y_end) // 2
            y_start = cy - max_h // 2
            y_end = y_start + max_h
        
        region = inverted_f[y_start:y_end, rb['x']:rb['x'] + rb['w']]
        vol = float(np.sum(region))
        mean_val = float(np.mean(region))
        max_intensity = float(np.max(region))
        
        bands.append({
            'x': rb['x'],
            'y': y_start,
            'width': rb['w'],
            'height': y_end - y_start,
            'peak_x': rb['peak_x'],
            'volume': vol,
            'mean': mean_val,
            'max_intensity': max_intensity,
            'strength': 'strong' if max_intensity > 30 else 'weak',
        })
    
    return bands


def create_overlay(img, bands):
    """検出結果のオーバーレイを作成"""
    overlay = img.copy()
    h, w = overlay.shape[:2]
    thickness = max(1, min(2, w // 200))
    font_scale = max(0.3, min(0.6, w / 500))
    
    for i, band in enumerate(bands):
        x, y, bw, bh = band['x'], band['y'], band['width'], band['height']
        color = (0, 255, 0) if band['strength'] == 'strong' else (0, 255, 255)
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, thickness)
        
        label_y = max(15, y - 5)
        cv2.putText(overlay, str(i + 1), (x + bw // 2 - 5, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), 1)
    
    return overlay


def create_profile_plot(gray, bands, target_y):
    """水平プロファイルとバンド位置を可視化"""
    h, w = gray.shape
    bg = np.percentile(gray, 95)
    inverted = np.maximum(0, bg - gray.astype(np.float64))
    inverted_smooth = cv2.GaussianBlur(inverted.astype(np.float32), (5, 5), 0)
    
    y_top = max(0, target_y - 18)
    y_bot = min(h, target_y + 18)
    h_profile = np.mean(inverted_smooth[y_top:y_bot, :], axis=0)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 3))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    ax.plot(h_profile, color='#0ea5e9', linewidth=1.5, alpha=0.8)
    ax.fill_between(range(len(h_profile)), h_profile, alpha=0.2, color='#0ea5e9')
    
    for i, band in enumerate(bands):
        cx = band['peak_x']
        ax.axvspan(band['x'], band['x'] + band['width'], alpha=0.1, color='#10b981')
        if cx < len(h_profile):
            ax.annotate(str(i+1), (cx, h_profile[min(cx, len(h_profile)-1)]),
                       textcoords="offset points", xytext=(0, 10),
                       ha='center', fontsize=9, fontweight='bold', color='#10b981')
    
    ax.set_title('水平プロファイル & バンド位置', fontweight='bold', color='white', fontsize=12)
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
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(df)))
    
    axes[0].bar(df['レーン'], df['Volume'], color=colors, edgecolor='none', width=0.7)
    axes[0].set_title('Band Volume', fontweight='bold', color='white', fontsize=14, pad=15)
    axes[0].set_xlabel('レーン', color='white', fontsize=11)
    axes[0].set_ylabel('Volume', color='white', fontsize=11)
    axes[0].grid(axis='y', alpha=0.3, color='#475569')
    
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

st.markdown('<h1 class="main-header">🧬 Western Blot 定量ツール</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Threshold Band Detection • 全自動バンド認識 • レーン数指定不要</p>', unsafe_allow_html=True)

# サイドバー
with st.sidebar:
    st.markdown("## ⚙️ 設定")
    
    merge_threshold = st.slider("マージ閾値", min_value=0.5, max_value=0.85, value=0.68, step=0.02,
                                help="バンド幅が中央値のこの割合未満なら隣とマージ")
    
    st.markdown("---")
    
    show_profile = st.checkbox("📈 プロファイル表示", value=True)
    
    st.markdown("---")
    
    with st.expander("📖 使い方"):
        st.markdown("""
        **1. 画像をアップロード**
        - Western Blotの画像をドラッグ&ドロップ
        - PNG, JPG, TIFF対応
        
        **2. バンド行を選択**
        - 複数行がある場合、解析したい行を選択
        
        **3. 「解析」をクリック**
        - 自動でバンドを検出・定量
        - 緑枠 = 濃いバンド、黄枠 = 薄いバンド
        
        **4. 結果を確認**
        - プロファイルでバンド位置を確認
        - グラフで相対強度を確認
        - CSVでデータをダウンロード
        
        **💡 うまく検出されない場合**
        - バンドが分離しすぎる → マージ閾値を上げる
        - バンドが統合される → マージ閾値を下げる
        """)
    
    st.markdown("---")
    st.markdown("### 🔗 リンク")
    st.markdown("[📦 GitHub](https://github.com/chibidragon-git/western-blot-quantifier)")
    st.markdown("---")
    st.markdown("**v5.0** • Multi-Threshold Detection")

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
            <div class="feature-title">🎯 Multi-Threshold検出</div>
            <div class="feature-desc">複数閾値でバンド位置を安定検出。vertical erosionで隣接バンドを分離</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📐 Tight Bounding Box</div>
            <div class="feature-desc">各バンドの上下左右をプロファイルから個別にフィット</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🔄 自動マージ</div>
            <div class="feature-desc">幅の一貫性チェックで分割されすぎたバンドを自動統合</div>
        </div>
        """, unsafe_allow_html=True)

else:
    img, gray = load_image(uploaded_file)
    h, w = gray.shape
    
    # バンド行検出
    band_rows = detect_band_rows(gray)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card-title">📷 元画像</div>', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)
        st.caption(f"サイズ: {w} × {h} px • {len(band_rows)} 行検出")
    
    with col2:
        st.markdown('<div class="card-title">🎯 検出結果</div>', unsafe_allow_html=True)
        result_placeholder = st.empty()
        result_placeholder.markdown("""
        <div style="display: flex; align-items: center; justify-content: center;
                    height: 200px; color: #94a3b8; font-style: italic;">
            「解析」をクリックしてバンドを検出
        </div>
        """, unsafe_allow_html=True)
    
    # バンド行選択
    if len(band_rows) > 1:
        row_options = [r['label'] for r in band_rows]
        selected_row = st.selectbox("📍 解析するバンド行を選択", row_options)
        target_y = band_rows[row_options.index(selected_row)]['y']
    else:
        target_y = band_rows[0]['y'] if band_rows else h // 2
    
    if st.button("🔬 解析", type="primary", use_container_width=True):
        with st.spinner("バンド検出中..."):
            bands = detect_bands_v7(gray, target_y, merge_threshold)
            
            if len(bands) == 0:
                st.error("❌ バンドが検出されませんでした。マージ閾値を調整してみてください。")
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
                
                if show_profile:
                    st.markdown('<div class="card-title">📈 水平プロファイル</div>', unsafe_allow_html=True)
                    profile_fig = create_profile_plot(gray, bands, target_y)
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
