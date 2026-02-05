#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Western Blot Quantifier - 使い方ガイド
"""

import streamlit as st

st.set_page_config(
    page_title="使い方ガイド - Western Blot Quantifier",
    page_icon="📖",
    layout="wide"
)

# カスタムCSS
st.markdown("""
<style>
.stApp {
    background: #0f172a;
}
.stApp, .stApp * {
    color: white !important;
}
[data-testid="stSidebar"] {
    background: #1e293b;
}
.step-card {
    background: #1e293b;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid #0ea5e9;
}
.step-number {
    background: linear-gradient(90deg, #0ea5e9 0%, #10b981 100%);
    color: white !important;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 1.2rem;
    margin-right: 1rem;
}
.step-title {
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.step-desc {
    color: #94a3b8 !important;
    line-height: 1.6;
}
.tip-card {
    background: #164e63;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.warning-card {
    background: #78350f;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 📖 使い方ガイド")
st.markdown("---")

# ステップ1
st.markdown("""
<div class="step-card">
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span class="step-number">1</span>
        <span class="step-title">画像をアップロード</span>
    </div>
    <div class="step-desc">
        Western Blotの画像をドラッグ&ドロップ、またはクリックして選択します。<br>
        <strong>対応形式:</strong> PNG, JPG, JPEG, TIFF
    </div>
</div>
""", unsafe_allow_html=True)

# サンプル画像表示エリア
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### ✅ 良い画像の例")
    st.markdown("""
    - 背景が均一（白または明るいグレー）
    - バンドがはっきり見える
    - 各レーンが区別できる
    """)

with col2:
    st.markdown("#### ❌ 避けたい画像")
    st.markdown("""
    - 背景にムラがある
    - 画像が暗すぎる/明るすぎる
    - ノイズや汚れが多い
    """)

st.markdown("---")

# ステップ2
st.markdown("""
<div class="step-card">
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span class="step-number">2</span>
        <span class="step-title">「解析」ボタンをクリック</span>
    </div>
    <div class="step-desc">
        バンドが自動検出されます。<br>
        <strong>🟢 緑枠</strong> = 濃いバンド（高閾値で検出）<br>
        <strong>🟡 黄枠</strong> = 薄いバンド（低閾値で検出）
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ステップ3
st.markdown("""
<div class="step-card">
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span class="step-number">3</span>
        <span class="step-title">結果を確認・調整</span>
    </div>
    <div class="step-desc">
        検出結果がおかしい場合は、サイドバーの閾値を調整してください。
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🎚️ 閾値の調整方法")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="tip-card">
        <strong>💡 薄いバンドが検出されない</strong><br>
        → 「低閾値」を下げる（10→5など）
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tip-card">
        <strong>💡 濃いバンドの枠が大きすぎる</strong><br>
        → 「高閾値」を上げる（20→30など）
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="tip-card">
        <strong>💡 ノイズを拾ってしまう</strong><br>
        → 「最小面積」を上げる（100→200など）
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tip-card">
        <strong>💡 濃いバンドが薄いと判定される</strong><br>
        → 「薄いバンド判定値」を下げる
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ステップ4
st.markdown("""
<div class="step-card">
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span class="step-number">4</span>
        <span class="step-title">データをダウンロード</span>
    </div>
    <div class="step-desc">
        「CSVダウンロード」ボタンで定量結果をダウンロードできます。<br>
        Excel等で開いて、さらに解析できます。
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# FAQ
st.markdown("## ❓ よくある質問")

with st.expander("バンドが1つも検出されない"):
    st.markdown("""
    - 低閾値を下げてみてください（5まで）
    - 最小面積を下げてみてください（50まで）
    - 画像のコントラストが低い可能性があります
    """)

with st.expander("バンドが分離して検出される"):
    st.markdown("""
    - 1つのバンドが2つに分かれる場合
    - 高閾値を下げてみてください
    - 画像の前処理（コントラスト調整）を試してください
    """)

with st.expander("相対値の計算方法は？"):
    st.markdown("""
    - 最も大きいVolumeを100%として計算
    - Volume = バンド領域の積分強度（背景を引いた値）
    """)

st.markdown("---")
st.markdown("### 🔙 [メインページに戻る](/)")
