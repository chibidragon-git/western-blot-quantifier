#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Western Blot Quantifier - ImageJ Automation Wrapper
ImageJのGel Analyzer機能を簡単に使えるようにしたツール

使い方:
    python quantify.py -i image.png
    python quantify.py -i image.png -l 12 -o results/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='Western Blot Quantifier - ImageJ Automation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用例:
  python quantify.py -i my_blot.png
  python quantify.py -i my_blot.png -l 12 -o ./results
  python quantify.py -i my_blot.png --interactive
        '''
    )
    
    parser.add_argument('-i', '--image', 
                       type=str, 
                       required=True,
                       help='画像ファイルのパス（必須）')
    
    parser.add_argument('-l', '--lanes', 
                       type=int, 
                       default=None,
                       help='レーン数（指定しない場合は自動検出）')
    
    parser.add_argument('-o', '--output', 
                       type=str, 
                       default=None,
                       help='出力ディレクトリ（デフォルト: 画像と同じ場所）')
    
    parser.add_argument('--interactive',
                       action='store_true',
                       help='インタラクティブモード（ImageJ GUIで手動選択）')
    
    parser.add_argument('--headless',
                       action='store_true',
                       help='ヘッドレスモード（GUI無しで自動処理）')
    
    parser.add_argument('--fiji-path',
                       type=str,
                       default=None,
                       help='Fiji/ImageJのインストールパス')
    
    return parser.parse_args()


def find_fiji():
    """Fijiのインストールパスを探す"""
    possible_paths = [
        '/Applications/Fiji.app',  # macOS
        os.path.expanduser('~/Fiji.app'),
        '/opt/fiji',  # Linux
        os.path.expanduser('~/fiji'),
        'C:\\Fiji.app',  # Windows
        os.path.expanduser('~\\Fiji.app'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def init_imagej(fiji_path=None, headless=False):
    """ImageJ (PyImageJ) を初期化"""
    try:
        import imagej
    except ImportError:
        print("❌ PyImageJがインストールされていません")
        print("   pip install pyimagej")
        sys.exit(1)
    
    if fiji_path is None:
        fiji_path = find_fiji()
    
    if fiji_path and os.path.exists(fiji_path):
        print(f"🔬 Fiji を起動中: {fiji_path}")
        mode = 'headless' if headless else 'interactive'
        ij = imagej.init(fiji_path, mode=mode)
    else:
        print("🔬 ImageJ を起動中（Fiji が見つからないためデフォルトを使用）")
        mode = 'headless' if headless else 'interactive'
        ij = imagej.init(mode=mode)
    
    print(f"   ImageJ バージョン: {ij.getVersion()}")
    return ij


def load_image(ij, image_path):
    """画像を読み込む"""
    print(f"📷 画像を読み込み中: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ 画像が見つかりません: {image_path}")
        sys.exit(1)
    
    # ImageJで画像を開く
    dataset = ij.io().open(image_path)
    print(f"   サイズ: {dataset.dimension(0)} x {dataset.dimension(1)}")
    
    return dataset


def run_gel_analyzer(ij, dataset, num_lanes=None):
    """Gel Analyzer を実行"""
    print("📊 Gel Analyzer を実行中...")
    
    # 画像をImageJのImagePlusに変換
    imp = ij.py.to_imageplus(dataset)
    
    # グレースケールに変換
    ij.IJ.run(imp, "8-bit", "")
    
    # レーン選択のためのマクロを実行
    if num_lanes:
        # レーン数が指定されている場合、等間隔で分割
        width = imp.getWidth()
        lane_width = width // num_lanes
        
        # 各レーンのROIを設定
        macro_code = f"""
        // レーン分割
        width = {width};
        numLanes = {num_lanes};
        laneWidth = width / numLanes;
        
        for (i = 0; i < numLanes; i++) {{
            x = i * laneWidth;
            makeRectangle(x, 0, laneWidth, getHeight());
            run("Measure");
        }}
        """
        ij.py.run_macro(macro_code)
    else:
        # インタラクティブモード: Gel Analyzerを開く
        ij.IJ.run(imp, "Gel Analyzer Options...", "")
        print("   💡 ImageJでレーンを選択してください")
        print("   1. 最初のレーンを選択 → Ctrl+1")
        print("   2. 次のレーンを選択 → Ctrl+2")
        print("   3. 最後のレーンを選択後 → Ctrl+3 でプロット")
    
    return imp


def extract_measurements(ij):
    """測定結果を抽出"""
    print("📈 測定結果を取得中...")
    
    # Results テーブルから値を取得
    rt = ij.ResultsTable.getResultsTable()
    
    if rt is None or rt.size() == 0:
        print("⚠️  測定結果がありません")
        return None
    
    # DataFrameに変換
    data = []
    for i in range(rt.size()):
        row = {
            'Lane': i + 1,
            'Area': rt.getValue("Area", i),
            'Mean': rt.getValue("Mean", i),
            'IntDen': rt.getValue("IntDen", i) if rt.columnExists("IntDen") else None,
            'RawIntDen': rt.getValue("RawIntDen", i) if rt.columnExists("RawIntDen") else None,
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 相対値を計算
    if 'IntDen' in df.columns and df['IntDen'].notna().any():
        max_val = df['IntDen'].max()
        df['Relative_%'] = (df['IntDen'] / max_val * 100).round(2)
    elif 'Mean' in df.columns:
        max_val = df['Mean'].max()
        df['Relative_%'] = (df['Mean'] / max_val * 100).round(2)
    
    return df


def save_results(df, output_dir, prefix):
    """結果を保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV保存
    csv_path = os.path.join(output_dir, f'{prefix}_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"💾 CSV保存: {csv_path}")
    
    # グラフ作成
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 絶対値
    value_col = 'IntDen' if 'IntDen' in df.columns and df['IntDen'].notna().any() else 'Mean'
    colors = plt.cm.plasma(df[value_col] / df[value_col].max())
    
    axes[0].bar(df['Lane'], df[value_col], color=colors, edgecolor='black')
    axes[0].set_title('Band Intensity', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Lane')
    axes[0].set_ylabel('Integrated Density')
    axes[0].set_xticks(df['Lane'])
    
    # 相対値
    axes[1].bar(df['Lane'], df['Relative_%'], color=colors, edgecolor='black')
    axes[1].set_title('Relative Intensity (%)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Lane')
    axes[1].set_ylabel('Relative %')
    axes[1].set_xticks(df['Lane'])
    axes[1].set_ylim(0, 110)
    axes[1].axhline(y=100, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{prefix}_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 グラフ保存: {plot_path}")
    plt.close()
    
    return csv_path, plot_path


def print_summary(df):
    """結果のサマリーを表示"""
    print("\n" + "="*60)
    print("定量化結果")
    print("="*60)
    
    value_col = 'IntDen' if 'IntDen' in df.columns and df['IntDen'].notna().any() else 'Mean'
    
    print(df[['Lane', value_col, 'Relative_%']].to_string(index=False))
    
    print("\n強度順:")
    sorted_df = df.sort_values(value_col, ascending=False)
    ranking = sorted_df['Lane'].tolist()
    print(f"  {ranking}")


def main():
    """メイン処理"""
    print("="*60)
    print("Western Blot Quantifier v1.0")
    print("="*60)
    
    args = parse_args()
    
    # 出力ディレクトリの設定
    if args.output is None:
        args.output = os.path.dirname(os.path.abspath(args.image))
    
    prefix = Path(args.image).stem
    
    # ImageJを初期化
    ij = init_imagej(fiji_path=args.fiji_path, headless=args.headless)
    
    try:
        # 画像を読み込み
        dataset = load_image(ij, args.image)
        
        # Gel Analyzerを実行
        imp = run_gel_analyzer(ij, dataset, num_lanes=args.lanes)
        
        if args.interactive:
            print("\n" + "="*60)
            print("📌 インタラクティブモード")
            print("="*60)
            print("ImageJで以下の操作を行ってください:")
            print("1. Analyze → Gels → Select First Lane")
            print("2. レーンを選択してドラッグ")
            print("3. Analyze → Gels → Select Next Lane (繰り返し)")
            print("4. 最後に Analyze → Gels → Plot Lanes")
            print("5. Wand ツールで各ピークをクリック")
            print("6. 完了したらこのウィンドウで Enter を押してください")
            input("\n[Enter] を押して結果を取得...")
        
        # 測定結果を抽出
        df = extract_measurements(ij)
        
        if df is not None:
            # 結果を保存
            save_results(df, args.output, prefix)
            
            # サマリーを表示
            print_summary(df)
        
        print("\n✅ 完了！")
        
    finally:
        # ImageJを終了（ヘッドレスモードの場合）
        if args.headless:
            ij.dispose()


if __name__ == "__main__":
    main()
