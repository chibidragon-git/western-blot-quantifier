# Western Blot Quantifier

ImageJを使ったウェスタンブロット定量化ツール。
ImageJの精度をそのまま活かしつつ、操作を簡単にしたラッパー。

## 特徴

- 🎯 ImageJのGel Analyzer機能を自動化
- 🖱️ バンド選択はImageJのUIで直感的に
- 📊 定量結果を自動でCSV出力
- 📈 グラフも自動生成

## 必要なもの

- Python 3.8+
- Fiji (ImageJ) - https://fiji.sc/

## インストール

```bash
pip install -r requirements.txt
```

## 使い方

### 基本的な使い方

```bash
python quantify.py -i your_image.png
```

### オプション

```bash
python quantify.py -i image.png -l 12 -o results/
```

- `-i, --image`: 画像ファイルのパス（必須）
- `-l, --lanes`: レーン数（オプション、自動検出も可能）
- `-o, --output`: 出力ディレクトリ（デフォルト: 画像と同じ場所）

## ワークフロー

1. コマンドを実行
2. ImageJが自動起動、画像が読み込まれる
3. レーン分割線が表示される
4. バンドをクリックで選択
5. 「Measure」で定量実行
6. 結果がCSVとグラフで出力される

## 出力ファイル

- `{image_name}_results.csv` - 定量データ
- `{image_name}_plot.png` - バーグラフ
- `{image_name}_normalized.csv` - ローディングコントロール正規化済みデータ

## ライセンス

MIT License
