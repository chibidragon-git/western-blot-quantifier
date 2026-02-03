#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageJマクロ - ウェスタンブロット定量化
このマクロをImageJにドラッグ&ドロップして実行
"""

IMAGEJ_MACRO = '''
// =====================================================
// Western Blot Quantification Macro
// =====================================================
// 使い方:
// 1. 画像を開く
// 2. このマクロを実行
// 3. プロンプトに従ってレーン数を入力
// 4. 結果が自動で保存される
// =====================================================

macro "Western Blot Quantifier" {
    // 画像が開いているか確認
    if (nImages == 0) {
        path = File.openDialog("画像を選択してください");
        open(path);
    }
    
    // 画像情報を取得
    title = getTitle();
    dir = getDirectory("image");
    if (dir == "") {
        dir = getDirectory("home");
    }
    
    // レーン数を取得
    Dialog.create("Western Blot Quantifier");
    Dialog.addNumber("レーン数:", 12);
    Dialog.addCheckbox("等間隔で自動分割", true);
    Dialog.addNumber("バンド領域の高さ (%):", 20);
    Dialog.show();
    
    numLanes = Dialog.getNumber();
    autoSplit = Dialog.getCheckbox();
    bandHeightPercent = Dialog.getNumber();
    
    // グレースケールに変換
    run("8-bit");
    
    // 画像サイズを取得
    width = getWidth();
    height = getHeight();
    laneWidth = floor(width / numLanes);
    bandHeight = floor(height * bandHeightPercent / 100);
    
    // 測定設定
    run("Set Measurements...", "area mean integrated redirect=None decimal=3");
    
    // 結果テーブルをクリア
    run("Clear Results");
    
    if (autoSplit) {
        // 自動分割モード
        // 各レーンの最も明るい（バンド）領域を検出
        
        for (i = 0; i < numLanes; i++) {
            x = i * laneWidth;
            
            // レーン全体のプロファイルを取得してバンド位置を検出
            makeRectangle(x, 0, laneWidth, height);
            
            // プロファイルを取得
            profile = getProfile();
            
            // 最大値の位置を見つける（バンド位置）
            maxVal = 0;
            maxPos = height / 2;
            for (j = 0; j < profile.length; j++) {
                if (profile[j] < maxVal || maxVal == 0) {  // 暗い=バンド
                    maxVal = profile[j];
                    maxPos = j;
                }
            }
            
            // バンド領域のROIを設定
            bandY = maxPos - bandHeight / 2;
            if (bandY < 0) bandY = 0;
            if (bandY + bandHeight > height) bandY = height - bandHeight;
            
            makeRectangle(x, bandY, laneWidth, bandHeight);
            
            // 測定
            run("Measure");
            
            // ROIをオーバーレイに追加
            Overlay.addSelection("red", 2);
        }
        
        Overlay.show();
        
    } else {
        // 手動選択モード
        for (i = 0; i < numLanes; i++) {
            x = i * laneWidth;
            makeRectangle(x, height/4, laneWidth, height/2);
            
            waitForUser("レーン " + (i+1) + " を調整してください\nOKを押して次へ");
            run("Measure");
            Overlay.addSelection("red", 2);
        }
        Overlay.show();
    }
    
    // 結果を保存
    baseName = replace(title, ".png", "");
    baseName = replace(baseName, ".jpg", "");
    baseName = replace(baseName, ".tif", "");
    
    // CSV保存
    saveAs("Results", dir + baseName + "_results.csv");
    
    // 結果を表示
    print("=====================================================");
    print("Western Blot Quantification Results");
    print("=====================================================");
    print("Image: " + title);
    print("Lanes: " + numLanes);
    print("Results saved to: " + dir + baseName + "_results.csv");
    print("=====================================================");
    
    // 相対値を計算して表示
    maxIntDen = 0;
    for (i = 0; i < nResults; i++) {
        intDen = getResult("IntDen", i);
        if (intDen > maxIntDen) maxIntDen = intDen;
    }
    
    print("\\nRelative Quantification (%):");
    for (i = 0; i < nResults; i++) {
        intDen = getResult("IntDen", i);
        relVal = intDen / maxIntDen * 100;
        print("Lane " + (i+1) + ": " + d2s(relVal, 1) + "%");
    }
    
    showMessage("完了", "定量化が完了しました\\n結果: " + dir + baseName + "_results.csv");
}
'''

def create_macro_file(output_path=None):
    """ImageJマクロファイルを作成"""
    if output_path is None:
        output_path = "WesternBlotQuantifier.ijm"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(IMAGEJ_MACRO)
    
    print(f"✅ マクロファイルを作成しました: {output_path}")
    print("   このファイルをImageJ/Fijiにドラッグ&ドロップして使用してください")
    return output_path


if __name__ == "__main__":
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else None
    create_macro_file(output)
