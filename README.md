# msot-cnn3

CMIP 系 GCM アンサンブル（42 モデル x 165 年）の 4 月の予測因子（msot = mrso, snc, sst, tsl など）から、
MJJASO などの季節降水量をグリッドごとの CNN で予測する実験コード群です。
各グリッド点に 1 つずつ CNN を学習させ（例: タイ域 20x20 = 400 モデル）、検証・可視化を行います。

## ディレクトリ構成

各ディレクトリは独立した実験スナップショットで、同じ名前のモジュール（`util.py`, `model.py`, `view.py` など）を
ディレクトリごとに持っています。スクリプトは **そのディレクトリに `cd` してから実行** してください
（`from util import ...` のようにカレントディレクトリのモジュールを import するため）。

| ディレクトリ | 内容 | 主なエントリポイント |
|---|---|---|
| `class/one/` | 1 点（one）の降水量をクラス分類 | `cnn3_main.py`（学習）, `evaluate.py` |
| `class/thailand/` | タイ域 20x20 グリッドのクラス分類 | `cnn3_main.py`, `evaluate.py`, `test.py` |
| `class/monsoon/` | モンスーン域 14x18 グリッドのクラス分類（EarlyStopping 付き） | `main.py`, `evaluate.py` |
| `continuous/one/` | 1 点の連続値予測とクラス境界作成 | `cnn3_main.py`, `mk_bnd.py`（境界作成）, `class_conversion.py`, `evaluate.py` |
| `continuous/thailand/default/` | タイ域の連続値予測（基本設定） | `cnn3_main.py`, `evaluate.py`, `transfer.py`（転移学習、作業途中） |
| `continuous/thailand/other_variable/` | 予測因子の組み合わせを変えた実験 | `main.py`, `evaluate.py` |
| `continuous/thailand/different_space/{1x1_land,5x5_land}/` | 入力領域を切り出した実験 | `main.py`, `evaluate.py` |
| `continuous/thailand/roc_comparison/` | 実験間の ROC / AUC 比較 | `roc_comparison.py` |
| `continuous/monsoon/other_variable/` | モンスーン域の連続値予測 | `main.py`, `evaluate.py` |
| `continuous/world/{1x1,5x5}/` | 全球の連続値予測 | `cnn3_main.py`, `evaluate.py` |
| `parallel_model/{thailand,monsoon}/` | クラス分類モデルと連続値モデルを並べて比較 | `train.py`, `evaluate.py` |

各ディレクトリの補助モジュール

* `util.py` / `preprocess.py` : データ読み込み (`load`)、欠損マスク (`mask`)、学習/検証分割 (`shuffle`)、pickle 読み込み
* `model.py` / `model3.py` / `class_model.py` / `continuous_model.py` : CNN 定義 (`build_model`, `init_model`)
* `view.py` : 地図・棒グラフ・箱ひげ図などの描画
* `gradcam.py` : Grad-CAM（TF1 互換モード `disable_eager_execution` を使用）
* `displaycallback.py` : 学習進捗を 1 行で表示する Keras コールバック

## 実行方法

```bash
cd class/thailand
python cnn3_main.py   # train_flag / overwrite_flag をスクリプト先頭で切り替え
python evaluate.py    # 各種 *_view_flag を __init__ で切り替え
```

* パラメータ（クラス数、epoch、解像度、予測因子名 `tors`、予測対象名 `tant` など）は各スクリプトの
  `main()` 冒頭またはクラスの `__init__` に直接書かれています。
* 入出力はすべて `/docker/mnt/d/research/D2/cnn3/` 以下（`predictors/`, `predictant/`, `train_val/`, `weights/`,
  `result/`, `boundaries/`, `heatmap/`）を前提としています。転移学習のみ `/docker/mnt/d/research/D3/cnn3/transfer/` を使用します。
* 必要なパッケージ: tensorflow / keras（`tf.keras.optimizers.legacy` と `keras.layers.core` を使うため TF 2.x の比較的古い版）,
  numpy, matplotlib, cartopy, opencv-python (cv2), scipy, scikit-learn, pandas。

## 既知の注意点

* 以前はすべての `shuffle()` で、検証データの抽出に `val_dct['model']` と `train_dct['year']` を組み合わせていました
  （`val_dct['year']` ではない）。このため検証サンプルが学習サンプルと重複する可能性がありました。
  現在は `val_dct['year']` を使うように修正済みです。修正前に保存した `train_val/` の pickle、学習済み重み、
  およびそれらから作った結果は修正後のコードと整合しないため、再生成してください。
* `parallel_model/*/train.py` の `main()` では、クラス用の予測対象を読み込んだ直後に連続値用で上書きしており、
  `train_class` に連続値の予測対象が渡ります（`train_flag = True` のときのみ影響）。
