# ai_explorer
AIに関連する処理を幅広く行うプロジェクト

## プロジェクト一覧

- `regression_analysis`: 回帰モデル
- `speech_recognition`: 音声認識処理
- `time_series_forecasting`: 時系列予測
- `shared`: 複数プロジェクトから参照する軽量な共通処理を配置

各子ディレクトリは、依存ライブラリの競合を避けるため、原則として個別の実行環境を持つ。


## リポジトリ構成

- 各サブプロジェクトは独立した依存関係を持つ
- ルートでは単一の Python 実行環境を管理しない
- Python の依存関係と実行環境は各サブプロジェクト配下で管理する

## 開発方針

- 共通化は最小限にとどめる
- 重い依存を持つ処理は各プロジェクト内に保持する
- サブプロジェクト間の直接 import は原則避ける
- notebook 出力や生成物は原則 Git 管理しない

## Python 実行環境

- スクリプトやテストは、対象コードを持つサブプロジェクトの環境で実行する
- 詳細なセットアップや実行方法は各サブプロジェクトの `README.md` を参照する

## 共通処理の扱い

- 共通処理は原則として軽量なライブラリに限定する
- 共通処理の実行は、利用側のサブプロジェクトの環境で行う
- BigQuery、AWS、機械学習フレームワークなどの重い依存を持つ処理は各プロジェクト内で管理する

## ドキュメント運用

- 全体で共通利用するドキュメントテンプレートは `docs/implementation_note_templates/` に置く
- 特定プロジェクトが主担当の実装メモは、そのプロジェクト配下の `docs/implementation_notes/` に置く
- CI 整理、モノレポ移行、共通基盤化のような横断テーマはルート `docs/implementation_notes/` に置く


## ディレクトリ構造
以下の構成になるように進める

```
ai_explorer/
├── README.md                        # プロジェクトの概要、目的、使用方法を記載
├── image_recognition/              # 画像認識に関連するコードとデータ
│   ├── models/                     # 画像認識モデルの定義とトレーニングスクリプト
│   ├── datasets/                   # 画像データセットを格納
│   └── scripts/                    # 画像認識の実行スクリプトやユーティリティ
├── natural_language_processing/    # 自然言語処理に関連するコードとデータ
│   ├── models/                     # NLPモデルの定義とトレーニングスクリプト
│   ├── datasets/                   # テキストデータセットを格納
│   └── scripts/                    # NLPの実行スクリプトやユーティリティ
├── speech_recognition/             # 音声認識に関連するコードとデータ
│   ├── models/                     # 音声認識モデルの定義とトレーニングスクリプト
│   ├── datasets/                   # 音声データセットを格納
│   └── scripts/                    # 音声認識の実行スクリプトやユーティリティ
├── reinforcement_learning/         # 強化学習に関連するコードとデータ
│   ├── models/                     # 強化学習エージェントの定義とトレーニングスクリプト
│   ├── environments/               # 強化学習環境の設定と管理
│   └── scripts/                    # 強化学習の実行スクリプトやユーティリティ
├── task_a/                         # タスクAに関連するコードとデータ
│   ├── models/                     # タスクAのモデル定義とトレーニングスクリプト
│   ├── datasets/                   # タスクAのデータセットを格納
│   └── scripts/                    # タスクAの実行スクリプトやユーティリティ
└── task_b/                         # タスクBに関連するコードとデータ
    ├── models/                     # タスクBのモデル定義とトレーニングスクリプト
    ├── datasets/                   # タスクBのデータセットを格納
    └── scripts/                    # タスクBの実行スクリプトやユーティリティ
```

各タスク内のディレクトリ内は以下の構成で作成。

```
task_x/
├── models/
│   ├── model_a/
│   │   ├── model_a.py        # Model Aの定義
│   │   └── model_a_utils.py        # Model A用のユーティリティ関数
│   ├── model_b/
│   │   ├── model_b.py        # Model Bの定義
│   │   └── model_b_utils.py        # Model B用のユーティリティ関数
├── datasets/
│   ├── raw/                  # 生データ
│   ├── processed/            # 前処理済みデータ
│   └── load_data.py          # データ読み込みスクリプト
├── scripts/
│   ├── exec_task.py          # タスクを実行するスクリプト
│   ├── train_model_a.py      # Model Aのトレーニングスクリプト
│   ├── train_model_b.py      # Model Bのトレーニングスクリプト
│   ├── evaluate_model_a.py   # Model Aの評価スクリプト
│   └── evaluate_model_b.py   # Model Bの評価スクリプト
├── settings/
│   ├── exec_task.yaml        # タスク実行用の設定ファイル
│   ├── train_model_a.yaml    # Model Aトレーニング用の設定ファイル
│   └── train_model_b.yaml    # Model Bトレーニング用の設定ファイル
├── logs/
│   ├── execution_logs/       # 実行時のログファイルを保存
│   └── settings_history/     # 実行時の設定ファイルを保存
│       ├── exec_task/        # タスク実行用ファイルの過去の設定ファイル
│       ├── train_model_a/    # Model Aトレーニング用の過去の設定ファイル
│       └── train_model_b/    # Model Bトレーニング用の過去の設定ファイル
└── utils/
    ├── preprocessing.py      # 共通の前処理関数を格納
    ├── postprocessing.py     # 共通の後処理関数を格納
    └── common_utils.py       # その他の共通ユーティリティ関数
```

## ブランチ戦略
ブランチ名は以下のように命名する
| ブランチ名     | 用途                                          | 命名規則例                 |
|--------------|-----------------------------------------------|--------------------------|
| main         | 安定したバージョンを保持。完成した機能や修正をマージ。 | main                     |
| feature/xxx  | 新しい機能の開発に使用。xxxは機能の名前を示す。      | feature/add-feature      |
| bugfix/xxx   | バグ修正に使用。xxxは修正内容を示す。              | bugfix/fix-issue         |
| refactor/xxx | 挙動を変えずにリファクタリングを行う場合に使用。      | refactor/rename-function |

## コミットメッセージのルール
以下のタグを使用する
| タグ | 用途 | 例 |
|--------|----------------------------------|------------------------------------|
| feature | 新しい機能の追加 | [feature]画像認識モデルを追加 |
| update | 既存の機能の更新 | [update]外れ値除去ロジックの更新 |
| bugfix | バグ修正 | [bugfix]データ読み込みエラーを修正 |
| docs | ドキュメントの変更 | [docs]READMEに使用方法を追加 |
| refactor | コードのリファクタリング（動作に影響しない） | [refactor]モデルの構造を整理 |
| chore | その他の変更（パッケージ更新、ビルドプロセスの変更など） | [chore]依存パッケージを更新 |
| test  | テストコードの追加・修正   | [test]estimateのユニットテスト追加 |

## 実行環境
以下の手順で環境をセットアップしてください。

`environment.yml`がない場合
```bash
conda create -n ai_explorer python=3.11 -y
conda activate ai_explorer

# `pip-tools`のinstall
pip install pip-tools
```

---

`environment.yml`がある場合
```bash
conda env create -f environment.yml -y
conda activate ai_explorer
```

## 実行環境の更新
新規でライブラリをinstallした場合、
`requirements.in`にライブラリを追加。

```bash
cd ~/workspace/ai_explorer
conda activate ai_explorer
# `pip-tools`, `pip`のアップグレード
pip install --upgrade pip-tools
pip install --upgrade pip

# requirements.txtの更新
pip-compile requirements.in --upgrade
# 現場のconda環境を落とし、新規でconda環境を作成し、`requirements.txt`からインストール
conda deactivate
conda create -n ai_explorer_x python=3.11 -y
conda activate ai_explorer_x
pip install -r requirements.txt
# 以下のコマンドでconda用の`enviromnent.yml`の更新
conda env export --no-builds | grep -v "^prefix:" > environment.yml
# pytest
conda deactivate
conda remove -n ai_explorer_x --all -y
```

`name`の`ai_explorer_x`を`ai_explorer`に更新してコミット

conda環境の更新

```bash
conda env create -f environment.yml -y
conda activate ai_explorter
```
