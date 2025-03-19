# ai_explorer
AIに関連する処理を幅広く行うプロジェクト

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

| ブランチ名 | 用途 | 命名規則例 |
|------------------|----------------------------------------------------------------------|---------------------|
| main | 安定したバージョンを保持。完成した機能や修正をマージ。 | main |
| feature/xxx | 新しい機能の開発に使用。xxxは機能の名前を示す。 | feature/add-feature |
| bugfix/xxx | バグ修正に使用。xxxは修正内容を示す。 | bugfix/fix-issue |

## コミットメッセージのルール
以下のタグを使用する

| タグ | 用途 | 例 |
|--------|----------------------------------|------------------------------------|
| feature | 新しい機能の追加 | feature: 画像認識モデルを追加 |
| bugfix | バグ修正 | bugfix: データ読み込みエラーを修正 |
| docs | ドキュメントの変更 | docs: READMEに使用方法を追加 |
| refactor | コードのリファクタリング（動作に影響しない） | refactor: モデルの構造を整理 |
| chore | その他の変更（パッケージ更新、ビルドプロセスの変更など） | chore: 依存パッケージを更新 |
