# Speech Recognition

このディレクトリは、音声認識を行うためのツールセットです。

## 実行環境
以下の手順で環境をセットアップしてください。

PyPI、condaで配布されてないReazonSpeechを利用するため、`environment.yml`ではなく、
`requirements.in`と`requirements.txt`で管理する運用にする。

### `requirements.txt`がない場合
```bash
conda create -n speech_recognition python=3.10 -y
conda activate speech_recognition
```

`pip-tools`のinstall

```bash
pip install pip-tools
``` 

### ReazonSpeechのインストール
ReazonSpeechの公式ページに従ってインストールします。
```bash
cd speech_recognition/libs
git clone https://github.com/reazon-research/ReazonSpeech
pip install ReazonSpeech/pkg/nemo-asr  # or k2-asr, espnet-asr or espnet-oneseg
```

---

### `requirements.txt`がある場合

現状のconda環境を停止して削除する。

```bash
conda deactivate
```

conda環境を削除

```bash
conda remove -n speech_recognition --all -y
```

## 実行環境の更新
新規でライブラリをinstallした場合、
`requirements.in`にライブラリを追加。

以下のコマンドを実行し`requirements.txt`を更新。

### ディレクトリ移動
```bash
cd speech_recognition
```

### ライブラリのバージョンを更新しなくて良い場合
```bash
pip-compile requirements.in
```

### ライブラリのバージョンを更新する場合
```bash
pip-compile requirements.in --upgrade
```

現場のconda環境を落とし、新規でconda環境を作成し、`requirements.txt`からインストール

```bash
conda deactivate
```

```bash
conda create -n speech_recognition python=3.10 -y
```

```bash
conda activate speech_recognition
```

```bash
pip install -r requirements.txt
```

動作確認後、`requirements.in`と`requirements.txt`をコミット

## 実行方法

音声認識を実行するには、ai_explore直下でscriptsディレクトリにあるファイルを実行します。
以下、run_batch_speech_recognition.pyを実行する例です。

```bash
python -m speech_recognition.scripts.run_batch_speech_recognition
```
