# Speech Recognition

このディレクトリは、音声認識を行うためのツールセットです。

## インストール

このプロジェクトを実行するには、以下の手順で環境をセットアップしてください。

```
conda env create -f environment.yml
conda activate speech_recognition
```

### ReazonSpeechのインストール
ReazonSpeechの公式ページに従ってインストールします。
```
cd speech_recognition/libs
git clone https://github.com/reazon-research/ReazonSpeech
pip install ReazonSpeech/pkg/nemo-asr  # or k2-asr, espnet-asr or espnet-oneseg
```

## 実行方法

音声認識を実行するには、ai_explore直下でscriptsディレクトリにあるファイルを実行します。
以下、run_batch_speech_recognition.pyを実行する例です。

```
python -m speech_recognition.scripts.run_batch_speech_recognition
```
