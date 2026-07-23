# Speech Recognition

このディレクトリは、音声認識を行うためのツールセットです。

## 実行環境の更新
以下の手順で環境をセットアップしてください。

### `environment.yml`がない場合
```bash
cd ~/workspace/ai_explorer/speech_recognition
conda create -n speech_recognition python=3.11 -y
conda activate speech_recognition

# `pip-tools`のinstall
pip install pip-tools

# ReazonSpeechの公式ページに従ってReazonSpeechをインストール
git clone https://github.com/reazon-research/ReazonSpeech libs/ReazonSpeech
python -m pip install -e ./libs/ReazonSpeech/pkg/nemo-asr  # or k2-asr, espnet-asr or espnet-oneseg
# pip install ReazonSpeech/pkg/nemo-asr  # or k2-asr, espnet-asr or espnet-oneseg
```

### 初回実行時

```bash
conda remove -n speech_recognition --all -y
cd ~/workspace/ai_explorer/speech_recognition
conda create -n speech_recognition python=3.11 -y
conda activate speech_recognition

pip install --upgrade pip pip-tools
mkdir -p libs
# ReazonSpeechの公式ページに従ってReazonSpeechをインストール
git clone https://github.com/reazon-research/ReazonSpeech libs/ReazonSpeech
pip install -e ./libs/ReazonSpeech/pkg/nemo-asr  # or k2-asr, espnet-asr or espnet-oneseg

# git -C libs/ReazonSpeech pull --ff-only
# pip install -e ./libs/ReazonSpeech/pkg/nemo-asr
pip-compile requirements.in --upgrade
pip install -r requirements.txt
```

### Reazonを入れ直す場合

```bash
conda remove -n speech_recognition --all -y
cd ~/workspace/ai_explorer/speech_recognition
conda create -n speech_recognition python=3.11 -y
conda activate speech_recognition

pip install --upgrade pip pip-tools

rm -rf libs/ReazonSpeech
# ReazonSpeechの公式ページに従ってReazonSpeechをインストール
git clone https://github.com/reazon-research/ReazonSpeech libs/ReazonSpeech
pip install -e ./libs/ReazonSpeech/pkg/nemo-asr  # or k2-asr, espnet-asr or espnet-oneseg

pip-compile requirements.in --upgrade
pip install -r requirements.txt
```

### ライブラリ更新_Reazonは入れ直さない

```bash
conda remove -n speech_recognition --all -y
cd ~/workspace/ai_explorer/speech_recognition
conda create -n speech_recognition python=3.11 -y
conda activate speech_recognition

pip install --upgrade pip pip-tools
git -C libs/ReazonSpeech pull --ff-only

rm -rf libs/ReazonSpeech
pip install -e ./libs/ReazonSpeech/pkg/nemo-asr  # or k2-asr, espnet-asr or espnet-oneseg
pip-compile requirements.in --upgrade
pip install -r requirements.txt
```


---

### `environment.yml`がある場合

```bash
conda remove -n speech_recognition --all -y
cd ~/workspace/ai_explorer/speech_recognition
conda create -n speech_recognition python=3.11 -y
conda activate speech_recognition

pip install --upgrade pip pip-tools
mkdir -p libs
git -C libs/ReazonSpeech pull --ff-only

rm -rf libs/ReazonSpeech
pip install -e ./libs/ReazonSpeech/pkg/nemo-asr  # or k2-asr, espnet-asr or espnet-oneseg
pip-compile requirements.in --upgrade
pip install -r requirements.txt
```



```bash
conda remove -n speech_recognition --all -y
cd ~/workspace/ai_explorer/speech_recognition
conda create -n speech_recognition python=3.11 -y
conda activate speech_recognition

pip install --upgrade pip pip-tools
mkdir -p libs
git -C libs/ReazonSpeech pull --ff-only
pip install -e ./libs/ReazonSpeech/pkg/nemo-asr
pip-compile requirements.in --upgrade
pip install -r requirements.txt
```

## 実行方法

音声認識を実行するには、ai_explore直下でscriptsディレクトリにあるファイルを実行します。
以下、run_batch_speech_recognition.pyを実行する例です。

```bash
python -m speech_recognition.scripts.run_batch_speech_recognition
```
