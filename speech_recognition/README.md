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
cd libs
git clone https://github.com/reazon-research/ReazonSpeech
pip install ReazonSpeech/pkg/nemo-asr  # or k2-asr, espnet-asr or espnet-oneseg
```

---

### `environment.yml`がある場合

```bash
cd ~/workspace/ai_explorer/speech_recognition
conda activate speech_recognition
pip install --upgrade pip-tools
pip install --upgrade pip
pip-compile requirements.in --upgrade
conda deactivate
conda create -n speech_recognition_x python=3.11 -y
conda activate speech_recognition_x
pip install -r requirements.txt
conda env export --no-builds | grep -v "^prefix:" > environment.yml
conda deactivate
conda remove -n speech_recognition_x --all -y
```

`name`の`speech_recognition_x`を`speech_recognition`に更新

```bash
conda env create -f environment.yml -y
conda activate speech_recognition
```


## 実行方法

音声認識を実行するには、ai_explore直下でscriptsディレクトリにあるファイルを実行します。
以下、run_batch_speech_recognition.pyを実行する例です。

```bash
python -m speech_recognition.scripts.run_batch_speech_recognition
```
