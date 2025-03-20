# import sys
import os

# ルート以外から実行する場合に利用
# プロジェクトのルートディレクトリをパスに追加
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import yaml
from speech_recognition.utils.speech_recognition import SpeechRecognition
from speech_recognition.utils.file_list_generator import FileListGenerator


def load_settings(base_dir, file_path):
    """設定ファイルを読み込む"""
    with open(os.path.join(base_dir, file_path), 'r', encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    return settings


def main():
    """バッチ音声認識を実行するメイン関数"""
    # speech_recognitionディレクトリ
    speech_recognition_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print("speech_recognition_dir: ", speech_recognition_dir)

    # 設定ファイルのパス
    settings_path = "settings/run_batch_speech_recognition.yaml"

    # 設定を読み込む
    settings = load_settings(speech_recognition_dir, settings_path)

    speech_recognition = SpeechRecognition()

    file_list_generator = FileListGenerator(speech_recognition_dir, settings)
    file_paths = file_list_generator.generate_file_paths()
    for file_path in file_paths:
        print("file_path: ", file_path)
        models = settings["models"]
        for model in models:
            print("model: ", model)
            recognition_result = speech_recognition.execute_speech_recognition(file_path, model, settings)
            speech_recognition.print_recognition_result(recognition_result, model)


if __name__ == "__main__":
    main()
