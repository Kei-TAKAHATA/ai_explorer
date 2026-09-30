# import sys
import os

# ルート以外から実行する場合に利用
# プロジェクトのルートディレクトリをパスに追加
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from speech_recognition.utils.file_list_generator import FileListGenerator
from speech_recognition.utils.settings_loader import load_settings
from speech_recognition.utils.speech_recognizer import SpeechRecognizer
from speech_recognition.utils.recognition_result_printer import RecognitionResultPrinter


def main():
    """バッチ音声認識を実行するメイン関数"""
    # speech_recognitionディレクトリ
    speech_recognition_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print("speech_recognition_dir: ", speech_recognition_dir)

    # 設定ファイルのパス
    settings_path = "settings/run_batch_speech_recognition.yaml"

    # 設定を読み込む
    settings = load_settings(speech_recognition_dir, settings_path)

    speech_recognizer = SpeechRecognizer()
    recognition_result_printer = RecognitionResultPrinter()

    file_list_generator = FileListGenerator(speech_recognition_dir)
    file_paths = file_list_generator.generate_file_paths(settings)
    for file_path in file_paths:
        print("file_path: ", file_path)
        models = settings["models"]
        for model_name in models:
            print("model_name: ", model_name)
            recognition_result = speech_recognizer.execute_speech_recognition(file_path, model_name, settings)
            recognition_result_printer.print_recognition_result(recognition_result, model_name)


if __name__ == "__main__":
    main()
