import whisper

from reazonspeech.nemo.asr import transcribe, audio_from_path, load_model


class SpeechRecognition:
    """
    音声認識を実施するクラス
    """

    def execute_whisper(self, file_path, settings, model_name=None):
        """
        Whisperモデルを使用して音声ファイルを音声認識

        Parameters
        ----------
        file_path : str
            トランスクリプトする音声ファイルのパス。
        settings : dict
            音声認識の設定を含む辞書。以下のキーを含むことができます:
            - whisper_device : str, optional
                モデルを実行するデバイス。デフォルトは "cpu"。
            - model_name : str, optional
                使用するWhisperモデルの名前。指定がない場合は "base" モデルを使用。

        Returns
        -------
        recognition_result : dict
            音声認識の結果
        """
        # デバイスの設定を取得。デフォルトは "cpu"
        device = settings.get("whisper_device", "cpu")

        # モデルをロード。指定がなければ "base" モデルをロード
        if model_name is None:
            model = whisper.load_model(name="base", device=device)
        else:
            model = whisper.load_model(name=model_name, device=device)

        # 音声ファイルを音声認識
        recognition_result = model.transcribe(file_path)

        # 音声認識結果を返す
        return recognition_result

    def execute_reazon(self, file_path, settings):
        """
        Reazonモデルを使用して音声ファイルを音声認識

        Parameters
        ----------
        file_path : str
            トランスクリプトする音声ファイルのパス。
        settings : dict
            音声認識の設定を含む辞書。以下のキーを含むことができます:
            - reazon_device : str
                モデルを実行するデバイス。

        Returns
        -------
        recognition_result
            音声認識の結果
        """
        # 音声ファイルをロード
        audio = audio_from_path(file_path)

        # デバイスの設定を取得
        device = settings["reazon_device"]

        # モデルをロード
        model = load_model(device=device)

        # 音声ファイルを音声認識
        recognition_result = transcribe(model, audio)

        # 音声認識結果を返す
        return recognition_result

    def execute_speech_recognition(self, file_path, model_name, settings):
        """
        指定されたモデルを使用して音声認識を実行します。

        Parameters
        ----------
        file_path : str
            音声ファイルのパス。
        model_name : str
            使用するモデルの名前。
        settings : dict
            音声認識の設定を含む辞書。

        Returns
        -------
        result
            音声認識の結果。
        """
        if "whisper" in model_name:
            model_name = model_name.replace("whisper_", "")
            recognition_result = self.execute_whisper(
                file_path=file_path,
                model_name=model_name,
                settings=settings,
            )
        elif model_name == "reazon":
            recognition_result = self.execute_reazon(
                file_path=file_path,
                settings=settings,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return recognition_result

    def print_recognition_result(self, recognition_result, model_name):
        """
        音声認識の結果を出力する

        Parameters
        ----------
        recognition_result : dict
            音声認識の結果
        """
        if "whisper" in model_name:
            for segment in recognition_result["segments"]:
                segment_id, start, end, text = [segment[key] for key in ["id", "start", "end", "text"]]
                print(f"{segment_id:03}: {start:5.1f} - {end:5.1f} | {text}")
        elif "reazon" in model_name:
            for segment in recognition_result.segments:
                print("%5.2f %5.2f %s" % (segment.start_seconds, segment.end_seconds, segment.text))
        else:
            raise ValueError(f"Unsupported model: {model_name}")
