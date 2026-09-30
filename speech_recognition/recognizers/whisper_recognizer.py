import whisper


class WhisperRecognizer:

    def recognize(self, file_path, settings, model_name=None):
        """
        Whisperモデルを使用して音声ファイルを音声認識を実行

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
