from reazonspeech.nemo.asr import transcribe, audio_from_path, load_model
# import reazonspeech.nemo.asr


class ReazonRecognizer:
    def recognize(self, file_path, settings):
        """
        Reazonモデルを使用して音声ファイルを音声認識を実行

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
