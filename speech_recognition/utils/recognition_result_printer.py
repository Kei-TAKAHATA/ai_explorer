class RecognitionResultPrinter:
    """
    音声認識の結果を出力するクラス
    """
    def print_recognition_result(self, recognition_result, model_name):
        """
        音声認識の結果を出力する
        """
        if "whisper" in model_name:
            for segment in recognition_result["segments"]:
                segment_id, start, end, text = [segment[key] for key in ["id", "start", "end", "text"]]
                print(f"{segment_id:03}: {start:5.1f} - {end:5.1f} | {text}")
                return
        elif "reazon" in model_name:
            for segment in recognition_result.segments:
                print("%5.2f %5.2f %s" % (segment.start_seconds, segment.end_seconds, segment.text))
                return
        raise ValueError(f"Unsupported model: {model_name}")
