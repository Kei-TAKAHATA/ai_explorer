import os
import glob


class FileListGenerator:
    """
    音声認識の対象ファイルのリストを生成するクラス
    """

    def __init__(self, base_dir, settings):
        self.base_dir = base_dir
        self.settings = settings

    def generate_file_paths(self):
        """
        設定からファイルパスのリストを生成する

        Returns
        -------
        list
            ファイルパスのリスト。
        """
        dir_path = os.path.join(self.base_dir, self.settings["dir"]["datasets_dir"], self.settings["dir"]["input_dir"])
        target_type = self.settings["target_type"]

        if target_type == "all":
            key = dir_path + "/*.wav"
            file_paths = glob.glob(key)
            file_paths.sort()
        elif target_type == "designation":
            file_names = self.settings["file_names"]
            file_paths = [os.path.join(dir_path, file_name) for file_name in file_names]
        return file_paths
