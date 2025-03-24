import os

import pandas as pd


class DataManager:
    def __init__(self, root_dir, settings):
        self.root_dir = root_dir
        self.settings = settings
        self.datasets_dir = settings["dir"]["datasets_dir"]
        self.input_dir = settings["dir"]["input_dir"]
        self.output_dir = settings["dir"]["output_dir"]
        # 出力先ディレクトリがない場合は作成して権限変更
        output_dir_path = os.path.join(self.root_dir, self.datasets_dir, self.output_dir)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
            os.chmod(output_dir_path, 0o777)
        self.train_data_name = settings["data"]["train_data_name"]
        self.test_data_name = settings["data"]["test_data_name"]
        self.output_file_base_name = settings["data"]["output_file_base_name"]

    def load_train_and_test_data(self):
        train_file_path = os.path.join(self.root_dir, self.datasets_dir, self.input_dir, self.train_data_name)
        test_file_path = os.path.join(self.root_dir, self.datasets_dir, self.input_dir, self.test_data_name)
        train_df = pd.read_csv(train_file_path, encoding="utf-8")
        test_df = pd.read_csv(test_file_path, encoding="utf-8")
        return train_df, test_df

    def save_df_data(self, df, prefix="", suffix=""):
        # 日付_時間_ベース名_追加オプション.csv
        file_name = f"{prefix}_{self.output_file_base_name}_{suffix}.csv"
        file_path = os.path.join(self.root_dir, self.datasets_dir, self.output_dir, file_name)
        # import pdb; pdb.set_trace()
        df.to_csv(file_path, index=False)
        # 権限変更
        os.chmod(file_path, 0o777)
