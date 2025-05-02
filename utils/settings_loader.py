import os
import yaml


def load_settings(base_dir, file_path):
    """設定ファイルを読み込む"""
    with open(os.path.join(base_dir, file_path), 'r', encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    return settings
