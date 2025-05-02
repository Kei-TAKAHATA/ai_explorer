import datetime
import os

from utils.settings_loader import load_settings
from time_series_forecasting.utils.data_manager import DataManager
from time_series_forecasting.utils.data_preprocessor import DataPreprocessor
from time_series_forecasting.utils.model_selector import ModelSelector


def main():
    """時系列予測を実行するメイン関数"""
    # 実行時の日時
    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    print("date_time_str: ", date_time_str)

    # time_series_forecastingディレクトリ
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print("root_dir: ", root_dir)

    # 設定ファイルのパス
    settings_path = "settings/run_time_series_forecast.yaml"
    # 設定を読み込む
    settings = load_settings(root_dir, settings_path)

    # データの読み込み
    data_manager = DataManager(root_dir=root_dir, settings=settings)
    train_data, test_data = data_manager.load_train_and_test_data()
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
    # 予測対象のテーブルごとのdictを作成
    train_dfs = create_id_dict(train_data)
    predict_template_df = generate_predict_template_df(start_date=datetime.date(2024, 10, 1), end_date=datetime.date(2024, 12, 31))
    # データの前処理
    data_preprocessor = DataPreprocessor(settings)
    predict_template_df = data_preprocessor.preprocess_data(predict_template_df)

    predict_result_dict = {}
    for id_, train_df in train_dfs.items():
        # データの前処理
        train_df = data_preprocessor.preprocess_data(train_df)
        predict_df = predict_template_df.copy()

        # モデルの定義
        model_selector = ModelSelector(settings)
        model = model_selector.get_model()
        # モデルの学習と予測
        train_x = data_preprocessor.select_train_columns(train_df).values
        train_y = data_preprocessor.select_predict_columns(train_df).values
        model.fit(train_x, train_y)
        predict_df = model.predict_from_df(df=predict_df, observation_column_names=["is_holiday", "is_day_before_holiday"])
        predict_df = adjust_predictions(predict_df)
        predict_result_dict[id_] = predict_df

    # 結果のまとめ
    date_predict_result_df = summarize_results(train_dfs, predict_result_dict)
    check_columns(date_predict_result_df)
    date_predict_result_df = merge_predict_and_ground_truth_df(date_predict_result_df, test_data)
    date_predict_result_df = add_necessary_columns_for_evaluation(date_predict_result_df)

    summary_result_df = evaluate(date_predict_result_df)
    # 全体結果
    print("全体結果")
    print(summary_result_df.describe())

    week_rmspe_median = summary_result_df["週単位RMSPE"].describe().loc["50%"]
    check_value = int(week_rmspe_median)
    # 予測結果の保存
    data_manager.save_df_data(df=date_predict_result_df, prefix=date_time_str, suffix=str(check_value))
    data_manager.save_df_data(df=summary_result_df, prefix=date_time_str, suffix="summary_" + str(check_value))


if __name__ == "__main__":
    main()
