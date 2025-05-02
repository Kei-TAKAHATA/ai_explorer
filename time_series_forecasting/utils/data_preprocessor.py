from time_series_forecasting.utils.date_processor import DateProcessor


class DataPreprocessor:
    def __init__(self, settings):
        self.settings = settings
        self.train_columns = settings["data"]["train_columns"]
        self.predict_columns = settings["data"]["predict_columns"]
        # train_columnsからpredict_columnsを除いたものをusing_predict_columnsとする
        self.using_predict_columns = [col for col in self.train_columns if col not in self.predict_columns]

    def preprocess_data(self, df):
        """
        トレーニングデータの DataFrame を前処理する。

        Parameters
        ----
        df : DataFrame
            トレーニングデータの DataFrame。

        Returns
        ----
        DataFrame
            前処理後のトレーニングデータの DataFrame。
        """
        date_processor = DateProcessor()
        df = date_processor.add_date_features(df)
        return df

    def select_train_columns(self, df):
        """
        トレーニングデータの DataFrame から予測に必要な列を抽出する。

        Parameters
        ----
        df : DataFrame
            トレーニングデータの DataFrame。
        """
        return df[self.train_columns]

    def select_predict_columns(self, df):
        """
        テストデータの DataFrame から予測に必要な列を抽出する。

        Parameters
        ----
        df : DataFrame
            テストデータの DataFrame。
        """
        return df[self.predict_columns]

    # def create_predict_df(self, df):
    #     """
    #     テストデータの DataFrame から予測に必要な列を抽出する。

    #     Parameters
    #     ----
    #     df : DataFrame
    #         テストデータの DataFrame。

    #     Returns
    #     ----
    #     DataFrame
    #         予測に必要な列の DataFrame。
    #     """
    #     df = df[["date"]]
    #     df.reset_index(drop=True, inplace=True)
    #     return df
