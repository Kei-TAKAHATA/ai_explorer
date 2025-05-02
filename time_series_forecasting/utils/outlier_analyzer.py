import pandas as pd


class OutlierAnalyzer:
    def __init__(self, settings: dict = None):
        """
        外れ値除去のための設定を行います。

        Parameters:
        method (str): 外れ値除去の方法。現状実装済みは以下
            iqr: IQRに基づく外れ値の範囲を決定する方法
            z_score: Zスコアに基づく外れ値の範囲を決定する方法
        iqr_coefficient (float): IQRに基づく外れ値の範囲を決定するための係数。デフォルトは1.5
        z_score_coefficient (float): Zスコアに基づく外れ値の範囲を決定するための係数。デフォルトは2.0
        """
        if settings is None:
            settings = {
                'method': 'iqr',
                'iqr_coefficient': 1.5,
                'z_score_coefficient': 2.0
            }
        self.method = settings.get('method', 'iqr')
        self.iqr_coefficient = settings.get('iqr_coefficient', 1.5)
        self.z_score_coefficient = settings.get('z_score_coefficient', 2.0)

    def calculate_min_max(self, data):
        """
        外れ値の基準となる最小値と最大値を計算する。

        Parameters:
        data (array-like): 外れ値を計算するためのデータ。リスト、NumPy配列、またはpandas.Seriesを受け付けます。

        Returns:
        tuple: 最小値と最大値のタプル。
        """
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        if self.method == 'iqr':
            # IQRに基づく計算
            q3 = data.quantile(0.75)
            q1 = data.quantile(0.25)
            iqr = q3 - q1
            calc_value = self.iqr_coefficient * iqr
            min_value = q1 - calc_value
            max_value = q3 + calc_value
            return min_value, max_value
        elif self.method == 'z_score':
            # Zスコアに基づく計算
            mean = data.mean()
            std_dev = data.std()
            min_value = mean - self.z_score_coefficient * std_dev
            max_value = mean + self.z_score_coefficient * std_dev
            return min_value, max_value
        else:
            raise ValueError("Unsupported method: {}".format(self.method))

    def iqr_filtered_series(self, df: pd.DataFrame, column):
        """
        指定された列から外れ値を除去したシリーズを作成します。

        Parameters:
        df (pandas.DataFrame): データフレーム。
        column (str): 外れ値を除去する対象の列名。

        Returns:
        pandas.Series: 外れ値を除去したシリーズ。
        """
        min_value, max_value = self.calculate_min_max(df[column])
        filtered_df = df[(min_value < df[column]) & (df[column] < max_value)]
        return filtered_df[column]

    def generate_outlier_removed_df(self, df, columns=None):
        """
        指定された列またはすべての列から外れ値を除去したデータフレームを作成します。

        Parameters:
        df (pandas.DataFrame): データフレーム。
        columns (list, optional): 外れ値を除去する対象の列名のリスト。デフォルトはすべての列。

        Returns:
        pandas.DataFrame: 外れ値を除去したデータフレーム。
        """
        if columns is None:
            columns = df.columns
        removed_df = pd.DataFrame()
        for column in columns:
            removed_series = self.iqr_filtered_series(df, column)
            # print(len(removed_series))
            removed_df = pd.concat([removed_df, removed_series], axis=1)
        return removed_df

    def get_outlier_df(self, df: pd.DataFrame, column):
        """
        指定された列の外れ値を検出します。

        Parameters:
        df (pandas.DataFrame): データフレーム。
        column (str): 外れ値を検出する対象の列名。

        Returns:
        tuple: 改善された外れ値と悪化した外れ値のデータフレームのタプル。
        """
        min_value, max_value = self.calculate_min_max(df[column])
        improved_outlier_df = df[(df[column] < min_value)]
        worsened_outlier_df = df[(max_value < df[column])]
        return improved_outlier_df, worsened_outlier_df
