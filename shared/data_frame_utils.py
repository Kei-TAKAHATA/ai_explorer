import pandas as pd


class DataFrameUtils:
    def group_df_by_column(self, df: pd.DataFrame, column_name: str) -> dict:
        """
        データフレームを指定した列のユニークな値でグループ化します。

        Parameters
        ----------
        df : pandas.DataFrame
            グループ化する対象のデータフレーム。

        column_name : str
            グループ化の基準となる列名。

        Returns
        -------
        dict
            ユニークな値をキーとし、それに対応するデータフレームを値とする辞書。
        """
        grouped_dataframes = {}
        unique_values = list(df[column_name].unique())
        for value in unique_values:
            filtered_df = df[df[column_name] == value]
            grouped_dataframes[value] = filtered_df
        return grouped_dataframes

    def concat_dataframes(self, dataframes: dict) -> pd.DataFrame:
        """
        辞書のデータフレームを結合する
        """
        return pd.concat(dataframes.values(), axis=0)

    def delete_str_columns(self, df: pd.DataFrame, delete_str: str) -> pd.DataFrame:
        """
        データフレームから指定された文字列を含む列を削除する。

        Parameters
        ----------
        df : pandas.DataFrame
            指定された文字列を含む列を削除する対象のデータフレーム。

        delete_str : str
            削除する文字列。

        Returns
        -------
        pandas.DataFrame
            指定された文字列を含む列が削除されたデータフレーム。
        """
        df_columns_list = list(df.columns)
        new_columns = []
        for column in df_columns_list:
            if delete_str in column:
                column = column.replace(delete_str, "")
            new_columns.append(column)
        df.columns = new_columns
        return df

    def sort_dataframe(self, df: pd.DataFrame, sort_columns: list) -> pd.DataFrame:
        """
        データフレームを指定されたカラムでソートする

        Parameters
        ----------
        df : pandas.DataFrame
            ソートする対象のデータフレーム。
        sort_columns : list
            ソートする対象のカラムのリスト。

        Returns
        -------
        pandas.DataFrame
            ソートされたデータフレーム。
        """
        # データフレームに存在するカラムだけを抽出
        sort_columns = [col for col in sort_columns if col in df.columns]

        df = df.sort_values(sort_columns)
        df.reset_index(inplace=True, drop=True)
        return df

    def replace_column_values_with_last(self, df: pd.DataFrame, id_column: str, target_column: str) -> pd.DataFrame:
        """
        データフレームの指定された列の値を、その列の最後の行の値に置換する。

        Parameters
        ----------
        df : pandas.DataFrame
            データフレーム。
        id_column : str
            置換対象の列名。
        target_column : str
            置換する列名。

        Returns
        -------
        pandas.DataFrame
            置換後のデータフレーム。
        """
        result_df = pd.DataFrame()
        unique_ids = list(df[id_column].unique())
        for unique_id in unique_ids:
            element_df = df[df[id_column] == unique_id].copy()  # スライスのコピーを作成
            last_value = element_df[target_column].iloc[-1]
            element_df.loc[:, target_column] = last_value  # .locを使用して値を設定
            result_df = pd.concat([result_df, element_df], axis=0)
        return result_df

    def filter_columns_by_keyword(self, key_word: str, columns: list) -> list:
        """
        指定されたキーワードを含む列名をリストで返す。

        Parameters
        ----------
        key_word : str
            検索するキーワード。
        columns : list
            検索対象の列名のリスト。

        Returns
        -------
        list
            キーワードを含む列名のリスト。
        """
        return [column for column in columns if key_word in column]

    def remove_duplicates(self, df: pd.DataFrame, subset: list = None, keep: str = 'first') -> pd.DataFrame:
        """
        データフレームから重複行を削除する。

        Parameters
        ----------
        df : pandas.DataFrame
            重複を削除する対象のデータフレーム。
        subset : list, optional
            重複を確認する列のリスト。デフォルトはNoneで、全ての列を対象とする。
        keep : {'first', 'last', False}, default 'first'
            重複行のうち、どの行を保持するかを指定する。
            'first' : 最初の出現を保持
            'last' : 最後の出現を保持
            False : 全ての重複行を削除

        Returns
        -------
        pandas.DataFrame
            重複が削除されたデータフレーム。
        """
        return df.drop_duplicates(subset=subset, keep=keep)
