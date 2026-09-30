import datetime

import pandas as pd

from utils.data_frame_utils import DataFrameUtils


class DataFrameDateUtils:
    """
    データフレームの日付操作を行うクラス。
    """
    def __init__(self):
        self.data_frame_utils = DataFrameUtils()

    def add_weekday_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データフレームに曜日情報を追加する。

        Parameters
        ----------
        df : pandas.DataFrame
            曜日情報を追加する対象のデータフレーム。

        Returns
        -------
        pandas.DataFrame
            曜日情報が追加されたデータフレーム。
        """
        columns = list(df.columns)
        if "date" not in columns:
            df.reset_index(inplace=True)

        date_list = list(df["date"])
        weekday_list = []
        for date in date_list:
            date_time = datetime.datetime.strptime(date, '%Y-%m-%d')
            weekday_list.append(date_time.strftime('%a'))

        column_name = "weekday"
        df.loc[:, column_name] = weekday_list
        return df

    def sort_weekday_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データフレームの曜日情報をソートする。

        Parameters
        ----------
        df : pandas.DataFrame
            曜日情報をソートする対象のデータフレーム。

        Returns
        -------
        pandas.DataFrame
            曜日情報がソートされたデータフレーム。
        """
        df_columns = df.columns
        weekday_info = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        other_weekday_list = []
        for column in df_columns:
            if column not in weekday_info:
                other_weekday_list.append(column)
        other_weekday_df = pd.DataFrame(df[other_weekday_list])
        for weekday in weekday_info:
            if weekday not in df_columns:
                df[weekday] = 0
        weekday_df = df[weekday_info]
        sort_weekday_df = pd.concat([other_weekday_df, weekday_df], axis=1)
        return sort_weekday_df

    def add_weekday_one_hot_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データフレームの曜日情報をワンホットエンコーディングする。

        Parameters
        ----------
        df : pandas.DataFrame
            ワンホットエンコーディングする対象のデータフレーム。

        Returns
        -------
        pandas.DataFrame
            ワンホットエンコーディングされたデータフレーム。
        """

        df = self.add_weekday_info(df)
        df = pd.get_dummies(df, columns=["weekday"], dtype=int)
        df = self.data_frame_utils.delete_str_columns(df, delete_str="weekday_")
        df = self.sort_weekday_columns(df)
        return df

    def add_month_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データフレームの月情報を追加する。

        Parameters
        ----------
        df : pandas.DataFrame
            月情報を追加する対象のデータフレーム。

        Returns
        -------
        pandas.DataFrame
            月情報が追加されたデータフレーム。
        """
        date_list = list(df["date"])
        month_list = []
        for date in date_list:
            date_time = datetime.datetime.strptime(date, '%Y-%m-%d')
            month = date_time.strftime('%m')
            month_list.append(month)
        column_name = "month"
        df[column_name] = month_list
        return df

    def sort_month_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データフレームの月情報をソートする。

        Parameters
        ----------
        df : pandas.DataFrame
            月情報をソートする対象のデータフレーム。

        Returns
        -------
        pandas.DataFrame
            月情報がソートされたデータフレーム。
        """
        df_columns = df.columns
        month_info = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        other_month_list = []
        for column in df_columns:
            if column not in month_info:
                other_month_list.append(column)
        other_month_df = pd.DataFrame(df[other_month_list])
        for month in month_info:
            if month not in df_columns:
                df[month] = 0
        month_df = df[month_info]
        sort_month_df = pd.concat([other_month_df, month_df], axis=1)
        return sort_month_df

    def add_month_one_hot_df(self, df: pd.DataFrame, settings: dict) -> pd.DataFrame:
        """
        データフレームの月情報をワンホットエンコーディングする。

        Parameters
        ----------
        df : pandas.DataFrame
            ワンホットエンコーディングする対象のデータフレーム。

        settings : dict
            ワンホットエンコーディングする設定。

        Returns
        -------
        pandas.DataFrame
            ワンホットエンコーディングされたデータフレーム。
        """
        if settings["using_month_info"] is False:
            return df
        df = self.add_month_info(df)
        df = pd.get_dummies(df, columns=["month"])
        df = self.data_frame_utils.delete_str_columns(df, delete_str="month_")
        df = self.sort_month_columns(df)
        return df

    def get_date_list(self, term_list: list[list[str]]) -> list[str]:
        """
        指定された期間内の日付を1日ずつリストにまとめる。

        Parameters
        ----------
        term_list : list
            日付リストを生成する期間のリストのリスト。
            各要素は [開始日, 終了日] のリスト
            [[start_date1, last_date1], [start_date2, last_date2], ...]

        Returns
        -------
        list
            日付リスト。
        """
        date_list = []
        for start_date, end_date in term_list:
            start_ = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_ = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            current_date = start_
            while current_date <= end_:
                date_list.append(current_date.strftime('%Y-%m-%d'))
                current_date = current_date + datetime.timedelta(days=1)
        return date_list

    def add_one_hot_flag(self, df: pd.DataFrame, date_list: list[str], new_column_name: str) -> pd.DataFrame:
        """
        データフレームの日付情報をワンホットエンコーディングする。

        Parameters
        ----------
        df : pandas.DataFrame
            ワンホットエンコーディングする対象のデータフレーム。

        date_list : list
            ワンホットエンコーディングする日付リスト。

        new_column_name : str
            ワンホットエンコーディングする新しい列名。

        Returns
        -------
        pandas.DataFrame
            ワンホットエンコーディングされたデータフレーム。
        """
        df = df.copy()
        df.loc[df["date"].isin(date_list), new_column_name] = 1
        df.loc[~df["date"].isin(date_list), new_column_name] = 0
        return df

    def add_one_hot_flags(self, df: pd.DataFrame, term_settings: dict) -> pd.DataFrame:
        """
        データフレームの日付情報をワンホットエンコーディングする。

        Parameters
        ----------
        df : pandas.DataFrame
            ワンホットエンコーディングする対象のデータフレーム。

        term_settings : dict
            ワンホットエンコーディングする設定。
            {
                "term_name1": [[start_date1, last_date1], [start_date2, last_date2], ...],
                "term_name2": [[start_date1, last_date1], [start_date2, last_date2], ...],
                ...
            }

        Returns
        -------
        pandas.DataFrame
            ワンホットエンコーディングされたデータフレーム。
        """
        for term_name, term_list in term_settings.items():
            if term_list:
                date_list = self.get_date_list(term_list)
                df = self.add_one_hot_flag(df, date_list, new_column_name=term_name)
        return df

    def generate_term_df(self, df: pd.DataFrame, start_date: str, last_date: str, date_column_name: str = "date") -> pd.DataFrame:
        column_list = list(df.columns)
        if date_column_name in column_list:
            term_df = df[(start_date <= df[date_column_name]) & (df[date_column_name] <= last_date)]
        else:
            term_df = df[(start_date <= df.index) & (df.index <= last_date)]
        return term_df

    def sort_date_df(self, df: pd.DataFrame, date_column_name: str = "date") -> pd.DataFrame:
        column_list = list(df.columns)
        if date_column_name in column_list:
            df = df.sort_values(by=date_column_name, ascending=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def select_term_df(
        self,
        df: pd.DataFrame,
        term_list: list[list[str]],
        is_sort: bool = True,
        reset_index: bool = True,
        date_column_name: str = "date"
    ) -> pd.DataFrame:
        """
        任意の期間を全てまとめたdfを作成

        Parameters
        ----------
        df: pd.DataFrame
            データフレーム
        term_list: list
            期間をまとめたリストのリスト
            [[start_date1, last_date1], [start_date2, last_date2], ..., [start_dateN, last_dateN]]
        is_sort: bool
            ソートするかどうか
        reset_index: bool
            True: インデックスをリセットする
            False: インデックスをリセットしない
        date_column_name: str
            日付の列名
            default: "date"

        Returns
        -------
        pd.DataFrame
            任意の期間を全てまとめたdf
        """
        select_df = pd.DataFrame()
        # column_list = list(df.columns)
        for term_element in term_list:
            start_date = term_element[0]
            # print(start_date)
            last_date = term_element[1]
            # print(last_date)
            term_df = self.generate_term_df(df, start_date, last_date, date_column_name=date_column_name)
            # print("len_term_df: ", len(term_df))
            select_df = pd.concat([select_df, term_df])
        if is_sort:
            select_df = self.sort_date_df(select_df)
        if reset_index:
            select_df.reset_index(drop=True, inplace=True)
        return select_df
