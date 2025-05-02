import pandas as pd
import datetime
import holidays


class DateProcessor:
    """
    日付に関する特徴量（祝日・休前日など）を付与するためのクラス。
    """
    def __init__(self):
        self.jp_holidays = holidays.country_holidays('JP')

    def add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        日付の特徴量を追加する
        休日、休前日フラグを追加する
        """
        df["pd_date"] = pd.to_datetime(df["date"])

        # 土曜日、日曜日、祝日を1、それ以外を0とする
        df["is_holiday"] = df["pd_date"].isin(self.jp_holidays) | df["pd_date"].dt.weekday.isin([5, 6])
        df["is_holiday"] = df["is_holiday"].astype(int)

        # 土曜日、日曜日、祝日の1日前を1、それ以外を0とする
        df["is_day_before_holiday"] = df["pd_date"].shift(-1).isin(self.jp_holidays) | df["pd_date"].dt.weekday.isin([4, 5])
        # 一番最後の行は別で処理
        last_date = df["pd_date"].iloc[-1]
        last_date_next_day = last_date + datetime.timedelta(days=1)
        df.loc[df.index[-1], "is_day_before_holiday"] = 1 if (last_date.weekday() in [4, 5]) or (last_date_next_day in self.jp_holidays) else 0
        df["is_day_before_holiday"] = df["is_day_before_holiday"].astype(int)
        return df
