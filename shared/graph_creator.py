import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.data_frame_date_utils import DataFrameDateUtils


class GraphCreator:
    """
    グラフを作成するクラス
    """

    def __init__(self):
        """
        GraphCreatorのインスタンスを作成
        """
        self.data_frame_date_utils = DataFrameDateUtils()

    def plot_hist(self, df: pd.DataFrame, plot_columns: list[str] = None, bins: int = 50, figsize: tuple[int, int] = (7, 5), alpha: float = None):
        """
        列ごとにヒストグラムを表示する（面積が1になるように正規化する）

        Parameters
        ----------
        df : pandas.DataFrame
            ヒストグラムを作成するデータフレーム。
        plot_columns : list of str, optional
            ヒストグラムを表示するカラムのリスト。デフォルトはすべてのカラム。
        bins : int, optional
            ビンの数。デフォルトは50。
        figsize : tuple of int, optional
            プロットのサイズ。デフォルトは(7, 5)。
        alpha : float, optional
            各ヒストグラムの透明度。デフォルトはカラム数に基づいて自動設定。
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("dfはpandasのDataFrameである必要があります。")

        if plot_columns is None:
            plot_columns = df.columns

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

        if alpha is None:
            alpha = 1 / len(plot_columns)

        for column in plot_columns:
            if column not in df.columns:
                raise ValueError(f"{column}はデータフレームに存在しません。")
            ax.hist(df[column], bins=bins, alpha=alpha, label=column, density=True)

        ax.legend()  # 凡例表示
        plt.show()
        plt.close()

    def get_default_colors(self):
        """
        デフォルトの色を取得

        Returns
        -------
        list of str
            matplotlibのデフォルトの色サイクルから取得した色のリスト。
        """
        # matplotlibのデフォルトの色サイクルから色を取得
        prop_cycle = plt.rcParams['axes.prop_cycle']
        return prop_cycle.by_key()['color']

    def plot_group_boxplot(
        self,
        ax: plt.Axes,
        group: pd.DataFrame,
        columns_to_plot: list[str],
        colors: list[str],
        alpha: float,
        positions: list[float]
    ):
        """
        特定のグループに対して箱ひげ図をプロット

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            箱ひげ図を描画するためのAxesオブジェクト。
        group : pandas.DataFrame
            プロットするデータを含むグループ。
        columns_to_plot : list of str
            プロットするカラムのリスト。
        colors : list of str
            各カラムに対応する色のリスト。
        alpha : float
            各箱ひげ図の透明度。
        positions : list of float
            各箱ひげ図のプロット位置。
        """
        # 各カラムに対して箱ひげ図を描画
        for j, column in enumerate(columns_to_plot):
            # NaNを除去してから箱ひげ図をプロット
            clean_data = group[column].dropna()
            if not clean_data.empty:
                box = ax.boxplot(clean_data, positions=[positions[j]], patch_artist=True, widths=1)
                for patch in box['boxes']:
                    patch.set_facecolor(colors[j % len(colors)])
                    patch.set_alpha(alpha)
                # 中央値の線を黒に設定
                for median in box['medians']:
                    median.set_color('black')

    def plot_boxplots(self, df: pd.DataFrame, columns_to_plot: list[str] = None, groupby_level: str = None, figsize: tuple[int, int] = (17, 6), title: str = "誤差率の比較", plot_legend: bool = True):
        """
        MultiIndex DataFrameの各グループに対して箱ひげ図をプロットする

        Parameters
        ----------
        df : pandas.DataFrame
            プロットするデータフレーム。
        columns_to_plot : list of str, optional
            プロットするカラムのリスト。デフォルトはすべてのカラム。
        groupby_level : str, optional
            グループ化に使用するレベル。デフォルトはNone。
        figsize : tuple of int, optional
            プロットのサイズ。デフォルトは(17, 6)。
        title : str, optional
            プロットのタイトル。デフォルトは"誤差率の比較"。
        """
        if columns_to_plot is None:
            columns_to_plot = df.columns

        # デフォルトの色と透明度を設定
        colors = self.get_default_colors()
        alpha = 1 / len(columns_to_plot)

        if groupby_level is not None:
            # 指定されたレベルでグループ化
            groups = df.groupby(level=groupby_level, dropna=False)
        else:
            # グループ化せずに全体を一つのグループとして扱う
            groups = [(None, df)]

        # プロットの幅を動的に設定
        if (groupby_level is None):
            figsize = (7, 5)  # グループ化しない場合は幅を小さく
        # figsize = (max(figsize[0], len(columns_to_plot) * 2), figsize[1])  # カラム数に応じて幅を調整

        # プロットの準備
        fig, ax = plt.subplots(figsize=figsize)

        for i, (name, group) in enumerate(groups):
            # 各グループのプロット位置を計算
            if groupby_level is not None:
                positions = np.arange(len(columns_to_plot)) * 2 + i * len(columns_to_plot) * 2.5
            else:
                # グループ化しない場合は中央に配置
                total_width = len(columns_to_plot) * 2
                offset = (figsize[0] - total_width) / 2
                positions = np.arange(len(columns_to_plot)) * 2 + offset

            self.plot_group_boxplot(ax, group, columns_to_plot, colors, alpha, positions)

        if groupby_level is not None:
            labels = df.index.get_level_values(groupby_level).unique()
            ax.set_xticks(np.arange(len(labels)) * len(columns_to_plot) * 2.5 + 0.5)
            ax.set_xticklabels(labels, rotation=90)
        else:
            ax.set_xticks(positions)  # 位置をpositionsに合わせる
            ax.set_xticklabels(columns_to_plot, rotation=90)

        ax.set_xlim(-1, len(columns_to_plot) * 2.5 * (len(groups) if groupby_level else 1))
        if plot_legend:
            ax.legend(columns_to_plot)  # 凡例表示
        ax.set_title(title)
        ax.set_ylabel('Values')

        # グリッド線を追加
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

        plt.suptitle('')
        plt.show()
        plt.close(fig)  # プロットを閉じてメモリを解放

    def plot_boxplot_no_group(self, df: pd.DataFrame, figsize: tuple[int, int] = (17, 6)):
        """
        グループ化しない箱ひげ図をプロットする
        """
        # プロットの準備
        fig, ax = plt.subplots(figsize=figsize)
        ax = df.plot.box(fontsize=6, ax=ax)
        plt.xticks(fontsize=5, rotation=90)
        plt.show()
        plt.close(fig)

    def display_dict(self, d: dict):
        """
        辞書の内容を表示する

        Parameters
        ----------
        d : dict
            表示する辞書。
        """
        for key, value in d.items():
            print(f"{key}: {value}")

    def get_x_data(self, df: pd.DataFrame, x_column: str = None) -> pd.Series:
        """
        x軸のデータを取得する

        Parameters
        ----------
        df : pandas.DataFrame
            データを含むデータフレーム。
        x_column : str or None
            x軸に使用するカラム名。Noneの場合は"date"カラムを使用。

        Returns
        -------
        pandas.Series
            x軸のデータ。
        """
        if x_column is None:
            x = df["date"]
        else:
            x = df[x_column]
        return x

    def select_columns_to_plot(self, df: pd.DataFrame, plot_columns: list[str] = None) -> list[str]:
        """
        プロットするカラムを選択する

        Parameters
        ----------
        df : pandas.DataFrame
            データを含むデータフレーム。
        plot_columns : list of str or None
            プロットするカラムのリスト。Noneの場合は"date"以外のすべてのカラムを選択。

        Returns
        -------
        list of str
            プロットするカラムのリスト。
        """
        if plot_columns is None:
            columns = df.columns.copy()
            columns_list = list(columns)
            columns_list.remove("date")
            plot_columns = columns_list.copy()
        return plot_columns

    def extract_plot_data(self, df: pd.DataFrame, plot_columns: list[str]) -> list[pd.Series]:
        """
        プロットするデータを抽出する

        Parameters
        ----------
        df : pandas.DataFrame
            データを含むデータフレーム。
        plot_columns : list of str
            プロットするカラムのリスト。

        Returns
        -------
        list of pandas.Series
            プロットするデータのリスト。
        """
        plot_data = []
        for column in plot_columns:
            data = df[column]
            plot_data.append(data)
        return plot_data

    def plot_line_graph(
        self,
        df: pd.DataFrame,
        x_column: str = None,
        plot_columns: list[str] = None,
        term_list: list[list[str]] = None,
        figsize: tuple[float, float] = (2.5, 7),
        x_ticks_fontsize: int = 5
    ):
        """
        線グラフをプロットする

        Parameters
        ----------
        df : pandas.DataFrame
            プロットするデータフレーム。
        x_column : str or None, optional
            x軸に使用するカラム名。Noneの場合は"date"カラムを使用。
        plot_columns : list of str or None, optional
            プロットするカラムのリスト。Noneの場合はx_column以外のすべてのカラムを使用。
        term_list : list of str or None, optional
            期間をまとめたリストのリスト。
            [[start_date1, last_date1], [start_date2, last_date2], ..., [start_dateN, last_dateN]]
            Noneの場合は全期間を使用。
            default: None
        figsize : tuple of float, optional
            プロットのサイズ。デフォルトは(2.5, 7)。
        x_ticks_fontsize : int, optional
            x軸のティックラベルのフォントサイズ。デフォルトは5。
        """
        # 期間の指定がある場合
        if term_list:
            df = self.data_frame_date_utils.select_term_df(df, term_list=term_list, date_column_name=x_column)

        x = self.get_x_data(df, x_column)
        plot_columns = self.select_columns_to_plot(df, plot_columns)
        plot_data = self.extract_plot_data(df, plot_columns)

        # colors = itertools.cycle(["r", "b", "g", "k", "m", "y", "c", "darkred", "navy", "limegreen", "orange", "gray", "purple", "cadetblue", "brown"])
        # colors = itertools.cycle(["red", "blue", "green", "black", "magenta", "yellow", "cyan",
        #                           "darkred", "navy", "limegreen", "orange", "gray", "purple", "cadetblue", "brown"])
        colors = itertools.cycle([
            "red",        # 1
            "blue",       # 2
            "green",      # 3
            "orange",     # 4（暖色で目立つ）
            "purple",     # 5（青・赤系の中間色）
            "black",      # 6（黒）
            "limegreen",  # 7（明るい緑系）
            "navy",       # 8（濃い青系）
            "magenta",    # 9（ピンク系で目立つ）
            "cadetblue",  # 10（青緑系）
            "darkred",    # 11（濃い赤系）
            "gray",       # 12（無彩色）
            "cyan",       # 13（明るい水色）
            "yellow",     # 14（背景によっては見えにくいので後ろに）
        ])

        # line_styles = [
        #     {"color": "red",   "linestyle": "-",  "linewidth": 2.5},  # 正解
        #     {"color": "blue",  "linestyle": "--", "linewidth": 1.8},  # 予測1
        #     {"color": "green", "linestyle": "-.", "linewidth": 1.8},  # 予測2
        #     {"color": "orange","linestyle": ":",  "linewidth": 1.5},  # 追加線
        #     # 以降は色だけ変える or 適宜繰り返し
        # ]

        # color_map = plt.get_cmap("tab20c")

        # figsizeの引数1が横、引数2が縦
        number_of_columns = 1
        unit_of_figure = 1
        figsize = (figsize[1] * unit_of_figure, figsize[0] * number_of_columns)
        fig, ax = plt.subplots(number_of_columns, unit_of_figure, figsize=figsize)

        for index in range(len(plot_columns)):
            label = plot_columns[index]
            data = plot_data[index]
            color = next(colors)
            ax.plot(x, data, color=color, label=label)
            # style = line_styles[index] if index < len(line_styles) else {"color": next(colors), "linestyle": "-", "linewidth": 1.5}
            # ax.plot(x, data, label=label, **style)

        ax.legend()  # 凡例表示
        plt.xticks(fontsize=x_ticks_fontsize, rotation=90)
        plt.tight_layout()
        plt.xlabel("date")
        plt.ylabel("quantity")
        plt.show()
        plt.close()
        return

    def plot_unit_line_graphs(self, unit_dfs: dict, settings: dict):
        """
        各ユニットの線グラフをプロットする

        Parameters
        ----------
        unit_dfs : dict
            各ユニットのデータフレームを含む辞書。
        settings : dict
            プロットの設定を含む辞書。
        """
        # プロットするユニットのキーのリスト
        # Noneの場合はすべてのキーを使用。
        key_list = settings.get("key_list", None)
        term_settings = settings["term_settings"]
        x_column = settings.get("x_column", None)
        plot_columns = settings["plot_columns"]
        print_info = settings.get("print_info", [])

        # key_listがNoneの場合、unit_dfsのすべてのキーを使用
        if key_list is None:
            key_list = unit_dfs.keys()

        for unit_name in key_list:
            print("----------------------------")
            print("unit_name: ", unit_name)
            df = unit_dfs[unit_name]

            # 存在するカラムを取得
            common_info = {col: df[col].iloc[0] for col in print_info if col in df.columns}
            self.display_dict(common_info)

            for term_name, term_list in term_settings.items():
                print()
                print("term_name: ", term_name)
                self.plot_line_graph(
                    df=df,
                    x_column=x_column,
                    plot_columns=plot_columns,
                    term_list=term_list,
                )
        return
