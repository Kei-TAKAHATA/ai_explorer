#!/usr/bin/env python
# coding: utf-8

# # 回帰モデルの評価指標比較_テックブログ用

# # ルートディレクトリの取得

# In[ ]:


import os
import sys

def get_ancestor_directory(path, levels=1):
    """
    指定した階層数だけ上の親ディレクトリを取得します。

    Parameters:
    path (str): 基準となるディレクトリのパス。
    levels (int): 上に移動する階層数。デフォルトは1。

    Returns:
    str: 指定した階層数だけ上の親ディレクトリのパス。
    """
    for _ in range(levels):
        path = os.path.dirname(path)
    return path

# 使用例
current_directory = os.getcwd()
# ai_explorer直下のライブラリをimportするため、2階層上のディレクトリを取得
root_directory = get_ancestor_directory(current_directory, levels=2)

print("現在のディレクトリ:", current_directory)
print("ルートディレクトリ:", root_directory)
sys.path.append(os.path.abspath(root_directory))


# # ライブラリのimport

# In[2]:


import numpy as np
import pandas as pd

from utils.graph_creator import GraphCreator

graph_creator = GraphCreator()


# ## 使用するメソッド

# In[3]:


# RMSEの計算
def calculate_rmse(df: pd.DataFrame, actual_column: str, predicted_column: str) -> float:
    """
    RMSE（Root Mean Square Error）を計算する関数
    RMSE = sqrt(1/n * sum((y - y_hat)^2))

    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    actual_column : str
        実測値の列名
    predicted_column : str
        予測値の列名

    Returns
    -------
    float
        RMSEの値
    """
    # 実測値と予測値の差を計算
    error = df[actual_column] - df[predicted_column]

    # 差の二乗を計算
    squared_error = error ** 2

    # 平均を計算
    mean_squared_error = squared_error.mean()

    # 平方根を計算
    rmse = np.sqrt(mean_squared_error)

    return rmse

# MAEの計算
def calculate_mae(df: pd.DataFrame, actual_column: str, predicted_column: str) -> float:
    """
    MAE（Mean Absolute Error）を計算する関数
    MAE = 1/n * sum(|y - y_hat|)

    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    actual_column : str
        実測値の列名
    predicted_column : str
        予測値の列名

    Returns
    -------
    float
        MAEの値
    """
    # 実測値と予測値の差を計算
    error = df[actual_column] - df[predicted_column]

    # 差の絶対値を計算
    absolute_error = np.abs(error)

    # 平均を計算
    mean_absolute_error = absolute_error.mean()

    return mean_absolute_error

# MAPEの計算
def calculate_mape(df: pd.DataFrame, actual_column: str, predicted_column: str) -> float:
    """
    MAPE（Mean Absolute Percentage Error）を計算する関数
    実績値が0の場合は計算できないため、0の行を削除して計算
    MAPE = 100 * 1/n * sum(| (y - y_hat) / y |)

    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    actual_column : str
        実測値の列名
    predicted_column : str
        予測値の列名

    Returns
    -------
    float
        MAPEの値
    """
    # 実測値が0の場合は0除算になるので、0の行を削除
    df = df[df[actual_column] != 0].copy()

    # 実測値と予測値の差を計算
    error = df[actual_column] - df[predicted_column]
    error_rate = error / df[actual_column]

    # 絶対値の比率を計算
    absolute_error_rate = np.abs(error_rate)

    # 平均を計算
    mean_absolute_error_rate = absolute_error_rate.mean()

    # 100を掛けてパーセントに変換
    mape = 100 * mean_absolute_error_rate

    return mape

# SMAPEの計算
def calculate_smape(df: pd.DataFrame, actual_column: str, predicted_column: str) -> float:
    """
    SMAPE（Symmetric Mean Absolute Percentage Error）を計算する関数
    実績値と予測値の絶対値の合計が0の場合は計算できないため、その場合は0の行を削除して計算
    SMAPE = 100 * 1/n * sum(| (y - y_hat) / ( |y| + |y_hat| ) / 2 |)

    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    actual_column : str
        実測値の列名
    predicted_column : str
        予測値の列名

    Returns
    -------
    float
        SMAPEの値
    """

    # 実測値と予測値の絶対値を計算
    absolute_actual = np.abs(df[actual_column])
    absolute_predicted = np.abs(df[predicted_column])

    # 実測値と予測値の絶対値の合計が0の場合は0除算になるので、0の行を削除
    df = df[absolute_actual + absolute_predicted != 0].copy()

    # 0の行を削除した後の実測値と予測値の絶対値を再計算
    absolute_actual = np.abs(df[actual_column])
    absolute_predicted = np.abs(df[predicted_column])

    # 実測値と予測値の差を計算
    error = df[actual_column] - df[predicted_column]

    # 絶対値の比率を計算
    absolute_error_rate = 2 * np.abs(error) / (absolute_actual + absolute_predicted)

    # 平均を計算
    mean_absolute_error_rate = absolute_error_rate.mean()

    # 100を掛けてパーセントに変換
    smape = 100 * mean_absolute_error_rate

    return smape

# MASEの計算
def calculate_mase(df: pd.DataFrame, actual_column: str, predicted_column: str, m: int = 1) -> float:
    """
    MASE（Mean Absolute Scaled Error）を計算する関数
    MASE = (1/n) * sum(|y - y_hat| / ( (1 / (n-1)) * sum(|y - y_i-1|)))

    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    actual_column : str
        実測値の列名
    predicted_column : str
        予測値の列名
    m : int, default 1
        季節性周期（非季節系列の場合は1）

    Returns
    -------
    float
        MASEの値
    """
    # 実測値と予測値の差を計算
    error = df[actual_column] - df[predicted_column]

    # 実測値と予測値の差の絶対値を計算
    absolute_error = np.abs(error)

    # 実測値と予測値の差の絶対値の平均を計算
    mean_absolute_error = absolute_error.mean()

    # naive法のMAE（ラグm差分の平均絶対値）
    if len(df[actual_column]) <= m:
        raise ValueError("データ数が季節性周期m以下のため、MASEを計算できません。")
    mae_naive = np.mean(np.abs(df[actual_column].values[m:] - df[actual_column].values[:-m]))

    # MASEを計算
    mase = mean_absolute_error / mae_naive

    return mase

# R^2の計算
def calculate_r2(df: pd.DataFrame, actual_column: str, predicted_column: str) -> float:
    """
    R^2（決定係数）を計算する関数
    R^2 = 1 - (sum((y - y_hat)^2) / sum((y - y_mean)^2))

    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    actual_column : str
        実測値の列名
    predicted_column : str
        予測値の列名

    Returns
    -------
    float
        R^2の値
    """

    # 実測値の平均を計算
    y_mean = df[actual_column].mean()

    # 実測値と平均の差の二乗の合計（全変動）を計算
    diff_mean_squared = (df[actual_column] - y_mean) ** 2
    total_sum_of_squares = diff_mean_squared.sum()

    # 実測値と予測値の差を計算
    error = df[actual_column] - df[predicted_column]

    # 実測値と予測値の差の二乗を計算
    squared_error = error ** 2

    # 実測値と予測値の差の二乗の合計（残渣平方和）を計算
    sum_of_squared_errors = squared_error.sum()

    # R^2を計算
    r2 = 1 - (sum_of_squared_errors / total_sum_of_squares)

    return r2


# In[4]:


def calculate_metrics(df: pd.DataFrame, actual_column: str, predicted_column: str) -> pd.DataFrame:
    """
    指標を行、予測カラム名を列にしたDataFrameを返す

    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    actual_column : str
        実測値の列名
    predicted_column : str
        予測値の列名

    Returns
    -------
    pd.DataFrame
        指標がindex、予測カラム名が列名のDataFrame
    """
    metrics = {
        "RMSE": calculate_rmse(df, actual_column, predicted_column),
        "MAE": calculate_mae(df, actual_column, predicted_column),
        "MAPE": calculate_mape(df, actual_column, predicted_column),
        "SMAPE": calculate_smape(df, actual_column, predicted_column),
        "MASE": calculate_mase(df, actual_column, predicted_column),
        "R^2": calculate_r2(df, actual_column, predicted_column)
    }
    # 指標をindex、予測カラム名を列名に
    return pd.DataFrame(metrics, index=[predicted_column]).T


# In[ ]:


data = {
    "date": ["2025-06-02", "2025-06-03", "2025-06-04", "2025-06-05", "2025-06-06", "2025-06-07", "2025-06-08"],
    "y": [10, 12, 0, 13, 20, 60, 50],
    "predicted_1": [10, 13, 1, 12, 18, 55, 40],
    "predicted_2": [7, 15, 2, 10, 23, 65, 55]
}

df = pd.DataFrame(data)
df


# In[ ]:


print(df.to_markdown(index=False))


# In[ ]:


graph_creator.plot_line_graph(
    df,
    x_column="date",
    plot_columns=["y", "predicted_1", "predicted_2"],
    figsize=(2.5, 7)
)


# In[ ]:


predicted_columns = [column for column in df.columns if column.startswith("predicted_")]

result = []
actual_column = "y"
for predicted_column in predicted_columns:
    result.append(calculate_metrics(df, actual_column, predicted_column))

result_df = pd.concat(result, axis=1)
result_df


# In[ ]:


print(result_df.round(2).to_markdown())


# In[ ]:




