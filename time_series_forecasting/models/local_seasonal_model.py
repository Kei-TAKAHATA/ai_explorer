import numpy as np
import pandas as pd

from time_series_forecasting.models import kalman_filter


class LocalSeasonalModel:
    """
    ローカル季節モデルのクラス

    状態空間方程式: x_t = F * x_t-1 + w_t
    観測方程式: y_t = H * x_t + v_t

    Attributes
    ----------
    y : np.ndarray
        観測データ
    sigma2_e : float
        観測誤差の分散
    sigma2_eta : float
        状態誤差の分散
    sigma2_s : float
        季節誤差の分散
    non_seasonal_dim : int
        季節成分以外の状態の次元数
    seasonal_periods : list[int]
        季節成分の周期
    """
    def __init__(
        self,
        seasonal_periods: list[int],
        include_trend: bool = True
    ):
        """
        コンストラクタ

        Parameters
        ----------
        seasonal_periods : list[int]
            季節成分の周期
        include_trend : bool, optional
            トレンドを含めるかどうかの設定
        """
        self.seasonal_periods = seasonal_periods  # 季節成分の周期
        self.include_trend = include_trend  # トレンドを含めるかどうかの設定
        self.filtered_states = []  # フィルタリング後の状態推定を保存するリスト

    def create_seasonal_weights(self, time_period: int):
        """周期成分のための2 * 2の回転行列を作成

        Parameters
        ----------
        time_period : int
            季節成分の周期

        Returns
        -------
        np.ndarray
            2x2の回転行列
        """
        theta = 2 * np.pi / time_period
        weight = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        return weight

    def create_state_transition_matrix(self, non_seasonal_dim: int, seasonal_periods: list[int]):
        """
        状態遷移行列を作成

        Parameters
        ----------
        seasonal_periods : list[int]
            季節成分の周期

        Returns
        -------
        np.ndarray
            状態遷移行列
        """
        seasonal_weights = [self.create_seasonal_weights(term) for term in seasonal_periods]

        # 状態空間の次元数を決定 (バイアス, 各季節成分: 2)
        num_seasonal_components = len(seasonal_weights)
        state_dim = non_seasonal_dim + 2 * num_seasonal_components  # バイアス成分 + 季節成分の合計

        # 状態遷移行列の構築
        F = np.eye(state_dim)  # 対角成分は単位行列で初期化
        if self.include_trend:
            F[0, 1] = 1  # トレンド項を含める
        for i, seasonal_weight in enumerate(seasonal_weights):
            start_idx = non_seasonal_dim + 2 * i  # 季節成分の開始インデックス
            F[start_idx:start_idx + 2, start_idx:start_idx + 2] = seasonal_weight  # 季節成分の遷移を設定

        return F

    def fit(self, train_x, train_y, sigma2_e: float = None, sigma2_eta: float = None, sigma2_s: float = None):
        """観測データに基づいてモデルを学習する"""
        # 状態誤差の分散
        sigma2_eta = sigma2_eta if sigma2_eta is not None else 0.01
        # 観測誤差の分散の計算（全く当てがないのでスケールも考慮した小さい値にする）
        sigma2_e = sigma2_e if sigma2_e is not None else np.std(train_y[-30:]) ** (1 / 4)
        # 季節誤差の分散
        sigma2_s = sigma2_s if sigma2_s is not None else 0.01  # 季節誤差の分散

        # 状態の次元数
        number_of_seasonal_periods = len(self.seasonal_periods)
        if self.include_trend:
            non_seasonal_dim = train_x.shape[1] + 1
        else:
            non_seasonal_dim = train_x.shape[1]
        state_dimension = non_seasonal_dim + 2 * number_of_seasonal_periods
        observation_dimension = train_y.shape[1]  # 観測データの次元数に基づく
        # print("state_dimension:", state_dimension)
        # print("observation_dimension:", observation_dimension)

        # 初期値の設定
        initial_state = np.zeros((state_dimension, 1))
        initial_covariance = np.eye(state_dimension) * 0.1

        self.F = self.create_state_transition_matrix(non_seasonal_dim, self.seasonal_periods)  # 状態遷移行列
        # print("F の shape:", self.F.shape)  # 確認
        self.H = np.ones((observation_dimension, state_dimension))  # 観測行列 H
        # print("H の shape:", self.H.shape)  # 確認
        self.Q = sigma2_eta * np.eye(state_dimension)  # 状態ノイズの共分散行列
        # print("Q の shape:", self.Q.shape)  # 確認
        self.R = sigma2_e * np.eye(observation_dimension)  # 観測ノイズの共分散行列
        # print("R の shape:", self.R.shape)  # 確認

        # カルマンフィルタの初期化
        self.kf = kalman_filter.KalmanFilter(
            F=self.F, H=self.H, Q=self.Q, R=self.R,
            initial_state=initial_state, initial_covariance=initial_covariance
        )

        for observation in train_y:
            if observation.ndim == 1:
                observation = observation.reshape(-1, 1)
            # print("observation:", observation)
            # print("observation.shape:", observation.shape)
            self.kf.predict()  # 予測ステップ
            self.kf.update(observation=observation)  # 更新ステップ
            self.filtered_states.append(self.kf.get_state_estimate())

    def predict_by_steps(self, steps: int = 1, decay_factor: float = 0.9) -> list[float]:
        """
        ステップ数分の予測を行う

        Parameters
        ----------
        target_columns : list[str]
            予測対象のカラム名
        steps : int, optional
            予測するステップ数
        decay_factor : float, optional
            トレンド項に適用する減衰因子

        Returns
        -------
        list[float]
            ターゲット列の予測された値のリスト
        """
        # 最後の状態推定（最初の状態推定は予測に含めない）
        last_state_estimate = self.filtered_states[-1]
        # 予測を行う
        forecasted_states = []  # 初期状態を省いて予測を開始
        for step in range(steps):
            # 状態遷移行列Fを使って、次の日の状態を予測
            next_state = self.F @ last_state_estimate
            if self.include_trend:
                next_state[1] *= decay_factor ** step  # トレンド項に減衰因子を適用
            forecasted_states.append(next_state)
            last_state_estimate = next_state  # 次の状態を更新

        # 予測結果を出力（ステップ数 * 観測データの次元数の行列 * 1）
        predict_values = [self.H @ state for state in forecasted_states]
        predict_values = np.array(predict_values)
        # ステップ数 * 観測データの次元数の行列に変換
        predict_values = predict_values.reshape(predict_values.shape[0], predict_values.shape[1])
        return predict_values

    def predict_from_df(self, df: pd.DataFrame, predict_column_names: list[str] = None) -> pd.DataFrame:
        """
        観測データのdfから予測を行い、予測結果を新しい列として追加したDataFrameを返す

        Parameters
        ----------
        df : pd.DataFrame
            観測データのDataFrame
        predict_column_names : list[str], optional
            予測結果を追加する列名
        """
        if predict_column_names is None:
            predict_column_names = ["predicted_values"]  # デフォルトの列名を設定

        steps = len(df)
        predict_values = self.predict_by_steps(steps=steps)

        # 予測結果を新しい列として追加
        predict_df = pd.DataFrame(predict_values, columns=predict_column_names)
        df = pd.concat([df, predict_df], axis=1)
        return df
