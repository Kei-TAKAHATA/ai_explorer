import numpy as np

import time_series_forecasting.models.kalman_filter as kalman_filter


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
    seasonal_periods : list[int]
        季節成分の周期
    """
    def __init__(self, y: np.ndarray, sigma2_e: float = None, sigma2_eta: float = None, sigma2_s: float = None, seasonal_periods: list[int] = None):
        """
        コンストラクタ

        Parameters
        ----------
        y : np.ndarray
            観測データ
        sigma2_e : float, optional
            観測誤差の分散
        sigma2_eta : float, optional
            状態誤差の分散
        sigma2_s : float, optional
            季節誤差の分散
        seasonal_periods : list[int], optional
            季節成分の周期
        """
        self.y = y  # 観測データ
        # 状態誤差の分散
        self.sigma2_eta = sigma2_eta if sigma2_eta is not None else 0.01
        # 観測誤差の分散の計算（全く当てがないのでスケールも考慮した小さい値にする）
        self.sigma2_e = sigma2_e if sigma2_e is not None else np.std(self.y[-30:]) ** (1 / 4)
        # 季節誤差の分散
        # self.sigma2_s = sigma2_s  # 季節誤差の分散

        self.seasonal_periods = seasonal_periods  # 季節成分の周期
        number_of_seasonal_periods = len(self.seasonal_periods)
        state_dimension = 1 + 1 + number_of_seasonal_periods * 2
        observation_dimension = 1
        print("state_dimension:", state_dimension)

        # 初期値の設定
        self.initial_state = np.zeros((state_dimension, 1))
        self.initial_covariance = np.eye(state_dimension) * 0.1

        self.F = self.create_state_transition_matrix(self.seasonal_periods)  # 状態遷移行列
        print("F の shape:", self.F.shape)  # 確認
        self.H = np.ones((1, state_dimension))  # 観測行列 H（出数を観測）
        print("H の shape:", self.H.shape)  # 確認

        self.Q = np.eye(state_dimension) * self.sigma2_eta  # 状態ノイズの共分散行列
        print("Q の shape:", self.Q.shape)  # 確認

        self.R = self.sigma2_e * np.eye(observation_dimension)  # 観測ノイズの共分散行列
        print("R の shape:", self.R.shape)  # 確認

        # カルマンフィルタの初期化
        self.kf = kalman_filter.KalmanFilter(
            F=self.F, H=self.H, Q=self.Q, R=self.R,
            initial_state=self.initial_state, initial_covariance=self.initial_covariance
        )
        # フィルタリング後の状態推定を保存するリスト
        self.filtered_states = []

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

    def create_state_transition_matrix(self, seasonal_periods: list[int]):
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

        # 状態空間の次元数を決定 (バイアスとトレンド: 2, 各季節成分: 2)
        num_seasonal_components = len(seasonal_weights)
        state_dim = 2 + 2 * num_seasonal_components  # バイアスとトレンド成分 + 季節成分の合計

        # ブロック行列の構築
        F = np.eye(state_dim)  # 対角成分は単位行列で初期化

        for i, seasonal_weight in enumerate(seasonal_weights):
            start_idx = 2 + 2 * i  # 季節成分の開始インデックス
            F[start_idx:start_idx + 2, start_idx:start_idx + 2] = seasonal_weight  # 季節成分の遷移を設定

        return F

    def fit(self):
        """観測データに基づいてモデルを学習する"""
        for observation in self.y:
            self.kf.predict()  # 予測ステップ
            self.kf.update(observation=observation)  # 更新ステップ
            self.filtered_states.append(self.kf.get_state_estimate())

    def predict(self, steps: int = 1) -> list[float]:
        """
        将来の値を予測

        Parameters
        ----------
        steps : int, optional
            予測するステップ数

        Returns
        -------
        list[float]
            予測された値のリスト
        """
        # 最後の状態推定（最初の状態推定は予測に含めない）
        last_state_estimate = self.filtered_states[-1]

        # 30日間の予測を行う
        forecasted_states = []  # 初期状態を省いて予測を開始
        for _ in range(steps):
            # 状態遷移行列Fを使って、次の日の状態を予測
            next_state = self.F @ last_state_estimate
            forecasted_states.append(next_state)
            last_state_estimate = next_state  # 次の状態を更新

        # 予測結果を出力（出数予測）
        predict_values = [self.H @ state for state in forecasted_states]
        predict_values = [value.item() for value in predict_values]  # スカラー値をリストに変換
        return predict_values
