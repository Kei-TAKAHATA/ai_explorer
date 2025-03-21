import numpy as np


class KalmanFilter:
    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray, initial_state: np.ndarray, initial_covariance: np.ndarray) -> None:
        """
        初期化

        Parameters
        ----------
        F : np.ndarray
            状態遷移行列
        H : np.ndarray
            観測行列
        Q : np.ndarray
            状態ノイズの共分散行列
        R : np.ndarray
            観測ノイズの共分散行列
        initial_state : np.ndarray
            初期状態ベクトル
        initial_covariance : np.ndarray
            初期状態の共分散行列
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.state_estimate = initial_state
        self.covariance_estimate = initial_covariance

    def predict(self) -> None:
        """
        予測ステップ

        Updates the state estimate and covariance estimate based on the model.
        """
        # 状態予測
        self.state_estimate = self.F @ self.state_estimate

        # 共分散予測
        self.covariance_estimate = self.F @ self.covariance_estimate @ self.F.T + self.Q

    def update(self, observation: np.ndarray) -> None:
        """
        更新ステップ

        Parameters
        ----------
        observation : np.ndarray
            観測値

        Updates the state estimate and covariance estimate using the observation.
        """
        # 観測の予測値
        observation_predict = self.H @ self.state_estimate

        # 観測予測の共分散
        S = self.H @ self.covariance_estimate @ self.H.T + self.R

        # カルマンゲイン（擬似逆行列を使用）
        K = self.covariance_estimate @ self.H.T @ np.linalg.pinv(S)

        # 状態更新
        self.state_estimate += K @ (observation - observation_predict)

        # 共分散更新
        identity_matrix = np.eye(self.covariance_estimate.shape[0])
        self.covariance_estimate = (identity_matrix - K @ self.H) @ self.covariance_estimate

    def get_state_estimate(self) -> np.ndarray:
        """
        現在の状態推定値を取得

        Returns
        -------
        np.ndarray
            現在の状態推定値
        """
        return self.state_estimate
