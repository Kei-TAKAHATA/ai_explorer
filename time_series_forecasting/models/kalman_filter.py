import numpy as np


class KalmanFilter:
    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 initial_state: np.ndarray, initial_covariance: np.ndarray) -> None:
        """
        カルマンフィルタの初期化

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
        self.F = F  # 状態遷移行列
        self.H = H  # 観測行列
        self.Q = Q  # 状態ノイズの共分散行列
        self.R = R  # 観測ノイズの共分散行列
        self.state_estimate = initial_state  # 状態推定値 (x)
        self.covariance_estimate = initial_covariance  # 共分散推定値 (P)

    def predict(self, control_input: np.ndarray = None, B: np.ndarray = None) -> None:
        """
        予測ステップ（Prediction Step）

        制御入力を考慮して、状態推定値と共分散を予測する。

        - 状態予測:
          \hat{x}_{k|k-1} = F x_{k-1} + B u_k
        - 共分散予測:
          P_{k|k-1} = F P_{k-1} F^T + Q

        Parameters
        ----------
        control_input : np.ndarray, optional
            制御入力ベクトル (u_k)
        B : np.ndarray, optional
            制御行列 (B)
        """
        # 状態予測（Xk_ = F * Xk-1 + B * u_k）
        self.state_estimate = self.F @ self.state_estimate
        if control_input is not None and B is not None:
            self.state_estimate += B @ control_input  # B u_k の計算

        # 共分散予測（Pk_ = F * Pk-1 * F^T + Q）
        self.covariance_estimate = self.F @ self.covariance_estimate @ self.F.T + self.Q

    def update(self, observation: np.ndarray, use_pinv: bool = True) -> None:
        """
        更新ステップ（Update Step）

        観測値を用いて推定値を修正する。

        - 観測の予測値（イノベーション）:
          y_k = z_k - H \hat{x}_{k|k-1}
        - 観測の予測共分散:
          S_k = H P_{k|k-1} H^T + R
        - カルマンゲイン:
          K_k = P_{k|k-1} H^T S_k^{-1}
        - 状態更新:
          x_k = \hat{x}_{k|k-1} + K_k y_k
        - 共分散更新:
          P_k = (I - K_k H) P_{k|k-1}

        Parameters
        ----------
        observation : np.ndarray
            観測値 (z_k)
        use_pinv : bool, optional
            逆行列の計算に擬似逆行列 (pinv) を使用するか。デフォルトは True。
        """
        # 観測の予測値（イノベーション）（yk = zk - H * Xk_）
        innovation = observation - self.H @ self.state_estimate

        # 観測の予測共分散（Sk = H * Pk_ * H^T + R）
        innovation_covariance = self.H @ self.covariance_estimate @ self.H.T + self.R

        # カルマンゲイン（Kk = Pk_ * H^T * (H * Pk_ * H^T + R)^-1）
        kalman_gain = self.covariance_estimate @ self.H.T @ (np.linalg.pinv(innovation_covariance) if use_pinv else np.linalg.inv(innovation_covariance))

        # 状態更新（Xk = Xk_ + Kk * zk - H * Xk_）
        self.state_estimate += kalman_gain @ innovation

        # 共分散更新（Pk = (I - Kk * H) * Pk_）
        identity_matrix = np.eye(self.covariance_estimate.shape[0])
        self.covariance_estimate = (identity_matrix - kalman_gain @ self.H) @ self.covariance_estimate

    def get_state_estimate(self) -> np.ndarray:
        """
        現在の状態推定値を取得

        Returns
        -------
        np.ndarray
            現在の状態推定値 (x_k)
        """
        return self.state_estimate
