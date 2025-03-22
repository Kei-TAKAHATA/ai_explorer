import numpy as np


class KalmanFilter:
    """
    カルマンフィルタのクラス
    状態方程式: x_k = F x_{k-1} + B u_k + G V_k
    観測方程式: z_k = H x_k + W_k
    """
    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 B: np.ndarray = None, G: np.ndarray = None,
                 initial_state: np.ndarray = None, initial_covariance: np.ndarray = None) -> None:
        """
        カルマンフィルタの初期化

        Parameters
        ----------
        F : np.ndarray
            状態遷移行列（状態の次元数 * 状態の次元数）
        H : np.ndarray
            観測行列（観測の次元数 * 状態の次元数）
        Q : np.ndarray
            プロセスノイズの共分散行列（状態の次元数 * 状態の次元数）
        R : np.ndarray
            観測ノイズの共分散行列（観測の次元数 * 観測の次元数）
        B : np.ndarray, optional
            制御行列（状態の次元数 * 制御の次元数）。デフォルトはゼロ行列。
        G : np.ndarray, optional
            状態ノイズの遷移行列（状態の次元数 * 状態の次元数）。デフォルトは単位行列。
        initial_state : np.ndarray, optional
            初期状態ベクトル（状態の次元数 * 状態の要素数）。デフォルトはゼロベクトル。
        initial_covariance : np.ndarray, optional
            初期状態の共分散行列（状態の次元数 * 状態の次元数）。デフォルトは単位行列。
        """
        self.F = F  # 状態遷移行列（状態の次元数 * 状態の次元数）
        self.H = H  # 観測行列（観測の次元数 * 状態の次元数）
        self.B = B if B is not None else np.zeros((F.shape[0], F.shape[0]))  # 制御行列（状態の次元数 * 制御の次元数）
        self.G = G if G is not None else np.eye(F.shape[0])  # 状態ノイズの遷移行列（状態の次元数 * 状態の次元数）
        self.Q = Q  # プロセスノイズの共分散行列（状態の次元数 * 状態の次元数）
        self.R = R  # 観測ノイズの共分散行列（観測の次元数 * 観測の次元数）
        self.state_estimate = initial_state if initial_state is not None else np.zeros((F.shape[0], 1))  # 状態推定値x（状態の次元数 * 1）
        self.covariance_estimate = initial_covariance if initial_covariance is not None else np.eye(F.shape[0])  # 共分散推定値（状態の次元数 * 状態の次元数）

    def predict(self, control_input: np.ndarray = None, process_noise: np.ndarray = None) -> None:
        """
        予測ステップ（Prediction Step）

        制御入力を考慮して、状態推定値と共分散を予測する。

        - 状態予測:
          \hat{x}_{k|k-1} = F x_{k-1} + B u_k + G V_k
        - 共分散予測:
          P_{k|k-1} = F P_{k-1} F^T + G Q G^T

        Parameters
        ----------
        control_input : np.ndarray, optional
            制御入力ベクトル (u_k)
        process_noise : np.ndarray, optional
            プロセスノイズベクトル (V_k)
        """
        if process_noise is None:
            process_noise = np.zeros(self.state_estimate.shape)  # ゼロベクトル

        # 状態予測（Xk_ = F * Xk-1 + B * u_k + G * V_k）
        # 制御入力がない場合は B * u_k を省略
        if control_input is not None:
            self.state_estimate = self.F @ self.state_estimate + self.B @ control_input + self.G @ process_noise
        else:
            self.state_estimate = self.F @ self.state_estimate + self.G @ process_noise

        # 共分散予測（Pk_ = F * Pk-1 * F^T + G * Q * G^T）
        self.covariance_estimate = self.F @ self.covariance_estimate @ self.F.T + self.G @ self.Q @ self.G.T

    def update(self, observation: np.ndarray | float | int, observation_noise: np.ndarray | float | int = None, use_pinv: bool = True) -> None:
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
        observation_noise : np.ndarray, optional
            観測ノイズベクトル (W_k)
        use_pinv : bool, optional
            逆行列の計算に擬似逆行列 (pinv) を使用するか。デフォルトは True。
        """
        # 観測値がスカラーの場合、1次元ベクトルに変換
        if np.isscalar(observation):
            observation = np.array([[observation]])

        if observation_noise is not None:
            if np.isscalar(observation_noise):
                observation_noise = np.array([[observation_noise]])

        # カルマンゲインの計算
        # 観測の予測共分散（Sk = H * Pk_ * H^T + R）
        innovation_covariance = self.H @ self.covariance_estimate @ self.H.T + self.R

        # カルマンゲイン（Kk = Pk_ * H^T * (H * Pk_ * H^T + R)^-1）
        kalman_gain = self.covariance_estimate @ self.H.T @ (np.linalg.pinv(innovation_covariance) if use_pinv else np.linalg.inv(innovation_covariance))

        if observation_noise is None:
            observation_noise = np.zeros(observation.shape)  # ゼロベクトル

        # 観測値にノイズを加える
        observation += observation_noise

        # 観測の予測値（イノベーション）（zk - H * Xk_）
        innovation = observation - self.H @ self.state_estimate

        # 状態更新（Xk = Xk_ + Kk * (zk - H * Xk_)）
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
