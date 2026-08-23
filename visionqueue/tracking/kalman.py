"""
Kalman Filter for ByteTrack - Constant Velocity Model

State: [x, y, w, h, vx, vy, vw, vh] (center x, center y, width, height, velocities)
Measurement: [x, y, w, h] (center x, center y, width, height)
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class KalmanFilter:
    """Constant velocity Kalman filter for bounding box tracking."""

    def __init__(self, dt: float = 1.0, std_weight_position: float = 1.0 / 20, std_weight_velocity: float = 1.0 / 160):
        self.dt = dt
        self.std_weight_position = std_weight_position
        self.std_weight_velocity = std_weight_velocity

        # State transition matrix (8x8)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = dt

        # Measurement matrix (4x8) - observe position only
        self.H = np.eye(4, 8, dtype=np.float32)

        # Process noise covariance (8x8)
        self.Q = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.Q[i, i] = std_weight_position ** 2
            self.Q[i + 4, i + 4] = std_weight_velocity ** 2

        # Measurement noise covariance (4x4)
        self.R = np.eye(4, dtype=np.float32) * (std_weight_position ** 2)

        # Initial state covariance
        self.P = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.P[i, i] = 10.0
            self.P[i + 4, i + 4] = 100.0

    def initiate(self, measurement: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Create initial state from measurement [x, y, w, h]."""
        mean = np.zeros(8, dtype=np.float32)
        mean[:4] = measurement
        mean[4:] = 0.0  # initial velocity = 0
        covariance = self.P.copy()
        return mean, covariance

    def predict(self, mean: NDArray[np.float32], covariance: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Predict step."""
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + self.Q
        return mean, covariance

    def project(self, mean: NDArray[np.float32], covariance: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Project state to measurement space."""
        mean = self.H @ mean
        covariance = self.H @ covariance @ self.H.T + self.R
        return mean, covariance

    def update(self, mean: NDArray[np.float32], covariance: NDArray[np.float32], measurement: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Update step with measurement."""
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain
        K = covariance @ self.H.T @ np.linalg.inv(projected_cov)

        # Update
        innovation = measurement - projected_mean
        mean = mean + K @ innovation
        covariance = (np.eye(8, dtype=np.float32) - K @ self.H) @ covariance

        return mean, covariance

    def gating_distance(self, mean: NDArray[np.float32], covariance: NDArray[np.float32], measurements: NDArray[np.float32], only_position: bool = False) -> NDArray[np.float32]:
        """Compute Mahalanobis distance for gating."""
        projected_mean, projected_cov = self.project(mean, covariance)

        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(projected_cov)
        d = measurements - projected_mean
        z = np.linalg.solve(cholesky_factor, d.T).T
        return np.sum(z * z, axis=1)

    @staticmethod
    def tlwh_to_xywh(tlwh: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert top-left-w-h to center-x-y-w-h."""
        x = tlwh[0] + tlwh[2] / 2.0
        y = tlwh[1] + tlwh[3] / 2.0
        return np.array([x, y, tlwh[2], tlwh[3]], dtype=np.float32)

    @staticmethod
    def xywh_to_tlwh(xywh: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert center-x-y-w-h to top-left-w-h."""
        x1 = xywh[0] - xywh[2] / 2.0
        y1 = xywh[1] - xywh[3] / 2.0
        return np.array([x1, y1, xywh[2], xywh[3]], dtype=np.float32)

    @staticmethod
    def tlwh_to_tlbr(tlwh: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert tlwh to tlbr (x1, y1, x2, y2)."""
        return np.array([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], dtype=np.float32)

    @staticmethod
    def tlbr_to_tlwh(tlbr: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert tlbr to tlwh."""
        return np.array([tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]], dtype=np.float32)