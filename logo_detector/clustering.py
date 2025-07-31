from typing import Tuple

import numpy as np
from sklearn.cluster import DBSCAN


class LogoClustering:
    """特徴点のクラスタリングを行うクラス"""

    def __init__(self, eps: float = 50, min_samples: int = 3):
        """
        パラメータ:
            eps (float): DBSCANのepsパラメータ（クラスタの距離閾値）
            min_samples (int): DBSCANのmin_samplesパラメータ（クラスタを形成する最小点数）
        """
        self.eps = eps
        self.min_samples = min_samples

    def cluster_matched_points(
        self, matched_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """マッチした特徴点をクラスタリング"""
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(
            matched_points
        )
        labels = clustering.labels_

        # ノイズを除去し、クラスタ別のIDを取得
        unique_clusters = np.unique(labels)
        unique_clusters = unique_clusters[unique_clusters != -1]

        return labels, unique_clusters

    def get_cluster_indices(self, labels: np.ndarray, cluster_id: int) -> np.ndarray:
        """指定クラスタに属するインデックスを取得"""
        return np.where(labels == cluster_id)[0]
