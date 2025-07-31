from typing import List, Tuple

import numpy as np


class PositionCalculator:
    """ロゴの位置を計算するクラス"""

    def __init__(self, position_threshold: float = 0.33):
        """
        パラメータ:
            position_threshold (float): 中央判定の閾値
        """
        self.position_threshold = position_threshold

    def determine_logo_position(
        self, keypoints: List, frame_shape: Tuple[int, int]
    ) -> str:
        """キーポイントの分布から画面内のロゴの位置を判定"""
        # キーポイントの座標を取得
        h, w = frame_shape
        points = np.array([kp.pt for kp in keypoints])

        # 画像の中心からの距離を計算
        center_x, center_y = w / 2, h / 2
        distances = np.sqrt(
            (points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2
        )
        avg_distance = np.mean(distances)

        # 画像の対角線の長さの半分
        max_distance = np.sqrt(w**2 + h**2) / 2

        # 中心からの平均距離を正規化
        relative_distance = avg_distance / max_distance

        # 位置の判定
        if relative_distance < self.position_threshold:
            return "center"
        else:
            return "edge"

    def determine_text_position(
        self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]
    ) -> str:
        """テキストの境界ボックスから画面内の位置を判定"""
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape

        # テキストの中心座標
        center_x = x + w / 2
        center_y = y + h / 2

        # 画像の中心座標
        frame_center_x = frame_w / 2
        frame_center_y = frame_h / 2

        # 中心からの距離を計算
        distance = np.sqrt(
            (center_x - frame_center_x) ** 2 + (center_y - frame_center_y) ** 2
        )

        # 画像の対角線の長さの半分
        max_distance = np.sqrt(frame_w**2 + frame_h**2) / 2

        # 中心からの相対距離
        relative_distance = distance / max_distance

        # 位置の判定
        if relative_distance < self.position_threshold:
            return "center"
        else:
            return "edge"
