import logging
import math
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PercentageCalculator:
    """画面占有率を計算するクラス"""

    def calculate_logo_percentage(
        self,
        src_keypoints: List,
        dst_keypoints: List,
        frame_shape: Tuple[int, int],
        logo_mask: np.ndarray,
        good_matches: List,
    ) -> float:
        """ロゴが画面に占める割合を計算"""
        if len(good_matches) < 4:
            return 0.0

        try:
            # マッチした点の対応を取得
            src_pts = np.float32(
                [src_keypoints[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [dst_keypoints[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)

            # ホモグラフィ行列を計算
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # ロゴのマスクを変換
            h, w = frame_shape
            transformed_mask = cv2.warpPerspective(logo_mask, M, (w, h))

            # ロゴのピクセル数と総ピクセル数を計算
            logo_pixels = np.sum(transformed_mask > 0)
            total_pixels = h * w

            # パーセンテージを計算（整数に切り上げ）
            percentage = math.ceil((logo_pixels / total_pixels) * 100)
            return percentage

        except Exception as e:
            logger.error(f"Error calculating logo percentage: {str(e)}")
            return 0.0

    def calculate_text_percentage(
        self,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int],
        correction_factor: float = 1.3,
    ) -> float:
        """テキストの境界ボックスから画面占有率を計算"""
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape

        # テキスト領域のピクセル数
        text_pixels = w * h

        # フレーム全体のピクセル数
        total_pixels = frame_h * frame_w

        # 占有率を計算して補正係数を適用
        raw_percentage = (text_pixels / total_pixels) * 100
        percentage = math.ceil(raw_percentage * correction_factor)

        return percentage
