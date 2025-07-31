import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FeatureMatcher:
    """特徴点検出とマッチングを行うクラス"""

    def __init__(self, match_threshold: float = 0.7, trees: int = 8, checks: int = 80):
        """
        パラメータ:
            match_threshold (float): 特徴点マッチングの品質閾値
            trees (int): FLANNマッチャーのtreesパラメータ
            checks (int): FLANNマッチャーのchecksパラメータ
        """
        self.match_threshold = match_threshold
        self.trees = trees
        self.checks = checks
        self.sift = cv2.SIFT_create()

    def extract_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """画像から特徴点と特徴量を抽出"""
        gray_img = self._ensure_grayscale(image)
        return self.sift.detectAndCompute(gray_img, None)

    def match_features(self, des_logo: np.ndarray, des_frame: np.ndarray) -> List:
        """特徴量同士をマッチング"""
        # FLANNマッチャーの設定
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=self.trees)
        search_params = dict(checks=self.checks)

        # FLANNマッチャーの初期化
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 最近傍マッチング
        matches = flann.knnMatch(des_logo, des_frame, k=2)

        # Ratio testで品質が高いマッチングを抽出
        good_matches = []
        for m, n in matches:
            if m.distance < self.match_threshold * n.distance:
                good_matches.append(m)

        return good_matches

    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """画像がグレースケールであることを保証"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
