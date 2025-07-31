import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import cv2
import numpy as np

from .clustering import LogoClustering
from .config import LOGO_DETECTION_PARAMS
from .feature_matcher import FeatureMatcher
from .image_utils import ImageUtils
from .percentage_calculator import PercentageCalculator
from .position_calculator import PositionCalculator
from .result_merger import ResultMerger
from .text_detector import TextDetector

# ロギング設定
logger = logging.getLogger(__name__)


class LogoDetector:
    """ロゴ検出を行うメインクラス"""

    def __init__(
        self,
        min_match_count: int = None,
        debug_mode: bool = False,
    ):
        # デバッグモードのフラグを設定（テストモードも兼ねる）
        self.debug_mode = debug_mode

        # 設定ファイルからパラメータを取得
        self.config = LOGO_DETECTION_PARAMS

        # パラメータ設定（引数が指定されていれば優先）
        self.min_match_count = (
            min_match_count
            if min_match_count is not None
            else self.config["min_match_count"]
        )

        # 各モジュールの初期化（パラメータを渡す）
        feature_matcher_config = self.config["feature_matcher"]
        clustering_config = self.config["clustering"]
        position_config = self.config["position"]
        text_detection_config = self.config["text_detection"]

        self.feature_matcher = FeatureMatcher(
            match_threshold=feature_matcher_config["match_threshold"],
            trees=feature_matcher_config["trees"],
            checks=feature_matcher_config["checks"],
        )
        self.clustering = LogoClustering(
            eps=clustering_config["eps"], min_samples=clustering_config["min_samples"]
        )
        self.position_calculator = PositionCalculator(
            position_threshold=position_config["position_threshold"]
        )
        self.percentage_calculator = PercentageCalculator()
        self.text_detector = TextDetector(
            min_confidence=text_detection_config["min_confidence"],
            min_size=text_detection_config["min_size"],
            debug_mode=debug_mode,
        )
        self.result_merger = ResultMerger()
        self.image_utils = ImageUtils(debug_mode=debug_mode)

    def detect_multiple_logos_in_frame(
        self, frame_path: str, logo_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """1つのフレーム画像内のロゴを検出する（同ロゴ複数可）"""
        try:
            # ロゴ情報を取得
            logo_id = logo_info.get("contracted_ad_item_id")

            # image_ad_itemsをリストとして取得
            logo_urls = logo_info.get("image_ad_items", [])

            all_results = []

            # 各ロゴ画像に対して検出処理を実行
            for logo_url in logo_urls:
                # デバッグモードではローカルパスを優先的に使用
                if self.debug_mode and "local_path" in logo_info:
                    logo_path = logo_info["local_path"]
                    logger.debug(f"Debug mode: Using local logo path {logo_path}")
                else:
                    # ロゴ画像をダウンロード
                    logo_path = self.image_utils.download_logo_image(logo_url, logo_id)

                # 画像の読み込み
                frame = cv2.imread(frame_path)
                logo = cv2.imread(logo_path)

                # 画像が読み込めない場合はスキップして次のロゴへ
                if frame is None or logo is None:
                    logger.error(
                        f"Failed to load images. Frame: {frame_path}, Logo: {logo_path}"
                    )
                    continue

                # グレースケールに変換
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)

                # ロゴのマスクを作成（画面占有率計算用）
                logo_mask = self.image_utils.create_logo_mask(logo_gray)

                # 特徴点と特徴量を抽出
                kp_logo, des_logo = self.feature_matcher.extract_features(logo_gray)
                kp_frame, des_frame = self.feature_matcher.extract_features(frame_gray)

                # 特徴量が不足している場合は次のロゴへ
                if (
                    des_logo is None
                    or des_frame is None
                    or len(des_logo) < 2
                    or len(des_frame) < 2
                ):
                    logger.debug(f"Not enough features for matching. Logo: {logo_id}")
                    continue

                # 特徴量のマッチング
                good_matches = self.feature_matcher.match_features(des_logo, des_frame)

                # マッチングの結果
                if len(good_matches) >= self.min_match_count:
                    # マッチしたフレーム上の特徴点座標を抽出
                    matched_points = np.array(
                        [kp_frame[m.trainIdx].pt for m in good_matches]
                    )

                    # DBSCANでクラスタリング
                    labels, unique_clusters = self.clustering.cluster_matched_points(
                        matched_points
                    )

                    # 各クラスタごとに処理
                    for cluster_id in unique_clusters:
                        # このクラスタに属するポイントのインデックス
                        cluster_indices = self.clustering.get_cluster_indices(
                            labels, cluster_id
                        )

                        # このクラスタに属するマッチを取得
                        cluster_matches = [good_matches[i] for i in cluster_indices]

                        # クラスタのサイズチェック
                        if (
                            len(cluster_matches) < self.min_match_count // 2
                        ):  # クラスタ用に閾値を下げる
                            continue

                        # クラスタに属するキーポイントを取得
                        cluster_keypoints = [
                            kp_frame[m.trainIdx] for m in cluster_matches
                        ]

                        # このクラスタ内のロゴ画像とフレーム画像の特徴点の対応を取得
                        src_pts = np.float32(
                            [kp_logo[m.queryIdx].pt for m in cluster_matches]
                        ).reshape(-1, 1, 2)
                        dst_pts = np.float32(
                            [kp_frame[m.trainIdx].pt for m in cluster_matches]
                        ).reshape(-1, 1, 2)

                        # ホモグラフィ行列を計算（傾きや歪みに対応）
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                        # 計算に失敗した場合はスキップ
                        if M is None:
                            continue

                        # ロゴの位置を判定
                        position = self.position_calculator.determine_logo_position(
                            cluster_keypoints, frame_gray.shape
                        )

                        # ロゴの画面占有率を計算
                        percentage = (
                            self.percentage_calculator.calculate_logo_percentage(
                                kp_logo,
                                kp_frame,
                                frame_gray.shape,
                                logo_mask,
                                cluster_matches,
                            )
                        )

                        # 特徴点（検出したロゴ）の中心座標を計算
                        keypoint_coords = np.array([kp.pt for kp in cluster_keypoints])
                        center_x = np.mean(keypoint_coords[:, 0])
                        center_y = np.mean(keypoint_coords[:, 1])

                        # 検出結果を追加
                        all_results.append(
                            {
                                "detected": True,
                                "contracted_ad_item_id": logo_id,
                                "position": position,
                                "screen_percentage": percentage,
                                "detection_method": "logo",
                                "center": (center_x, center_y),
                            }
                        )
                else:
                    logger.debug(
                        f"Logo not detected for image: {logo_url}, matches: {len(good_matches)}"
                    )

            return all_results

        except Exception as e:
            logger.error(f"Error detecting multiple logos in frame: {str(e)}")
            return []

    def detect_logos(
        self, frame_info: List[Dict[str, Any]], logos: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """複数のフレームから複数のロゴを検出する"""
        logger.info(
            f"Starting logo detection for {len(frame_info)} frames and {len(logos)} logos"
        )

        # フレームごとの検出結果を格納するリスト
        results = []

        # 各フレームに対して処理
        for frame in frame_info:
            frame_number = frame.get("frame_id")
            frame_path = frame.get("local_path")
            frame_s3_url = frame.get("s3_url")

            # S3のURLをHTTPSアクセス可能な形式に変換
            frame_https_url = frame_s3_url
            if frame_s3_url and frame_s3_url.startswith("s3://"):
                bucket_key = frame_s3_url.replace("s3://", "").split("/", 1)
                if len(bucket_key) == 2:
                    bucket, key = bucket_key
                    frame_https_url = f"https://{bucket}.s3.amazonaws.com/{key}"

            frame_result = {
                "frame_number": frame_number,
                "frame_image_url": frame_https_url,
                "detected_logos": [],
            }

            all_detections = []

            # 画像ベースのロゴ検出とテキストベースのロゴ検出を並列実行
            with ThreadPoolExecutor(
                max_workers=min(len(logos), self.config["max_workers"])
            ) as executor:
                # 画像ベースの検出タスク
                futures = []
                for logo in logos:
                    # 画像ベースの検出
                    futures.append(
                        (
                            executor.submit(
                                self.detect_multiple_logos_in_frame, frame_path, logo
                            ),
                            "image",
                        )
                    )

                # テキストベースの検出タスク
                futures.append(
                    (
                        executor.submit(
                            self.text_detector.detect_text_logos, frame_path, logos
                        ),
                        "text",
                    )
                )

                # 完了したタスクから順次結果を取得
                for future, task_type in futures:
                    try:
                        detection_results = future.result()
                        if detection_results:
                            all_detections.extend(detection_results)
                    except Exception as e:
                        logger.error(f"Error in {task_type} detection thread: {str(e)}")

            # 重複した検出の統合
            if all_detections:
                # フレーム画像を読み込んでサイズを取得
                frame_img = cv2.imread(frame_path)
                if frame_img is not None:
                    frame_shape = frame_img.shape[:2]
                    merged_detections = self.result_merger.merge_duplicate_detections(
                        all_detections, frame_shape
                    )

                    # 内部処理用のフィールドを削除し、必要なフィールドのみを含める
                    for detection in merged_detections:
                        # デバッグモードでない場合はdetection_methodフィールドを含めない
                        if not self.debug_mode:
                            if "detection_method" in detection:
                                del detection["detection_method"]

                        # 内部処理用フィールドを削除
                        if "center" in detection:
                            del detection["center"]
                        if "bbox" in detection:
                            del detection["bbox"]
                        if "matched_text" in detection:
                            del detection["matched_text"]

                    frame_result["detected_logos"] = merged_detections
                else:
                    # フレーム画像が読み込めない場合は統合処理を通さずにそのまま追加
                    for detection in all_detections:
                        if detection.get("detected", False):
                            # 結果オブジェクトを作成
                            result_obj = {
                                "contracted_ad_item_id": detection[
                                    "contracted_ad_item_id"
                                ],
                                "position": detection["position"],
                                "screen_percentage": detection["screen_percentage"],
                            }

                            # デバッグモードの場合のみdetection_methodを含める
                            if self.debug_mode:
                                result_obj["detection_method"] = detection.get(
                                    "detection_method", "logo"
                                )

                            # 内部処理用のフィールドを削除
                            if "center" in detection:
                                del detection["center"]
                            if "bbox" in detection:
                                del detection["bbox"]
                            if "matched_text" in detection:
                                del detection["matched_text"]

                            frame_result["detected_logos"].append(result_obj)

            # 検出結果があるフレームのみ追加
            if frame_result["detected_logos"]:
                results.append(frame_result)

        logger.info(f"Logo detection completed. Found logos in {len(results)} frames")
        return results
