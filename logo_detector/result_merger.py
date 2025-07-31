from typing import Any, Dict, List, Tuple

import numpy as np


class ResultMerger:
    """検出結果を統合するクラス"""

    def __init__(self, duplicate_threshold: float = 0.02):
        self.duplicate_threshold = duplicate_threshold
        # 同じロゴイメージを使用した場合のより厳密な閾値（対角線の0.5%）
        self.same_method_threshold = 0.005

    def merge_duplicate_detections(
        self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """重複して検出された同一のロゴを統合する"""
        # 空の場合はそのまま返す
        if not detections:
            return []

        merged_results = []
        processed_indices = set()

        h, w = frame_shape

        # 対角線の長さを計算
        diagonal = np.sqrt(w**2 + h**2)

        # 異なる検出方法間の統合用の距離閾値
        cross_method_threshold = self.duplicate_threshold * diagonal

        # 同じ検出方法内での統合用の距離閾値（より厳密：フレーム対角線の0.5%）
        same_method_threshold = self.same_method_threshold * diagonal

        # 検出結果をcontracted_ad_item_idでグループ化
        id_groups = {}
        for i, detection in enumerate(detections):
            ad_item_id = detection["contracted_ad_item_id"]
            if ad_item_id not in id_groups:
                id_groups[ad_item_id] = []
            id_groups[ad_item_id].append((i, detection))

        # 各広告アイテムIDのグループ内で処理
        for ad_item_id, group in id_groups.items():
            # ロゴ検出とテキスト検出でさらにグループ化
            logo_group = [
                (idx, det) for idx, det in group if det["detection_method"] == "logo"
            ]
            text_group = [
                (idx, det) for idx, det in group if det["detection_method"] == "text"
            ]

            # ロゴ検出の結果をまず処理（近接したロゴ検出を統合）- より厳密な閾値を使用
            logo_merged = self.merge_same_method_detections(
                logo_group, same_method_threshold
            )
            # テキスト検出の結果を処理 - より厳密な閾値を使用
            text_merged = self.merge_same_method_detections(
                text_group, same_method_threshold
            )

            # 処理済みのインデックスを更新
            for idx, _ in logo_group + text_group:
                processed_indices.add(idx)

            # ロゴ検出とテキスト検出の統合結果が近接している場合はさらに統合
            all_merged = logo_merged + text_merged
            if len(all_merged) > 1:
                final_merged = []
                remaining = list(range(len(all_merged)))

                while remaining:
                    current_idx = remaining.pop(0)
                    current = all_merged[current_idx]
                    current_center = current["center"]
                    current_method = current["detection_method"]
                    same_group = [current]

                    to_remove = []
                    for j in remaining:
                        candidate = all_merged[j]
                        candidate_center = candidate["center"]
                        candidate_method = candidate["detection_method"]

                        distance = np.sqrt(
                            (current_center[0] - candidate_center[0]) ** 2
                            + (current_center[1] - candidate_center[1]) ** 2
                        )

                        # 異なる検出方法間ではより広い閾値、同じ検出方法ではより厳密な閾値を使用
                        threshold = (
                            cross_method_threshold
                            if current_method != candidate_method
                            else same_method_threshold
                        )

                        if distance <= threshold:
                            same_group.append(candidate)
                            to_remove.append(j)

                    # 削除対象のインデックスを残りのリストから除外
                    remaining = [j for j in remaining if j not in to_remove]

                    # グループを統合
                    if len(same_group) > 1:
                        merged = self.merge_group_detections(same_group)
                        final_merged.append(merged)
                    else:
                        final_merged.append(current)

                # 結果を追加
                merged_results.extend(final_merged)
            else:
                # 統合なしで結果を追加
                merged_results.extend(all_merged)

        # 未処理の検出結果を追加
        for i, detection in enumerate(detections):
            if i not in processed_indices:
                merged_results.append(
                    {
                        "contracted_ad_item_id": detection["contracted_ad_item_id"],
                        "position": detection["position"],
                        "screen_percentage": detection["screen_percentage"],
                        "detection_method": detection["detection_method"],
                        "center": detection["center"],  # 後で削除する
                    }
                )

        # 最終処理で不要なフィールドを削除
        for result in merged_results:
            if "center" in result:
                del result["center"]

        return merged_results

    def merge_same_method_detections(
        self, group: List[Tuple[int, Dict[str, Any]]], distance_threshold: float
    ) -> List[Dict[str, Any]]:
        """同じ検出方法のグループ内で近接した検出を統合"""
        if not group:
            return []

        merged = []
        remaining = list(range(len(group)))

        while remaining:
            current_idx = remaining.pop(0)
            _, current = group[current_idx]
            current_center = current["center"]
            same_position_group = [current]

            to_remove = []
            for j in remaining:
                _, candidate = group[j]
                candidate_center = candidate["center"]

                distance = np.sqrt(
                    (current_center[0] - candidate_center[0]) ** 2
                    + (current_center[1] - candidate_center[1]) ** 2
                )

                if distance <= distance_threshold:
                    same_position_group.append(candidate)
                    to_remove.append(j)

            # 削除対象のインデックスを残りのリストから除外
            remaining = [j for j in remaining if j not in to_remove]

            # 最も画面占有率の高い検出を選択
            best_detection = max(
                same_position_group, key=lambda d: d["screen_percentage"]
            )
            merged.append(
                {
                    "contracted_ad_item_id": best_detection["contracted_ad_item_id"],
                    "position": best_detection["position"],
                    "screen_percentage": best_detection["screen_percentage"],
                    "detection_method": best_detection["detection_method"],
                    "center": best_detection["center"],  # 後で参照するために保持
                }
            )

        return merged

    def merge_group_detections(
        self, group_detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """同じグループの検出結果を統合"""
        result = {"contracted_ad_item_id": group_detections[0]["contracted_ad_item_id"]}

        # 検出方法の優先順位: logo > text
        logo_detections = [
            d for d in group_detections if d["detection_method"] == "logo"
        ]

        if logo_detections:
            best_detection = max(logo_detections, key=lambda d: d["screen_percentage"])
            result.update(
                {
                    "position": best_detection["position"],
                    "screen_percentage": best_detection["screen_percentage"],
                    "detection_method": "logo",
                    "center": best_detection["center"],
                }
            )
        else:
            text_detection = max(group_detections, key=lambda d: d["screen_percentage"])
            result.update(
                {
                    "position": text_detection["position"],
                    "screen_percentage": text_detection["screen_percentage"],
                    "detection_method": "text",
                    "center": text_detection["center"],
                }
            )

        return result
