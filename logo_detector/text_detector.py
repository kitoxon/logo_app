import logging
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)


class TextDetector:
    def __init__(
        self, min_confidence: float = 0.5, min_size: int = 320, debug_mode: bool = False
    ):
        """
        パラメータ:
            min_confidence (float): テキスト検出の最小信頼度
            min_size (int): テキスト検出のための最小画像サイズ
            debug_mode（bool）: デバッグモードの有効/無効
        """
        # EASTモデルの設定
        model_path = Path(__file__).parent / "models" / "frozen_east_text_detection.pb"
        if not model_path.exists():
            raise FileNotFoundError(
                "EASTモデルが見つかりません。以下のURLからダウンロードしてください：\n"
                "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
            )
        self.net = cv2.dnn.readNet(str(model_path))
        self.min_confidence = min_confidence
        self.min_size = min_size
        self.debug_mode = debug_mode

        # デバッグ画像の保存先
        self.debug_dir = Path(__file__).parent / "debug_images"
        self.debug_dir.mkdir(exist_ok=True)

    def detect_text_regions(self, image_path: str) -> List[Dict[str, Any]]:
        """EASTを使用してテキスト領域を検出"""
        try:
            # フレーム名を取得
            frame_name = Path(image_path).stem

            # 画像を読み込み
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return []

            orig = image.copy()
            (H, W) = image.shape[:2]

            # アスペクト比を保持しながら、適切なサイズに調整
            # 短辺が320px以上になるようにリサイズ
            scale = max(self.min_size / min(H, W), 1.0)
            newW = int(W * scale)
            newH = int(H * scale)

            # 32の倍数に調整（EASTモデルの入力サイズに合わせる）
            newW = (newW // 32) * 32
            newH = (newH // 32) * 32

            # リサイズ
            image = cv2.resize(image, (newW, newH))
            rW = W / float(newW)
            rH = H / float(newH)

            # 前処理を追加
            # コントラストを強調
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            # EASTモデルの入力データを準備
            blob = cv2.dnn.blobFromImage(
                image,
                1.0,
                (newW, newH),
                (123.68, 116.78, 103.94),
                swapRB=True,
                crop=False,
            )
            self.net.setInput(blob)

            # モデルの出力レイヤー
            layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
            (scores, geometry) = self.net.forward(layerNames)

            # テキスト領域を検出
            rectangles = []
            confidences = []

            for y in range(0, scores.shape[2]):
                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]

                for x in range(0, scores.shape[3]):
                    if scoresData[x] < self.min_confidence:
                        continue

                    (offsetX, offsetY) = (x * 4.0, y * 4.0)

                    angle = anglesData[x]
                    cos = np.cos(angle)
                    sin = np.sin(angle)

                    h = xData0[x] + xData2[x]
                    w = xData1[x] + xData3[x]

                    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                    startX = int(endX - w)
                    startY = int(endY - h)

                    rectangles.append((startX, startY, endX, endY))
                    confidences.append(scoresData[x])

            # 重複する検出を統合
            boxes = cv2.dnn.NMSBoxes(rectangles, confidences, self.min_confidence, 0.3)

            text_regions = []
            if len(boxes) > 0:
                # OpenCV 4.8.0では、boxesは直接numpy配列として返される
                for i in boxes.flatten():
                    (startX, startY, endX, endY) = rectangles[i]

                    # 元の画像サイズに座標を戻す
                    startX = int(startX * rW)
                    startY = int(startY * rH)
                    endX = int(endX * rW)
                    endY = int(endY * rH)

                    # 領域の中心座標を計算
                    center_x = (startX + endX) / 2
                    center_y = (startY + endY) / 2

                    # 画面占有率を計算
                    area = (endX - startX) * (endY - startY)
                    total_area = orig.shape[1] * orig.shape[0]
                    screen_percentage = (
                        area / total_area
                    ) * 100  # 修正: 小数点のままにする

                    # 画面占有率でフィルタリング
                    # 極端に小さいまたは大きい領域を除外
                    if screen_percentage < 0.5 or screen_percentage > 80.0:
                        logger.debug(
                            f"Filtered out region with screen_percentage: {screen_percentage:.2f}%"
                        )
                        continue

                    # 検出したテキスト領域を切り出してOCR処理
                    detected_text, confidence = self._perform_ocr(
                        orig, startX, startY, endX, endY, frame_name
                    )

                    # デバッグ情報の強化: 座標情報と併せてOCR結果を出力
                    logger.info(
                        f"Region at ({startX},{startY},{endX},{endY}) - OCR text: '{detected_text}',"
                        f"Confidence: {confidence:.2f}"
                    )
                    logger.info(
                        f"EAST Confidence: {confidences[i]:.3f}, Screen %: {screen_percentage:.2f}%"
                    )

                    text_regions.append(
                        {
                            "bbox": (startX, startY, endX, endY),
                            "center": (center_x, center_y),
                            "screen_percentage": screen_percentage,
                            "confidence": confidences[i],
                            "detected_text": detected_text,  # OCRで認識したテキスト
                            "ocr_confidence": confidence,  # OCRの信頼度
                        }
                    )

            # 全体の検出結果をログに出力
            logger.info(f"Total text regions detected: {len(text_regions)}")
            return text_regions

        except Exception as e:
            logger.error(f"Error in text detection: {str(e)}")

            logger.error(traceback.format_exc())
            return []

    def _perform_ocr(self, image, startX, startY, endX, endY, frame_name=None):
        """領域からテキストを抽出するOCR処理"""
        try:
            # フレーム名が指定されていない場合はデフォルト名を設定
            if frame_name is None:
                frame_name = f"frame_{self.frame_count}"

            # 領域番号を取得（同一フレーム内での領域を区別するため）
            region_id = self._get_next_region_id(frame_name)

            # 最終的なベース名を作成
            base_name = f"{frame_name}_region{region_id}"

            # バウンディングボックスの座標が画像の範囲内に収まるよう調整
            height, width = image.shape[:2]

            # 領域を少し拡大（テキストの全体を確実に含めるため）
            padding = 5
            startX = max(0, startX - padding)
            startY = max(0, startY - padding)
            endX = min(width, endX + padding)
            endY = min(height, endY + padding)

            # 領域を切り出し
            roi = image[startY:endY, startX:endX]
            if roi.size == 0:
                logger.warning("ROI size is zero, skipping OCR")
                return "", 0.0

            # 元の切り出し領域を保存
            roi_path = str(self.debug_dir / f"{base_name}_roi.jpg")
            cv2.imwrite(roi_path, roi)

            # OCR処理のための前処理
            # グレースケール変換
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # グレースケール画像を保存
            gray_path = str(self.debug_dir / f"{base_name}_gray.jpg")
            cv2.imwrite(gray_path, gray)

            # グレースケール画像を白黒の2値画像に変換しテキスト領域を背景から明確に区別
            _, binary_otsu = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # 画像の各領域ごとに最適な閾値を計算し、影やグラデーションがある場合でもテキストを検出
            binary_adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # エッジ検出
            edges = cv2.Canny(gray, 50, 150)
            # 膨張処理で輪郭を太くする
            dilated_edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

            # 輪郭を検出しテキスト以外の領域を除外
            contours, _ = cv2.findContours(
                dilated_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            mask = np.zeros_like(gray)
            for cnt in contours:
                if cv2.contourArea(cnt) > 30:  # 小さすぎる輪郭を除外
                    cv2.drawContours(mask, [cnt], -1, 255, -1)

            # マスクを使用して二値化画像を生成
            binary_contour = cv2.bitwise_and(binary_otsu, binary_otsu, mask=mask)

            # 白黒を反転し他の手法で検出が難しいテキスちおを検出する
            _, binary_inv = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

            # デバッグモードが有効な場合のみ画像を保存
            if self.debug_mode:
                cv2.imwrite(
                    str(self.debug_dir / f"{base_name}_binary_otsu.jpg"), binary_otsu
                )
                cv2.imwrite(
                    str(self.debug_dir / f"{base_name}_binary_adaptive.jpg"),
                    binary_adaptive,
                )
                cv2.imwrite(str(self.debug_dir / f"{base_name}_edges.jpg"), edges)
                cv2.imwrite(
                    str(self.debug_dir / f"{base_name}_dilated_edges.jpg"),
                    dilated_edges,
                )
                cv2.imwrite(str(self.debug_dir / f"{base_name}_mask.jpg"), mask)
                cv2.imwrite(
                    str(self.debug_dir / f"{base_name}_binary_contour.jpg"),
                    binary_contour,
                )
                cv2.imwrite(
                    str(self.debug_dir / f"{base_name}_binary_inv.jpg"), binary_inv
                )
                logger.info(f"Debug images saved with base name: {base_name}")

            # 各前処理方法に対してOCRを実行
            ocr_results = []

            # OCR設定のバリエーション
            # PSM 6: 単一テキストブロック
            # PSM 7: 単一テキスト行
            # PSM 8: 単一単語
            # PSM 10: 単一文字
            # PSM 11: 密なテキスト+バイアスなし
            # PSM 13: 生のライン
            configs = [
                # 英語・数字用の設定
                "--psm 7 --oem 1 -l eng",
                "--psm 8 --oem 1 -l eng",
                "--psm 6 --oem 1 -l eng",
                # 日本語用の設定
                "--psm 7 --oem 1 -l jpn",
                "--psm 8 --oem 1 -l jpn",
                "--psm 6 --oem 1 -l jpn",
                # 日本語+英語の混合設定
                "--psm 7 --oem 1 -l jpn+eng",
                "--psm 8 --oem 1 -l jpn+eng",
                "--psm 6 --oem 1 -l jpn+eng",
                # 特殊なケース用
                "--psm 10 --oem 1 -l jpn+eng",
                "--psm 13 --oem 1 -l jpn+eng",
                "--psm 11 --oem 1 -l jpn+eng",
            ]

            # 処理画像のリスト
            processed_images = [
                (binary_otsu, "binary_otsu"),
                (binary_adaptive, "binary_adaptive"),
                (binary_contour, "binary_contour"),
                (binary_inv, "binary_inv"),
            ]

            # 各画像と設定の組み合わせでOCRを実行
            best_confidence = 0.0
            best_text = ""
            best_config = ""
            best_image_type = ""

            for img, img_type in processed_images:
                # OCR用に画像をリサイズ（拡大）
                resized = cv2.resize(
                    img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
                )

                for config in configs:
                    try:
                        # OCR実行とデータ取得
                        ocr_data = pytesseract.image_to_data(
                            resized,
                            config=config,
                            output_type=pytesseract.Output.DICT,
                        )

                        # 信頼度が高いテキストを選択
                        confidences = [
                            float(conf) for conf in ocr_data["conf"] if conf != "-1"
                        ]
                        if confidences:
                            avg_conf = sum(confidences) / len(confidences)
                            texts = [
                                ocr_data["text"][i]
                                for i in range(len(ocr_data["text"]))
                                if ocr_data["conf"][i] != "-1"
                                and ocr_data["text"][i].strip()
                            ]

                            if texts:
                                text = " ".join(texts)
                                # 空白を除去
                                cleaned_text = "".join(text.split())

                                if cleaned_text:
                                    ocr_results.append(
                                        (cleaned_text, avg_conf, config, img_type)
                                    )
                                    logger.info(
                                        f"OCR [{img_type}] with {config}: '{cleaned_text}', conf: {avg_conf:.2f}"
                                    )

                                    # より高い信頼度の結果を記録
                                    if avg_conf > best_confidence:
                                        best_confidence = avg_conf
                                        best_text = cleaned_text
                                        best_config = config
                                        best_image_type = img_type

                    except Exception as e:
                        logger.error(f"OCR error with {img_type} and {config}: {e}")

            # 最終的なOCR結果を記録
            if best_text:
                logger.info(
                    f"Best OCR result: '{best_text}', conf: {best_confidence:.2f} [{best_image_type}] {best_config}"
                )
                return best_text, best_confidence
            else:
                logger.warning("No text detected with any OCR configuration")
                return "", 0.0

        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            logger.error(traceback.format_exc())
            return "", 0.0

    def detect_text_logos(
        self, frame_path: str, logos: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """テキストベースのロゴ検出を実行"""
        results = []

        # テキストラベルを持つロゴのみを処理（logo_textフィールドを使用）
        text_logos = [
            logo for logo in logos if "logo_text" in logo and logo["logo_text"]
        ]
        if not text_logos:
            logger.debug("No logos with logo_text field found")
            return results

        # テキスト領域を検出（OCR処理も含む）
        text_regions = self.detect_text_regions(frame_path)
        if not text_regions:
            logger.debug(f"No text regions detected in frame: {frame_path}")
            return results

        # 各テキスト領域に対して、ロゴのテキストとマッチングを行う
        for region in text_regions:
            detected_text = region.get("detected_text", "").strip()
            ocr_confidence = region.get("ocr_confidence", 0.0)

            if not detected_text:
                logger.debug("Empty text detected in region, skipping")
                continue

            logger.debug(f"Processing region with detected text: '{detected_text}'")

            matched_logos = []

            # 各ロゴとのテキスト一致度を計算
            for logo in text_logos:
                logo_id = logo.get("contracted_ad_item_id")
                logo_text = logo.get("logo_text", "").strip()

                # テキスト一致度を計算
                match_score, match_type = self._calculate_text_similarity(
                    detected_text, logo_text
                )

                # OCR信頼度を考慮した最終スコア
                final_score = (
                    match_score * (ocr_confidence / 100.0)
                    if ocr_confidence > 0
                    else match_score
                )

                if final_score > 0.1:  # 一致度の閾値を調整（OCR信頼度を考慮）
                    matched_logos.append(
                        {
                            "logo_id": logo_id,
                            "logo_text": logo_text,
                            "match_score": match_score,
                            "final_score": final_score,
                            "match_type": match_type,
                        }
                    )
                    logger.debug(
                        f"Matched '{detected_text}' with logo_text '{logo_text}', score: {match_score:.2f},",
                        f"final: {final_score:.2f}",
                    )

            # マッチしたロゴがない場合はスキップ
            if not matched_logos:
                logger.debug(f"No logo text matched with '{detected_text}'")
                continue

            # 最も一致度の高いロゴを選択
            best_match = max(matched_logos, key=lambda x: x["final_score"])

            # 画面上の位置を判定
            center_x, center_y = region["center"]
            position = self._determine_position(center_x, center_y, frame_path)

            # 画面占有率を小数点切り上げで整数に変換
            screen_percentage = math.ceil(region["screen_percentage"])

            # 検出結果を追加
            results.append(
                {
                    "detected": True,
                    "contracted_ad_item_id": best_match["logo_id"],
                    "position": position,
                    "screen_percentage": screen_percentage,
                    "detection_method": "text_ocr",
                    "center": region["center"],
                    "bbox": region["bbox"],
                    "confidence": region.get("confidence", 0.0)
                    * best_match["final_score"],
                    "detected_text": detected_text,
                    "logo_text": best_match["logo_text"],
                    "match_score": best_match["match_score"],
                    "ocr_confidence": ocr_confidence,
                    "final_score": best_match["final_score"],
                    "match_type": best_match["match_type"],
                }
            )

        return results

    def _calculate_text_similarity(self, detected_text, logo_text):
        """テキストの類似度を計算"""
        # 完全一致の場合
        if detected_text == logo_text:
            return 1.0, "exact"

        # 大文字小文字を区別しない比較（英語部分のみ）
        detected_upper = "".join(
            [c.upper() if c.isascii() else c for c in detected_text]
        )
        logo_upper = "".join([c.upper() if c.isascii() else c for c in logo_text])

        # 大文字小文字を無視した完全一致
        if detected_upper == logo_upper:
            return 0.98, "case_insensitive_exact"

        # ロゴテキストが検出テキストに含まれる場合
        if logo_upper in detected_upper:
            # ロゴテキストの長さが検出テキストの長さに対する割合を計算
            score = len(logo_upper) / len(detected_upper)
            return min(score + 0.3, 0.95), "contains"  # 0.95を上限とする

        # 検出テキストがロゴテキストに含まれる場合
        if detected_upper in logo_upper:
            # 検出テキストの長さがロゴテキストの長さに対する割合を計算
            score = len(detected_upper) / len(logo_upper)
            return min(score + 0.2, 0.9), "substring"  # 0.9を上限とする

        # レーベンシュタイン距離を使用した部分一致の計算
        distance = self._levenshtein_distance(detected_upper, logo_upper)
        max_len = max(len(detected_upper), len(logo_upper))
        if max_len > 0:
            similarity = 1 - (distance / max_len)

            # 類似度が高い場合のみ
            if similarity > 0.6:
                return similarity, "partial"

        # ほとんど一致しない場合
        return 0.0, "none"

    def _levenshtein_distance(self, s1, s2):
        """2つの文字列間のレーベンシュタイン距離を計算"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _determine_position(self, x: float, y: float, frame_path: str) -> str:
        """画面上の位置を判定"""
        try:
            if frame_path:
                image = cv2.imread(frame_path)
                height, width = image.shape[:2]
            else:
                # フレームパスが提供されていない場合、最後に読み込んだ画像のサイズを使用
                # または適当なデフォルト値
                height, width = 1080, 1920

            # 画面を3x3のグリッドに分割
            x_ratio = x / width
            y_ratio = y / height

            if y_ratio < 0.33:
                if x_ratio < 0.33:
                    return "top_left"
                elif x_ratio < 0.66:
                    return "top_center"
                else:
                    return "top_right"
            elif y_ratio < 0.66:
                if x_ratio < 0.33:
                    return "middle_left"
                elif x_ratio < 0.66:
                    return "center"
                else:
                    return "middle_right"
            else:
                if x_ratio < 0.33:
                    return "bottom_left"
                elif x_ratio < 0.66:
                    return "bottom_center"
                else:
                    return "bottom_right"
        except Exception as e:
            logger.error(f"Error determining position: {str(e)}")
            return "unknown"

    def _get_next_region_id(self, frame_name):
        """フレーム内で一意の領域IDを取得"""
        # この関数はインスタンス変数を使って、同じフレーム内での領域に連番を振る
        if not hasattr(self, "_region_counters"):
            self._region_counters = {}

        if frame_name not in self._region_counters:
            self._region_counters[frame_name] = 0

        self._region_counters[frame_name] += 1
        return self._region_counters[frame_name]
