#!/usr/bin/env python3
import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List

import cv2

from logo_detector import detect_logos

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_test_data(json_path: str) -> Dict[str, Any]:
    """テスト用JSONデータを読み込む"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ロゴ定義ファイルが見つかりません: {json_path}")

    with open(json_path, "r") as f:
        return json.load(f)


def prepare_frame_info(frames_dir: str) -> List[Dict[str, Any]]:
    """フレーム情報を準備する"""
    if not os.path.exists(frames_dir) or not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"フレームディレクトリが見つかりません: {frames_dir}")

    frame_info = []

    # フレームディレクトリ内のPNG画像をリストアップ
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )

    for i, frame_file in enumerate(frame_files):
        frame_id = int(os.path.splitext(frame_file)[0])
        frame_path = os.path.join(frames_dir, frame_file)

        # 画像が読み込めるか確認
        img = cv2.imread(frame_path)
        if img is None:
            logger.warning(f"Failed to load image: {frame_path}")
            continue

        frame_info.append(
            {
                "frame_id": frame_id,
                "local_path": frame_path,
                "s3_url": f"test/dummy/path/{frame_id}.png",  # テスト用のダミーURL
                "timestamp": f"{frame_id // 60:02d}:{frame_id % 60:02d}",  # MM:SS形式
            }
        )

    return frame_info


def test_logo_detection(
    frames_dir: str = None,
    logos_json: str = None,
    output_json: str = None,
    debug_mode: bool = True,
):
    """ロゴ検出テストを実行する"""
    start_time = time.time()

    # テストデータの読み込み
    test_data = load_test_data(logos_json)
    logos = test_data.get("data", {}).get("logos", [])

    if not logos:
        logger.error("No logos found in test data")
        return

    # フレーム情報の準備
    frame_info = prepare_frame_info(frames_dir)
    logger.info(f"Found {len(frame_info)} frames in {frames_dir}")

    # ロゴ検出の実行
    logger.info(f"Starting logo detection with {len(logos)} logos")
    detection_results = detect_logos(frame_info, logos, debug_mode=debug_mode)

    # 結果の表示
    elapsed_time = time.time() - start_time
    logger.info(f"Detection completed in {elapsed_time:.2f} seconds")
    logger.info(f"Found logos in {len(detection_results)} frames")

    # 検出方法の統計を集計
    detection_methods_summary = {"logo": 0, "text": 0}
    for frame_result in detection_results:
        for logo in frame_result.get("detected_logos", []):
            method = logo.get("detection_method", "logo")  # デフォルトは"logo"
            detection_methods_summary[method] = (
                detection_methods_summary.get(method, 0) + 1
            )

    # 結果をJSONファイルに保存（指定がある場合）
    if output_json:
        results = {
            "detection_results": detection_results,
            "processing_time": elapsed_time,
            "frames_processed": len(frame_info),
            "frames_with_logos": len(detection_results),
            "detection_methods_summary": detection_methods_summary,  # 更新された統計情報
        }

        with open(output_json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_json}")
        logger.info(f"Detection methods summary: {detection_methods_summary}")

    # 簡易結果表示
    for frame_result in detection_results:
        frame_id = frame_result.get("frame_number")
        logos_count = len(frame_result.get("detected_logos", []))
        logger.info(f"Frame {frame_id}: {logos_count} logos detected")

        # 詳細表示
        for logo in frame_result.get("detected_logos", []):
            logo_id = logo.get("contracted_ad_item_id")
            percentage = logo.get("screen_percentage")
            method = logo.get("detection_method")
            logger.info(f"  - Logo: {logo_id}, Size: {percentage}%, Method: {method}")

    return detection_results


def visualize_results(
    frames_dir: str, detection_results: List[Dict[str, Any]], output_dir: str
):
    """検出結果を可視化する"""
    os.makedirs(output_dir, exist_ok=True)

    # フレームごとの検出結果を処理
    for frame_result in detection_results:
        frame_id = frame_result.get("frame_number")
        frame_path = os.path.join(frames_dir, f"{frame_id}.png")

        # フレーム画像を読み込む
        frame = cv2.imread(frame_path)
        if frame is None:
            logger.warning(f"Failed to load frame: {frame_path}")
            continue

        # 検出したロゴを描画
        for logo in frame_result.get("detected_logos", []):
            logo_id = logo.get("contracted_ad_item_id")
            percentage = logo.get("screen_percentage")
            method = logo.get("detection_method")

            # テキスト情報の準備
            text = f"{logo_id} ({percentage}%)"

            # 画像の中央に情報を表示（実際のロゴの位置情報はAPIからは取得できない）
            h, w = frame.shape[:2]
            cv2.putText(
                frame,
                text,
                (w // 2, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # 検出方法によって色を変える
            color = (0, 255, 0) if method == "logo" else (0, 0, 255)
            cv2.rectangle(frame, (10, 10), (w - 10, h - 10), color, 3)

        # 結果を保存
        output_path = os.path.join(output_dir, f"result_{frame_id}.png")
        cv2.imwrite(output_path, frame)
        logger.info(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    """
    ロゴ検出ローカルテスト用
    そのままpython test_logo_detection.pyで実行することでローカルのファイルのみでテスト可能
    """
    parser = argparse.ArgumentParser(description="Test logo detection logic")
    parser.add_argument(
        "--frames",
        help="Directory containing frame images (デフォルト: ./test_data/frames)",
        default="./test_data/frames",
    )
    parser.add_argument(
        "--logos",
        help="JSON file with logo definitions (デフォルト: ./test_data/logos.json)",
        default="./test_data/test_input.json",
    )
    parser.add_argument(
        "--output",
        help="Output JSON file for results",
        default="./test_data/result.json",
    )
    parser.add_argument("--visualize", help="Directory for visualization output")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for local testing",
        default=True,  # デフォルトでデバッグモードを有効に
    )

    args = parser.parse_args()

    # ロゴ検出テストを実行
    results = test_logo_detection(
        args.frames,
        args.logos,
        args.output,
        debug_mode=args.debug,
    )

    # 可視化（オプション）
    if args.visualize and results:
        visualize_results(
            args.frames
            or prepare_frame_info(args.frames)[0]["local_path"].rsplit("/", 1)[0],
            results,
            args.visualize,
        )
