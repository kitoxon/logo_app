import atexit
import json
import logging
import os
import shutil
import sys
from typing import Any, Dict, List

import requests

from google_drive import cleanup_credentials, fetch_video
from logo_detector import detect_logos
from video_processor import split_video_to_frames

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# 環境変数から設定を取得
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

# 終了時に認証情報をクリーンアップするために登録
atexit.register(cleanup_credentials)


def handle_error(e, is_fatal=False, callback_url=None, game_video_id=None):
    """共通エラーハンドリング"""
    prefix = "Fatal error" if is_fatal else "Error"
    logger.error(f"{prefix}: {str(e)}", exc_info=True)

    error_response = {
        "statusCode": 500,
        "body": {
            "success": False,
            "message": f"{prefix}: {str(e)}",
            "error_type": type(e).__name__,
        },
    }

    # コールバックURLがある場合、エラーを通知
    if callback_url and game_video_id:
        try:
            send_callback(
                callback_url, game_video_id, status="failed", error_message=str(e)
            )
        except Exception as callback_error:
            logger.error(f"Failed to send error callback: {str(callback_error)}")

    return error_response


def validate_input(event):
    """入力データの検証と変換を行う"""
    data = event.get("data", {})
    video_url = data.get("video_url")
    game_video_id = data.get("game_video_id")
    logos = data.get("logos", [])
    callback_url = event.get("callback_url")

    # 必須パラメータのバリデーション
    missing_params = []
    if not video_url:
        missing_params.append("video_url")
    if not game_video_id:
        missing_params.append("game_video_id")
    if not logos:
        missing_params.append("logos")
    if not callback_url:
        missing_params.append("callback_url")

    if missing_params:
        raise ValueError(
            f"Required parameters are missing: {', '.join(missing_params)}"
        )

    # ロゴ情報の検証
    for i, logo in enumerate(logos):
        if not logo.get("contracted_ad_item_id"):
            raise ValueError(f"Logo at index {i} is missing 'contracted_ad_item_id'")
        if not logo.get("image_ad_items") or not isinstance(
            logo.get("image_ad_items"), list
        ):
            raise ValueError(
                f"Logo at index {i} is missing 'image_ad_items' or it is not a list"
            )
        if len(logo.get("image_ad_items", [])) == 0:
            raise ValueError(f"Logo at index {i} has empty 'image_ad_items' list")

    # コールバックURLの形式検証を追加
    if callback_url and not callback_url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid callback URL format: {callback_url}")

    # 検証済みの入力データを返す
    return {
        "video_url": video_url,
        "game_video_id": game_video_id,
        "logos": logos,
        "callback_url": callback_url,
    }


def cleanup_temp_files(temp_paths: List[str]):
    """一時ファイルを削除する"""
    for path in temp_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                    logger.info(f"Removed temporary directory: {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove directory {path}: {str(e)}")
            else:
                try:
                    os.remove(path)
                    logger.info(f"Removed temporary file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove file {path}: {str(e)}")


def format_detection_results(
    frame_detections: List[Dict[str, Any]], game_video_id: str
) -> Dict[str, Any]:
    """
    検出結果を指定されたJSON形式にフォーマットする

    Args:
        frame_detections (List[Dict[str, Any]]): フレームごとの検出結果
        game_video_id (str): ゲーム動画ID

    Returns:
        Dict[str, Any]: フォーマットされた結果
    """
    return {
        "video_info": {"game_video_id": game_video_id},
        "frame_detections": frame_detections,
    }


def send_callback(
    callback_url: str,
    game_video_id: str,
    results=None,
    status="completed",
    error_message=None,
) -> bool:
    """
    コールバックURLに処理結果を送信する

    Args:
        callback_url (str): コールバックURL
        game_video_id (str): ゲーム動画ID
        results (Dict, optional): 検出結果
        status (str): 処理状態 ("completed" または "failed")
        error_message (str, optional): エラーメッセージ

    Returns:
        bool: 送信成功したかどうか
    """
    try:
        payload = {"status": status, "game_video_id": game_video_id}

        # 正常終了時は結果を含める
        if status == "completed" and results:
            payload["success"] = True
            payload["results"] = results

        # エラー時はエラーメッセージを含める
        if status == "failed" and error_message:
            payload["success"] = False
            payload["error"] = error_message

        # コールバックURLに結果を送信
        response = requests.post(
            callback_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        response.raise_for_status()
        logger.info(f"Callback sent successfully to: {callback_url}")
        return True

    except Exception as e:
        logger.error(f"Failed to send callback: {str(e)}")
        return False


def main(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    メイン処理を実行する関数

    Args:
        event (Dict[str, Any]): 入力JSONデータ
            - data.video_url: 動画URL
            - data.game_video_id: 試合動画ID
            - data.logos: ロゴ情報のリスト
            - callback_url: コールバックURL

    Returns:
        Dict[str, Any]: 処理結果
    """
    temp_paths = []  # クリーンアップ対象の一時ファイル
    input_data = None

    try:
        # 入力データの検証
        input_data = validate_input(event)
        game_video_id = input_data["game_video_id"]

        # 処理開始ログ
        logger.info(f"Starting logo detection for game video ID: {game_video_id}")
        logger.info(f"Processing {len(input_data['logos'])} logos")

        # Google Driveから動画を取得
        logger.info(f"Fetching video from Google Drive: {input_data['video_url']}")
        video_path = fetch_video(input_data["video_url"], game_video_id)
        temp_paths.append(video_path)  # クリーンアップリストに追加

        # 動画をフレームに分割
        logger.info("Splitting video into frames")
        frame_info = split_video_to_frames(video_path, game_video_id)

        # フレーム保存ディレクトリをクリーンアップリストに追加
        if frame_info and len(frame_info) > 0:
            frame_dir = os.path.dirname(frame_info[0]["local_path"])
            temp_paths.append(frame_dir)

        # ロゴを検出
        logger.info(f"Detecting logos in {len(frame_info)} frames")
        detection_results = detect_logos(frame_info, input_data["logos"])

        # 結果をフォーマット
        formatted_results = format_detection_results(detection_results, game_video_id)

        # コールバックで結果を通知
        logger.info(f"Sending results to callback URL: {input_data['callback_url']}")
        callback_success = send_callback(
            input_data["callback_url"], game_video_id, results=formatted_results
        )

        if not callback_success:
            logger.warning("Callback failed, but processing completed successfully")

        # 処理結果を返す
        return {
            "statusCode": 200,
            "body": {
                "message": "Processing completed successfully",
                "game_video_id": game_video_id,
                "frames_processed": len(frame_info),
                "frames_with_logos": len(detection_results),
                "callback_sent": callback_success,
            },
        }

    except Exception as e:
        callback_url = input_data.get("callback_url") if input_data else None
        game_video_id = input_data.get("game_video_id") if input_data else None
        return handle_error(
            e, is_fatal=True, callback_url=callback_url, game_video_id=game_video_id
        )

    finally:
        # 一時ファイルのクリーンアップ
        cleanup_temp_files(temp_paths)


if __name__ == "__main__":
    try:
        # コマンドライン引数からJSONファイルパスを取得（開発用）
        if len(sys.argv) > 1:
            # JSONファイルとして読み込む
            input_file = sys.argv[1]
            logger.info(f"Reading input from file: {input_file}")
            with open(input_file, "r") as f:
                event = json.load(f)
        # fargateの場合
        elif os.environ.get("INPUT_DATA"):
            # 環境変数からJSON文字列を取得
            logger.info("Reading input from environment variable")
            event = json.loads(os.environ.get("INPUT_DATA"))
        # 標準入力から読み込む場合（パイプラインなど）
        else:
            # 標準入力から読み込む
            logger.info("Reading input from stdin")
            event = json.load(sys.stdin)

        # メイン処理実行
        result = main(event)

        # 結果を標準出力に出力
        print(json.dumps(result))

        # 終了コード0でコンテナを停止（エラー時は1）
        sys.exit(0 if result.get("statusCode") < 400 else 1)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)

        callback_url = event.get("callback_url") if "event" in locals() else None
        game_video_id = (
            event.get("data", {}).get("game_video_id") if "event" in locals() else None
        )
        error_result = handle_error(
            e, is_fatal=True, callback_url=callback_url, game_video_id=game_video_id
        )
        # 終了コード1（エラー）でコンテナを停止
        sys.exit(1)
