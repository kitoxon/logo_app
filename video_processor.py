import concurrent.futures
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError

# ロギング設定
logger = logging.getLogger(__name__)

# S3クライアント
s3_client = boto3.client("s3")

# 環境変数から設定を取得
S3_BUCKET = os.environ.get("S3_BUCKET")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))


def split_video_to_frames(video_path: str, game_video_id: str) -> List[Dict[str, Any]]:
    """
    FFmpegを使用して動画を1秒ごとにフレーム分割し、S3にアップロードする

    Args:
        video_path (str): 動画ファイルのパス
        game_video_id (str): ゲーム動画ID（S3のプレフィックスとして使用）

    Returns:
        List[Dict[str, Any]]: 分割されたフレーム情報のリスト
    """
    logger.info(f"Splitting video {video_path} into frames")

    # 一時ディレクトリを作成
    temp_dir = tempfile.mkdtemp(prefix=f"frames_{game_video_id}_")
    frames_info = []
    failed_uploads = []

    try:
        # 動画の長さを取得（秒単位）
        duration_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        duration = float(subprocess.check_output(duration_cmd).decode("utf-8").strip())
        logger.info(f"Video duration: {duration} seconds")

        # 1秒ごとにフレームを抽出
        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            "fps=1",
            "-q:v",
            "1",  # 高品質（1-31で1が最高品質）
            f"{temp_dir}/%d.png",
        ]

        logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, check=True)

        # 生成されたフレームを確認
        frame_files = sorted(
            [f for f in os.listdir(temp_dir) if f.endswith(".png")],
            key=lambda x: int(os.path.splitext(x)[0]),
        )

        logger.info(f"Generated {len(frame_files)} frame images")

        # フレーム情報の作成とS3アップロードの準備
        upload_tasks = []
        for frame_file in frame_files:
            frame_id = int(os.path.splitext(frame_file)[0])
            timestamp = frame_id  # 1秒ごとなので、フレームIDが秒数に対応

            frame_path = os.path.join(temp_dir, frame_file)
            s3_key = f"{game_video_id}/{frame_id}.png"

            # アップロードタスクの準備
            upload_tasks.append(
                {
                    "frame_id": frame_id,
                    "timestamp": f"{timestamp // 60:02d}:{timestamp % 60:02d}",  # MM:SS形式
                    "local_path": frame_path,
                    "s3_key": s3_key,
                }
            )

        # バッチに分割してS3にアップロード
        batch_size = 5  # 一度に処理するバッチサイズ

        for i in range(0, len(upload_tasks), batch_size):
            batch = upload_tasks[i : i + batch_size]
            logger.info(
                f"Uploading batch {i // batch_size + 1} / {(len(upload_tasks) + batch_size - 1) // batch_size}"
            )

            # 並列アップロード（バッチ内で同時実行）
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=batch_size
            ) as executor:
                # 各フレームのアップロード処理を並列実行
                future_to_task = {
                    executor.submit(
                        upload_to_s3, task["local_path"], task["s3_key"]
                    ): task
                    for task in batch
                }

                # 結果の収集
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        s3_url = future.result()
                        # 成功したフレーム情報を記録
                        frames_info.append(
                            {
                                "frame_id": task["frame_id"],
                                "timestamp": task["timestamp"],
                                "local_path": task["local_path"],
                                "s3_url": s3_url,
                            }
                        )
                    except Exception as e:
                        logger.error(
                            f"Error uploading frame {task['frame_id']}: {str(e)}"
                        )
                        # 失敗したアップロードを記録
                        failed_uploads.append(task)

        # 失敗したアップロードを再試行
        if failed_uploads:
            logger.info(f"Retrying {len(failed_uploads)} failed uploads...")
            retry_count = 0
            max_retry_attempts = 3

            while failed_uploads and retry_count < max_retry_attempts:
                retry_count += 1
                logger.info(f"Retry attempt {retry_count}/{max_retry_attempts}")

                still_failed = []
                for task in failed_uploads:
                    try:
                        logger.info(f"Retrying upload for frame {task['frame_id']}")
                        s3_url = upload_to_s3(task["local_path"], task["s3_key"])
                        frames_info.append(
                            {
                                "frame_id": task["frame_id"],
                                "timestamp": task["timestamp"],
                                "local_path": task["local_path"],
                                "s3_url": s3_url,
                            }
                        )
                    except Exception as e:
                        logger.error(
                            f"Retry failed for frame {task['frame_id']}: {str(e)}"
                        )
                        still_failed.append(task)

                failed_uploads = still_failed

                if failed_uploads:
                    logger.info(
                        f"Still have {len(failed_uploads)} failed uploads. Waiting before next retry..."
                    )
                    import time

                    time.sleep(5)  # 次の再試行前に少し待機

            if failed_uploads:
                logger.error(
                    f"Could not upload {len(failed_uploads)} frames after all retries"
                )
                frame_ids = [task["frame_id"] for task in failed_uploads]
                raise RuntimeError(
                    f"Failed to upload all frames. Missing frames: {frame_ids}"
                )

        # フレームIDでソート
        frames_info.sort(key=lambda x: x["frame_id"])

        logger.info(
            f"Successfully split video into {len(frames_info)} frames and uploaded to S3"
        )
        return frames_info

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {str(e)}")
        raise RuntimeError(f"Failed to split video: {str(e)}")
    except Exception as e:
        logger.error(f"Error splitting video: {str(e)}")
        raise e
    finally:
        # メイン処理でクリーンアップするので一時ディレクトリのクリーンアップはここではしない
        pass


def upload_to_s3(local_path: str, s3_key: str) -> str:
    """
    ファイルをS3にアップロードする

    Args:
        local_path (str): ローカルファイルパス
        s3_key (str): S3のキー

    Returns:
        str: S3のURL
    """
    if not S3_BUCKET:
        raise ValueError("S3_BUCKET environment variable is not set")

    retries = 0
    while retries < MAX_RETRIES:
        try:
            s3_client.upload_file(
                local_path,
                S3_BUCKET,
                s3_key,
                ExtraArgs={"ContentType": "image/png"},
            )

            s3_url = f"s3://{S3_BUCKET}/{s3_key}"
            return s3_url

        except ClientError as e:
            logger.error(f"S3 upload error: {str(e)}")
            retries += 1
            if retries >= MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to upload to S3 after {MAX_RETRIES} attempts"
                )
        except Exception as e:
            logger.error(f"S3 upload error: {str(e)}")
            retries += 1
            if retries >= MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to upload to S3 after {MAX_RETRIES} attempts"
                )

    # 到達しない
    return None


# テスト用のメイン処理
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # テスト用のパラメータ
    test_video_path = "/tmp/test_video.mp4"
    test_game_id = "test_game_001"

    # 環境変数の設定を確認
    if not os.environ.get("S3_BUCKET"):
        logger.warning("S3_BUCKET environment variable not set. Using default.")
        os.environ["S3_BUCKET"] = "test-logo-detection-bucket"

    try:
        frames = split_video_to_frames(test_video_path, test_game_id)
        logger.info(f"Processed {len(frames)} frames")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
