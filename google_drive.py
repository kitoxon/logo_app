import logging
import os
import urllib.parse

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from secret_manager import get_google_credentials

# ロギング
logger = logging.getLogger(__name__)

# 環境変数から設定を取得
TEMP_DIR = os.environ.get("TEMP_DIR", "/tmp")
USE_SECRETS_MANAGER = os.environ.get("USE_SECRETS_MANAGER", "true").lower() == "true"
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
_credentials_path = None  # キャッシュされた認証情報ファイルパス


def extract_file_id_from_url(url: str) -> str:
    """Google Drive URLからファイルIDを抽出する

    Args:
        url (str): Google Drive URL

    Returns:
        str: ファイルID

    Raises:
        ValueError: 有効なGoogle Drive URLではない場合

    """

    try:
        # ファイルIDを抽出
        if "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
        # idパラメータがある場合
        elif "id=" in url:
            parsed_url = urllib.parse.urlparse(url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            file_id = query_params.get("id", [""])[0]
        else:
            raise ValueError(f"Unsupported Google Drive URL format: {url}")

        # ファイルIDがない場合
        if not file_id:
            raise ValueError(f"Could not extract file ID from URL: {url}")

        return file_id

    except Exception as e:
        logger.error(f"Error extracting file ID from URL: {str(e)}")
        raise ValueError(f"Invalid Google Drive URL: {url}. Error: {str(e)}")


def get_drive_service():
    """Google Drive APIサービスを取得する"""
    global _credentials_path

    try:
        # 認証情報を設定
        if USE_SECRETS_MANAGER:
            # AWS Secrets Managerから認証情報を取得
            if not _credentials_path:
                _credentials_path = get_google_credentials()
            credentials_file = _credentials_path

            # debug-start
            try:
                import json

                with open(credentials_file, "r") as f:
                    file_content = f.read()
                logger.info(
                    f"Credentials file content first 100 chars: {file_content[:100]}..."
                )
                parsed_creds = json.loads(file_content)
                logger.info(f"Parsed credentials keys: {list(parsed_creds.keys())}")
                if (
                    "client_email" not in parsed_creds
                    or "token_uri" not in parsed_creds
                ):
                    logger.error("Missing required fields in credentials JSON")
            except Exception as e:
                logger.error(f"Error checking credentials file: {str(e)}")
            # debug-end

        else:
            # 認証情報を取得
            credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if not credentials_file or not os.path.exists(credentials_file):
                raise FileNotFoundError(
                    f"Credentials file not found: {credentials_file}"
                )

        # debug
        logger.info(f"Loading credentials from file: {credentials_file}")

        # サービスアカウント認証情報を読み込む
        credentials = service_account.Credentials.from_service_account_file(
            credentials_file,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )

        logger.info("Credentials loaded successfully")

        # APIサービスを取得
        service = build("drive", "v3", credentials=credentials)
        return service

    except Exception as e:
        logger.error(f"Error creating Google Drive service: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to initialize Google Drive API: {str(e)}")


def download_file_to_temp(file_id: str, game_video_id: str) -> str:
    """Google Driveからファイルをダウンロードして一時ディレクトリに保存する

    Args:
        file_id (str): Google DriveファイルID
        game_video_id (str): 試合動画ID（ファイル名に使用）

    Returns:
        str: ダウンロードしたファイルのパス
    """

    service = get_drive_service()

    try:
        # ファイル情報の取得
        file_metadata = (
            service.files()
            .get(fileId=file_id, fields="name,mimeType,size", supportsAllDrives=True)
            .execute()
        )
        logger.info(f"File metadata: {file_metadata}")

        # ファイル名を取得
        file_name = file_metadata.get("name", f"{game_video_id}.mp4")

        # 拡張子を取得
        file_extension = os.path.splitext(file_name)[1] if "." in file_name else ".mp4"

        # game_video_idに基づいて一時ファイルパスを作成
        # 元のファイル名は使用せず、game_video_idを使って保存する
        temp_file_path = os.path.join(TEMP_DIR, f"{game_video_id}{file_extension}")

        # ディレクトリがなければ作成
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        # 大きなファイルはストリーミングダウンロード
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)

        retries = 0

        while retries < MAX_RETRIES:
            try:
                # バイナリ書き込みモードでファイルを開く
                with open(temp_file_path, "wb") as f:
                    # ダウンロードを実行
                    downloader = MediaIoBaseDownload(
                        f, request, chunksize=1024 * 1024 * 5
                    )
                    done = False

                    # 完了するまでチャンクごとにダウンロード
                    while not done:
                        status, done = downloader.next_chunk()
                        if status:
                            # ダウンロードの進捗をログに記録
                            logger.info(
                                f"Download progress: {int(status.progress() * 100)}%"
                            )

                # ダウンロードファイルサイズ取得
                actual_size = os.path.getsize(temp_file_path)
                # metadataからファイルサイズ取得
                expected_size = int(file_metadata.get("size", 0))

                # APIのファイルサイズとダウンロードしたファイルのサイズが一致しない場合
                if expected_size > 0 and actual_size != expected_size:
                    logger.warning(
                        f"File size mismatch: expected {expected_size}, got {actual_size}"
                    )
                    retries += 1

                    # ダウンロードを再試行
                    continue

                logger.info(f"File downloaded successfully to {temp_file_path}")
                return temp_file_path

            except Exception as e:
                logger.error(f"Download attempt {retries+1} failed: {str(e)}")
                retries += 1
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

                if retries >= MAX_RETRIES:
                    logger.error(
                        f"Failed to download file after {MAX_RETRIES} attempts"
                    )
                    raise RuntimeError(
                        f"Failed to download file from Google Drive: {str(e)}"
                    )

                logger.info(f"Retrying download... ({retries}/{MAX_RETRIES})")

    except HttpError as e:
        if e.resp.status == 404:  # 動画ファイルがドライブに見つからない場合
            logger.error(f"File with ID {file_id} not found in Google Drive")
            raise FileNotFoundError(f"File with ID {file_id} not found in Google Drive")
        else:
            logger.error(f"HTTP error accessing Google Drive: {e.resp.status} {str(e)}")
            raise RuntimeError(
                f"HTTP error accessing Google Drive: {e.resp.status} {str(e)}"
            )

    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise RuntimeError(f"Failed to download file from Google Drive: {str(e)}")


def fetch_video(video_url: str, game_video_id: str) -> str:
    """Google Driveから動画を取得する関数。
    URLで指定されたファイルをダウンロードし、game_video_idを使って保存する。
    ファイル名とgame_video_idの一致は確認しない。

    Args:
        video_url (str): 動画のURL
        game_video_id (str): 試合動画ID (ローカルでの保存に使用)

    Returns:
        str: ダウンロードした動画のパス (ローカルパス)
    """
    logger.info(f"Fetching video from URL: {video_url}")

    try:
        # URLからファイルIDを抽出
        file_id = extract_file_id_from_url(video_url)
        logger.info(f"Extracted file ID: {file_id}")

        # ファイルをダウンロード (ファイル名の検証なし)
        local_path = download_file_to_temp(file_id, game_video_id)
        return local_path

    except HttpError as e:
        if e.resp.status == 404:
            logger.error(f"File with ID {file_id} not found in Google Drive")
            raise FileNotFoundError(f"File with ID {file_id} not found in Google Drive")
        else:
            logger.error(f"HTTP error accessing Google Drive: {e.resp.status} {str(e)}")
            raise RuntimeError(
                f"HTTP error accessing Google Drive: {e.resp.status} {str(e)}"
            )
    except Exception as e:
        logger.error(f"Failed to fetch video: {str(e)}")
        raise e


# 終了時の認証情報クリーンアップ処理
def cleanup_credentials():
    """一時的に保存した認証情報ファイルを削除する"""
    global _credentials_path

    if _credentials_path and os.path.exists(_credentials_path):
        try:
            os.remove(_credentials_path)
            logger.info(f"Removed temporary credentials file: {_credentials_path}")
            _credentials_path = None
        except Exception as e:
            logger.error(f"Failed to remove credentials file: {str(e)}")
            raise e


# テスト用
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # テスト用のURLとゲームID
    test_url = "https://drive.google.com/file/d/EXAMPLE_FILE_ID/view"
    test_game_id = "test_game_001"

    try:
        video_path = fetch_video(test_url, test_game_id)
        print(f"Video downloaded to: {video_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
