import logging
import os
import tempfile

import boto3
from botocore.exceptions import ClientError

# ロギング設定
logger = logging.getLogger(__name__)

# 環境変数から設定を取得
SECRET_NAME = os.environ.get("GOOGLE_CREDENTIALS_SECRET_NAME")
AWS_REGION = os.environ.get(
    "AWS_REGION", "ap-northeast-1"
)  # デフォルトは東京リージョン


def get_google_credentials():
    """
    AWS Secrets Manager から Google サービスアカウント認証情報を取得し、
    一時ファイルとして保存する

    Returns:
        str: 認証情報ファイルのパス
    """
    if not SECRET_NAME:
        raise ValueError(
            "GOOGLE_CREDENTIALS_SECRET_NAME environment variable is not set"
        )

    try:
        # Secrets Manager クライアントを作成
        session = boto3.session.Session()
        client = session.client(service_name="secretsmanager", region_name=AWS_REGION)

        # シークレットを取得
        logger.info(f"Retrieving secret {SECRET_NAME} from AWS Secrets Manager")
        response = client.get_secret_value(SecretId=SECRET_NAME)

        if "SecretString" not in response:
            raise ValueError("Secret does not contain a SecretString")

        # JSON形式の認証情報を取得
        secret_data = response["SecretString"]

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            credentials_path = tmp_file.name
            tmp_file.write(secret_data)

        # debug-start
        try:
            import json

            with open(credentials_path, "r") as f:
                file_content = f.read()
            # JSONとして解析できるか確認
            parsed_json = json.loads(file_content)
            logger.info(f"Parsed JSON: {parsed_json}")
            # 重要なキーが存在するか確認
            logger.info(
                f"Credentials file created. Keys present: {list(parsed_json.keys())}"
            )
            logger.info(f"client_email exists: {'client_email' in parsed_json}")
            logger.info(f"token_uri exists: {'token_uri' in parsed_json}")
        except Exception as e:
            logger.error(f"Debug check failed: {str(e)}")
        # debug-end

        logger.info(f"Google credentials saved to temporary file: {credentials_path}")

        # 環境変数を設定して他のコンポーネントが使用できるようにする
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        return credentials_path

    except ClientError as e:
        logger.error(f"Failed to retrieve secret from AWS Secrets Manager: {str(e)}")
        raise RuntimeError(f"Could not retrieve Google credentials: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing Google credentials: {str(e)}")
        raise
