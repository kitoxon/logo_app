import logging
import os
from typing import Optional

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


class ImageUtils:
    """画像処理に関するユーティリティクラス"""

    def __init__(
        self,
        cache_dir: str = "/tmp/logos",
        max_retries: int = 3,
        debug_mode: bool = False,
    ):
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.debug_mode = debug_mode

        # キャッシュディレクトリの作成
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_logo_image(self, logo_url: str, logo_id: str) -> Optional[str]:
        """ロゴ画像をダウンロードしキャッシュ"""
        # デバッグモードの場合、URLがローカルパスならそのまま使用
        if self.debug_mode and (os.path.exists(logo_url) or logo_url.startswith("./")):
            logger.debug(f"Debug mode: Using local logo path {logo_url}")
            return logo_url

        # キャッシュパスを設定
        extension = os.path.splitext(logo_url.split("/")[-1])[-1]
        if not extension:
            extension = ".png"  # デフォルト拡張子
        local_path = os.path.join(self.cache_dir, f"{logo_id}{extension}")

        # キャッシュにすでに存在する場合はそのまま返す
        if os.path.exists(local_path):
            logger.info(f"Using cached logo: {local_path}")
            return local_path

        # ロゴをダウンロード
        logger.info(f"Downloading logo from: {logo_url}")
        retries = 0
        while retries < self.max_retries:
            try:
                response = requests.get(logo_url, stream=True, timeout=30)
                response.raise_for_status()

                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.info(f"Logo downloaded to: {local_path}")
                return local_path

            except Exception as e:
                logger.error(
                    f"Error downloading logo (attempt {retries + 1}): {str(e)}"
                )
                retries += 1
                if retries >= self.max_retries:
                    raise RuntimeError(
                        f"Failed to download logo after {self.max_retries} attempts"
                    )

        return None

    def create_logo_mask(self, logo_gray: np.ndarray) -> np.ndarray:
        """ロゴのマスクを作成（画面占有率計算用）"""
        _, logo_mask = cv2.threshold(logo_gray, 1, 255, cv2.THRESH_BINARY)
        return logo_mask
