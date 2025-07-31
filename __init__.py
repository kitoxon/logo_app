from typing import Any, Dict, List

from .logo_detector import LogoDetector

__all__ = ["LogoDetector"]

# メインインスタンスを作成
detector = LogoDetector()


# main.pyなどから呼び出せるよう公開APIとして定義
def detect_logos(
    frame_info: List[Dict[str, Any]],
    logos: List[Dict[str, Any]],
    debug_mode: bool = False,
) -> List[Dict[str, Any]]:
    """ロゴ検出のAPI"""
    detector = LogoDetector(debug_mode=debug_mode)
    return detector.detect_logos(frame_info, logos)
