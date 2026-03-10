"""入力ファイル検証."""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def validate_input(file_path: str | Path) -> Path:
    """入力ファイルの存在と形式を検証する.

    Args:
        file_path: 入力ファイルパス

    Returns:
        検証済みの Path オブジェクト

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        ValueError: サポートされていない形式の場合
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {path}")

    if not path.is_file():
        raise ValueError(f"ディレクトリではなくファイルを指定してください: {path}")

    supported = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS
    if path.suffix.lower() not in supported:
        raise ValueError(
            f"サポートされていないファイル形式です: {path.suffix}\n"
            f"対応形式: {', '.join(sorted(supported))}"
        )

    logger.info("入力ファイル検証OK", path=str(path), size_mb=path.stat().st_size / 1024 / 1024)
    return path
