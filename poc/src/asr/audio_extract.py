"""ffmpeg 音声抽出・正規化（16kHz mono WAV）."""

from __future__ import annotations

from pathlib import Path

import ffmpeg
import structlog

logger = structlog.get_logger(__name__)


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """動画ファイルから音声を抽出し、正規化されたWAVファイルを生成する.

    Args:
        video_path: 入力動画ファイルパス
        output_path: 出力WAVファイルパス
        sample_rate: サンプリングレート（デフォルト: 16kHz）
        channels: チャンネル数（デフォルト: 1=モノラル）

    Returns:
        生成されたWAVファイルのパス

    Raises:
        FileNotFoundError: 入力ファイルが存在しない場合
        RuntimeError: ffmpeg 処理に失敗した場合
    """
    if not video_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("音声抽出開始", input=str(video_path), output=str(output_path))

    try:
        (
            ffmpeg.input(str(video_path))
            .output(
                str(output_path),
                acodec="pcm_s16le",
                ar=sample_rate,
                ac=channels,
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"音声抽出に失敗しました: {e.stderr}") from e

    logger.info("音声抽出完了", output=str(output_path))
    return output_path
