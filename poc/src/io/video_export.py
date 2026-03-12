"""感情ラベル付き動画エクスポート."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import structlog

from poc.src.pipeline.models import EmotionTimeline, TranscriptSegment

logger = structlog.get_logger(__name__)

# 感情ラベルに対応する色 (ffmpeg color format)
EMOTION_COLORS = {
    "happy": "yellow",
    "calm": "cyan",
    "angry": "red",
    "sad": "blue",
    "neutral": "white",
}

# fontconfig 用フォント名（OS 別）
_FONT_NAMES = {
    "Windows": "Yu Gothic",
    "Darwin": "Hiragino Sans",
    "Linux": "DejaVu Sans",
}


def _default_font_name() -> str:
    """OS に応じたデフォルト fontconfig フォント名を返す."""
    return _FONT_NAMES.get(platform.system(), "sans-serif")


def _escape_drawtext_value(value: str) -> str:
    r"""drawtext フィルタのオプション値をエスケープする.

    filter_script 用: \, \:, \\ をエスケープ。
    シングルクォートは使わず、特殊文字を直接エスケープする。
    """
    value = value.replace("\\", "\\\\")
    value = value.replace(":", "\\:")
    value = value.replace("'", "\\'")
    return value


def export_video_with_emotions(
    video_path: Path,
    output_path: Path,
    emotions: EmotionTimeline,
    *,
    transcript_segments: list[TranscriptSegment] | None = None,
    font_name: str | None = None,
    font_size: int = 48,
    transcript_font_size: int = 32,
) -> Path:
    """感情ラベルとトランスクリプト字幕をオーバーレイした動画をエクスポートする.

    各感情区間中、画面中央に fused_label テキストを表示する。
    トランスクリプトがある場合、画面下部に字幕テキストを表示する。

    Args:
        video_path: 入力動画ファイルパス
        output_path: 出力動画ファイルパス
        emotions: 感情タイムライン
        transcript_segments: トランスクリプトのセグメントリスト
        font_name: fontconfig フォント名 (None で OS デフォルト)
        font_size: 感情ラベルのフォントサイズ
        transcript_font_size: トランスクリプト字幕のフォントサイズ

    Returns:
        生成された動画ファイルのパス

    Raises:
        RuntimeError: ffmpeg 処理に失敗した場合
    """
    if not emotions.entries:
        logger.warning("感情エントリが空のため、動画エクスポートをスキップ")
        return output_path

    resolved_font = font_name or _default_font_name()

    logger.info(
        "感情ラベル付き動画エクスポート開始",
        input=str(video_path),
        output=str(output_path),
        entries=len(emotions.entries),
        transcript_segments=len(transcript_segments) if transcript_segments else 0,
        font=resolved_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # drawtext フィルタチェーンを構築
    # filter_script 用: シングルクォートではなく \エスケープ を使用
    escaped_font = _escape_drawtext_value(resolved_font)

    drawtext_parts = []

    # --- 感情ラベル（画面中央）---
    for entry in emotions.entries:
        label = entry.fused_label
        if not label:
            continue
        color = EMOTION_COLORS.get(label, "white")
        escaped_label = _escape_drawtext_value(label)

        dt = (
            f"drawtext="
            f"font={escaped_font}"
            f":text={escaped_label}"
            f":fontsize={font_size}"
            f":fontcolor={color}"
            f":borderw=2:bordercolor=black"
            f":x=(w-text_w)/2:y=(h-text_h)/2"
            f":box=1:boxcolor=black@0.5:boxborderw=10"
            f":enable=between(t\\,{entry.start}\\,{entry.end})"
        )
        drawtext_parts.append(dt)

    # --- トランスクリプト字幕（画面下部）---
    if transcript_segments:
        for seg in transcript_segments:
            if not seg.text.strip():
                continue
            escaped_text = _escape_drawtext_value(seg.text.strip())

            dt = (
                f"drawtext="
                f"font={escaped_font}"
                f":text={escaped_text}"
                f":fontsize={transcript_font_size}"
                f":fontcolor=white"
                f":borderw=2:bordercolor=black"
                f":x=(w-text_w)/2:y=h-text_h-40"
                f":box=1:boxcolor=black@0.6:boxborderw=8"
                f":enable=between(t\\,{seg.start}\\,{seg.end})"
            )
            drawtext_parts.append(dt)

    if not drawtext_parts:
        logger.warning("有効な感情ラベルがないため、動画エクスポートをスキップ")
        return output_path

    filter_str = ",".join(drawtext_parts)

    # フィルタ文字列をファイルに書き出してコマンドライン長制限を回避
    filter_script = output_path.parent / "_emotion_filter.txt"
    filter_script.write_text(filter_str, encoding="utf-8")

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-filter_script:v",
        str(filter_script),
        "-c:a",
        "copy",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(output_path),
    ]

    logger.debug("ffmpeg コマンド実行", filter_count=len(drawtext_parts))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr_tail = proc.stderr[-1000:] if proc.stderr else "no stderr"
            raise RuntimeError(f"動画エクスポートに失敗しました: {stderr_tail}")
    finally:
        filter_script.unlink(missing_ok=True)

    logger.info("感情ラベル付き動画エクスポート完了", output=str(output_path))
    return output_path
