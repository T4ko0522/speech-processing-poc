"""SRT/VTT タイムスタンプ変換."""

from __future__ import annotations

import srt
from datetime import timedelta

from poc.src.pipeline.models import TranscriptSegment


def _seconds_to_timedelta(seconds: float) -> timedelta:
    """秒数を timedelta に変換."""
    return timedelta(seconds=seconds)


def _seconds_to_vtt_timestamp(seconds: float) -> str:
    """秒数を VTT タイムスタンプ形式に変換."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def segments_to_srt(segments: list[TranscriptSegment]) -> str:
    """TranscriptSegment リストを SRT 形式の文字列に変換."""
    srt_subs = []
    for seg in segments:
        srt_subs.append(
            srt.Subtitle(
                index=seg.id + 1,
                start=_seconds_to_timedelta(seg.start),
                end=_seconds_to_timedelta(seg.end),
                content=seg.text,
            )
        )
    return srt.compose(srt_subs)


def segments_to_vtt(segments: list[TranscriptSegment]) -> str:
    """TranscriptSegment リストを WebVTT 形式の文字列に変換."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start_ts = _seconds_to_vtt_timestamp(seg.start)
        end_ts = _seconds_to_vtt_timestamp(seg.end)
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)
