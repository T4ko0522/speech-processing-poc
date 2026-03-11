"""Prosody特徴抽出: 発話速度・F0・エネルギー."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import structlog

from poc.src.pipeline.models import ProsodyFeatures, TranscriptSegment

logger = structlog.get_logger(__name__)


def analyze_prosody(
    audio_path: Path | str,
    segments: list[TranscriptSegment],
    *,
    sr: int = 16000,
    preloaded_audio: tuple | None = None,
) -> dict[int, ProsodyFeatures]:
    """音声セグメントごとにprosody特徴を抽出する.

    Args:
        audio_path: 音声ファイルパス
        segments: 字幕セグメントリスト
        sr: サンプリングレート
        preloaded_audio: (audio_array, sample_rate) のタプル。指定時はファイル読み込みをスキップ

    Returns:
        segment_id → ProsodyFeatures のマッピング
    """
    logger.info("prosody特徴抽出開始", segments=len(segments))

    if preloaded_audio is not None:
        y, sr_actual = preloaded_audio
    else:
        y, sr_actual = librosa.load(str(audio_path), sr=sr)
    duration_total = len(y) / sr_actual

    results: dict[int, ProsodyFeatures] = {}

    for seg in segments:
        start_sample = int(seg.start * sr_actual)
        end_sample = int(min(seg.end, duration_total) * sr_actual)

        if end_sample <= start_sample:
            results[seg.id] = ProsodyFeatures()
            continue

        segment_audio = y[start_sample:end_sample]
        segment_duration = (end_sample - start_sample) / sr_actual

        # 発話速度 (文字/秒)
        text_len = len(seg.text.replace(" ", ""))
        speech_rate = text_len / segment_duration if segment_duration > 0 else 0.0

        # F0抽出 (pyin)
        f0, voiced_flag, _ = librosa.pyin(
            segment_audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr_actual,
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        voiced_f0 = (
            voiced_f0[~np.isnan(voiced_f0)] if len(voiced_f0) > 0 else np.array([])
        )

        if len(voiced_f0) > 0:
            f0_mean = float(np.mean(voiced_f0))
            f0_std = float(np.std(voiced_f0))
            f0_range = float(np.max(voiced_f0) - np.min(voiced_f0))
        else:
            f0_mean = 0.0
            f0_std = 0.0
            f0_range = 0.0

        # RMSエネルギー
        rms = librosa.feature.rms(y=segment_audio)[0]
        energy_mean = float(np.mean(rms)) if len(rms) > 0 else 0.0
        energy_std = float(np.std(rms)) if len(rms) > 0 else 0.0

        results[seg.id] = ProsodyFeatures(
            speech_rate=round(speech_rate, 2),
            f0_mean=round(f0_mean, 2),
            f0_std=round(f0_std, 2),
            f0_range=round(f0_range, 2),
            energy_mean=round(energy_mean, 6),
            energy_std=round(energy_std, 6),
        )

    logger.info("prosody特徴抽出完了", segments_processed=len(results))
    return results
