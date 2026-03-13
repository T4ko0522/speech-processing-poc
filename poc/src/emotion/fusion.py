"""Late Fusion: SER + 次元感情 + prosody → EmotionTimeline."""

from __future__ import annotations

import structlog

from poc.src.pipeline.models import (
    DimensionalEmotion,
    EmotionTimeline,
    FusedEmotion,
    ProsodyFeatures,
    SpeechEmotion,
    TranscriptSegment,
)

logger = structlog.get_logger(__name__)

# SER 8ラベル → valence マッピング
EMOTION_VALENCE_MAP = {
    "happy": 0.8,
    "calm": 0.65,
    "surprised": 0.55,
    "neutral": 0.5,
    "fearful": 0.3,
    "sad": 0.2,
    "angry": 0.2,
    "disgust": 0.15,
}

# SER 8ラベル → arousal マッピング
EMOTION_AROUSAL_MAP = {
    "angry": 0.85,
    "surprised": 0.8,
    "happy": 0.7,
    "fearful": 0.7,
    "disgust": 0.55,
    "neutral": 0.45,
    "calm": 0.25,
    "sad": 0.2,
}


def _ser_to_va(top_label: str) -> tuple[float, float]:
    """SER の top_label から valence/arousal を直接マッピングする."""
    valence = EMOTION_VALENCE_MAP.get(top_label, 0.5)
    arousal = EMOTION_AROUSAL_MAP.get(top_label, 0.5)
    return valence, arousal


def _compute_prosody_modifier(prosody: ProsodyFeatures) -> float:
    """F0・エネルギーから arousal 調整値を算出する.

    F0 が高い／エネルギーが大きい → arousal を上げる方向 (+)
    F0 が低い／エネルギーが小さい → arousal を下げる方向 (-)

    Returns:
        -1.0 〜 +1.0 の範囲の調整係数（±0.15 にスケールされる前の値）
    """
    modifier = 0.0

    # F0 による調整: 200Hz を基準、±100Hz で ±0.5
    if prosody.f0_mean > 0:
        f0_centered = (prosody.f0_mean - 200.0) / 100.0
        modifier += max(-0.5, min(0.5, f0_centered))

    # エネルギーによる調整: 0.02 を基準、±0.02 で ±0.5
    if prosody.energy_mean > 0:
        energy_centered = (prosody.energy_mean - 0.02) / 0.02
        modifier += max(-0.5, min(0.5, energy_centered))

    return max(-1.0, min(1.0, modifier))


def fuse_emotions(
    segments: list[TranscriptSegment],
    dimensional_emotions: dict[int, DimensionalEmotion] | None = None,
    speech_emotions: dict[int, SpeechEmotion] | None = None,
    prosody_results: dict[int, ProsodyFeatures] | None = None,
    *,
    speech_weight: float = 0.5,
    dimensional_weight: float = 0.3,
    prosody_boost: float = 0.2,
    neutral_zone: list[float],
) -> EmotionTimeline:
    """SER と次元感情を Late Fusion で統合する.

    Args:
        segments: 字幕セグメントリスト
        dimensional_emotions: 次元感情
        speech_emotions: 音声カテゴリカル感情 (SER)
        prosody_results: prosody特徴量
        speech_weight: SER の重み (0-1)
        dimensional_weight: 次元感情の重み (0-1)
        prosody_boost: prosody 調整の最大幅
        neutral_zone: neutralゾーンの範囲 [lower, upper]

    Returns:
        EmotionTimeline: 融合済み感情タイムライン
    """
    neutral_lower, neutral_upper = neutral_zone

    logger.info(
        "感情融合開始",
        speech_weight=speech_weight,
        dimensional_weight=dimensional_weight,
        prosody_boost=prosody_boost,
        neutral_zone=neutral_zone,
    )

    entries: list[FusedEmotion] = []

    for seg in segments:
        dim = dimensional_emotions.get(seg.id) if dimensional_emotions else None
        speech_emo = speech_emotions.get(seg.id) if speech_emotions else None
        prosody = prosody_results.get(seg.id) if prosody_results else None

        # Valence/Arousal 融合
        has_speech = speech_emo is not None and bool(speech_emo.top_label)
        has_dim = dim is not None

        if has_speech:
            ser_v, ser_a = _ser_to_va(speech_emo.top_label)
        else:
            ser_v, ser_a = 0.5, 0.5

        if has_speech and has_dim:
            w_sum = speech_weight + dimensional_weight
            sw = speech_weight / w_sum
            dw = dimensional_weight / w_sum
            fused_valence = ser_v * sw + dim.valence * dw
            fused_arousal = ser_a * sw + dim.arousal * dw
        elif has_speech:
            fused_valence, fused_arousal = ser_v, ser_a
        elif has_dim:
            fused_valence, fused_arousal = dim.valence, dim.arousal
        else:
            fused_valence, fused_arousal = 0.5, 0.5

        # prosody による arousal 調整
        if prosody is not None:
            prosody_modifier = _compute_prosody_modifier(prosody)
            fused_arousal += prosody_modifier * prosody_boost

        # 0〜1 にクリップ
        fused_valence = max(0.0, min(1.0, fused_valence))
        fused_arousal = max(0.0, min(1.0, fused_arousal))

        # 融合ラベル決定（拡大neutralゾーン）
        fused_label = _determine_fused_label(
            fused_valence, fused_arousal, neutral_lower, neutral_upper
        )

        entries.append(
            FusedEmotion(
                start=seg.start,
                end=seg.end,
                dimensional=dim,
                speech_emotions=speech_emo,
                prosody=prosody,
                fused_label=fused_label,
                fused_valence=round(fused_valence, 4),
                fused_arousal=round(fused_arousal, 4),
            )
        )

    logger.info("感情融合完了", entries_count=len(entries))
    return EmotionTimeline(entries=entries)


def _determine_fused_label(
    valence: float,
    arousal: float,
    neutral_lower: float,
    neutral_upper: float,
) -> str:
    """valence/arousal の値から感情ラベルを決定（Russell's circumplex model）."""
    if valence >= neutral_upper and arousal >= neutral_upper:
        return "happy"
    elif valence >= neutral_upper and arousal < neutral_lower:
        return "calm"
    elif valence < neutral_lower and arousal >= neutral_upper:
        return "angry"
    elif valence < neutral_lower and arousal < neutral_lower:
        return "sad"
    else:
        return "neutral"
