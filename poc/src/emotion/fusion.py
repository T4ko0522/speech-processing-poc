"""Late Fusion: 3モデル統合 → EmotionTimeline."""

from __future__ import annotations

import structlog

from poc.src.pipeline.models import (
    DimensionalEmotion,
    EmotionCategory,
    EmotionTimeline,
    FusedEmotion,
    TextEmotion,
    TranscriptSegment,
)

logger = structlog.get_logger(__name__)

# 感情ラベル → valence マッピング（簡易）
EMOTION_VALENCE_MAP = {
    "happy": 0.8,
    "joy": 0.8,
    "trust": 0.6,
    "anticipation": 0.6,
    "surprise": 0.5,
    "neutral": 0.5,
    "fear": 0.3,
    "fearful": 0.3,
    "sad": 0.2,
    "sadness": 0.2,
    "angry": 0.2,
    "anger": 0.2,
    "disgusted": 0.15,
    "disgust": 0.15,
}

# 感情ラベル → arousal マッピング（簡易）
EMOTION_AROUSAL_MAP = {
    "angry": 0.8,
    "anger": 0.8,
    "surprise": 0.75,
    "surprised": 0.75,
    "happy": 0.7,
    "joy": 0.7,
    "fear": 0.7,
    "fearful": 0.7,
    "anticipation": 0.6,
    "disgusted": 0.5,
    "disgust": 0.5,
    "trust": 0.4,
    "neutral": 0.3,
    "sad": 0.25,
    "sadness": 0.25,
}


def _get_dominant_text_emotion(text_emotion: TextEmotion) -> tuple[str, float]:
    """テキスト感情から支配的なラベルとスコアを取得."""
    if not text_emotion.scores:
        return "neutral", 0.0
    label = max(text_emotion.scores, key=text_emotion.scores.get)
    return label, text_emotion.scores[label]


def fuse_emotions(
    segments: list[TranscriptSegment],
    speech_emotions: dict[int, EmotionCategory] | None = None,
    dimensional_emotions: dict[int, DimensionalEmotion] | None = None,
    text_emotions: dict[int, TextEmotion] | None = None,
    *,
    speech_weight: float = 0.4,
    text_weight: float = 0.6,
) -> EmotionTimeline:
    """3モデルの感情推定結果を Late Fusion で統合する.

    重み付け: speech_weight (デフォルト 0.4) / text_weight (デフォルト 0.6)
    次元感情は arousal/valence として直接使用

    Args:
        segments: 字幕セグメントリスト
        speech_emotions: 音声カテゴリカル感情
        dimensional_emotions: 次元感情
        text_emotions: テキスト感情
        speech_weight: 音声感情の重み
        text_weight: テキスト感情の重み

    Returns:
        EmotionTimeline: 融合済み感情タイムライン
    """
    logger.info("感情融合開始", speech_weight=speech_weight, text_weight=text_weight)

    entries: list[FusedEmotion] = []

    for seg in segments:
        speech_cat = speech_emotions.get(seg.id) if speech_emotions else None
        dim = dimensional_emotions.get(seg.id) if dimensional_emotions else None
        text_emo = text_emotions.get(seg.id) if text_emotions else None

        # Valence 融合
        fused_valence = 0.5
        fused_arousal = 0.5
        sources = 0

        if speech_cat:
            sv = EMOTION_VALENCE_MAP.get(speech_cat.label, 0.5)
            sa = EMOTION_AROUSAL_MAP.get(speech_cat.label, 0.5)
            fused_valence = sv * speech_weight
            fused_arousal = sa * speech_weight
            sources += speech_weight

        if text_emo:
            text_label, _ = _get_dominant_text_emotion(text_emo)
            tv = EMOTION_VALENCE_MAP.get(text_label, 0.5)
            ta = EMOTION_AROUSAL_MAP.get(text_label, 0.5)
            fused_valence += tv * text_weight
            fused_arousal += ta * text_weight
            sources += text_weight

        if sources > 0:
            fused_valence /= sources
            fused_arousal /= sources

        # 次元感情がある場合は加重平均で統合
        if dim:
            fused_valence = fused_valence * 0.5 + dim.valence * 0.5
            fused_arousal = fused_arousal * 0.5 + dim.arousal * 0.5

        # 融合ラベル決定
        fused_label = _determine_fused_label(fused_valence, fused_arousal)

        entries.append(
            FusedEmotion(
                start=seg.start,
                end=seg.end,
                speech_category=speech_cat,
                dimensional=dim,
                text_emotions=text_emo,
                fused_label=fused_label,
                fused_valence=round(fused_valence, 4),
                fused_arousal=round(fused_arousal, 4),
            )
        )

    logger.info("感情融合完了", entries_count=len(entries))
    return EmotionTimeline(entries=entries)


def _determine_fused_label(valence: float, arousal: float) -> str:
    """valence/arousal の値から感情ラベルを決定（Russell's circumplex model）."""
    if valence >= 0.6 and arousal >= 0.6:
        return "happy"
    elif valence >= 0.6 and arousal < 0.4:
        return "calm"
    elif valence < 0.4 and arousal >= 0.6:
        return "angry"
    elif valence < 0.4 and arousal < 0.4:
        return "sad"
    else:
        return "neutral"
