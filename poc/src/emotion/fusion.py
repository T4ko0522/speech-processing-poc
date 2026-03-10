"""Late Fusion: 3モデル統合 → EmotionTimeline."""

from __future__ import annotations

import structlog

from poc.src.pipeline.models import (
    DimensionalEmotion,
    EmotionCategory,
    EmotionTimeline,
    FusedEmotion,
    ProsodyFeatures,
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


def _text_scores_to_va(scores: dict[str, float]) -> tuple[float, float]:
    """全8感情スコアの加重和で valence/arousal を算出する."""
    total = sum(scores.values())
    if total == 0:
        return 0.5, 0.5
    valence = sum(
        EMOTION_VALENCE_MAP.get(label, 0.5) * (score / total)
        for label, score in scores.items()
    )
    arousal = sum(
        EMOTION_AROUSAL_MAP.get(label, 0.5) * (score / total)
        for label, score in scores.items()
    )
    return valence, arousal


def fuse_emotions(
    segments: list[TranscriptSegment],
    speech_emotions: dict[int, EmotionCategory] | None = None,
    dimensional_emotions: dict[int, DimensionalEmotion] | None = None,
    text_emotions: dict[int, TextEmotion] | None = None,
    prosody_results: dict[int, ProsodyFeatures] | None = None,
    *,
    speech_weight: float = 0.4,
    text_weight: float = 0.6,
    dimensional_weight: float = 0.2,
    neutral_zone: list[float] | None = None,
) -> EmotionTimeline:
    """3モデルの感情推定結果を Late Fusion で統合する.

    Args:
        segments: 字幕セグメントリスト
        speech_emotions: 音声カテゴリカル感情
        dimensional_emotions: 次元感情
        text_emotions: テキスト感情
        prosody_results: prosody特徴量
        speech_weight: 音声感情の重み
        text_weight: テキスト感情の重み
        dimensional_weight: 次元感情のブレンド比率 (0-1)
        neutral_zone: neutralゾーンの範囲 [lower, upper]

    Returns:
        EmotionTimeline: 融合済み感情タイムライン
    """
    if neutral_zone is None:
        neutral_zone = [0.35, 0.65]
    neutral_lower, neutral_upper = neutral_zone

    logger.info(
        "感情融合開始",
        speech_weight=speech_weight,
        text_weight=text_weight,
        dimensional_weight=dimensional_weight,
        neutral_zone=neutral_zone,
    )

    entries: list[FusedEmotion] = []

    for seg in segments:
        speech_cat = speech_emotions.get(seg.id) if speech_emotions else None
        dim = dimensional_emotions.get(seg.id) if dimensional_emotions else None
        text_emo = text_emotions.get(seg.id) if text_emotions else None
        prosody = prosody_results.get(seg.id) if prosody_results else None

        # Valence/Arousal 融合
        fused_valence = 0.5
        fused_arousal = 0.5
        sources = 0

        if speech_cat:
            sv = EMOTION_VALENCE_MAP.get(speech_cat.label, 0.5)
            sa = EMOTION_AROUSAL_MAP.get(speech_cat.label, 0.5)
            fused_valence = sv * speech_weight
            fused_arousal = sa * speech_weight
            sources += speech_weight

        if text_emo and text_emo.scores:
            # スコア加重方式: 全8感情の分布情報を活用
            tv, ta = _text_scores_to_va(text_emo.scores)
            fused_valence += tv * text_weight
            fused_arousal += ta * text_weight
            sources += text_weight

        if sources > 0:
            fused_valence /= sources
            fused_arousal /= sources

        # 次元感情がある場合は低い比率でブレンド（信頼性が低いため）
        if dim:
            cat_weight = 1.0 - dimensional_weight
            fused_valence = fused_valence * cat_weight + dim.valence * dimensional_weight
            fused_arousal = fused_arousal * cat_weight + dim.arousal * dimensional_weight

        # 融合ラベル決定（拡大neutralゾーン）
        fused_label = _determine_fused_label(
            fused_valence, fused_arousal, neutral_lower, neutral_upper
        )

        entries.append(
            FusedEmotion(
                start=seg.start,
                end=seg.end,
                speech_category=speech_cat,
                dimensional=dim,
                text_emotions=text_emo,
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
    neutral_lower: float = 0.35,
    neutral_upper: float = 0.65,
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
