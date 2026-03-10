"""Late Fusion: テキスト感情 + 次元感情 → EmotionTimeline."""

from __future__ import annotations

import structlog

from poc.src.pipeline.models import (
    DimensionalEmotion,
    EmotionTimeline,
    FusedEmotion,
    ProsodyFeatures,
    TextEmotion,
    TranscriptSegment,
)

logger = structlog.get_logger(__name__)

# テキスト感情ラベル → valence マッピング（WRIME 8感情用）
EMOTION_VALENCE_MAP = {
    "joy": 0.8,
    "trust": 0.6,
    "anticipation": 0.6,
    "surprise": 0.5,
    "fear": 0.3,
    "sadness": 0.2,
    "anger": 0.2,
    "disgust": 0.15,
}

# テキスト感情ラベル → arousal マッピング（WRIME 8感情用）
EMOTION_AROUSAL_MAP = {
    "anger": 0.8,
    "surprise": 0.75,
    "joy": 0.7,
    "fear": 0.7,
    "anticipation": 0.6,
    "disgust": 0.5,
    "trust": 0.4,
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
    dimensional_emotions: dict[int, DimensionalEmotion] | None = None,
    text_emotions: dict[int, TextEmotion] | None = None,
    prosody_results: dict[int, ProsodyFeatures] | None = None,
    *,
    text_weight: float = 0.6,
    dimensional_weight: float = 0.2,
    neutral_zone: list[float] | None = None,
) -> EmotionTimeline:
    """テキスト感情と次元感情を Late Fusion で統合する.

    Args:
        segments: 字幕セグメントリスト
        dimensional_emotions: 次元感情
        text_emotions: テキスト感情
        prosody_results: prosody特徴量
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
        text_weight=text_weight,
        dimensional_weight=dimensional_weight,
        neutral_zone=neutral_zone,
    )

    entries: list[FusedEmotion] = []

    for seg in segments:
        dim = dimensional_emotions.get(seg.id) if dimensional_emotions else None
        text_emo = text_emotions.get(seg.id) if text_emotions else None
        prosody = prosody_results.get(seg.id) if prosody_results else None

        # Valence/Arousal 融合
        fused_valence = 0.5
        fused_arousal = 0.5

        if text_emo and text_emo.scores:
            # スコア加重方式: 全8感情の分布情報を活用
            fused_valence, fused_arousal = _text_scores_to_va(text_emo.scores)

        # 次元感情がある場合はブレンド
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
