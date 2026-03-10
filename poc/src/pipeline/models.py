"""Pydantic データモデル — 全ステージ間の型定義."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WordSegment(BaseModel):
    """単語レベルのセグメント."""

    word: str
    start: float
    end: float
    score: float = 0.0


class TranscriptSegment(BaseModel):
    """字幕セグメント（1発話単位）."""

    id: int
    start: float
    end: float
    text: str
    words: list[WordSegment] = Field(default_factory=list)
    speaker: str | None = None


class RawTranscript(BaseModel):
    """WhisperX 出力の生字幕."""

    language: str
    segments: list[TranscriptSegment]


class CorrectionDiff(BaseModel):
    """誤字補正の差分情報."""

    segment_id: int
    original: str
    corrected: str
    confidence: float = 1.0


class FixedTranscript(BaseModel):
    """誤字補正済み字幕."""

    language: str
    segments: list[TranscriptSegment]
    diffs: list[CorrectionDiff] = Field(default_factory=list)


class SceneBoundary(BaseModel):
    """シーン境界."""

    scene_id: int
    start: float
    end: float
    start_timecode: str = ""
    end_timecode: str = ""
    frame_path: str | None = None


class SceneSummary(BaseModel):
    """シーン要約."""

    scene_id: int
    summary: str
    keywords: list[str] = Field(default_factory=list)


class ScenesResult(BaseModel):
    """シーン分析結果."""

    boundaries: list[SceneBoundary]
    summaries: list[SceneSummary] = Field(default_factory=list)


class EmotionCategory(BaseModel):
    """カテゴリカル感情."""

    label: str
    score: float


class DimensionalEmotion(BaseModel):
    """次元感情（arousal/valence/dominance）."""

    arousal: float
    valence: float
    dominance: float


class TextEmotion(BaseModel):
    """テキスト感情（WRIME 8感情）."""

    scores: dict[str, float] = Field(default_factory=dict)


class FusedEmotion(BaseModel):
    """融合済み感情."""

    start: float
    end: float
    speech_category: EmotionCategory | None = None
    dimensional: DimensionalEmotion | None = None
    text_emotions: TextEmotion | None = None
    fused_label: str = ""
    fused_valence: float = 0.0
    fused_arousal: float = 0.0


class EmotionTimeline(BaseModel):
    """感情タイムライン."""

    entries: list[FusedEmotion] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """パイプライン全体の出力."""

    input_file: str
    raw_transcript: RawTranscript | None = None
    fixed_transcript: FixedTranscript | None = None
    scenes: ScenesResult | None = None
    emotions: EmotionTimeline | None = None
