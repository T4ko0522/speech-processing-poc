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


class DimensionalEmotion(BaseModel):
    """次元感情（arousal/valence/dominance）."""

    arousal: float
    valence: float
    dominance: float


class TextEmotion(BaseModel):
    """テキスト感情（WRIME 8感情）."""

    scores: dict[str, float] = Field(default_factory=dict)


class ProsodyFeatures(BaseModel):
    """prosody特徴量（発話の緩急・抑揚）."""

    speech_rate: float = 0.0  # 文字/秒
    f0_mean: float = 0.0  # 基本周波数の平均 (Hz)
    f0_std: float = 0.0  # 基本周波数の標準偏差
    f0_range: float = 0.0  # 基本周波数のレンジ (max - min)
    energy_mean: float = 0.0  # RMSエネルギーの平均
    energy_std: float = 0.0  # RMSエネルギーの標準偏差


class FusedEmotion(BaseModel):
    """融合済み感情."""

    start: float
    end: float
    dimensional: DimensionalEmotion | None = None
    text_emotions: TextEmotion | None = None
    prosody: ProsodyFeatures | None = None
    fused_label: str = ""
    fused_valence: float = 0.0
    fused_arousal: float = 0.0


class EmotionTimeline(BaseModel):
    """感情タイムライン."""

    entries: list[FusedEmotion] = Field(default_factory=list)


class StepTiming(BaseModel):
    """パイプラインステップの計測情報."""

    step_name: str
    duration_seconds: float
    status: str = "completed"  # "completed" / "skipped" / "failed"
    skip_reason: str | None = None
    retry_count: int = 0


class PipelineResult(BaseModel):
    """パイプライン全体の出力."""

    input_file: str
    raw_transcript: RawTranscript | None = None
    fixed_transcript: FixedTranscript | None = None
    scenes: ScenesResult | None = None
    emotions: EmotionTimeline | None = None
    step_timings: list[StepTiming] = Field(default_factory=list)
