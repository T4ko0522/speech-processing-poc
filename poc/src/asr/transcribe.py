"""WhisperX 字幕生成 + word-level alignment + 話者分離."""

from __future__ import annotations

from pathlib import Path

import structlog
import whisperx

from poc.src.pipeline.models import RawTranscript, TranscriptSegment, WordSegment

logger = structlog.get_logger(__name__)


def transcribe(
    audio_path: Path,
    *,
    model_name: str = "medium",
    language: str = "ja",
    batch_size: int = 16,
    compute_type: str = "int8",
    device: str = "cpu",
    hf_token: str | None = None,
    diarization: dict | None = None,
) -> RawTranscript:
    """WhisperX で音声ファイルを文字起こしし、word-level alignment と話者分離を行う.

    Args:
        audio_path: 入力WAVファイルパス
        model_name: WhisperXモデル名
        language: 言語コード
        batch_size: バッチサイズ
        compute_type: 計算精度
        device: デバイス (cpu/cuda)
        hf_token: HuggingFace トークン
        diarization: 話者分離設定 (enabled, min_speakers, max_speakers)

    Returns:
        RawTranscript: タイムスタンプ付き字幕データ
    """
    logger.info(
        "文字起こし開始",
        model=model_name,
        language=language,
        device=device,
    )

    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )

    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    logger.info("アライメント実行中")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device=device,
    )

    # 話者分離
    diar_cfg = diarization or {}
    if diar_cfg.get("enabled", False) and hf_token:
        result = _run_diarization(
            audio,
            result,
            hf_token=hf_token,
            device=device,
            min_speakers=diar_cfg.get("min_speakers"),
            max_speakers=diar_cfg.get("max_speakers"),
        )

    segments = []
    for i, seg in enumerate(result["segments"]):
        words = []
        for w in seg.get("words", []):
            words.append(
                WordSegment(
                    word=w.get("word", ""),
                    start=w.get("start", seg["start"]),
                    end=w.get("end", seg["end"]),
                    score=w.get("score", 0.0),
                )
            )
        segments.append(
            TranscriptSegment(
                id=i,
                start=seg["start"],
                end=seg["end"],
                text=seg.get("text", ""),
                words=words,
                speaker=seg.get("speaker"),
            )
        )

    logger.info("文字起こし完了", segments_count=len(segments))
    return RawTranscript(language=language, segments=segments)


def _run_diarization(
    audio,
    aligned_result: dict,
    *,
    hf_token: str,
    device: str = "cpu",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
    """WhisperX DiarizationPipeline で話者分離を実行し、セグメントに話者を割り当てる."""
    from whisperx.diarize import DiarizationPipeline

    logger.info(
        "話者分離開始",
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    diarize_model = DiarizationPipeline(
        use_auth_token=hf_token,
        device=device,
    )

    diarize_segments = diarize_model(
        audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    result = whisperx.assign_word_speakers(diarize_segments, aligned_result)

    speakers = {seg.get("speaker") for seg in result["segments"] if seg.get("speaker")}
    logger.info("話者分離完了", speakers_detected=len(speakers))

    return result
