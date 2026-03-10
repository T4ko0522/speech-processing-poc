"""WhisperX 字幕生成 + word-level alignment."""

from __future__ import annotations

import functools
from pathlib import Path

import structlog
import torch
import whisperx

from poc.src.pipeline.models import RawTranscript, TranscriptSegment, WordSegment

logger = structlog.get_logger(__name__)


_torch_load_patched = False


def _patch_torch_load() -> None:
    """PyTorch 2.6+ の weights_only=True デフォルトを回避する.

    pyannote.audio が omegaconf 型を含むチェックポイントを読み込むため、
    weights_only=False をデフォルトにする。
    """
    global _torch_load_patched
    if _torch_load_patched:
        return
    _original = torch.load

    @functools.wraps(_original)
    def _patched(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original(*args, **kwargs)

    torch.load = _patched
    _torch_load_patched = True


def transcribe(
    audio_path: Path,
    *,
    model_name: str = "large-v3",
    language: str = "ja",
    batch_size: int = 16,
    compute_type: str = "float32",
    device: str = "cpu",
    hf_token: str | None = None,
) -> RawTranscript:
    """WhisperX で音声ファイルを文字起こしし、word-level alignment を行う.

    Args:
        audio_path: 入力WAVファイルパス
        model_name: WhisperXモデル名
        language: 言語コード
        batch_size: バッチサイズ
        compute_type: 計算精度
        device: デバイス (cpu/cuda)
        hf_token: HuggingFace トークン

    Returns:
        RawTranscript: タイムスタンプ付き字幕データ
    """
    logger.info(
        "文字起こし開始",
        model=model_name,
        language=language,
        device=device,
    )

    _patch_torch_load()

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
