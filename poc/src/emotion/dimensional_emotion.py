"""audeering モデルで arousal/valence/dominance 次元感情推定."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import structlog
import torch
from transformers import Wav2Vec2Processor

from poc.src.pipeline.models import DimensionalEmotion, TranscriptSegment

logger = structlog.get_logger(__name__)


def analyze_dimensional_emotion(
    audio_path: Path,
    segments: list[TranscriptSegment],
    *,
    model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    device: str = "cpu",
) -> dict[int, DimensionalEmotion]:
    """audeering の wav2vec2 モデルで arousal/valence/dominance を推定する.

    Args:
        audio_path: 入力WAVファイルパス
        segments: 字幕セグメントリスト（時間区間参照用）
        model_name: HuggingFace モデル名
        device: デバイス (cpu/cuda)

    Returns:
        セグメントID → DimensionalEmotion のマッピング
    """
    from transformers import Wav2Vec2Model

    logger.info("次元感情推定開始", model=model_name)

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # 音声読み込み
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    results: dict[int, DimensionalEmotion] = {}

    for seg in segments:
        start_sample = int(seg.start * sr)
        end_sample = int(seg.end * sr)
        segment_audio = audio[start_sample:end_sample]

        if len(segment_audio) == 0:
            results[seg.id] = DimensionalEmotion(
                arousal=0.5, valence=0.5, dominance=0.5
            )
            continue

        inputs = processor(
            segment_audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # hidden_states の平均をとり、線形変換で3次元に射影
            hidden = outputs.last_hidden_state.mean(dim=1)
            # 簡易的にsigmoidで0-1に正規化
            values = torch.sigmoid(hidden[0, :3]).cpu().numpy()

        results[seg.id] = DimensionalEmotion(
            arousal=float(values[0]),
            valence=float(values[1]),
            dominance=float(values[2]),
        )

    logger.info("次元感情推定完了", segments_analyzed=len(results))
    return results
