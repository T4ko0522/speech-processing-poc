"""audeering モデルで arousal/valence/dominance 次元感情推定."""

from __future__ import annotations

from pathlib import Path

import librosa
import structlog
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor

from poc.src.pipeline.models import DimensionalEmotion, TranscriptSegment

logger = structlog.get_logger(__name__)


class _RegressionHead(nn.Module):
    """audEERING カスタム回帰ヘッド."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class _EmotionModel(Wav2Vec2PreTrainedModel):
    """audEERING wav2vec2 + 回帰ヘッド.

    モデルカード: https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
    回帰ヘッドが arousal/dominance/valence を直接出力する。
    """

    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.post_init()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


def analyze_dimensional_emotion(
    audio_path: Path,
    segments: list[TranscriptSegment],
    *,
    model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    device: str = "cpu",
) -> dict[int, DimensionalEmotion]:
    """audeering の wav2vec2 モデルで arousal/valence/dominance を推定する.

    正しい回帰ヘッド付きモデル (_EmotionModel) を使用し、
    arousal/dominance/valence を直接出力する。

    Args:
        audio_path: 入力WAVファイルパス
        segments: 字幕セグメントリスト（時間区間参照用）
        model_name: HuggingFace モデル名
        device: デバイス (cpu/cuda)

    Returns:
        セグメントID → DimensionalEmotion のマッピング
    """
    logger.info("次元感情推定開始", model=model_name)

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = _EmotionModel.from_pretrained(model_name)
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
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            _, logits = model(input_values)
            vals = logits.squeeze().cpu().numpy()

        # モデル出力順序: arousal, dominance, valence
        results[seg.id] = DimensionalEmotion(
            arousal=float(vals[0]),
            dominance=float(vals[1]),
            valence=float(vals[2]),
        )

    logger.info("次元感情推定完了", segments_analyzed=len(results))
    return results
