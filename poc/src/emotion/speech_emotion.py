"""カテゴリカル Speech Emotion Recognition (SER)."""

from __future__ import annotations

from pathlib import Path

import librosa
import structlog
import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from poc.src.pipeline.models import SpeechEmotion, TranscriptSegment

logger = structlog.get_logger(__name__)

# モデルの出力ラベル (8感情)
SER_LABELS = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]


class _ClassificationHead(nn.Module):
    """ehcalabres カスタム分類ヘッド.

    チェックポイントのキー: classifier.dense / classifier.output
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dense(features)
        x = torch.relu(x)
        x = self.output(x)
        return x


class _SERModel(Wav2Vec2PreTrainedModel):
    """ehcalabres wav2vec2-lg-xlsr + 分類ヘッド.

    モデルカード: https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
    """

    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _ClassificationHead(config)
        self.post_init()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return logits


def analyze_speech_emotion(
    audio_path: Path,
    segments: list[TranscriptSegment],
    *,
    model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    device: str = "cpu",
    preloaded_audio: tuple | None = None,
    temperature: float = 0.5,
) -> dict[int, SpeechEmotion]:
    """Wav2Vec2 SER モデルで音声セグメントからカテゴリカル感情を推定する.

    Args:
        audio_path: 入力WAVファイルパス
        segments: 字幕セグメントリスト（時間区間参照用）
        model_name: HuggingFace モデル名
        device: デバイス (cpu/cuda)
        preloaded_audio: (audio_array, sample_rate) のタプル。指定時はファイル読み込みをスキップ

    Returns:
        セグメントID → SpeechEmotion のマッピング
    """
    logger.info("音声感情推定開始", model=model_name)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = _SERModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # 音声読み込み（プリロード済みならスキップ）
    if preloaded_audio is not None:
        audio, sr = preloaded_audio
    else:
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    results: dict[int, SpeechEmotion] = {}

    for seg in segments:
        start_sample = int(seg.start * sr)
        end_sample = int(seg.end * sr)
        segment_audio = audio[start_sample:end_sample]

        if len(segment_audio) == 0:
            scores = {label: 0.0 for label in SER_LABELS}
            scores["neutral"] = 1.0
            results[seg.id] = SpeechEmotion(scores=scores, top_label="neutral")
            continue

        inputs = feature_extractor(
            segment_audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            logits = model(input_values)
            probs = torch.softmax(logits / temperature, dim=-1).squeeze().cpu().numpy()

        scores = {}
        for i, label in enumerate(SER_LABELS):
            if i < len(probs):
                scores[label] = float(probs[i])
            else:
                scores[label] = 0.0

        top_label = max(scores, key=scores.get)
        results[seg.id] = SpeechEmotion(scores=scores, top_label=top_label)

    logger.info("音声感情推定完了", segments_analyzed=len(results))
    return results
