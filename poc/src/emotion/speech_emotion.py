"""SenseVoiceSmall カテゴリカル感情推定."""

from __future__ import annotations

from pathlib import Path

import structlog

from poc.src.pipeline.models import EmotionCategory, TranscriptSegment

logger = structlog.get_logger(__name__)


def analyze_speech_emotion(
    audio_path: Path,
    segments: list[TranscriptSegment],
    *,
    model_name: str = "FunAudioLLM/SenseVoiceSmall",
    language: str = "ja",
    device: str = "cpu",
) -> dict[int, EmotionCategory]:
    """SenseVoiceSmall で音声からカテゴリカル感情を推定する.

    Args:
        audio_path: 入力WAVファイルパス
        segments: 字幕セグメントリスト（時間区間参照用）
        model_name: FunASR モデル名
        language: 言語コード
        device: デバイス (cpu/cuda)

    Returns:
        セグメントID → EmotionCategory のマッピング
    """
    from funasr import AutoModel

    logger.info("音声感情推定開始", model=model_name)

    model = AutoModel(model=model_name, device=device)

    # SenseVoiceSmall の感情ラベル
    emotion_labels = ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"]

    results: dict[int, EmotionCategory] = {}

    # 音声全体で推論
    res = model.generate(input=str(audio_path), language=language)

    if res and len(res) > 0:
        # SenseVoiceSmall はテキスト出力に感情タグを含む
        # 各セグメントに対して最も近い結果をマッピング
        for seg in segments:
            # デフォルトは neutral
            best_label = "neutral"
            best_score = 0.5

            # res の出力からテキスト内の感情タグを解析
            for r in res:
                text = r.get("text", "")
                for label in emotion_labels:
                    tag = f"<|{label.upper()}|>"
                    if tag in text.upper() or f"<|{label}|>" in text:
                        best_label = label
                        best_score = 0.8
                        break

            results[seg.id] = EmotionCategory(label=best_label, score=best_score)
    else:
        # 推論失敗時はデフォルト値
        for seg in segments:
            results[seg.id] = EmotionCategory(label="neutral", score=0.5)

    logger.info("音声感情推定完了", segments_analyzed=len(results))
    return results
