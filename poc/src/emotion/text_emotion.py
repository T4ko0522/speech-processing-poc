"""LUKE WRIME テキスト感情推定."""

from __future__ import annotations

import structlog
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from poc.src.pipeline.models import TextEmotion, TranscriptSegment

# transformers 5.x の MLukeTokenizer バグ回避:
# vocab が dict で渡されるが Unigram は list[tuple] を期待する
from transformers.models.mluke import tokenization_mluke as _mluke_mod

_orig_mluke_init = _mluke_mod.MLukeTokenizer.__init__


def _patched_mluke_init(self, *args, **kwargs):
    if "vocab" in kwargs and isinstance(kwargs["vocab"], dict):
        kwargs["vocab"] = list(kwargs["vocab"].items())
    _orig_mluke_init(self, *args, **kwargs)


_mluke_mod.MLukeTokenizer.__init__ = _patched_mluke_init

logger = structlog.get_logger(__name__)

# WRIME 8感情ラベル
WRIME_LABELS = [
    "joy",  # 喜び
    "sadness",  # 悲しみ
    "anticipation",  # 期待
    "surprise",  # 驚き
    "anger",  # 怒り
    "fear",  # 恐れ
    "disgust",  # 嫌悪
    "trust",  # 信頼
]


def analyze_text_emotion(
    segments: list[TranscriptSegment],
    *,
    model_name: str = "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime",
    device: str = "cpu",
) -> dict[int, TextEmotion]:
    """LUKE モデルで字幕テキストから WRIME 8感情を推定する.

    Args:
        segments: 字幕セグメントリスト
        model_name: HuggingFace モデル名
        device: デバイス (cpu/cuda)

    Returns:
        セグメントID → TextEmotion のマッピング
    """
    import torch

    logger.info("テキスト感情推定開始", model=model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    results: dict[int, TextEmotion] = {}

    for seg in segments:
        if not seg.text.strip():
            results[seg.id] = TextEmotion(scores={label: 0.0 for label in WRIME_LABELS})
            continue

        inputs = tokenizer(
            seg.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # ラベル数とモデル出力数を合わせる
        scores = {}
        for i, label in enumerate(WRIME_LABELS):
            if i < len(probs):
                scores[label] = float(probs[i])
            else:
                scores[label] = 0.0

        results[seg.id] = TextEmotion(scores=scores)

    logger.info("テキスト感情推定完了", segments_analyzed=len(results))
    return results
