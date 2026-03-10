"""LLM チャンク単位誤字補正、diff生成、confidence判定."""

from __future__ import annotations

import json
from pathlib import Path

import structlog
from openai import OpenAI
from tenacity import stop_after_attempt, wait_exponential

from poc.src.pipeline.models import (
    CorrectionDiff,
    FixedTranscript,
    RawTranscript,
    TranscriptSegment,
)

logger = structlog.get_logger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "typo_correction.txt"


def _load_prompt() -> str:
    """誤字補正プロンプトを読み込む."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def _call_llm(
    client: OpenAI,
    prompt: str,
    chunk: list[dict],
    model: str,
    temperature: float,
    max_retries: int = 3,
) -> dict:
    """LLM に誤字補正リクエストを送信."""
    from tenacity import Retrying

    retryer = Retrying(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    for attempt in retryer:
        with attempt:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(chunk, ensure_ascii=False)},
                ],
            )
            content = response.choices[0].message.content

            # JSON部分を抽出（Ollamaはマークダウンで囲む場合がある）
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                text = "\n".join(lines)
            return json.loads(text)


def correct_transcript(
    transcript: RawTranscript,
    *,
    client: OpenAI,
    model: str,
    chunk_size: int = 10,
    temperature: float = 0.1,
    confidence_threshold: float = 0.7,
    max_retries: int = 3,
) -> FixedTranscript:
    """字幕テキストの誤字を LLM でチャンク単位に補正する.

    Args:
        transcript: 生字幕データ
        client: OpenAI 互換クライアント
        model: 使用するモデル名
        chunk_size: 一度に補正するセグメント数
        temperature: 生成温度
        confidence_threshold: この閾値未満の補正は適用しない
        max_retries: LLM呼び出しの最大リトライ回数

    Returns:
        FixedTranscript: 補正済み字幕 + diff ログ
    """
    prompt = _load_prompt()

    fixed_segments: list[TranscriptSegment] = []
    diffs: list[CorrectionDiff] = []

    # チャンク分割
    chunks: list[list[TranscriptSegment]] = []
    for i in range(0, len(transcript.segments), chunk_size):
        chunks.append(transcript.segments[i : i + chunk_size])

    consecutive_failures = 0
    max_consecutive_failures = 3

    for chunk_idx, chunk in enumerate(chunks):
        chunk_data = [{"id": seg.id, "text": seg.text} for seg in chunk]
        logger.info("誤字補正チャンク処理中", chunk=chunk_idx + 1, total=len(chunks))

        try:
            result = _call_llm(client, prompt, chunk_data, model, temperature, max_retries)
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            logger.warning(
                "誤字補正API失敗、未補正で通過",
                chunk=chunk_idx + 1,
                error=str(e),
                consecutive_failures=consecutive_failures,
            )
            fixed_segments.extend(chunk)
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    "連続失敗のため誤字補正を打ち切り",
                    remaining_chunks=len(chunks) - chunk_idx - 1,
                )
                for remaining in chunks[chunk_idx + 1 :]:
                    fixed_segments.extend(remaining)
                break
            continue

        result_map = {item["id"]: item for item in result.get("segments", [])}

        for seg in chunk:
            if seg.id in result_map:
                item = result_map[seg.id]
                corrected_text = item.get("text", seg.text)
                confidence = item.get("confidence", 1.0)
                was_corrected = item.get("corrected", False)

                if was_corrected and confidence >= confidence_threshold and corrected_text != seg.text:
                    diffs.append(
                        CorrectionDiff(
                            segment_id=seg.id,
                            original=seg.text,
                            corrected=corrected_text,
                            confidence=confidence,
                        )
                    )
                    new_seg = seg.model_copy(update={"text": corrected_text})
                    fixed_segments.append(new_seg)
                else:
                    fixed_segments.append(seg)
            else:
                fixed_segments.append(seg)

    logger.info("誤字補正完了", total_diffs=len(diffs))
    return FixedTranscript(
        language=transcript.language,
        segments=fixed_segments,
        diffs=diffs,
    )
