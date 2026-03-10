"""LLM フレーム画像 + 字幕で要約."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import structlog
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from poc.src.pipeline.models import (
    SceneBoundary,
    SceneSummary,
    ScenesResult,
    TranscriptSegment,
)

logger = structlog.get_logger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "scene_summary.txt"


def _load_prompt() -> str:
    """シーン要約プロンプトを読み込む."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def _encode_image(image_path: str) -> str:
    """画像をBase64エンコードする."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_subtitles_for_scene(
    segments: list[TranscriptSegment],
    start: float,
    end: float,
) -> str:
    """シーン区間に含まれる字幕テキストを取得."""
    texts = []
    for seg in segments:
        if seg.start >= start and seg.end <= end:
            texts.append(seg.text)
        elif seg.start < end and seg.end > start:
            texts.append(seg.text)
    return "\n".join(texts)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _call_llm_vision(
    client: OpenAI,
    prompt: str,
    image_b64: str | None,
    subtitle_text: str,
    start: float,
    end: float,
    model: str,
    max_tokens: int,
    supports_vision: bool = True,
) -> dict:
    """LLM にシーン要約リクエストを送信."""
    user_content: list[dict] = []

    # Vision 対応モデルのみ画像を送信
    if image_b64 and supports_vision:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )

    context = f"シーン時間: {start:.1f}s ~ {end:.1f}s\n"
    if subtitle_text:
        context += f"字幕テキスト:\n{subtitle_text}"
    else:
        context += "字幕テキスト: なし"

    user_content.append({"type": "text", "text": context})

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
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


def summarize_scenes(
    boundaries: list[SceneBoundary],
    segments: list[TranscriptSegment],
    *,
    client: OpenAI,
    model: str,
    max_tokens: int = 500,
    supports_vision: bool = True,
) -> ScenesResult:
    """各シーンの代表フレームと字幕から要約を生成する.

    Args:
        boundaries: シーン境界リスト
        segments: 字幕セグメントリスト
        client: OpenAI 互換クライアント
        model: 使用するモデル名
        max_tokens: 最大トークン数
        supports_vision: Vision（画像入力）対応かどうか

    Returns:
        ScenesResult: シーン境界 + 要約
    """
    prompt = _load_prompt()
    summaries: list[SceneSummary] = []
    consecutive_failures = 0
    max_consecutive_failures = 3

    for boundary in boundaries:
        logger.info("シーン要約生成中", scene_id=boundary.scene_id)

        image_b64 = None
        if boundary.frame_path and Path(boundary.frame_path).exists():
            image_b64 = _encode_image(boundary.frame_path)

        subtitle_text = _get_subtitles_for_scene(
            segments, boundary.start, boundary.end
        )

        try:
            result = _call_llm_vision(
                client, prompt, image_b64, subtitle_text,
                boundary.start, boundary.end, model, max_tokens,
                supports_vision=supports_vision,
            )
            summaries.append(
                SceneSummary(
                    scene_id=boundary.scene_id,
                    summary=result.get("summary", ""),
                    keywords=result.get("keywords", []),
                )
            )
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            logger.warning(
                "シーン要約失敗、スキップ",
                scene_id=boundary.scene_id,
                error=str(e),
                consecutive_failures=consecutive_failures,
            )
            summaries.append(
                SceneSummary(scene_id=boundary.scene_id, summary="", keywords=[])
            )
            if consecutive_failures >= max_consecutive_failures:
                remaining = len(boundaries) - boundary.scene_id - 1
                logger.error(
                    "連続失敗のためシーン要約を打ち切り",
                    remaining_scenes=remaining,
                )
                break

    logger.info("シーン要約完了", count=len(summaries))
    return ScenesResult(boundaries=boundaries, summaries=summaries)
