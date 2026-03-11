"""LLM プロバイダー共通ヘルパー（OpenAI / Ollama 切り替え）."""

from __future__ import annotations

import json

from openai import OpenAI


def parse_llm_json(content: str) -> dict:
    """LLM レスポンスから JSON を抽出・パースする.

    Ollama 等がマークダウンコードブロックで囲む場合に対応する。

    Args:
        content: LLM のレスポンステキスト

    Returns:
        パース済みの dict
    """
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def create_llm_client(
    config: dict,
    openai_api_key: str | None = None,
) -> OpenAI:
    """設定に基づいて LLM クライアントを返す.

    Args:
        config: llm セクションの設定 dict（default.yaml の llm セクション）
        openai_api_key: OpenAI API キー（provider=openai 時に使用）

    Returns:
        OpenAI 互換クライアント
    """
    provider = config["provider"]

    if provider == "ollama":
        ollama_cfg = config["ollama"]
        return OpenAI(base_url=ollama_cfg["base_url"], api_key="ollama")

    # OpenAI
    if not openai_api_key:
        raise ValueError("OpenAI プロバイダーには OPENAI_API_KEY が必要です")
    return OpenAI(api_key=openai_api_key)


def get_model_for_task(
    config: dict,
    task: str,
) -> str:
    """タスク別のモデル名を取得する.

    Args:
        config: llm セクションの設定 dict（default.yaml の llm セクション）
        task: "correction" or "summary"

    Returns:
        モデル名
    """
    provider = config["provider"]

    if provider == "ollama":
        return config["ollama"]["model"]

    openai_cfg = config["openai"]
    if task == "correction":
        return openai_cfg["model_correction"]
    elif task == "summary":
        return openai_cfg["model_summary"]
    raise ValueError(f"未知のタスク: {task}")
