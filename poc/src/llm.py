"""LLM プロバイダー共通ヘルパー（OpenAI / Ollama 切り替え）."""

from __future__ import annotations

from openai import OpenAI


def create_llm_client(
    config: dict,
    openai_api_key: str | None = None,
) -> tuple[OpenAI, str]:
    """設定に基づいて LLM クライアントとモデル名を返す.

    Args:
        config: llm セクションの設定 dict
        openai_api_key: OpenAI API キー（provider=openai 時に使用）

    Returns:
        (OpenAI クライアント, モデル名) のタプル
    """
    provider = config.get("provider", "openai")

    if provider == "ollama":
        ollama_cfg = config.get("ollama", {})
        base_url = ollama_cfg.get("base_url", "http://localhost:11434/v1")
        model = ollama_cfg.get("model", "gemma3:4b")
        client = OpenAI(base_url=base_url, api_key="ollama")
        return client, model

    # OpenAI
    if not openai_api_key:
        raise ValueError("OpenAI プロバイダーには OPENAI_API_KEY が必要です")
    client = OpenAI(api_key=openai_api_key)
    return client, ""


def get_model_for_task(
    config: dict,
    task: str,
    default: str = "",
) -> str:
    """タスク別のモデル名を取得する.

    Args:
        config: llm セクションの設定 dict
        task: "correction" or "summary"
        default: フォールバックモデル名

    Returns:
        モデル名
    """
    provider = config.get("provider", "openai")

    if provider == "ollama":
        return config.get("ollama", {}).get("model", "gemma3:4b")

    openai_cfg = config.get("openai", {})
    if task == "correction":
        return openai_cfg.get("model_correction", default or "gpt-4.1")
    elif task == "summary":
        return openai_cfg.get("model_summary", default or "gpt-4o")
    return default
