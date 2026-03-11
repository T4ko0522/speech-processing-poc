"""click CLI エントリポイント."""

from __future__ import annotations

import functools
import os
import sys

# PyTorch 2.6+ 互換パッチ: pyannote.audio のチェックポイントが omegaconf の
# 多数の内部型（ListConfig, DictConfig, ContainerMetadata 等）を含むため、
# weights_only=True では読み込めない。読み込み対象は HuggingFace の信頼済み
# モデルのみであるため、weights_only=False をデフォルトにする。
import torch

_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

import click  # noqa: E402
import structlog  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402

console = Console()


def _setup_logging() -> None:
    """structlog のセットアップ."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    default="poc/output",
    help="出力ディレクトリ (デフォルト: poc/output)",
)
@click.option(
    "--device",
    "-d",
    default="cpu",
    type=click.Choice(["cpu", "cuda"]),
    help="デバイス (デフォルト: cpu)",
)
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True),
    help="設定ファイルパス (デフォルト: poc/configs/default.yaml)",
)
@click.option(
    "--export-video",
    is_flag=True,
    default=False,
    help="感情ラベル付き動画をエクスポートする",
)
def main(
    input_file: str,
    output_dir: str,
    device: str,
    config: str | None,
    export_video: bool,
) -> None:
    """VODAI 動画解析PoC CLI.

    INPUT_FILE: 入力動画ファイルパス
    """
    load_dotenv()
    _setup_logging()

    console.print(
        Panel(
            "[bold]VODAI 動画解析 PoC[/bold]\n"
            f"入力: {input_file}\n"
            f"出力: {output_dir}\n"
            f"デバイス: {device}",
            title="VODAI",
            border_style="blue",
        )
    )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    hf_token = os.getenv("HF_TOKEN")

    if not openai_api_key:
        console.print(
            "[yellow]注意: OPENAI_API_KEY 未設定。Ollama設定があればローカルLLMを使用します。[/yellow]"
        )

    from poc.src.pipeline.runner import run_pipeline

    try:
        result = run_pipeline(
            input_file,
            output_dir=output_dir,
            device=device,
            config_path=config,
            openai_api_key=openai_api_key,
            hf_token=hf_token,
            export_video=export_video,
        )
        console.print(f"\n[green]完了！[/green] 出力ディレクトリ: {output_dir}")

        # サマリー表示
        if result.raw_transcript:
            console.print(f"  字幕セグメント数: {len(result.raw_transcript.segments)}")
        if result.fixed_transcript:
            console.print(f"  誤字補正数: {len(result.fixed_transcript.diffs)}")
        if result.scenes:
            console.print(f"  シーン数: {len(result.scenes.boundaries)}")
        if result.emotions:
            console.print(f"  感情エントリ数: {len(result.emotions.entries)}")

    except FileNotFoundError as e:
        console.print(f"[red]エラー: {e}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]入力エラー: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]処理エラー: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
