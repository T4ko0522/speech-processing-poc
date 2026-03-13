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

import warnings

# --- 既知の互換性警告を抑制 ---
# speechbrain 1.0: pretrained → inference のリダイレクト警告
warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated")
# pytorch-lightning: チェックポイント自動マイグレーション通知
warnings.filterwarnings("ignore", message="Lightning automatically upgraded your loaded checkpoint")
# pyannote.audio: モデル学習時バージョンとの差異（動作に影響なし）
warnings.filterwarnings("ignore", message="Model was trained with pyannote.audio")
warnings.filterwarnings("ignore", message="Model was trained with torch")

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


class _DefaultGroup(click.Group):
    """未知の引数をデフォルトコマンド (run) にフォールバックする Group."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # --help やオプションはそのまま、それ以外はデフォルトの run に振り分け
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["run"] + args
        return super().parse_args(ctx, args)


@click.group(cls=_DefaultGroup)
def main() -> None:
    """VODAI 動画解析PoC CLI."""


# ===== run コマンド（デフォルト: vodai <file> で呼ばれる） =====


@main.command()
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
def run(
    input_file: str,
    output_dir: str,
    device: str,
    config: str | None,
    export_video: bool,
) -> None:
    """パイプライン全体を実行する.

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


# ===== dev サブコマンドグループ =====


@main.group()
def dev() -> None:
    """開発用サブコマンド."""


@dev.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    default="poc/output/transcript_raw.json",
    type=click.Path(exists=True),
    help="入力 transcript JSON (デフォルト: poc/output/transcript_raw.json)",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    default="poc/output/transcript_fixed.json",
    help="出力先パス (デフォルト: poc/output/transcript_fixed.json)",
)
@click.option(
    "--config",
    "-c",
    default=None,
    type=click.Path(exists=True),
    help="設定ファイルパス",
)
def correct(input_path: str, output_path: str, config: str | None) -> None:
    """既存の transcript_raw.json に対して誤字補正だけを実行する."""
    import json
    from pathlib import Path

    load_dotenv()
    _setup_logging()

    from poc.src.correction.typo_corrector import correct_transcript
    from poc.src.llm import check_llm_connection, create_llm_client, get_model_for_task
    from poc.src.pipeline.models import RawTranscript
    from poc.src.pipeline.runner import load_config

    cfg = load_config(Path(config) if config else None)
    llm_cfg = cfg["llm"]
    corr_cfg = cfg["correction"]

    # LLM 初期化
    openai_api_key = os.getenv("OPENAI_API_KEY")
    try:
        client = create_llm_client(llm_cfg, openai_api_key)
        check_llm_connection(llm_cfg, client)
    except (ConnectionError, ValueError) as e:
        console.print(f"[red]LLM接続エラー: {e}[/red]")
        sys.exit(1)

    # transcript 読み込み
    with open(input_path, encoding="utf-8") as f:
        raw = RawTranscript.model_validate(json.load(f))

    console.print(
        f"[bold]誤字補正 dev モード[/bold]  segments={len(raw.segments)}  "
        f"chunk_size={corr_cfg['chunk_size']}"
    )

    model = get_model_for_task(llm_cfg, "correction")
    fixed, retries = correct_transcript(
        raw,
        client=client,
        model=model,
        chunk_size=corr_cfg["chunk_size"],
        temperature=corr_cfg["temperature"],
        confidence_threshold=corr_cfg["confidence_threshold"],
        max_retries=corr_cfg["max_retries"],
    )

    # 結果書き出し
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixed.model_dump(), f, ensure_ascii=False, indent=2)

    console.print(f"\n[green]完了！[/green] {output_path}")
    console.print(f"  補正数: {len(fixed.diffs)}  リトライ: {retries}")
    for d in fixed.diffs:
        console.print(f"  [dim]seg {d.segment_id}:[/dim] {d.original}")
        console.print(f"         → {d.corrected}  [dim](conf={d.confidence})[/dim]")


if __name__ == "__main__":
    main()
