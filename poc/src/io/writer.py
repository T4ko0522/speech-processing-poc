"""JSON/SRT/VTT/report 書き出し."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from poc.src.io.subtitle_format import segments_to_srt, segments_to_vtt
from poc.src.pipeline.models import PipelineResult

logger = structlog.get_logger(__name__)


def write_results(result: PipelineResult, output_dir: str | Path) -> dict[str, Path]:
    """パイプライン結果を各種形式で書き出す.

    Args:
        result: パイプライン結果
        output_dir: 出力ディレクトリ

    Returns:
        生成されたファイルパスの辞書
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    # transcript_raw.json
    if result.raw_transcript:
        p = out / "transcript_raw.json"
        p.write_text(
            result.raw_transcript.model_dump_json(indent=2),
            encoding="utf-8",
        )
        files["transcript_raw"] = p
        logger.info("出力完了", file=str(p))

    # transcript_fixed.json
    if result.fixed_transcript:
        p = out / "transcript_fixed.json"
        p.write_text(
            result.fixed_transcript.model_dump_json(indent=2),
            encoding="utf-8",
        )
        files["transcript_fixed"] = p
        logger.info("出力完了", file=str(p))

    # scenes.json
    if result.scenes:
        p = out / "scenes.json"
        p.write_text(
            result.scenes.model_dump_json(indent=2),
            encoding="utf-8",
        )
        files["scenes"] = p
        logger.info("出力完了", file=str(p))

    # emotions.json
    if result.emotions:
        p = out / "emotions.json"
        p.write_text(
            result.emotions.model_dump_json(indent=2),
            encoding="utf-8",
        )
        files["emotions"] = p
        logger.info("出力完了", file=str(p))

    # SRT / VTT — 補正済みがあればそれ、なければ生字幕
    transcript = result.fixed_transcript or result.raw_transcript
    if transcript:
        # output.srt
        p = out / "output.srt"
        p.write_text(segments_to_srt(transcript.segments), encoding="utf-8")
        files["srt"] = p
        logger.info("出力完了", file=str(p))

        # output.vtt
        p = out / "output.vtt"
        p.write_text(segments_to_vtt(transcript.segments), encoding="utf-8")
        files["vtt"] = p
        logger.info("出力完了", file=str(p))

    # report.json
    report = {
        "input_file": result.input_file,
        "raw_segments": len(result.raw_transcript.segments) if result.raw_transcript else 0,
        "fixed_segments": len(result.fixed_transcript.segments) if result.fixed_transcript else 0,
        "correction_diffs": len(result.fixed_transcript.diffs) if result.fixed_transcript else 0,
        "scenes": len(result.scenes.boundaries) if result.scenes else 0,
        "emotion_entries": len(result.emotions.entries) if result.emotions else 0,
        "output_files": {k: str(v) for k, v in files.items()},
    }
    p = out / "report.json"
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    files["report"] = p
    logger.info("出力完了", file=str(p))

    return files
