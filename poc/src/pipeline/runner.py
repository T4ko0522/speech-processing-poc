"""パイプライン全体のオーケストレーション."""

from __future__ import annotations

import time
from pathlib import Path

import structlog
import yaml

from poc.src.pipeline.models import (
    EmotionTimeline,
    PipelineResult,
    ScenesResult,
    StepTiming,
)

logger = structlog.get_logger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"


def load_config(config_path: Path | None = None) -> dict:
    """設定ファイルを読み込む."""
    path = config_path or CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _record_timing(
    timings: list[StepTiming],
    step_name: str,
    start_time: float,
    status: str = "completed",
    skip_reason: str | None = None,
    retry_count: int = 0,
) -> None:
    """ステップの計測結果を記録する."""
    duration = time.monotonic() - start_time
    timings.append(
        StepTiming(
            step_name=step_name,
            duration_seconds=round(duration, 3),
            status=status,
            skip_reason=skip_reason,
            retry_count=retry_count,
        )
    )


def run_pipeline(
    input_file: str,
    *,
    output_dir: str = "poc/output",
    device: str = "cpu",
    config_path: str | None = None,
    openai_api_key: str | None = None,
    hf_token: str | None = None,
    export_video: bool = False,
) -> PipelineResult:
    """パイプライン全体を実行する."""
    from poc.src.asr.audio_extract import extract_audio
    from poc.src.asr.transcribe import transcribe
    from poc.src.correction.typo_corrector import correct_transcript
    from poc.src.emotion.dimensional_emotion import analyze_dimensional_emotion
    from poc.src.emotion.fusion import fuse_emotions
    from poc.src.emotion.speech_emotion import analyze_speech_emotion
    from poc.src.io.reader import validate_input
    from poc.src.io.writer import write_results
    from poc.src.llm import check_llm_connection, create_llm_client, get_model_for_task
    from poc.src.scene.detector import detect_scenes
    from poc.src.scene.summarizer import summarize_scenes

    pipeline_start = time.monotonic()
    config = load_config(Path(config_path) if config_path else None)
    result = PipelineResult(input_file=input_file)
    timings: list[StepTiming] = []

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ===== LLM クライアント初期化 =====
    llm_cfg = config["llm"]
    provider = llm_cfg["provider"]
    llm_client = None
    try:
        llm_client = create_llm_client(llm_cfg, openai_api_key)
        check_llm_connection(llm_cfg, llm_client)
        logger.info("LLMプロバイダー初期化", provider=provider)
    except ConnectionError as e:
        llm_client = None
        logger.warning("LLM接続失敗、補正・要約はスキップ", error=str(e))
    except ValueError as e:
        logger.warning("LLMクライアント初期化失敗、補正・要約はスキップ", error=str(e))

    # ===== 入力検証 =====
    video_path = validate_input(input_file)
    logger.info("パイプライン開始", input=str(video_path))

    # ===== Step 1: 音声抽出 (Fatal) =====
    audio_cfg = config["audio"]
    audio_path = out_path / "audio.wav"
    t0 = time.monotonic()
    try:
        extract_audio(
            video_path,
            audio_path,
            sample_rate=audio_cfg["sample_rate"],
            channels=audio_cfg["channels"],
        )
        _record_timing(timings, "audio_extract", t0)
    except Exception as e:
        _record_timing(
            timings, "audio_extract", t0, status="failed", skip_reason=str(e)
        )
        logger.error("音声抽出失敗（Fatal）", error=str(e))
        raise

    # ===== Step 2: 文字起こし (Fatal) =====
    asr_cfg = config["asr"]
    t0 = time.monotonic()
    try:
        raw_transcript = transcribe(
            audio_path,
            model_name=asr_cfg["model_name"],
            language=asr_cfg["language"],
            batch_size=asr_cfg["batch_size"],
            compute_type=asr_cfg["compute_type"],
            device=device,
            hf_token=hf_token,
            diarization=asr_cfg.get("diarization"),
        )
        result.raw_transcript = raw_transcript
        _record_timing(timings, "transcribe", t0)
    except Exception as e:
        _record_timing(timings, "transcribe", t0, status="failed", skip_reason=str(e))
        logger.error("文字起こし失敗（Fatal）", error=str(e))
        raise

    segments = raw_transcript.segments

    # ===== Step 3: 誤字補正 (リトライ後スキップ可) =====
    t0 = time.monotonic()
    if llm_client:
        corr_cfg = config["correction"]
        correction_model = get_model_for_task(llm_cfg, "correction")
        try:
            fixed, corr_retries = correct_transcript(
                raw_transcript,
                client=llm_client,
                model=correction_model,
                chunk_size=corr_cfg["chunk_size"],
                temperature=corr_cfg["temperature"],
                confidence_threshold=corr_cfg["confidence_threshold"],
                max_retries=corr_cfg["max_retries"],
            )
            result.fixed_transcript = fixed
            segments = fixed.segments
            _record_timing(timings, "correction", t0, retry_count=corr_retries)
        except Exception:
            _record_timing(
                timings, "correction", t0, status="failed", skip_reason="API失敗"
            )
            logger.warning("誤字補正スキップ（API失敗）")
    else:
        _record_timing(
            timings, "correction", t0, status="skipped", skip_reason="LLM未設定"
        )
        logger.warning("LLM未設定、誤字補正スキップ")

    # ===== Step 4: シーン検出 (スキップ可) =====
    scene_cfg = config["scene"]
    boundaries = []
    t0 = time.monotonic()
    try:
        boundaries = detect_scenes(
            video_path,
            out_path,
            threshold=scene_cfg["threshold"],
            min_scene_len=scene_cfg["min_scene_len"],
            merge_threshold=scene_cfg["merge_threshold"],
        )
        _record_timing(timings, "scene_detect", t0)
    except Exception:
        _record_timing(
            timings, "scene_detect", t0, status="failed", skip_reason="検出エラー"
        )
        logger.warning("シーン検出スキップ（エラー）")

    # ===== Step 5: シーン要約 (スキップ可) =====
    t0 = time.monotonic()
    if boundaries and llm_client:
        summary_cfg = config["scene_summary"]
        summary_model = get_model_for_task(llm_cfg, "summary")
        # Ollama のローカルモデルは基本 Vision 非対応
        supports_vision = provider == "openai"
        try:
            scenes_result, summary_retries = summarize_scenes(
                boundaries,
                segments,
                client=llm_client,
                model=summary_model,
                max_tokens=summary_cfg["max_tokens"],
                supports_vision=supports_vision,
            )
            result.scenes = scenes_result
            _record_timing(timings, "scene_summary", t0, retry_count=summary_retries)
        except Exception:
            _record_timing(
                timings, "scene_summary", t0, status="failed", skip_reason="API失敗"
            )
            logger.warning("シーン要約スキップ（API失敗）")
            result.scenes = ScenesResult(boundaries=boundaries)
    elif boundaries:
        _record_timing(
            timings, "scene_summary", t0, status="skipped", skip_reason="LLM未設定"
        )
        result.scenes = ScenesResult(boundaries=boundaries)
    else:
        _record_timing(
            timings, "scene_summary", t0, status="skipped", skip_reason="シーン未検出"
        )

    # ===== Step 6: 感情推定 (スキップ可) =====
    emotion_cfg = config["emotion"]
    dimensional_emotions = None
    speech_emotions = None

    # 感情推定用に音声を一度だけロード
    import librosa as _librosa

    _emotion_audio: tuple | None = None
    try:
        _emotion_audio = _librosa.load(
            str(audio_path), sr=audio_cfg["sample_rate"], mono=True
        )
    except Exception:
        # librosa (PySoundFile/audioread) が失敗した場合、wave stdlib でフォールバック
        import wave as _wave

        import numpy as _np

        try:
            with _wave.open(str(audio_path), "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                audio_data = wf.readframes(wf.getnframes())
                audio_arr = (
                    _np.frombuffer(audio_data, dtype=_np.int16).astype(_np.float32)
                    / 32768.0
                )
                if n_channels > 1:
                    audio_arr = audio_arr.reshape(-1, n_channels).mean(axis=1)
                target_sr = audio_cfg["sample_rate"]
                if sr != target_sr:
                    audio_arr = _librosa.resample(
                        audio_arr, orig_sr=sr, target_sr=target_sr
                    )
                    sr = target_sr
                _emotion_audio = (audio_arr, sr)
            logger.info("音声ロード完了（waveフォールバック）")
        except Exception:
            logger.warning("感情推定用の音声ロード失敗")

    # 次元感情
    t0 = time.monotonic()
    try:
        dim_cfg = emotion_cfg["dimensional"]
        dimensional_emotions = analyze_dimensional_emotion(
            audio_path,
            segments,
            model_name=dim_cfg["model"],
            device=device,
            preloaded_audio=_emotion_audio,
        )
        _record_timing(timings, "emotion_dimensional", t0)
    except Exception:
        _record_timing(
            timings,
            "emotion_dimensional",
            t0,
            status="failed",
            skip_reason="推定エラー",
        )
        logger.warning("次元感情推定スキップ")

    # 音声感情 (SER)
    t0 = time.monotonic()
    try:
        speech_cfg = emotion_cfg["speech"]
        speech_emotions = analyze_speech_emotion(
            audio_path,
            segments,
            model_name=speech_cfg["model"],
            device=device,
            preloaded_audio=_emotion_audio,
            temperature=speech_cfg.get("temperature", 0.5),
        )
        _record_timing(timings, "emotion_speech", t0)
    except Exception:
        _record_timing(
            timings, "emotion_speech", t0, status="failed", skip_reason="推定エラー"
        )
        logger.warning("音声感情推定スキップ")

    # prosody
    t0 = time.monotonic()
    prosody_results = None
    prosody_cfg = emotion_cfg["prosody"]
    if prosody_cfg["enabled"]:
        try:
            from poc.src.emotion.prosody import analyze_prosody

            prosody_results = analyze_prosody(
                audio_path, segments, preloaded_audio=_emotion_audio
            )
            _record_timing(timings, "emotion_prosody", t0)
        except Exception:
            _record_timing(
                timings,
                "emotion_prosody",
                t0,
                status="failed",
                skip_reason="推定エラー",
            )
            logger.warning("prosody推定スキップ")
    else:
        _record_timing(
            timings, "emotion_prosody", t0, status="skipped", skip_reason="設定で無効"
        )
        logger.info("prosody推定スキップ（emotion.prosody.enabled=false）")

    # 融合
    t0 = time.monotonic()
    fusion_cfg = emotion_cfg["fusion"]
    if dimensional_emotions or speech_emotions:
        timeline = fuse_emotions(
            segments,
            dimensional_emotions=dimensional_emotions,
            speech_emotions=speech_emotions,
            prosody_results=prosody_results,
            speech_weight=fusion_cfg["speech_weight"],
            dimensional_weight=fusion_cfg["dimensional_weight"],
            prosody_boost=fusion_cfg["prosody_boost"],
            neutral_zone=fusion_cfg["neutral_zone"],
        )
        result.emotions = timeline
        _record_timing(timings, "emotion_fusion", t0)
    else:
        result.emotions = EmotionTimeline(entries=[])
        _record_timing(
            timings,
            "emotion_fusion",
            t0,
            status="skipped",
            skip_reason="感情データなし",
        )

    # ===== Step 7: 感情ラベル付き動画エクスポート (スキップ可) =====
    if export_video:
        t0 = time.monotonic()
        if result.emotions and result.emotions.entries:
            try:
                from poc.src.io.video_export import export_video_with_emotions

                video_out = out_path / "emotion_overlay.mp4"
                export_cfg = config.get("video_export", {})
                # トランスクリプトセグメントを取得（fixed > raw の優先順）
                transcript_segs = None
                if result.fixed_transcript and result.fixed_transcript.segments:
                    transcript_segs = result.fixed_transcript.segments
                elif result.raw_transcript and result.raw_transcript.segments:
                    transcript_segs = result.raw_transcript.segments

                export_video_with_emotions(
                    video_path,
                    video_out,
                    result.emotions,
                    transcript_segments=transcript_segs,
                    font_name=export_cfg.get("font_name"),
                    font_size=export_cfg.get("font_size", 48),
                    transcript_font_size=export_cfg.get("transcript_font_size", 32),
                )
                _record_timing(timings, "video_export", t0)
            except Exception as e:
                _record_timing(
                    timings,
                    "video_export",
                    t0,
                    status="failed",
                    skip_reason=str(e),
                )
                logger.warning("動画エクスポートスキップ", error=str(e))
        else:
            _record_timing(
                timings,
                "video_export",
                t0,
                status="skipped",
                skip_reason="感情データなし",
            )

    # ===== Step 8: 出力書き出し =====
    t0 = time.monotonic()
    result.step_timings = timings
    files = write_results(result, out_path)
    _record_timing(timings, "output", t0)

    total_duration = round(time.monotonic() - pipeline_start, 3)
    logger.info(
        "パイプライン完了", output_files=len(files), total_duration=total_duration
    )

    return result
