"""パイプライン全体のオーケストレーション."""

from __future__ import annotations

from pathlib import Path

import structlog
import yaml

from poc.src.pipeline.models import (
    EmotionTimeline,
    PipelineResult,
    ScenesResult,
)

logger = structlog.get_logger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"


def load_config(config_path: Path | None = None) -> dict:
    """設定ファイルを読み込む."""
    path = config_path or CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    input_file: str,
    *,
    output_dir: str = "poc/output",
    device: str = "cpu",
    config_path: str | None = None,
    openai_api_key: str | None = None,
    hf_token: str | None = None,
) -> PipelineResult:
    """パイプライン全体を実行する."""
    from poc.src.asr.audio_extract import extract_audio
    from poc.src.asr.transcribe import transcribe
    from poc.src.correction.typo_corrector import correct_transcript
    from poc.src.emotion.dimensional_emotion import analyze_dimensional_emotion
    from poc.src.emotion.fusion import fuse_emotions
    from poc.src.emotion.speech_emotion import analyze_speech_emotion
    from poc.src.emotion.text_emotion import analyze_text_emotion
    from poc.src.io.reader import validate_input
    from poc.src.io.writer import write_results
    from poc.src.llm import create_llm_client, get_model_for_task
    from poc.src.scene.detector import detect_scenes
    from poc.src.scene.summarizer import summarize_scenes

    config = load_config(Path(config_path) if config_path else None)
    result = PipelineResult(input_file=input_file)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ===== LLM クライアント初期化 =====
    llm_cfg = config.get("llm", {})
    provider = llm_cfg.get("provider", "openai")
    llm_client = None
    try:
        llm_client, _ = create_llm_client(llm_cfg, openai_api_key)
        logger.info("LLMプロバイダー初期化", provider=provider)
    except ValueError as e:
        logger.warning("LLMクライアント初期化失敗、補正・要約はスキップ", error=str(e))

    # ===== 入力検証 =====
    video_path = validate_input(input_file)
    logger.info("パイプライン開始", input=str(video_path))

    # ===== Step 1: 音声抽出 (Fatal) =====
    audio_cfg = config.get("audio", {})
    audio_path = out_path / "audio.wav"
    try:
        extract_audio(
            video_path,
            audio_path,
            sample_rate=audio_cfg.get("sample_rate", 16000),
            channels=audio_cfg.get("channels", 1),
        )
    except Exception as e:
        logger.error("音声抽出失敗（Fatal）", error=str(e))
        raise

    # ===== Step 2: 文字起こし (Fatal) =====
    asr_cfg = config.get("asr", {})
    try:
        raw_transcript = transcribe(
            audio_path,
            model_name=asr_cfg.get("model_name", "large-v3"),
            language=asr_cfg.get("language", "ja"),
            batch_size=asr_cfg.get("batch_size", 16),
            compute_type=asr_cfg.get("compute_type", "float32"),
            device=device,
            hf_token=hf_token,
        )
        result.raw_transcript = raw_transcript
    except Exception as e:
        logger.error("文字起こし失敗（Fatal）", error=str(e))
        raise

    segments = raw_transcript.segments

    # ===== Step 3: 誤字補正 (リトライ後スキップ可) =====
    if llm_client:
        corr_cfg = config.get("correction", {})
        correction_model = get_model_for_task(llm_cfg, "correction")
        try:
            fixed = correct_transcript(
                raw_transcript,
                client=llm_client,
                model=correction_model,
                chunk_size=corr_cfg.get("chunk_size", 10),
                temperature=corr_cfg.get("temperature", 0.1),
                confidence_threshold=corr_cfg.get("confidence_threshold", 0.7),
            )
            result.fixed_transcript = fixed
            segments = fixed.segments
        except Exception:
            logger.warning("誤字補正スキップ（API失敗）")
    else:
        logger.warning("LLM未設定、誤字補正スキップ")

    # ===== Step 4: シーン検出 (スキップ可) =====
    scene_cfg = config.get("scene", {})
    boundaries = []
    try:
        boundaries = detect_scenes(
            video_path,
            out_path,
            threshold=scene_cfg.get("threshold", 27.0),
            min_scene_len=scene_cfg.get("min_scene_len", 15),
        )
    except Exception:
        logger.warning("シーン検出スキップ（エラー）")

    # ===== Step 5: シーン要約 (スキップ可) =====
    if boundaries and llm_client:
        summary_cfg = config.get("scene_summary", {})
        summary_model = get_model_for_task(llm_cfg, "summary")
        # Ollama のローカルモデルは基本 Vision 非対応
        supports_vision = provider == "openai"
        try:
            scenes_result = summarize_scenes(
                boundaries,
                segments,
                client=llm_client,
                model=summary_model,
                max_tokens=summary_cfg.get("max_tokens", 500),
                supports_vision=supports_vision,
            )
            result.scenes = scenes_result
        except Exception:
            logger.warning("シーン要約スキップ（API失敗）")
            result.scenes = ScenesResult(boundaries=boundaries)
    elif boundaries:
        result.scenes = ScenesResult(boundaries=boundaries)

    # ===== Step 6: 感情推定 (スキップ可) =====
    emotion_cfg = config.get("emotion", {})
    speech_emotions = None
    dimensional_emotions = None
    text_emotions = None

    # 音声感情
    try:
        speech_cfg = emotion_cfg.get("speech", {})
        speech_emotions = analyze_speech_emotion(
            audio_path,
            segments,
            model_name=speech_cfg.get("model", "FunAudioLLM/SenseVoiceSmall"),
            language=speech_cfg.get("language", "ja"),
            device=device,
        )
    except Exception:
        logger.warning("音声感情推定スキップ")

    # 次元感情
    try:
        dim_cfg = emotion_cfg.get("dimensional", {})
        dimensional_emotions = analyze_dimensional_emotion(
            audio_path,
            segments,
            model_name=dim_cfg.get(
                "model",
                "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
            ),
            device=device,
        )
    except Exception:
        logger.warning("次元感情推定スキップ")

    # テキスト感情
    try:
        text_cfg = emotion_cfg.get("text", {})
        text_emotions = analyze_text_emotion(
            segments,
            model_name=text_cfg.get(
                "model",
                "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime",
            ),
            device=device,
        )
    except Exception:
        logger.warning("テキスト感情推定スキップ")

    # 融合
    fusion_cfg = emotion_cfg.get("fusion", {})
    if speech_emotions or dimensional_emotions or text_emotions:
        timeline = fuse_emotions(
            segments,
            speech_emotions=speech_emotions,
            dimensional_emotions=dimensional_emotions,
            text_emotions=text_emotions,
            speech_weight=fusion_cfg.get("speech_weight", 0.4),
            text_weight=fusion_cfg.get("text_weight", 0.6),
        )
        result.emotions = timeline
    else:
        result.emotions = EmotionTimeline(entries=[])

    # ===== Step 7: 出力書き出し =====
    files = write_results(result, out_path)
    logger.info("パイプライン完了", output_files=len(files))

    return result
