"""PySceneDetect シーン境界検出 + 代表フレーム抽出."""

from __future__ import annotations

from pathlib import Path

import cv2
import structlog
from scenedetect import ContentDetector, SceneManager, open_video

from poc.src.pipeline.models import SceneBoundary

logger = structlog.get_logger(__name__)


def _format_timecode(seconds: float) -> str:
    """秒数を HH:MM:SS.mmm 形式に変換."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _merge_short_scenes(
    boundaries: list[SceneBoundary],
    merge_threshold: float,
) -> list[SceneBoundary]:
    """merge_threshold秒未満の短シーンを次のシーンに吸収する."""
    if not boundaries:
        return boundaries

    merged: list[SceneBoundary] = []
    for b in boundaries:
        duration = b.end - b.start
        if merged and duration < merge_threshold:
            # 短シーンを前のシーンに吸収（endを延長）
            prev = merged[-1]
            merged[-1] = prev.model_copy(
                update={
                    "end": b.end,
                    "end_timecode": b.end_timecode,
                }
            )
        else:
            merged.append(b)

    # scene_id を振り直し
    result = []
    for i, b in enumerate(merged):
        result.append(b.model_copy(update={"scene_id": i}))

    logger.info(
        "短シーンマージ完了",
        before=len(boundaries),
        after=len(result),
        merge_threshold=merge_threshold,
    )
    return result


def detect_scenes(
    video_path: Path,
    output_dir: Path,
    *,
    threshold: float = 40.0,
    min_scene_len: int = 45,
    merge_threshold: float = 2.0,
) -> list[SceneBoundary]:
    """動画からシーン境界を検出し、代表フレームを抽出する.

    Args:
        video_path: 入力動画ファイルパス
        output_dir: フレーム画像の出力先
        threshold: ContentDetector の閾値
        min_scene_len: 最小シーン長（フレーム数）
        merge_threshold: この秒数未満の短シーンをマージ

    Returns:
        シーン境界のリスト
    """
    logger.info("シーン検出開始", threshold=threshold, merge_threshold=merge_threshold)

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        logger.warning("シーンが検出されませんでした")
        return []

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    boundaries = []
    cap = cv2.VideoCapture(str(video_path))

    for i, (start, end) in enumerate(scene_list):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        mid_sec = (start_sec + end_sec) / 2

        # 中間フレームを代表フレームとして抽出
        frame_path = frames_dir / f"scene_{i:04d}.png"
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_sec * 1000)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(frame_path), frame)
        else:
            frame_path = None

        boundaries.append(
            SceneBoundary(
                scene_id=i,
                start=start_sec,
                end=end_sec,
                start_timecode=_format_timecode(start_sec),
                end_timecode=_format_timecode(end_sec),
                frame_path=str(frame_path) if frame_path else None,
            )
        )

    cap.release()
    logger.info("シーン検出完了（マージ前）", scenes_count=len(boundaries))

    # 短シーンのマージ
    boundaries = _merge_short_scenes(boundaries, merge_threshold)

    logger.info("シーン検出完了", scenes_count=len(boundaries))
    return boundaries
