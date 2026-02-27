#!/usr/bin/env python3
"""
sample_pose_filter_from_videos_batch.py  (batch YOLO for speed).

Based on the original sample_pose_filter_from_videos.py. fileciteturn0file0

Main change:
- Run YOLO pose inference in batches (BATCH_SIZE frames at a time) to reduce overhead and improve GPU utilization.

What this script does:
- For each video under VIDEOS_DIR:
  - Read frames sequentially with OpenCV.
  - Sample frames at a constant interval (INTERVAL_SEC) by using a stride in frames.
  - Run YOLO pose on sampled frames in batches.
  - Keep frames that contain >=2 "fencers".
    * "Fencer" is defined by checkForFencer(): BOTH ankles visible with sufficient keypoint confidence.
  - Optional: correlation filter vs previous SAVED final frame (to avoid near-duplicates).
- Output:
  * If SAVE_ALL_FINAL_TO_ONE_FOLDER=True: all final images go to GLOBAL_FINAL_DIR (no debug images).
  * Else: per-video subfolders under OUT_DIR; optional debug overlays under OUT_DIR/<video>/debug/.

Notes on speed:
- Batch inference reduces per-call overhead (CUDA launch, preprocessing) and typically speeds up runtime significantly.
- If you still need more speed, increase INTERVAL_SEC (fewer frames) and/or raise BATCH_SIZE.

"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from fencer_heuristics import checkForFencer
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
# VIDEOS_DIR = Path("/media/yin/Seagate8T1/youtube_fencing/October_NAC_SaltLakeCityUT_2025_H264")            # input videos folder
# VIDEOS_DIR = Path("/media/yin/Seagate8T1/youtube_fencing/fencing_list_001_H264")            # input videos folder  October_NAC_SaltLakeCityUT_2025_H264
VIDEOS_DIR = Path("/media/yin/Seagate8T1/youtube_fencing/Fencing_MostlyFoilSuperPlaylist_h264")  # input videos folder
OUT_DIR = VIDEOS_DIR / Path("sample_images")  # per-video output root (used only if SAVE_ALL_FINAL_TO_ONE_FOLDER=False)

SAVE_ALL_FINAL_TO_ONE_FOLDER = False
GLOBAL_FINAL_DIR = VIDEOS_DIR / Path("final_all")  # where all final images go (when above is True)

# Model
MODEL_PATH = "yolov8n-pose.pt"
DEVICE = 0  # 0 for CUDA:0, or "cpu"
IMGSZ = 640
BOX_CONF = 0.25
MAX_PERSONS = 10  # number of top-conf people to consider

# Sampling
INTERVAL_SEC = 2.0  # constant sampling interval in seconds (increase for speed)
MAX_CANDIDATES_PER_VIDEO = 1000  # cap sampled frames per video

# Batch inference
BATCH_SIZE = 8  # increase (8/16/32) for better throughput if GPU memory allows

# Saving
MIN_SAVE = 1
MAX_SAVE = 100
JPEG_QUALITY_FINAL = 99

# Correlation anti-duplicate (vs previous SAVED final)
ENABLE_CORR_FILTER = True
MAX_CORR_WITH_PREV_SAVED = 0.3  # higher = more similar; skip if >= this
CORR_W, CORR_H = 96, 54

# Debug overlays (auto disabled in global-output mode)
DEBUG_SAVE = True
DEBUG_SAVE_ONLY_WHEN_FAIL = False
DEBUG_MAX_DEBUG_PER_VIDEO = 800
DEBUG_JPEG_QUALITY = 92

if SAVE_ALL_FINAL_TO_ONE_FOLDER:
    DEBUG_SAVE = False

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


# =========================
# Utilities
# =========================
def list_videos(root: Path) -> list[Path]:
    vids: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            vids.append(p)
    return vids


def _prep_for_corr(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (CORR_W, CORR_H), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32).reshape(-1)


def frame_corr(a_vec: np.ndarray, b_vec: np.ndarray) -> float:
    a = a_vec - a_vec.mean()
    b = b_vec - b_vec.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(a, b) / denom)


# =========================
# Pose / fencer logic (from YOLO result)
# =========================
def make_pose(box_xyxy: np.ndarray, kxy: np.ndarray, kcf: np.ndarray, score: float, idx: int) -> dict:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    kpts: list[float] = []
    for i in range(17):
        kpts.extend([float(kxy[i, 0]), float(kxy[i, 1]), float(kcf[i])])
    return {"box": [x1, y1, x2 - x1, y2 - y1], "keypoints": kpts, "score": float(score), "idx": int(idx)}


def size_ok(pose: dict, min_h: float = 80.0) -> bool:
    return float(pose["box"][3]) >= float(min_h)


# Simple skeleton edges for debug visualization
COCO17_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def _draw_pose_overlay(
    frame_bgr: np.ndarray,
    poses: list[dict],
    is_fencer: list[bool],
    fencer_reasons: list[str] | None,
    ok_frame: bool,
    reason: str,
    corr_val: float | None = None,
) -> np.ndarray:
    img = frame_bgr.copy()

    header = f"OK={ok_frame}"
    if corr_val is not None:
        header += f" | corr={corr_val:.4f}"
    if reason:
        header += f" | {reason}"

    cv2.putText(img, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    for idx, (p, f_ok) in enumerate(zip(poses, is_fencer)):
        x, y, bw, bh = p["box"]
        x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        rsn = ""
        if fencer_reasons is not None and idx < len(fencer_reasons):
            rsn = str(fencer_reasons[idx] or "")
        tag = f"fencer={int(bool(f_ok))}"
        if not f_ok and rsn:
            tag += f" | {rsn}"
        cv2.putText(img, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        k = p["keypoints"]
        pts = []
        for i in range(17):
            xi = float(k[i * 3 + 0])
            yi = float(k[i * 3 + 1])
            ci = float(k[i * 3 + 2])
            pts.append((xi, yi, ci))

        for xi, yi, ci in pts:
            if ci > 0.0:
                cv2.circle(img, (int(xi), int(yi)), 3, (255, 255, 255), -1)

        for a, b in COCO17_EDGES:
            xa, ya, ca = pts[a]
            xb, yb, cb = pts[b]
            if ca > 0.0 and cb > 0.0:
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (255, 255, 255), 2)

    return img


def evaluate_fencing_from_result(frame_bgr: np.ndarray, res) -> tuple[bool, dict]:
    """Return (good, debug_dict) for a single frame, given a YOLO Result."""
    dbg: dict = {"reason": "", "poses": [], "is_fencer": []}

    if res is None or res.boxes is None or res.keypoints is None:
        dbg["reason"] = "no_boxes_or_keypoints"
        return False, dbg

    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    kxy = res.keypoints.xy.cpu().numpy()
    kcf = res.keypoints.conf.cpu().numpy()

    if len(boxes) < 2:
        dbg["reason"] = "persons_lt_2"
        return False, dbg

    order = np.argsort(-confs)[:MAX_PERSONS]

    poses: list[dict] = []
    for rank, i in enumerate(order):
        p = make_pose(boxes[i], kxy[i], kcf[i], confs[i], rank)
        if size_ok(p):
            poses.append(p)

    dbg["poses"] = poses
    if len(poses) < 2:
        dbg["reason"] = "size_filter_left_lt_2"
        return False, dbg

    results = [checkForFencer(p, frame_bgr, return_reason=True) for p in poses]
    flags = [ok for (ok, _) in results]
    reasons = [rsn for (_, rsn) in results]
    dbg["is_fencer"] = flags
    dbg["fencer_reasons"] = reasons

    if sum(flags) < 2:
        fail_reasons = [r for (ok, r) in results if not ok]
        if fail_reasons:
            from collections import Counter

            c = Counter(fail_reasons)
            summary = ", ".join([f"{k}×{v}" for k, v in c.most_common()])
            dbg["reason"] = f"fencers_lt_2 ({sum(flags)}/{len(flags)}) | {summary}"
        else:
            dbg["reason"] = f"fencers_lt_2 ({sum(flags)}/{len(flags)})"
        return False, dbg

    dbg["reason"] = "pass"
    return True, dbg


# =========================
# Main
# =========================
def run() -> None:
    videos = list_videos(VIDEOS_DIR)
    total_videos = len(videos)

    if not videos:
        print(f"[WARN] No videos found under: {VIDEOS_DIR.resolve()}")
        return

    if SAVE_ALL_FINAL_TO_ONE_FOLDER:
        if GLOBAL_FINAL_DIR.exists():
            while True:
                print(f"'{GLOBAL_FINAL_DIR}' exists. Please select delete or keep the contents, or abort the program.")
                print("[d] Delete – delete existing sample images")
                print("[k] Keep – keep images")
                print("[a] Abort – exit program")

                choice = input("Choose [d/k/a]: ").strip().lower()

                if choice == "d":
                    print("Delete existing sample images ...")
                    shutil.rmtree(GLOBAL_FINAL_DIR)
                    GLOBAL_FINAL_DIR.mkdir(parents=True, exist_ok=True)
                    break
                elif choice == "k":
                    print("Keep existing sample images.")
                    break

                elif choice == "a":
                    print("Aborted.")
                    exit(1)
                else:
                    print("Invalid choice. Please try again.\n")
        else:
            GLOBAL_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    else:
        if OUT_DIR.exists():
            print(f"WARNING: This action will overwrite existing files in {OUT_DIR}.")
            input("Press Enter to continue, or Ctrl+C to abort...")
            shutil.rmtree(OUT_DIR)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    vid_pre = time.perf_counter()
    for vid_idx, vid in enumerate(videos, start=1):
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"[WARN] Cannot open: {vid}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f"[WARN] {vid.name}: cannot read FPS; skipping")
            cap.release()
            continue

        frames_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = float(frames_total / fps) if frames_total > 0 else 0.0
        if duration <= 0:
            print(f"[WARN] {vid.name}: non-positive duration; skipping")
            cap.release()
            continue

        # Cap sampling count by adapting interval when needed
        interval_sec = float(INTERVAL_SEC)
        est_samples = duration / max(interval_sec, 1e-6)
        if est_samples > MAX_CANDIDATES_PER_VIDEO:
            interval_sec = duration / MAX_CANDIDATES_PER_VIDEO

        stride = max(1, round(fps * interval_sec))

        # Output location
        if SAVE_ALL_FINAL_TO_ONE_FOLDER:
            save_dir = GLOBAL_FINAL_DIR
        else:
            save_dir = OUT_DIR / vid.stem
            save_dir.mkdir(parents=True, exist_ok=True)

        # Debug setup (only when enabled AND not global mode)
        if DEBUG_SAVE:
            debug_dir = save_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_written = 0

        saved = 0
        processed = 0
        prev_saved_vec = None

        # Batch buffers
        batch_frames: list[np.ndarray] = []
        batch_meta: list[tuple[int, float]] = []  # (frame_idx, t_sec)

        frame_idx = -1

        def flush_batch() -> None:
            nonlocal saved, processed, prev_saved_vec, debug_written, batch_frames, batch_meta

            if not batch_frames:
                return

            # YOLO batch inference (returns list of Result in same order)
            results = model.predict(
                source=batch_frames,
                imgsz=IMGSZ,
                conf=BOX_CONF,
                device=DEVICE,
                half=True,
                verbose=False,
            )

            for frame_bgr, (fidx, t_sec), res in zip(batch_frames, batch_meta, results):
                if saved >= MAX_SAVE:
                    break

                good, dbg = evaluate_fencing_from_result(frame_bgr, res)

                # Correlation filter vs previous SAVED final
                corr_val = None
                if good and ENABLE_CORR_FILTER:
                    curr_vec = _prep_for_corr(frame_bgr)
                    if prev_saved_vec is not None:
                        corr_val = frame_corr(prev_saved_vec, curr_vec)
                        if corr_val >= MAX_CORR_WITH_PREV_SAVED:
                            good = False
                            dbg["reason"] = "too_similar_prev_saved"

                # Debug write (reason is on-image)
                if DEBUG_SAVE and debug_written < DEBUG_MAX_DEBUG_PER_VIDEO:
                    should_write = (not DEBUG_SAVE_ONLY_WHEN_FAIL) or (not good)
                    if should_write:
                        overlay = _draw_pose_overlay(
                            frame_bgr=frame_bgr,
                            poses=dbg.get("poses", []),
                            is_fencer=dbg.get("is_fencer", []),
                            fencer_reasons=dbg.get("fencer_reasons", None),
                            ok_frame=good,
                            reason=dbg.get("reason", ""),
                            corr_val=corr_val,
                        )
                        dbg_path = debug_dir / f"cand{fidx:06d}_t{t_sec:.2f}_{'OK' if good else 'NO'}.jpg"
                        okw = cv2.imwrite(
                            str(dbg_path),
                            overlay,
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(DEBUG_JPEG_QUALITY)],
                        )
                        if not okw:
                            print(f"[WARN] Debug imwrite failed: {dbg_path}")
                        debug_written += 1

                if good:
                    if SAVE_ALL_FINAL_TO_ONE_FOLDER:
                        out_name = f"{vid.stem}_t{t_sec:.2f}_s{saved:03d}.jpg"
                    else:
                        out_name = f"{saved:03d}_t{t_sec:.2f}.jpg"
                    out_path = save_dir / out_name

                    okf = cv2.imwrite(
                        str(out_path),
                        frame_bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY_FINAL)],
                    )
                    if not okf:
                        print(f"[WARN] Final imwrite failed: {out_path}")

                    saved += 1
                    if ENABLE_CORR_FILTER:
                        prev_saved_vec = _prep_for_corr(frame_bgr)

                processed += 1

            batch_frames = []
            batch_meta = []

        # Read video sequentially and sample with stride
        while saved < MAX_SAVE:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_idx += 1
            if frame_idx % stride != 0:
                continue

            t_sec = frame_idx / fps
            batch_frames.append(frame)
            batch_meta.append((frame_idx, t_sec))

            if len(batch_frames) >= BATCH_SIZE:
                flush_batch()

        # Flush remaining
        flush_batch()

        cap.release()

        vid_end = time.perf_counter()
        print(
            f"[VIDEO {vid_idx}/{total_videos}] Done in {vid_end - vid_pre:.0f}s "
            f"| saved={saved}/{processed} | {vid.name}"
        )
        vid_pre = vid_end

        # If per-video mode, optionally remove empty folders
        if (not SAVE_ALL_FINAL_TO_ONE_FOLDER) and saved < MIN_SAVE:
            if not DEBUG_SAVE:
                try:
                    for p in save_dir.glob("*"):
                        p.unlink()
                    save_dir.rmdir()
                except Exception:
                    pass

    out_msg = str(GLOBAL_FINAL_DIR if SAVE_ALL_FINAL_TO_ONE_FOLDER else OUT_DIR)
    print(f"Done. Output: {out_msg}")


if __name__ == "__main__":
    start = time.perf_counter()
    run()
    end = time.perf_counter()
    print(f"Runtime: {end - start:.6f} seconds")
