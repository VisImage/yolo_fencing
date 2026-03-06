#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import time
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

# ---------- CONFIG ----------
WORK_DIR = Path(r"/path/to/your/images")  # CHANGE THIS

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RESIZE_FOR_COMPARE = (256, 256)
TOP_K = 50

WINDOW_NAME = "Compare Pairs (h=help, y/u/b delete, s skip, q quit)"
MAX_DISPLAY_SIZE = (1400, 800)

SAFE_MOVE_TO_SUBFOLDER = True
DELETED_SUBFOLDER_NAME = "_deleted"
PROGRESS_EVERY_SECONDS = 2.0

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


HELP_TEXT = """
Interactive image comparison (grayscale Pearson correlation)

KEYS (interactive window):
  h  : toggle help overlay
  y  : delete/move LEFT image
  u  : delete/move RIGHT image
  b  : delete/move BOTH images
  s  : skip this pair
  q  : quit interactive mode

DELETE MODE:
  By default images are MOVED into "_deleted/" under WORK_DIR.
  Set SAFE_MOVE_TO_SUBFOLDER = False for permanent deletion.

WORKFLOW:
  1) Script computes Pearson correlation for all image pairs (O(N^2))
  2) Top-K most similar pairs are shown
  3) You decide what to delete interactively
""".strip()


# ---------- CORE ----------
def list_images(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def load_gray_vector(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    return img.reshape(-1)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _fmt_time(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    return f"{sec / 3600:.1f}h"


def build_pairs_gray(folder: Path, top_k: int):
    paths = list_images(folder)
    n = len(paths)
    if n < 2:
        return []

    print(f"[INFO] Found {n} images. Extracting grayscale vectors...")
    t0 = time.time()

    feats = {}
    it = tqdm(paths, desc="Features", unit="img") if tqdm else paths
    for p in it:
        try:
            feats[p] = load_gray_vector(p, RESIZE_FOR_COMPARE)
        except Exception as e:
            print(f"[SKIP] {p}: {e}")

    items = list(feats.items())
    n2 = len(items)
    if n2 < 2:
        return []

    print(f"[INFO] Feature extraction done in {_fmt_time(time.time() - t0)}")

    total_pairs = n2 * (n2 - 1) // 2
    print(f"[INFO] Comparing {total_pairs:,} pairs (Pearson correlation)...")

    results = []

    if tqdm:
        for (p1, v1), (p2, v2) in tqdm(combinations(items, 2), total=total_pairs, unit="pair", desc="Pairs"):
            results.append((pearson_corr(v1, v2), p1, p2))
    else:
        start = time.time()
        last = start
        done = 0
        for (p1, v1), (p2, v2) in combinations(items, 2):
            results.append((pearson_corr(v1, v2), p1, p2))
            done += 1
            now = time.time()
            if now - last >= PROGRESS_EVERY_SECONDS:
                rate = done / (now - start)
                remain = (total_pairs - done) / rate if rate > 0 else float("inf")
                print(f"[PROGRESS] {done:,}/{total_pairs:,} ({done / total_pairs * 100:.1f}%) ETA={_fmt_time(remain)}")
                last = now

    results.sort(key=lambda x: x[0], reverse=True)
    return results[: max(top_k, 1)]


# ---------- DISPLAY ----------
def read_for_display(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image for display: {path}")
    return img


def resize_to_fit(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def pad_to_same_height(left: np.ndarray, right: np.ndarray):
    h = max(left.shape[0], right.shape[0])
    if left.shape[0] < h:
        left = cv2.copyMakeBorder(left, 0, h - left.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    if right.shape[0] < h:
        right = cv2.copyMakeBorder(right, 0, h - right.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    return left, right


def overlay_help(img: np.ndarray) -> np.ndarray:
    lines = [
        "Help (press 'h' to hide/show):",
        " y = delete/move LEFT image",
        " u = delete/move RIGHT image",
        " b = delete/move BOTH images",
        " s = skip",
        " q = quit",
        f" Delete mode: {'MOVE to _deleted/' if SAFE_MOVE_TO_SUBFOLDER else 'PERMANENT DELETE'}",
    ]
    out = img.copy()
    y = 30
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        y += 26
    return out


def safe_delete(path: Path, deleted_dir: Path):
    if not path.exists():
        return
    if SAFE_MOVE_TO_SUBFOLDER:
        deleted_dir.mkdir(parents=True, exist_ok=True)
        target = deleted_dir / path.name
        if target.exists():
            stem, suf = target.stem, target.suffix
            k = 1
            while True:
                alt = deleted_dir / f"{stem}__{k}{suf}"
                if not alt.exists():
                    target = alt
                    break
                k += 1
        shutil.move(str(path), str(target))
    else:
        path.unlink()


def interactive_review(pairs, folder: Path):
    deleted_dir = folder / DELETED_SUBFOLDER_NAME
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    show_help = True

    for score, p1, p2 in pairs:
        if not p1.exists() or not p2.exists():
            continue

        while True:
            img1 = resize_to_fit(read_for_display(p1), MAX_DISPLAY_SIZE[0] // 2, MAX_DISPLAY_SIZE[1])
            img2 = resize_to_fit(read_for_display(p2), MAX_DISPLAY_SIZE[0] // 2, MAX_DISPLAY_SIZE[1])
            img1, img2 = pad_to_same_height(img1, img2)

            joined = np.hstack([img1, img2])

            cv2.putText(
                joined,
                f"Pearson corr: {score:.4f}",
                (10, joined.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if show_help:
                joined = overlay_help(joined)

            cv2.imshow(WINDOW_NAME, joined)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("h"):
                show_help = not show_help
                continue
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            if key == ord("s"):
                break
            if key == ord("y"):
                safe_delete(p1, deleted_dir)
                break
            if key == ord("u"):
                safe_delete(p2, deleted_dir)
                break
            if key == ord("b"):
                safe_delete(p1, deleted_dir)
                safe_delete(p2, deleted_dir)
                break
            else:
                break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help-only", action="store_true", help="Show help text and exit")
    args, _ = parser.parse_known_args()

    if args.help_only:
        print(HELP_TEXT)
        return

    if not WORK_DIR.exists():
        raise SystemExit(f"Folder not found: {WORK_DIR}")

    print(f"[INFO] Working folder: {WORK_DIR}")
    pairs = build_pairs_gray(WORK_DIR, TOP_K)
    print(f"[INFO] Reviewing {len(pairs)} pairs")
    interactive_review(pairs, WORK_DIR)


if __name__ == "__main__":
    main()
