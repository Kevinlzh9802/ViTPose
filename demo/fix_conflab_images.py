"""
fix_conflab_images.py

One-time repair utility for conflab batches where images.zip was extracted
directly into bbox_kp/<batch>/ instead of bbox_kp/<batch>/images/.

For each affected batch this script:
  1. Creates bbox_kp/<batch>/images/ if it does not already exist.
  2. Moves every *.jpg found directly in bbox_kp/<batch>/ into images/,
     renaming each file to an 8-digit zero-padded integer (e.g. 00000100.jpg).
  3. Skips the batch silently if images/ already exists and contains jpg files.

Usage
-----
# Dry run first (no files moved):
python demo/fix_conflab_images.py /path/to/bbox_kp --dry_run

# Fix all batches:
python demo/fix_conflab_images.py /path/to/bbox_kp

# Fix specific batches:
python demo/fix_conflab_images.py /path/to/bbox_kp --batch=228,431
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _fix_batch(batch_dir: Path, dry_run: bool = False) -> int:
    """Move misplaced *.jpg from *batch_dir* root into *batch_dir*/images/.

    Returns the number of files moved (or that would be moved in dry-run mode).
    """
    images_dir = batch_dir / "images"

    if images_dir.is_dir() and any(images_dir.glob("*.jpg")):
        print(f"  [{batch_dir.name}] images/ already populated, skipping")
        return 0

    jpg_files = sorted(batch_dir.glob("*.jpg"))
    if not jpg_files:
        print(f"  [{batch_dir.name}] no .jpg files in batch root, nothing to do")
        return 0

    print(f"  [{batch_dir.name}] moving {len(jpg_files)} file(s) → images/")
    if not dry_run:
        images_dir.mkdir(exist_ok=True)

    moved = 0
    for src in jpg_files:
        try:
            frame_num = int(src.stem)
            dst_name = f"{frame_num:08d}.jpg"
        except ValueError:
            dst_name = src.name  # keep original name if stem is not an integer
        dst = images_dir / dst_name
        if not dry_run:
            shutil.move(str(src), str(dst))
        else:
            print(f"    {src.name} -> images/{dst_name}")
        moved += 1

    return moved


def _resolve_batch_dirs(root: Path, batches_raw: str) -> list[Path]:
    if batches_raw.strip().lower() == "all":
        return sorted(d for d in root.iterdir() if d.is_dir())
    dirs = []
    for b in batches_raw.split(","):
        b = b.strip()
        if not b:
            continue
        d = root / b
        if not d.is_dir():
            print(f"  Warning: batch dir not found: {d}")
        else:
            dirs.append(d)
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Move conflab frame jpegs from bbox_kp/<batch>/*.jpg into "
            "bbox_kp/<batch>/images/<frame:08d>.jpg."
        )
    )
    parser.add_argument(
        "bbox_kp_root",
        help="Root directory containing per-batch subfolders (e.g. .../conflab/bbox_kp).",
    )
    parser.add_argument(
        "--batch",
        default="all",
        metavar="BATCHES",
        help="Comma-separated batch numbers (e.g. 228,431) or 'all' (default).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would happen without moving any files.",
    )
    args = parser.parse_args()

    root = Path(args.bbox_kp_root)
    if not root.is_dir():
        raise SystemExit(f"Error: directory not found: {root}")

    batch_dirs = _resolve_batch_dirs(root, args.batch)
    if not batch_dirs:
        raise SystemExit("No batch directories found.")

    if args.dry_run:
        print("DRY RUN — no files will be moved\n")

    total = 0
    for batch_dir in batch_dirs:
        total += _fix_batch(batch_dir, dry_run=args.dry_run)

    verb = "would be moved" if args.dry_run else "moved"
    print(f"\nDone. Total files {verb}: {total}")


if __name__ == "__main__":
    main()
