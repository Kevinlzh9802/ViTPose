"""
vitpose_to_dataframe.py

Convert ViTPose keypoints into a per-frame pandas dataframe that matches the
input shape expected by demo/dante_transfer.py.

Input layout
------------
The script mirrors demo/vitpose_to_world_coords.py and scans:

    <results_dir>/
        camXX*/
            vitpose_keypoints.json

Expected JSON schema:
    {
        "annotations": {
            "<frame_id>": {
                "keypoints": {
                    "<person_id>": [[x, y, score], ... 17 COCO keypoints ...]
                }
            }
        }
    }

Output
------
For each camera folder, the script writes a pickled dataframe. Each dataframe
row represents one timestamp/frame and contains:

    - timestamp: frame id
    - spaceFeat: dict with keys {head, shoulder, hip, foot}
    - groups: empty list placeholder
    - group_ids: empty list placeholder

Each spaceFeat entry is an (n_people, 4) object array containing:
    [person_id, x, y, orientation]

At this stage x/y are image-plane coordinates derived directly from the 2D
keypoints. When camera intrinsics/extrinsics are available, this script can be
updated to swap those values for calibrated world coordinates while preserving
the same dataframe contract.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


CONF_THRESHOLD = 0.0

# Orientation pairs: (left_idx, right_idx)
ORIENTATION_PAIRS = {
    "head": (1, 2),       # left_eye -> right_eye
    "shoulder": (5, 6),   # left_shoulder -> right_shoulder
    "hip": (11, 12),      # left_hip -> right_hip
    "foot": (15, 16),     # left_ankle -> right_ankle
}

# Fallback keypoints used to recover an x/y center when the main pair is missing.
SEGMENT_FALLBACKS = {
    "head": [0, 1, 2, 3, 4],
    "shoulder": [5, 6],
    "hip": [11, 12],
    "foot": [15, 16],
}


def cam_number_from_name(cam_name: str) -> str | None:
    """Extract camera number from a folder name, e.g. 'cam06_batch01' -> '06'."""
    import re

    match = re.search(r"cam(\d+)", cam_name)
    return match.group(1) if match else None


def numeric_sort_key(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def orientation_from_pair(left_xy, right_xy) -> float | None:
    """Heading angle in the image plane from a left/right keypoint pair."""
    if left_xy is None or right_xy is None:
        return None

    dx = right_xy[0] - left_xy[0]
    dy = right_xy[1] - left_xy[1]
    if dx == 0.0 and dy == 0.0:
        return None

    # 90 deg CCW: (dx, dy) -> (-dy, dx)
    return math.atan2(dx, -dy)


def valid_xy(raw_kps: list, kp_idx: int, conf_thresh: float) -> tuple[float, float] | None:
    """Return (x, y) for a keypoint if it passes the confidence threshold."""
    kp = raw_kps[kp_idx]
    if kp[2] < conf_thresh:
        return None
    return float(kp[0]), float(kp[1])


def segment_xy_and_orientation(raw_kps: list, segment_name: str, conf_thresh: float) -> tuple[float, float, float]:
    """
    Build an image-plane proxy for a segment.

    x/y use the midpoint of the left/right pair when both are visible. When the
    pair is incomplete, the position falls back to the mean of visible fallback
    keypoints. Orientation is defined only when both paired keypoints are
    visible; otherwise NaN is stored.
    """
    left_idx, right_idx = ORIENTATION_PAIRS[segment_name]
    left_xy = valid_xy(raw_kps, left_idx, conf_thresh)
    right_xy = valid_xy(raw_kps, right_idx, conf_thresh)

    if left_xy is not None and right_xy is not None:
        x = (left_xy[0] + right_xy[0]) / 2.0
        y = (left_xy[1] + right_xy[1]) / 2.0
        theta = orientation_from_pair(left_xy, right_xy)
        return x, y, float(theta) if theta is not None else math.nan

    fallback_points = []
    for kp_idx in SEGMENT_FALLBACKS[segment_name]:
        xy = valid_xy(raw_kps, kp_idx, conf_thresh)
        if xy is not None:
            fallback_points.append(xy)

    if fallback_points:
        x = float(np.mean([pt[0] for pt in fallback_points]))
        y = float(np.mean([pt[1] for pt in fallback_points]))
        return x, y, math.nan

    return math.nan, math.nan, math.nan


def process_person_keypoints(raw_kps: list, person_id: str, conf_thresh: float = CONF_THRESHOLD) -> dict[str, list]:
    """
    Convert one person's 17 COCO keypoints into DANTE-style segment rows.

    Returns a dict mapping each segment name to:
        [person_id, x, y, orientation]
    """
    if len(raw_kps) != 17:
        raise ValueError("Expected 17 COCO keypoints, got {}".format(len(raw_kps)))

    segment_rows = {}
    for segment_name in ORIENTATION_PAIRS:
        x, y, theta = segment_xy_and_orientation(raw_kps, segment_name, conf_thresh)
        segment_rows[segment_name] = [str(person_id), x, y, theta]
    return segment_rows


def process_vitpose_json(input_path: str | Path, conf_thresh: float = CONF_THRESHOLD) -> pd.DataFrame:
    """
    Process a single vitpose_keypoints.json into a dataframe indexed by frame id.

    Columns:
        - timestamp
        - spaceFeat
        - groups
        - group_ids
    """
    input_path = Path(input_path)
    with open(input_path) as f:
        data = json.load(f)

    records = []
    annotations = data.get("annotations", {})
    for frame_id in sorted(annotations.keys(), key=numeric_sort_key):
        frame_data = annotations[frame_id]
        keypoints_by_track = frame_data.get("keypoints", {})
        sorted_track_ids = sorted(keypoints_by_track.keys(), key=numeric_sort_key)

        segment_rows = {segment_name: [] for segment_name in ORIENTATION_PAIRS}
        for track_id in sorted_track_ids:
            person_segments = process_person_keypoints(
                keypoints_by_track[track_id],
                person_id=str(track_id),
                conf_thresh=conf_thresh,
            )
            for segment_name, row in person_segments.items():
                segment_rows[segment_name].append(row)

        spacefeat = {}
        for segment_name, rows in segment_rows.items():
            if rows:
                spacefeat[segment_name] = np.array(rows, dtype=object)
            else:
                spacefeat[segment_name] = np.empty((0, 4), dtype=object)

        records.append(
            {
                "timestamp": str(frame_id),
                "spaceFeat": spacefeat,
                "groups": [],
                "group_ids": [],
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df.index = df["timestamp"]
        df.index.name = "timestamp"

    return df


def process_results_directory(
    results_dir: str | Path,
    output_dir: str | Path | None = None,
    output_name: str = "vitpose_dataframe.pkl",
    conf_thresh: float = CONF_THRESHOLD,
    intrinsics_dir: str | Path | None = None,
    extrinsics_dir: str | Path | None = None,
) -> None:
    """
    Walk every cam*/vitpose_keypoints.json under results_dir and write a
    dataframe alongside the input, or under output_dir/<cam_name>/.
    """
    results_dir = Path(results_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if intrinsics_dir is not None or extrinsics_dir is not None:
        print("Calibration directories were provided but are not used yet; exporting image-plane x/y coordinates.")
        if intrinsics_dir is not None:
            print(f"  intrinsics_dir={intrinsics_dir}")
        if extrinsics_dir is not None:
            print(f"  extrinsics_dir={extrinsics_dir}")

    json_files = sorted(results_dir.glob("*/vitpose_keypoints.json"))
    print(f"Found {len(json_files)} result files under {results_dir}")

    for json_file in json_files:
        cam_name = json_file.parent.name
        cam_number = cam_number_from_name(cam_name)
        if cam_number is None:
            print(f"  Skipping {cam_name}: could not parse camera number")
            continue

        print(f"  Processing {json_file.relative_to(results_dir)}")
        df = process_vitpose_json(json_file, conf_thresh=conf_thresh)

        if output_dir is not None:
            out_path = output_dir / cam_name / output_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = json_file.parent / output_name

        df.to_pickle(out_path)
        print(f"    -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ViTPose keypoints into DANTE-style dataframe pickles."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=r"c:\Users\sotir\Desktop\Uni\Master\Thesis\vitpose_results",
        help="Root directory containing cam*/vitpose_keypoints.json files",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional root directory for output dataframes. Defaults to each input folder.",
    )
    parser.add_argument(
        "--output_name",
        default="vitpose_dataframe.pkl",
        help="Filename for each per-camera dataframe pickle.",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=CONF_THRESHOLD,
        help="Minimum keypoint confidence used when building segment positions.",
    )
    parser.add_argument(
        "--intrinsics_dir",
        default=None,
        help="Reserved for future calibrated conversion. Currently unused.",
    )
    parser.add_argument(
        "--extrinsics_dir",
        default=None,
        help="Reserved for future calibrated conversion. Currently unused.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_results_directory(
        args.results_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
        conf_thresh=args.conf_thresh,
        intrinsics_dir=args.intrinsics_dir,
        extrinsics_dir=args.extrinsics_dir,
    )
