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

The saved x/y values are world-floor coordinates obtained by back-projecting
the 2D keypoints with per-camera intrinsic/extrinsic calibration loaded from:

    <camera_params_root>/
        camera_XX/
            intrinsic.json
            extrinsic.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


BODY_HEIGHT = 1.7
CONF_THRESHOLD = 0.0
CAMERA_PARAMS_ROOT = Path(
    "/tudelft.net/staff-umbrella/neon/ingroup_dataset/processed_data/"
    "gopro_data/camera_calibration/camera_params"
)

# COCO-17 keypoint heights as a fraction of body height above the floor.
KP_HEIGHT_RATIOS = np.array([
    0.95,   #  0 nose
    0.97,   #  1 left_eye
    0.97,   #  2 right_eye
    0.95,   #  3 left_ear
    0.95,   #  4 right_ear
    0.85,   #  5 left_shoulder
    0.85,   #  6 right_shoulder
    0.68,   #  7 left_elbow
    0.68,   #  8 right_elbow
    0.55,   #  9 left_wrist
    0.55,   # 10 right_wrist
    0.50,   # 11 left_hip
    0.50,   # 12 right_hip
    0.27,   # 13 left_knee
    0.27,   # 14 right_knee
    0.02,   # 15 left_ankle
    0.02,   # 16 right_ankle
], dtype=np.float64)

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

# --- Time-mapping and GT-group constants --------------------------------- #
FPS = 60
FRAMES_PER_BATCH = 18000  # 60 fps × 300 s = 5 min per batch


def batch_number_from_name(cam_name: str) -> int | None:
    """Extract 1-based batch number, e.g. 'cam06_batch03' -> 3."""
    match = re.search(r"batch(\d+)", cam_name)
    return int(match.group(1)) if match else None


def _camera_start_seconds(cam_number: str) -> int | None:
    """Seconds since midnight for the first frame of a camera's batch-1."""
    num = int(cam_number)
    if 6 <= num <= 10:
        return 13 * 3600 + 45 * 60  # 13:45:00
    elif 1 <= num <= 5:
        return 14 * 3600 + 52 * 60  # 14:52:00
    return None


def _frame_to_time_str(
    cam_number: str,
    batch_number: int,
    frame_index: int,
    fps: int = FPS,
) -> str:
    """Map a frame index within a batch to an absolute 'HH:MM:SS;FF' string."""
    start = _camera_start_seconds(cam_number)
    if start is None:
        return ""
    total_frame = start * fps + (batch_number - 1) * FRAMES_PER_BATCH + frame_index
    ff = total_frame % fps
    total_secs = total_frame // fps
    hh = total_secs // 3600
    mm = (total_secs % 3600) // 60
    ss = total_secs % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d};{ff:02d}"


def _normalize_time_key(raw: str) -> str:
    """Normalize 'HH:MM:SS;FF' so the FF field is always two digits."""
    raw = raw.strip()
    if ";" in raw:
        base, ff = raw.rsplit(";", 1)
        return f"{base};{int(ff):02d}"
    return raw


def _load_gt_groups(csv_path: str | Path) -> dict[str, str]:
    """Read a GT groups CSV and return {normalized_time_str: raw_groups_str}."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        print(f"  Warning: GT groups CSV not found: {csv_path}")
        return {}
    df_csv = pd.read_csv(csv_path)
    gt_map: dict[str, str] = {}
    for _, row in df_csv.iterrows():
        t = _normalize_time_key(str(row["time_association"]))
        g = str(row["conversational_groups"]).strip()
        gt_map[t] = g
    return gt_map


def _parse_groups_string(groups_str: str) -> list[set[int]]:
    """Parse '{1,2} {3,4,5,6}' into [set(1,2), set(3,4,5,6)]."""
    if not groups_str or groups_str.lower() == "nan":
        return []
    result: list[set[int]] = []
    for m in re.finditer(r"\{([^}]+)\}", groups_str):
        members = {int(x.strip()) for x in m.group(1).split(",")}
        result.append(members)
    return result


def _gt_csv_name_for_camera(cam_number: str) -> str | None:
    """Return the expected GT CSV filename for a camera, or None."""
    num = int(cam_number)
    if 6 <= num <= 10:
        return "mingle_1_groups.csv"
    elif 1 <= num <= 5:
        return "mingle_2_groups.csv"
    return None


_camera_params_cache: dict[tuple[str, str], dict] = {}


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


def parse_camera_numbers(camera_numbers) -> set[str] | None:
    """
    Normalize camera number input into a zero-padded string set, e.g. {"06","08"}.
    """
    if camera_numbers is None:
        return None

    if isinstance(camera_numbers, str):
        raw_values = [item.strip() for item in camera_numbers.split(",")]
    else:
        raw_values = [str(item).strip() for item in camera_numbers]

    normalized = set()
    for value in raw_values:
        if not value:
            continue
        if value.lower().startswith("cam"):
            value = value[3:]
        normalized.add(value.zfill(2))

    return normalized or None


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


def normalize_distortion_coefficients(coeffs) -> np.ndarray:
    """
    Convert a short distortion vector into an OpenCV-compatible array.

    The provided JSON examples use [k1, k2, k3]. OpenCV's pinhole model expects
    [k1, k2, p1, p2, k3], so tangential terms are padded with zeros.
    """
    dist = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    if dist.size == 3:
        dist = np.array([dist[0], dist[1], 0.0, 0.0, dist[2]], dtype=np.float64)
    elif dist.size not in {4, 5, 8, 12, 14}:
        raise ValueError("Unsupported distortion coefficient length: {}".format(dist.size))
    return dist


def load_camera_params(
    cam_number: str | int,
    camera_params_root: str | Path = CAMERA_PARAMS_ROOT,
) -> dict:
    """Load per-camera intrinsics and extrinsics from camera_XX/*.json."""
    cam_id = str(cam_number).zfill(2)
    cache_key = (str(Path(camera_params_root)), cam_id)
    if cache_key in _camera_params_cache:
        return _camera_params_cache[cache_key]

    camera_dir = Path(camera_params_root) / f"camera_{cam_id}"
    intrinsic_path = camera_dir / "intrinsic.json"
    extrinsic_path = camera_dir / "extrinsic.json"

    with open(intrinsic_path) as f:
        intrinsic_data = json.load(f)
    with open(extrinsic_path) as f:
        extrinsic_data = json.load(f)

    params = {
        "K": np.asarray(intrinsic_data["intrinsic"], dtype=np.float64),
        "D": normalize_distortion_coefficients(intrinsic_data["distortion_coefficients"]),
        "rvec": np.asarray(extrinsic_data["rvec"], dtype=np.float64).reshape(3, 1),
        "tvec": np.asarray(extrinsic_data["tvec"], dtype=np.float64).reshape(3, 1),
    }
    _camera_params_cache[cache_key] = params
    return params


def undistort_points(pts_uv: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Undistort pixel coordinates into normalized camera coordinates."""
    pts = pts_uv.reshape(-1, 1, 2).astype(np.float64)
    undistorted = cv2.undistortPoints(pts, K, D)
    return undistorted.reshape(-1, 2)


def backproject_to_world(xn: float, yn: float, z_kp: float, R: np.ndarray, tvec: np.ndarray) -> tuple[float | None, float | None]:
    """
    Back-project normalized image coordinates to world-floor X/Y at a known Z.
    """
    camera_center_world = -(R.T @ tvec.reshape(3))
    ray_cam = np.array([xn, yn, 1.0], dtype=np.float64)
    ray_world = R.T @ ray_cam

    if abs(ray_world[2]) < 1e-9:
        return None, None

    t = (z_kp - camera_center_world[2]) / ray_world[2]
    x_world = float(camera_center_world[0] + t * ray_world[0])
    y_world = float(camera_center_world[1] + t * ray_world[1])
    return x_world, y_world


def valid_world_xy(kp_world: list, kp_idx: int) -> tuple[float, float] | None:
    """Return (x, y) for a world keypoint if available."""
    kp = kp_world[kp_idx]
    if kp is None:
        return None
    return float(kp[0]), float(kp[1])


def segment_xy_and_orientation(kp_world: list, segment_name: str) -> tuple[float, float, float]:
    """
    Build a world-plane segment descriptor from projected keypoints.

    x/y use the midpoint of the left/right pair when both are visible. When the
    pair is incomplete, the position falls back to the mean of visible fallback
    keypoints. Orientation is defined only when both paired keypoints are
    visible; otherwise NaN is stored.
    """
    left_idx, right_idx = ORIENTATION_PAIRS[segment_name]
    left_xy = valid_world_xy(kp_world, left_idx)
    right_xy = valid_world_xy(kp_world, right_idx)

    if left_xy is not None and right_xy is not None:
        x = (left_xy[0] + right_xy[0]) / 2.0
        y = (left_xy[1] + right_xy[1]) / 2.0
        theta = orientation_from_pair(left_xy, right_xy)
        return x, y, float(theta) if theta is not None else math.nan

    fallback_points = []
    for kp_idx in SEGMENT_FALLBACKS[segment_name]:
        xy = valid_world_xy(kp_world, kp_idx)
        if xy is not None:
            fallback_points.append(xy)

    if fallback_points:
        x = float(np.mean([pt[0] for pt in fallback_points]))
        y = float(np.mean([pt[1] for pt in fallback_points]))
        return x, y, math.nan

    return math.nan, math.nan, math.nan


def project_person_keypoints_to_world(
    raw_kps: list,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    tvec: np.ndarray,
    body_height: float,
    conf_thresh: float,
) -> list:
    """Project one person's 17 COCO keypoints to world coordinates."""
    if len(raw_kps) != 17:
        raise ValueError("Expected 17 COCO keypoints, got {}".format(len(raw_kps)))

    valid_idx = [i for i in range(17) if raw_kps[i][2] >= conf_thresh]
    kp_world = [None] * 17
    if not valid_idx:
        return kp_world

    pts_uv = np.array([[raw_kps[i][0], raw_kps[i][1]] for i in valid_idx], dtype=np.float64)
    norm_xy = undistort_points(pts_uv, K, D)

    for j, kp_idx in enumerate(valid_idx):
        z_kp = body_height * KP_HEIGHT_RATIOS[kp_idx]
        xn, yn = norm_xy[j]
        xw, yw = backproject_to_world(xn, yn, z_kp, R, tvec)
        if xw is not None and yw is not None:
            kp_world[kp_idx] = (xw, yw, z_kp)

    return kp_world


def process_person_keypoints(
    raw_kps: list,
    person_id: str,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    tvec: np.ndarray,
    body_height: float = BODY_HEIGHT,
    conf_thresh: float = CONF_THRESHOLD,
) -> dict[str, list]:
    """
    Convert one person's 17 COCO keypoints into DANTE-style segment rows.

    Returns a dict mapping each segment name to:
        [person_id, x, y, orientation]
    """
    kp_world = project_person_keypoints_to_world(
        raw_kps,
        K=K,
        D=D,
        R=R,
        tvec=tvec,
        body_height=body_height,
        conf_thresh=conf_thresh,
    )

    segment_rows = {}
    for segment_name in ORIENTATION_PAIRS:
        x, y, theta = segment_xy_and_orientation(kp_world, segment_name)
        segment_rows[segment_name] = [str(person_id), x, y, theta]
    return segment_rows


def process_vitpose_json(
    input_path: str | Path,
    cam_number: str | int,
    camera_params_root: str | Path = CAMERA_PARAMS_ROOT,
    body_height: float = BODY_HEIGHT,
    conf_thresh: float = CONF_THRESHOLD,
    batch_number: int | None = None,
    gt_groups: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Process a single vitpose_keypoints.json into a dataframe indexed by frame id.

    Columns:
        - timestamp
        - time        (HH:MM:SS;FF wall-clock time, empty if batch info unavailable)
        - spaceFeat
        - groups      (list of sets from GT CSV, empty if time not in CSV)
        - group_ids
    """
    input_path = Path(input_path)
    with open(input_path) as f:
        data = json.load(f)

    camera_params = load_camera_params(cam_number, camera_params_root=camera_params_root)
    K = camera_params["K"]
    D = camera_params["D"]
    rvec = camera_params["rvec"]
    tvec = camera_params["tvec"]
    R, _ = cv2.Rodrigues(rvec)

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
                K=K,
                D=D,
                R=R,
                tvec=tvec,
                body_height=body_height,
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

        # Compute wall-clock time and look up GT groups
        time_str = ""
        gt_group: list[set[int]] = []
        if batch_number is not None:
            try:
                fidx = int(frame_id)
            except (ValueError, TypeError):
                fidx = None
            if fidx is not None:
                time_str = _frame_to_time_str(
                    str(cam_number).zfill(2), batch_number, fidx
                )
                if time_str and gt_groups:
                    raw = gt_groups.get(time_str, "")
                    gt_group = _parse_groups_string(raw)

        records.append(
            {
                "timestamp": str(frame_id),
                "time": time_str,
                "spaceFeat": spacefeat,
                "groups": gt_group,
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
    camera_params_root: str | Path = CAMERA_PARAMS_ROOT,
    body_height: float = BODY_HEIGHT,
    camera_numbers=None,
    gt_groups_root: str | Path | None = None,
    plot_dir: str | Path | None = None,
) -> None:
    """
    Walk every cam*/vitpose_keypoints.json under results_dir and write a
    dataframe alongside the input, or under output_dir/<cam_name>/.
    """
    results_dir = Path(results_dir)
    selected_camera_numbers = parse_camera_numbers(camera_numbers)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load GT group CSVs keyed by csv filename
    _gt_csv_cache: dict[str, dict[str, str]] = {}
    if gt_groups_root is not None:
        gt_groups_root = Path(gt_groups_root)

    json_files = sorted(results_dir.glob("*/vitpose_keypoints.json"))
    print(f"Found {len(json_files)} result files under {results_dir}")

    for json_file in json_files:
        cam_name = json_file.parent.name
        cam_number = cam_number_from_name(cam_name)
        if cam_number is None:
            print(f"  Skipping {cam_name}: could not parse camera number")
            continue
        cam_number = str(cam_number).zfill(2)
        if selected_camera_numbers is not None and cam_number not in selected_camera_numbers:
            print(f"  Skipping {cam_name}: cam{cam_number} not in requested set")
            continue

        batch_num = batch_number_from_name(cam_name)

        # Load GT groups CSV for this camera (cached per csv file)
        gt_groups: dict[str, str] | None = None
        if gt_groups_root is not None:
            csv_name = _gt_csv_name_for_camera(cam_number)
            if csv_name is not None:
                if csv_name not in _gt_csv_cache:
                    _gt_csv_cache[csv_name] = _load_gt_groups(
                        gt_groups_root / csv_name
                    )
                gt_groups = _gt_csv_cache[csv_name]

        print(f"  Processing {json_file.relative_to(results_dir)}")
        df = process_vitpose_json(
            json_file,
            cam_number=cam_number,
            camera_params_root=camera_params_root,
            body_height=body_height,
            conf_thresh=conf_thresh,
            batch_number=batch_num,
            gt_groups=gt_groups,
        )

        if output_dir is not None:
            out_path = output_dir / cam_name / output_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = json_file.parent / output_name

        df.to_pickle(out_path)
        print(f"    -> {out_path}")

        # Plot position/orientation every 60 seconds (3600 frames at 60 fps)
        if plot_dir is not None:
            from demo.plot_person import plot_dataframe_positions

            cam_plot_dir = Path(plot_dir) / cam_name
            plot_dataframe_positions(
                df,
                source_tag=cam_name,
                output_dir=cam_plot_dir,
                frame_interval=FPS * 60,  # every 60 seconds
            )


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
        help="Minimum keypoint confidence used when projecting keypoints.",
    )
    parser.add_argument(
        "--body_height",
        type=float,
        default=BODY_HEIGHT,
        help="Assumed body height in meters for back-projection.",
    )
    parser.add_argument(
        "--camera_params_root",
        default=str(CAMERA_PARAMS_ROOT),
        help="Root containing camera_XX/intrinsic.json and extrinsic.json.",
    )
    parser.add_argument(
        "--camera_numbers",
        default=None,
        help="Optional comma-separated camera numbers to process, e.g. '06,08,10'.",
    )
    parser.add_argument(
        "--gt_groups_root",
        default=None,
        help=(
            "Directory containing mingle_1_groups.csv (cam 6-10) and/or "
            "mingle_2_groups.csv (cam 1-5) with GT conversational groups. "
            "Default on DAIC: /tudelft.net/staff-umbrella/neon/ingroup_dataset/"
            "B2_pipeline/cgroup_annotation/"
        ),
    )
    parser.add_argument(
        "--plot_dir",
        default=None,
        help=(
            "Optional directory for position/orientation plots. "
            "If set, plots are written every 60 seconds per camera."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_results_directory(
        args.results_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
        conf_thresh=args.conf_thresh,
        camera_params_root=args.camera_params_root,
        body_height=args.body_height,
        camera_numbers=args.camera_numbers,
        gt_groups_root=args.gt_groups_root,
        plot_dir=args.plot_dir,
    )
