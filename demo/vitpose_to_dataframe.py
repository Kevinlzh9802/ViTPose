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
                    "<person_id>": [[x, y, score], ... 17 Conflab keypoints ...]
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

# --- Resolution bookkeeping ---------------------------------------------- #
# The ViTPose JSONs store keypoints in the resolution that was actually fed to
# the network (960 x 540 in this pipeline). The camera intrinsics, however,
# were calibrated at the full GoPro resolution (1920 x 1080). Before we can
# undistort / back-project the keypoints we therefore need to map them into
# the intrinsic's coordinate system by multiplying (u, v) with the ratio
# ``INTRINSIC_IMAGE_SIZE / KEYPOINT_IMAGE_SIZE``.
KEYPOINT_IMAGE_SIZE = (960, 540)      # (width, height) of VitPose output
INTRINSIC_IMAGE_SIZE = (1920, 1080)   # (width, height) at which K/D live

# Conflab-17 keypoint heights as a fraction of body height above the floor.
# This matches configs/_base_/datasets/conflab.py, which is the dataset_info
# used by configs/ViTPose_coco_plus_conflab_w_bg_256x192.py.
KP_HEIGHT_RATIOS = np.array([
    1.00,   #  0 head
    0.97,   #  1 nose
    0.86,   #  2 neck
    0.85,   #  3 right_shoulder
    0.68,   #  4 right_elbow
    0.55,   #  5 right_wrist
    0.85,   #  6 left_shoulder
    0.68,   #  7 left_elbow
    0.55,   #  8 left_wrist
    0.50,   #  9 right_hip
    0.27,   # 10 right_knee
    0.02,   # 11 right_ankle
    0.50,   # 12 left_hip
    0.27,   # 13 left_knee
    0.02,   # 14 left_ankle
    0.00,   # 15 right_foot
    0.00,   # 16 left_foot
], dtype=np.float64)

# Orientation pairs in Conflab-17 index order.
# The lateral segments use (left_idx, right_idx) and are rotated 90 degrees to
# estimate heading. Head has no left/right pair in Conflab, so it uses the
# direct head -> nose vector.
ORIENTATION_PAIRS = {
    "head": (0, 1),       # head -> nose
    "shoulder": (6, 3),   # left_shoulder -> right_shoulder
    "hip": (12, 9),       # left_hip -> right_hip
    "foot": (16, 15),     # left_foot -> right_foot
}
ORIENTATION_MODES = {
    "head": "direct",
    "shoulder": "lateral",
    "hip": "lateral",
    "foot": "lateral",
}

# Fallback keypoints used to recover an x/y center when the main pair is missing.
SEGMENT_FALLBACKS = {
    "head": [0, 1, 2],
    "shoulder": [6, 3],
    "hip": [12, 9],
    "foot": [16, 15, 14, 11],
}

# --- Time-mapping and GT-group constants --------------------------------- #
FPS = 60
FRAMES_PER_BATCH = 18000  # 60 fps × 300 s = 5 min per batch
FRAMES_PER_SEG = 600       # each raw video segment is 10 s at 60 fps (210 segs / 35 min)

# Root directory holding cam<XX>/cam<XX>_seg<YYY>_frame0.jpg preview images.
FRAMES_ROOT_DEFAULT = Path(
    "/tudelft.net/staff-umbrella/neon/ingroup_dataset/"
    "B2_pipeline/video_segs_raw"
)

# How often (in frames) to render diagnostic plots when --plot_dir is set.
PLOT_FRAME_INTERVAL = 1200  # every 20 s at 60 fps

# Bird's-eye plot window, expressed as (half_width_x, half_width_y) in metres
# around the camera's (X, Y) world position. A window of ±4 m × ±3 m covers an
# 8 m × 6 m floor area centred under each ceiling camera.
BEV_HALF_WIDTH_XY = (4.0, 3.0)

# 2D rotation applied to every back-projected world (X, Y) BEFORE it enters the
# dataframe / BEV plot. The current matrix corresponds to a 90 degrees CW
# rotation of the world plane (equivalently, a 90 degrees CCW rotation of the
# displayed image):
#
#     plot_x  =  +world_Y
#     plot_y  =  -world_X
#
# Both the saved pkl's ``spaceFeat`` columns and all BEV plots use this rotated
# frame; set ``WORLD_REORIENTATION_2D = np.eye(2)`` to disable.
WORLD_REORIENTATION_2D = np.array(
    [[ 0.0, 1.0],
     [-1.0, 0.0]],
    dtype=np.float64,
)


def apply_world_reorientation_xy(x: float, y: float) -> tuple[float, float]:
    """Rotate a world ``(X, Y)`` pair into the plot/dataframe frame."""
    vec = WORLD_REORIENTATION_2D @ np.array([x, y], dtype=np.float64)
    return float(vec[0]), float(vec[1])


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
    """Heading angle from a left/right keypoint pair."""
    if left_xy is None or right_xy is None:
        return None

    dx = right_xy[0] - left_xy[0]
    dy = right_xy[1] - left_xy[1]
    if dx == 0.0 and dy == 0.0:
        return None

    # 90 deg CCW: (dx, dy) -> (-dy, dx)
    return math.atan2(dx, -dy)


def orientation_from_vector(start_xy, end_xy) -> float | None:
    """Angle of the direct vector from ``start_xy`` to ``end_xy``."""
    if start_xy is None or end_xy is None:
        return None

    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    if dx == 0.0 and dy == 0.0:
        return None

    return math.atan2(dy, dx)


DEFAULT_CAMERA_MODEL = "fisheye"  # "fisheye" (Kannala-Brandt) or "pinhole"


def normalize_distortion_coefficients(
    coeffs, model: str = DEFAULT_CAMERA_MODEL
) -> np.ndarray:
    """
    Convert a distortion vector into the shape expected by the OpenCV routine
    for the requested ``model``.

    Fisheye (Kannala-Brandt): exactly 4 radial coefficients ``[k1, k2, k3, k4]``.
    Pinhole: OpenCV expects ``[k1, k2, p1, p2, k3]`` (5 values). Older data with
    a 3-element vector ``[k1, k2, k3]`` is padded with zero tangential terms.
    """
    dist = np.asarray(coeffs, dtype=np.float64).reshape(-1)

    if model == "fisheye":
        if dist.size == 4:
            return dist
        if dist.size == 3:
            # libCalib sometimes drops the last k4 when it was fixed at zero.
            return np.array([dist[0], dist[1], dist[2], 0.0], dtype=np.float64)
        if dist.size > 4:
            return dist[:4].copy()
        raise ValueError(
            f"Fisheye model needs at least 3 coefficients, got {dist.size}."
        )

    if model == "pinhole":
        if dist.size == 3:
            return np.array([dist[0], dist[1], 0.0, 0.0, dist[2]], dtype=np.float64)
        if dist.size in {4, 5, 8, 12, 14}:
            return dist
        raise ValueError(
            f"Unsupported pinhole distortion length: {dist.size}"
        )

    raise ValueError(f"Unknown camera model: {model!r}")


def load_camera_params(
    cam_number: str | int,
    camera_params_root: str | Path = CAMERA_PARAMS_ROOT,
) -> dict:
    """Load per-camera intrinsics and extrinsics from camera_XX/*.json.

    The intrinsic JSON may include a ``"model"`` field set to ``"fisheye"``
    (Kannala-Brandt, matches ``libCalib::CameraModelOpenCVFisheye``) or
    ``"pinhole"``. When the field is absent we default to
    :data:`DEFAULT_CAMERA_MODEL` (fisheye).
    """
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

    model = str(intrinsic_data.get("model", DEFAULT_CAMERA_MODEL)).lower()
    if model not in {"fisheye", "pinhole"}:
        raise ValueError(
            f"Unknown camera model {model!r} in {intrinsic_path}; "
            "expected 'fisheye' or 'pinhole'."
        )

    params = {
        "model": model,
        "K": np.asarray(intrinsic_data["intrinsic"], dtype=np.float64),
        "D": normalize_distortion_coefficients(
            intrinsic_data["distortion_coefficients"], model=model
        ),
        "rvec": np.asarray(extrinsic_data["rvec"], dtype=np.float64).reshape(3, 1),
        "tvec": np.asarray(extrinsic_data["tvec"], dtype=np.float64).reshape(3, 1),
    }
    _camera_params_cache[cache_key] = params
    return params


def undistort_points(
    pts_uv: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    model: str = DEFAULT_CAMERA_MODEL,
) -> np.ndarray:
    """Undistort pixel coordinates into normalized camera coordinates.

    Dispatches to the fisheye (Kannala-Brandt) or pinhole undistortion routine
    based on ``model``. Fisheye expects D of length 4; pinhole expects 4/5/8/12/14.
    """
    pts = pts_uv.reshape(-1, 1, 2).astype(np.float64)
    if model == "fisheye":
        D_fisheye = np.asarray(D, dtype=np.float64).reshape(-1, 1)[:4]
        undistorted = cv2.fisheye.undistortPoints(pts, K, D_fisheye)
    elif model == "pinhole":
        undistorted = cv2.undistortPoints(pts, K, D)
    else:
        raise ValueError(f"Unknown camera model: {model!r}")
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
    start_idx, end_idx = ORIENTATION_PAIRS[segment_name]
    start_xy = valid_world_xy(kp_world, start_idx)
    end_xy = valid_world_xy(kp_world, end_idx)

    if start_xy is not None and end_xy is not None:
        x = (start_xy[0] + end_xy[0]) / 2.0
        y = (start_xy[1] + end_xy[1]) / 2.0
        if ORIENTATION_MODES[segment_name] == "direct":
            theta = orientation_from_vector(start_xy, end_xy)
        else:
            theta = orientation_from_pair(start_xy, end_xy)
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


def keypoint_to_intrinsic_scale(
    keypoint_size: tuple[int, int] = KEYPOINT_IMAGE_SIZE,
    intrinsic_size: tuple[int, int] = INTRINSIC_IMAGE_SIZE,
) -> tuple[float, float]:
    """Return (sx, sy) that map a keypoint pixel into the intrinsic's frame."""
    return (
        intrinsic_size[0] / keypoint_size[0],
        intrinsic_size[1] / keypoint_size[1],
    )


def project_person_keypoints_to_world(
    raw_kps: list,
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    tvec: np.ndarray,
    body_height: float,
    conf_thresh: float,
    keypoint_size: tuple[int, int] = KEYPOINT_IMAGE_SIZE,
    intrinsic_size: tuple[int, int] = INTRINSIC_IMAGE_SIZE,
    model: str = DEFAULT_CAMERA_MODEL,
) -> list:
    """Project one person's 17 Conflab keypoints to world coordinates.

    ``raw_kps`` are expected in ``keypoint_size`` pixel coordinates; they are
    rescaled to ``intrinsic_size`` before being undistorted via the given
    camera ``model`` (fisheye or pinhole).
    """
    if len(raw_kps) != 17:
        raise ValueError("Expected 17 Conflab keypoints, got {}".format(len(raw_kps)))

    valid_idx = [i for i in range(17) if raw_kps[i][2] >= conf_thresh]
    kp_world = [None] * 17
    if not valid_idx:
        return kp_world

    sx, sy = keypoint_to_intrinsic_scale(keypoint_size, intrinsic_size)
    pts_uv = np.array(
        [[raw_kps[i][0] * sx, raw_kps[i][1] * sy] for i in valid_idx],
        dtype=np.float64,
    )
    norm_xy = undistort_points(pts_uv, K, D, model=model)

    for j, kp_idx in enumerate(valid_idx):
        z_kp = body_height * KP_HEIGHT_RATIOS[kp_idx]
        xn, yn = norm_xy[j]
        xw, yw = backproject_to_world(xn, yn, z_kp, R, tvec)
        if xw is not None and yw is not None:
            xw_r, yw_r = apply_world_reorientation_xy(xw, yw)
            kp_world[kp_idx] = (xw_r, yw_r, z_kp)

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
    model: str = DEFAULT_CAMERA_MODEL,
) -> dict[str, list]:
    """
    Convert one person's 17 Conflab keypoints into DANTE-style segment rows.

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
        model=model,
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
    plot_frame_interval: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Process a single vitpose_keypoints.json into a dataframe indexed by frame id.

    Columns:
        - timestamp
        - time        (HH:MM:SS;FF wall-clock time, empty if batch info unavailable)
        - spaceFeat
        - groups      (list of sets from GT CSV, empty if time not in CSV)
        - group_ids

    When ``plot_frame_interval`` is set, for every Nth frame (by position, not
    by id) the raw Conflab keypoints and their back-projected world counterparts
    are captured in the returned ``plot_samples`` dict, keyed by the local frame
    id. The mapping has the shape::

        {frame_id: {"raw": {track_id: 17x3 list},
                    "world": {track_id: 17-list of (x, y, z) | None}}}
    """
    input_path = Path(input_path)
    with open(input_path) as f:
        data = json.load(f)

    camera_params = load_camera_params(cam_number, camera_params_root=camera_params_root)
    K = camera_params["K"]
    D = camera_params["D"]
    rvec = camera_params["rvec"]
    tvec = camera_params["tvec"]
    camera_model = camera_params.get("model", DEFAULT_CAMERA_MODEL)
    R, _ = cv2.Rodrigues(rvec)

    records = []
    plot_samples: dict[str, dict] = {}
    annotations = data.get("annotations", {})
    sorted_frame_ids = sorted(annotations.keys(), key=numeric_sort_key)
    for frame_idx, frame_id in enumerate(sorted_frame_ids):
        frame_data = annotations[frame_id]
        keypoints_by_track = frame_data.get("keypoints", {})
        sorted_track_ids = sorted(keypoints_by_track.keys(), key=numeric_sort_key)

        capture_plot = (
            plot_frame_interval is not None
            and plot_frame_interval > 0
            and frame_idx % plot_frame_interval == 0
        )
        raw_by_track: dict[str, list] = {}
        world_by_track: dict[str, list] = {}

        segment_rows = {segment_name: [] for segment_name in ORIENTATION_PAIRS}
        for track_id in sorted_track_ids:
            raw_kps = keypoints_by_track[track_id]
            kp_world = project_person_keypoints_to_world(
                raw_kps,
                K=K,
                D=D,
                R=R,
                tvec=tvec,
                body_height=body_height,
                conf_thresh=conf_thresh,
                model=camera_model,
            )
            for segment_name in ORIENTATION_PAIRS:
                x, y, theta = segment_xy_and_orientation(kp_world, segment_name)
                segment_rows[segment_name].append([str(track_id), x, y, theta])

            if capture_plot:
                raw_by_track[str(track_id)] = raw_kps
                world_by_track[str(track_id)] = kp_world

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

        if capture_plot:
            plot_samples[str(frame_id)] = {
                "raw": raw_by_track,
                "world": world_by_track,
            }

    df = pd.DataFrame(records)
    if not df.empty:
        df.index = df["timestamp"]
        df.index.name = "timestamp"

    return df, plot_samples


def _camera_center_world_xy(rvec: np.ndarray, tvec: np.ndarray) -> tuple[float, float]:
    """Return the camera centre's (X, Y) in the plot/dataframe frame.

    Uses the standard OpenCV convention where ``rvec`` / ``tvec`` map world
    points into the camera frame, so ``C_world = -R.T @ tvec``. The same
    :data:`WORLD_REORIENTATION_2D` rotation that is applied to keypoints is
    applied here so the BEV window stays centred under the camera after the
    rotation.
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    c = -(R.T @ np.asarray(tvec, dtype=np.float64).reshape(3))
    return apply_world_reorientation_xy(float(c[0]), float(c[1]))


def _bev_bounds_around(
    center_xy: tuple[float, float],
    half_widths: tuple[float, float] = BEV_HALF_WIDTH_XY,
) -> tuple[float, float, float, float]:
    """Axis-aligned bounds ``(xmin, xmax, ymin, ymax)`` centred on ``center_xy``."""
    cx, cy = center_xy
    hx, hy = half_widths
    return (cx - hx, cx + hx, cy - hy, cy + hy)


def _global_frame_index(batch_number: int | None, local_frame: int) -> int:
    """Map a local per-batch frame index to a global frame index across batches."""
    if batch_number is None:
        return local_frame
    return (batch_number - 1) * FRAMES_PER_BATCH + local_frame


def _seg_info_for_global_frame(global_frame: int) -> tuple[int, int]:
    """Return ``(seg_number_1_indexed, frame_within_seg)`` for a global frame."""
    seg_num = global_frame // FRAMES_PER_SEG + 1
    local = global_frame % FRAMES_PER_SEG
    return seg_num, local


def _seg_frame_image_path(
    frames_root: Path | None, cam_number: str, seg_num: int
) -> Path | None:
    """Build the path to ``camXX/camXX_segYYY_frame0.jpg`` if frames_root is set."""
    if frames_root is None:
        return None
    cam_tag = f"cam{cam_number}"
    return (
        Path(frames_root) / cam_tag / f"{cam_tag}_seg{seg_num:03d}_frame0.jpg"
    )


def _render_position_plots(
    df: pd.DataFrame,
    cam_number: str,
    batch_number: int | None,
    cam_plot_dir: Path,
    frame_interval: int,
    shared_bounds_override: tuple[float, float, float, float] | None = None,
) -> None:
    """Render the existing top-down position plots into ``cam_plot_dir``.

    Filenames use the global frame index so plots from all batches of the same
    camera can coexist in one folder and sort chronologically.
    """
    from demo.plot_person import (
        DEFAULT_SEGMENT,
        compute_plot_bounds,
        plot_single_frame,
    )

    if len(df) == 0:
        return

    bounds = (
        shared_bounds_override
        if shared_bounds_override is not None
        else compute_plot_bounds(df, segment=DEFAULT_SEGMENT)
    )
    cam_tag = f"cam{cam_number}"
    selected = df.iloc[::frame_interval]

    for frame_id, row in selected.iterrows():
        try:
            local_frame = int(frame_id)
        except (ValueError, TypeError):
            local_frame = 0
        global_frame = _global_frame_index(batch_number, local_frame)
        out_path = (
            cam_plot_dir
            / f"{cam_tag}__position_frame_{global_frame:07d}.png"
        )

        groups = row.get("groups") if "groups" in row.index else None
        time_str = row.get("time", "") if "time" in row.index else ""
        batch_part = (
            f"batch{batch_number:02d}" if batch_number is not None else "batch??"
        )
        time_part = f"  |  {time_str}" if time_str else ""
        title = (
            f"{cam_tag}  |  {batch_part}  |  global frame {global_frame}"
            f"  (local {local_frame}){time_part}"
        )
        plot_single_frame(
            frame_id=str(frame_id),
            spacefeat=row["spaceFeat"],
            source_tag=cam_tag,
            output_dir=cam_plot_dir,
            bounds=bounds,
            groups=groups if isinstance(groups, list) and groups else None,
            time_str=str(time_str) if time_str else "",
            out_path=out_path,
            title_override=title,
        )


def _render_keypoints_plots(
    df: pd.DataFrame,
    plot_samples: dict,
    cam_number: str,
    batch_number: int | None,
    cam_plot_dir: Path,
    frames_root: Path | None,
    shared_bounds_override: tuple[float, float, float, float] | None = None,
) -> None:
    """Render per-sample combined keypoint figures (pixel + bird's-eye)."""
    from demo.plot_person import (
        DEFAULT_SEGMENT,
        compute_plot_bounds,
        plot_keypoints_subplots,
    )

    if not plot_samples:
        return

    bounds = (
        shared_bounds_override
        if shared_bounds_override is not None
        else compute_plot_bounds(df, segment=DEFAULT_SEGMENT)
    )
    cam_tag = f"cam{cam_number}"

    for frame_id, sample in plot_samples.items():
        try:
            local_frame = int(frame_id)
        except (ValueError, TypeError):
            local_frame = 0
        global_frame = _global_frame_index(batch_number, local_frame)
        seg_num, seg_local = _seg_info_for_global_frame(global_frame)
        image_path = _seg_frame_image_path(frames_root, cam_number, seg_num)

        time_str = ""
        if frame_id in df.index:
            try:
                time_str = str(df.loc[frame_id, "time"])
            except KeyError:
                time_str = ""

        out_path = (
            cam_plot_dir
            / f"{cam_tag}__keypoints_frame_{global_frame:07d}.png"
        )
        seg_info = f"seg{seg_num:03d} (offset {seg_local} frames)"
        batch_part = (
            f"batch{batch_number:02d}" if batch_number is not None else "batch??"
        )
        source_tag = f"{cam_tag}  |  {batch_part}  |  global frame {global_frame}"
        plot_keypoints_subplots(
            frame_id=str(frame_id),
            raw_by_track=sample["raw"],
            world_by_track=sample["world"],
            image_path=image_path,
            out_path=out_path,
            source_tag=source_tag,
            world_bounds=bounds,
            time_str=time_str,
            seg_info=seg_info,
            keypoint_image_size=KEYPOINT_IMAGE_SIZE,
        )


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
    frames_root: str | Path | None = None,
    plot_frame_interval: int = PLOT_FRAME_INTERVAL,
) -> None:
    """
    Walk every cam*/vitpose_keypoints.json under results_dir and write a
    dataframe alongside the input, or under output_dir/<cam_name>/.

    When ``plot_dir`` is set, two kinds of diagnostic plots are written under
    ``plot_dir/cam<XX>/`` every ``plot_frame_interval`` frames (1200 by default,
    i.e. every 20 s at 60 fps):

        * ``cam<XX>__position_frame_<global>.png`` – top-down position plot.
        * ``cam<XX>__keypoints_frame_<global>.png`` – two-subplot figure with
          raw pixel keypoints overlaid on the source video frame (left) and
          the same keypoints back-projected to the bird's-eye view (right).

    Batches are assumed to be consecutive 5-minute blocks, so global frames are
    computed as ``(batch - 1) * 18000 + local_frame``. The seg image used for
    the overlay is ``cam<XX>_seg<YYY>_frame0.jpg`` from ``frames_root``, where
    each seg is a 10-second / 600-frame block.
    """
    results_dir = Path(results_dir)
    selected_camera_numbers = parse_camera_numbers(camera_numbers)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if frames_root is not None:
        frames_root = Path(frames_root)

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
        df, plot_samples = process_vitpose_json(
            json_file,
            cam_number=cam_number,
            camera_params_root=camera_params_root,
            body_height=body_height,
            conf_thresh=conf_thresh,
            batch_number=batch_num,
            gt_groups=gt_groups,
            plot_frame_interval=plot_frame_interval if plot_dir is not None else None,
        )

        if output_dir is not None:
            out_path = output_dir / cam_name / output_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = json_file.parent / output_name

        df.to_pickle(out_path)
        print(f"    -> {out_path}")

        if plot_dir is not None:
            cam_plot_dir = Path(plot_dir) / f"cam{cam_number}"
            cam_plot_dir.mkdir(parents=True, exist_ok=True)

            cam_params = load_camera_params(
                cam_number, camera_params_root=camera_params_root
            )
            cam_world_xy = _camera_center_world_xy(
                cam_params["rvec"], cam_params["tvec"]
            )
            bev_bounds = _bev_bounds_around(cam_world_xy)
            print(
                f"    [plot] bird's-eye window centred on camera "
                f"(X={cam_world_xy[0]:.2f} m, Y={cam_world_xy[1]:.2f} m) "
                f"-> X in [{bev_bounds[0]:.2f}, {bev_bounds[1]:.2f}], "
                f"Y in [{bev_bounds[2]:.2f}, {bev_bounds[3]:.2f}]"
            )

            _render_position_plots(
                df=df,
                cam_number=cam_number,
                batch_number=batch_num,
                cam_plot_dir=cam_plot_dir,
                frame_interval=plot_frame_interval,
                shared_bounds_override=bev_bounds,
            )
            _render_keypoints_plots(
                df=df,
                plot_samples=plot_samples,
                cam_number=cam_number,
                batch_number=batch_num,
                cam_plot_dir=cam_plot_dir,
                frames_root=frames_root,
                shared_bounds_override=bev_bounds,
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
            "Optional directory for diagnostic plots. If set, position and "
            "keypoint plots are written every PLOT_FRAME_INTERVAL frames under "
            "<plot_dir>/cam<XX>/ (no batch sub-folders)."
        ),
    )
    parser.add_argument(
        "--frames_root",
        default=str(FRAMES_ROOT_DEFAULT),
        help=(
            "Root containing cam<XX>/cam<XX>_seg<YYY>_frame0.jpg preview "
            "images used as background for keypoint plots. Default on DAIC: "
            f"{FRAMES_ROOT_DEFAULT}"
        ),
    )
    parser.add_argument(
        "--plot_frame_interval",
        type=int,
        default=PLOT_FRAME_INTERVAL,
        help=(
            "Frames between consecutive diagnostic plots (default: "
            f"{PLOT_FRAME_INTERVAL} = every 20 s at 60 fps)."
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
        frames_root=args.frames_root,
        plot_frame_interval=args.plot_frame_interval,
    )
