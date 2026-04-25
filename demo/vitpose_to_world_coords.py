"""
vitpose_to_world_coords.py

Converts ViTPose 2D keypoints (Conflab-17) to 2D world floor coordinates and
orientations, suitable for Stephanie / DANTE pedestrian trajectory models.

Assumptions
-----------
* Camera intrinsics: OpenCV fisheye model loaded from
  camera_calibration/fisheye/intrinsics/parameters-camera-XX.json.
* Camera extrinsics (rvec, tvec): currently trivial (camera at height H,
  looking straight down). Update CAMERA_RVEC / CAMERA_TVEC per camera
  once you have real extrinsic calibration.
* Each person has body height BODY_HEIGHT (m).
* Each COCO keypoint sits at a fixed fraction of that height above the floor
  (see KP_HEIGHT_RATIOS below).

Back-projection
---------------
For each keypoint at known world height Z_kp:
  1. Fisheye-undistort pixel (u,v) -> normalised coords (xn, yn)
  2. Build ray in camera space:  d_cam  = [xn, yn, 1]
  3. Transform to world space:   d_world = R^T @ d_cam
  4. Camera centre in world:     C       = -R^T @ tvec
  5. Intersect ray with plane Z = Z_kp:
       t       = (Z_kp - C.z) / d_world.z
       X_world = C.x + t * d_world.x
       Y_world = C.y + t * d_world.y

Trivial extrinsics (camera at (0,0,H) looking straight down)
-------------------------------------------------------------
  rvec = [pi, 0, 0]   (180 deg rotation around X — flips Y and Z axes so
                        the camera Z axis points downward into the scene)
  tvec = [0,  0, H]   (world origin at camera centre gives C = [0,0,H])

Orientation
-----------
For each body segment (head / shoulder / hip / foot), the orientation is the
angle of the vector obtained by rotating (right_kp - left_kp) 90 deg CCW:

    v       = right_kp_xy - left_kp_xy      (world coords)
    v_rot   = (-v.y, v.x)                   (90 deg CCW)
    theta   = atan2(v_rot.y, v_rot.x)       (radians)
"""

import json
import math
import numpy as np
import cv2
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

CAMERA_HEIGHT  = 3.5    # metres  (update once you have the exact value)
BODY_HEIGHT    = 1.7    # metres
CONF_THRESHOLD = 0.0    # minimum keypoint confidence

# Camera extrinsics — trivial case: camera at (0,0,H) looking straight down.
# rvec=[pi,0,0] rotates 180 deg around X so the camera Z axis points downward.
# Replace these per-camera once you have real extrinsic calibration.
CAMERA_RVEC = np.array([np.pi, 0.0, 0.0], dtype=np.float64)
CAMERA_TVEC = np.array([0.0, 0.0, CAMERA_HEIGHT], dtype=np.float64)

# Root of the intrinsics files  (parameters-camera-XX.json)
INTRINSICS_DIR = Path(
    r"c:\Users\sotir\Desktop\Uni\Master\Thesis"
    r"\camera_calibration\fisheye\intrinsics"
)

# ---------------------------------------------------------------------------
# CONFLAB-17 KEYPOINT HEIGHT RATIOS  (fraction of body height above the floor)
# ---------------------------------------------------------------------------
# Index  name
#   0    head           1  nose           2  neck
#   3    right_shoulder 4  right_elbow    5  right_wrist
#   6    left_shoulder  7  left_elbow     8  left_wrist
#   9    right_hip     10  right_knee    11  right_ankle
#  12    left_hip      13  left_knee     14  left_ankle
#  15    right_foot    16  left_foot

KP_HEIGHT_RATIOS = np.array([
    1.00,   #  0  head
    0.97,   #  1  nose
    0.86,   #  2  neck
    0.85,   #  3  right_shoulder
    0.68,   #  4  right_elbow
    0.55,   #  5  right_wrist
    0.85,   #  6  left_shoulder
    0.68,   #  7  left_elbow
    0.55,   #  8  left_wrist
    0.50,   #  9  right_hip
    0.27,   # 10  right_knee
    0.02,   # 11  right_ankle
    0.50,   # 12  left_hip
    0.27,   # 13  left_knee
    0.02,   # 14  left_ankle
    0.00,   # 15  right_foot
    0.00,   # 16  left_foot
])

# Orientation pairs in Conflab-17 index order.
ORIENTATION_PAIRS = {
    "head":     (0,  1),   # head         -> nose
    "shoulder": (6,  3),   # left_shoulder -> right_shoulder
    "hip":      (12, 9),   # left_hip     -> right_hip
    "foot":     (16, 15),  # left_foot    -> right_foot
}
ORIENTATION_MODES = {
    "head": "direct",
    "shoulder": "lateral",
    "hip": "lateral",
    "foot": "lateral",
}

# ---------------------------------------------------------------------------
# CAMERA INTRINSICS LOADER
# ---------------------------------------------------------------------------

def load_fisheye_intrinsics(cam_number: str | int) -> dict:
    """
    Load OpenCV-fisheye intrinsics for the given camera number from
    parameters-camera-XX.json.

    Returns dict with keys: fx, fy, cx, cy, K (3x3), D (4,)
    """
    cam_id = str(cam_number).zfill(2)
    path   = INTRINSICS_DIR / f"parameters-camera-{cam_id}.json"
    with open(path) as f:
        data = json.load(f)

    params = (data["Calibration"]["cameras"][0]
              ["model"]["ptr_wrapper"]["data"]["parameters"])

    f_val = params["f"]["val"]
    ar    = params["ar"]["val"]     # fy = f * ar  (ar = 1.0 for all three cameras)
    fx    = f_val
    fy    = f_val * ar
    cx    = params["cx"]["val"]
    cy    = params["cy"]["val"]
    k1    = params["k1"]["val"]
    k2    = params["k2"]["val"]
    k3    = params["k3"]["val"]
    k4    = params["k4"]["val"]

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy,  cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([k1, k2, k3, k4], dtype=np.float64)

    return dict(fx=fx, fy=fy, cx=cx, cy=cy, K=K, D=D)


# Cache so we don't reload the same file repeatedly
_intrinsics_cache: dict[str, dict] = {}

def get_intrinsics(cam_number: str | int) -> dict:
    key = str(cam_number).zfill(2)
    if key not in _intrinsics_cache:
        _intrinsics_cache[key] = load_fisheye_intrinsics(key)
    return _intrinsics_cache[key]


def cam_number_from_name(cam_name: str) -> str | None:
    """Extract camera number from directory name, e.g. 'cam06_batch01' -> '06'."""
    import re
    m = re.search(r"cam(\d+)", cam_name)
    return m.group(1) if m else None

# ---------------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------------

def undistort_points_fisheye(pts_uv: np.ndarray,
                              K: np.ndarray,
                              D: np.ndarray) -> np.ndarray:
    """
    Undistort N pixel points using the OpenCV fisheye model.
    Returns normalised image coordinates (xn, yn) — shape (N, 2).
    """
    pts = pts_uv.reshape(-1, 1, 2).astype(np.float64)
    # cv2.fisheye.undistortPoints returns normalised coords (no K applied)
    undist = cv2.fisheye.undistortPoints(pts, K, D)
    return undist.reshape(-1, 2)


def backproject_to_world(xn: float, yn: float, z_kp: float,
                          R: np.ndarray, tvec: np.ndarray) -> tuple:
    """
    Back-project normalised camera coords (xn, yn) to world (X, Y) by
    intersecting the camera ray with the horizontal plane Z_world = z_kp.

    Parameters
    ----------
    xn, yn  : normalised (undistorted) image coords
    z_kp    : known world height of the keypoint (metres)
    R       : 3x3 rotation matrix  (world -> camera)
    tvec    : (3,) translation vector (world -> camera)

    Returns
    -------
    (X_world, Y_world) or (None, None) if ray is parallel to the floor plane
    """
    C_world  = -(R.T @ tvec)                       # camera centre in world
    d_cam    = np.array([xn, yn, 1.0])
    d_world  = R.T @ d_cam                          # ray direction in world

    if abs(d_world[2]) < 1e-9:                      # ray parallel to floor
        return None, None

    t = (z_kp - C_world[2]) / d_world[2]
    return float(C_world[0] + t * d_world[0]), float(C_world[1] + t * d_world[1])


def orientation_from_pair(left_xy, right_xy) -> float | None:
    """
    Heading angle (rad) from a left/right world-coord pair.
    Rotate vector (right - left) by 90 deg CCW.
    """
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


def process_person_keypoints(raw_kps: list,
                              K: np.ndarray,
                              D: np.ndarray,
                              R: np.ndarray,
                              tvec: np.ndarray,
                              body_height: float = BODY_HEIGHT,
                              conf_thresh: float = CONF_THRESHOLD) -> dict:
    """
    Convert one person's 17 Conflab keypoints to world coordinates and
    compute body-segment orientations.

    raw_kps : list of 17 items, each [u, v, confidence]

    Returns dict:
        pos_x, pos_y        -- hip-midpoint world position (m)
        orient_head/shoulder/hip/foot  -- heading angles (rad), or None
        keypoints_world     -- list of 17 {x, y, z, conf} or None
    """
    assert len(raw_kps) == 17

    valid_idx = [i for i in range(17) if raw_kps[i][2] >= conf_thresh]
    kp_world  = [None] * 17

    if valid_idx:
        pts_uv = np.array([[raw_kps[i][0], raw_kps[i][1]]
                           for i in valid_idx], dtype=np.float64)
        norm_xy = undistort_points_fisheye(pts_uv, K, D)   # (M, 2)

        for j, idx in enumerate(valid_idx):
            z_kp       = body_height * KP_HEIGHT_RATIOS[idx]
            xn, yn     = norm_xy[j]
            xw, yw     = backproject_to_world(xn, yn, z_kp, R, tvec)
            if xw is not None:
                kp_world[idx] = (xw, yw, z_kp)

    # Build output keypoint list
    keypoints_world = []
    for i in range(17):
        if kp_world[i] is not None:
            keypoints_world.append({
                "x":    round(kp_world[i][0], 4),
                "y":    round(kp_world[i][1], 4),
                "z":    round(kp_world[i][2], 4),
                "conf": round(float(raw_kps[i][2]), 4),
            })
        else:
            keypoints_world.append(None)

    # Position: hip midpoint (fallback to visible keypoint average)
    lh, rh = kp_world[12], kp_world[9]
    if lh is not None and rh is not None:
        pos_x = (lh[0] + rh[0]) / 2
        pos_y = (lh[1] + rh[1]) / 2
    elif lh is not None:
        pos_x, pos_y = lh[0], lh[1]
    elif rh is not None:
        pos_x, pos_y = rh[0], rh[1]
    else:
        visible = [kp for kp in kp_world if kp is not None]
        if visible:
            pos_x = float(np.mean([k[0] for k in visible]))
            pos_y = float(np.mean([k[1] for k in visible]))
        else:
            pos_x = pos_y = None

    # Orientations
    orientations = {}
    for name, (start_idx, end_idx) in ORIENTATION_PAIRS.items():
        start_xy = (
            (kp_world[start_idx][0], kp_world[start_idx][1])
            if kp_world[start_idx] else None
        )
        end_xy = (
            (kp_world[end_idx][0], kp_world[end_idx][1])
            if kp_world[end_idx] else None
        )
        if ORIENTATION_MODES[name] == "direct":
            theta = orientation_from_vector(start_xy, end_xy)
        else:
            theta = orientation_from_pair(start_xy, end_xy)
        orientations[name] = round(theta, 6) if theta is not None else None

    return {
        "pos_x":           round(pos_x, 4) if pos_x is not None else None,
        "pos_y":           round(pos_y, 4) if pos_y is not None else None,
        "orient_head":     orientations["head"],
        "orient_shoulder": orientations["shoulder"],
        "orient_hip":      orientations["hip"],
        "orient_foot":     orientations["foot"],
        "keypoints_world": keypoints_world,
    }


def process_vitpose_json(input_path: str | Path,
                          cam_number: str | int,
                          rvec: np.ndarray = CAMERA_RVEC,
                          tvec: np.ndarray = CAMERA_TVEC,
                          body_height: float = BODY_HEIGHT,
                          conf_thresh: float = CONF_THRESHOLD) -> dict:
    """
    Process a single vitpose_keypoints.json and return world-coordinate
    annotations in the same {frame_id: {track_id: ...}} structure.

    rvec / tvec : camera extrinsics (world -> camera).  Defaults to the
                  trivial top-down case; replace per camera as needed.
    """
    input_path = Path(input_path)
    with open(input_path) as f:
        data = json.load(f)

    intr     = get_intrinsics(cam_number)
    K, D     = intr["K"], intr["D"]
    R, _     = cv2.Rodrigues(rvec)

    output = {}
    for frame_id, frame_data in data["annotations"].items():
        frame_out = {}
        for track_id, raw_kps in frame_data.get("keypoints", {}).items():
            frame_out[track_id] = process_person_keypoints(
                raw_kps, K, D, R, tvec, body_height, conf_thresh
            )
        output[frame_id] = frame_out

    return output


def process_results_directory(results_dir: str | Path,
                               output_dir: str | Path | None = None,
                               **kwargs) -> None:
    """
    Walk every cam*/vitpose_keypoints.json under results_dir, process each,
    and write world_coords.json alongside the input (or under output_dir).
    """
    results_dir = Path(results_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(results_dir.glob("*/vitpose_keypoints.json"))
    print(f"Found {len(json_files)} result files under {results_dir}")

    for jf in json_files:
        cam_name   = jf.parent.name          # e.g. "cam06_batch01"
        cam_number = cam_number_from_name(cam_name)
        if cam_number is None:
            print(f"  Skipping {cam_name}: could not parse camera number")
            continue

        print(f"  Processing {jf.relative_to(results_dir)} "
              f"(cam{cam_number} fisheye intrinsics)")
        world_data = process_vitpose_json(jf, cam_number=cam_number, **kwargs)

        if output_dir is not None:
            out_path = output_dir / cam_name / "world_coords.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = jf.parent / "world_coords.json"

        with open(out_path, "w") as f:
            json.dump(world_data, f)
        print(f"    -> {out_path}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert ViTPose keypoints to world floor coordinates."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=r"c:\Users\sotir\Desktop\Uni\Master\Thesis\vitpose_results",
        help="Root directory containing cam*/vitpose_keypoints.json files",
    )
    parser.add_argument("--output_dir",    default=None)
    parser.add_argument("--camera_height", type=float, default=CAMERA_HEIGHT,
                        help="Camera height (m); updates the trivial-extrinsics tvec")
    parser.add_argument("--body_height",   type=float, default=BODY_HEIGHT)
    parser.add_argument("--conf_thresh",   type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    # Build tvec from the supplied camera height (rvec stays trivial)
    tvec = np.array([0.0, 0.0, args.camera_height], dtype=np.float64)

    process_results_directory(
        args.results_dir,
        output_dir=args.output_dir,
        tvec=tvec,
        body_height=args.body_height,
        conf_thresh=args.conf_thresh,
    )
