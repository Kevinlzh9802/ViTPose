"""
Draw ViTPose keypoint figures with a small blur around each detected head.

This script intentionally does not generate dataframes. It scans the existing
``cam*/vitpose_keypoints.json`` outputs, samples frames at a fixed interval,
loads the corresponding raw preview image, blurs a region centred on Conflab-17
keypoint 0 (``head``), and then draws the keypoints and skeleton on top.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.vitpose_to_dataframe import (
    BODY_HEIGHT,
    CONF_THRESHOLD,
    FRAMES_ROOT_DEFAULT,
    KEYPOINT_IMAGE_SIZE,
    PLOT_FRAME_INTERVAL,
    _bev_bounds_around,
    _camera_center_world_xy,
    _global_frame_index,
    _seg_frame_image_path,
    _seg_info_for_global_frame,
    batch_number_from_name,
    cam_number_from_name,
    load_camera_params,
    numeric_sort_key,
    parse_camera_numbers,
    project_person_keypoints_to_world,
)


# Conflab-17 skeleton in index space. See configs/_base_/datasets/conflab.py.
SKELETON = (
    (14, 13),
    (13, 12),
    (11, 10),
    (10, 9),
    (12, 9),
    (6, 12),
    (3, 9),
    (6, 7),
    (3, 4),
    (7, 8),
    (4, 5),
    (16, 14),
    (15, 11),
    (1, 0),
    (2, 0),
    (6, 2),
    (3, 2),
)

KEYPOINT_NAMES = (
    "head",
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_foot",
    "left_foot",
)


def _person_color(index: int) -> tuple[float, float, float, float]:
    cmap = plt.get_cmap("tab20")
    return cmap(index % cmap.N)


def _valid_xy_from_raw(raw_kps: list, kp_idx: int, conf_thresh: float) -> tuple[float, float] | None:
    if raw_kps is None or len(raw_kps) <= kp_idx:
        return None
    kp = raw_kps[kp_idx]
    if len(kp) < 3 or float(kp[2]) < conf_thresh:
        return None
    x, y = float(kp[0]), float(kp[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _valid_xy_from_world(kp_world: list, kp_idx: int) -> tuple[float, float] | None:
    if kp_world is None or len(kp_world) <= kp_idx:
        return None
    kp = kp_world[kp_idx]
    if kp is None:
        return None
    x, y = float(kp[0]), float(kp[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _odd_kernel_size(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _load_image_rgb(image_path: Path | None) -> np.ndarray | None:
    if image_path is None or not Path(image_path).is_file():
        return None
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _blur_head_regions(
    image_rgb: np.ndarray,
    raw_by_track: dict[str, list],
    img_scale_x: float,
    img_scale_y: float,
    conf_thresh: float,
    radius_px: int,
    kernel_size: int,
) -> np.ndarray:
    """Blur square image regions centred on keypoint 0 (head)."""
    blurred_image = image_rgb.copy()
    img_h, img_w = blurred_image.shape[:2]
    scaled_radius = int(round(radius_px * (img_scale_x + img_scale_y) / 2.0))
    scaled_radius = max(1, scaled_radius)
    kernel = _odd_kernel_size(kernel_size)

    for raw_kps in raw_by_track.values():
        head_xy = _valid_xy_from_raw(raw_kps, 0, conf_thresh)
        if head_xy is None:
            continue
        cx = int(round(head_xy[0] * img_scale_x))
        cy = int(round(head_xy[1] * img_scale_y))
        if cx < 0 or cx >= img_w or cy < 0 or cy >= img_h:
            continue

        x0 = max(0, cx - scaled_radius)
        x1 = min(img_w, cx + scaled_radius + 1)
        y0 = max(0, cy - scaled_radius)
        y1 = min(img_h, cy + scaled_radius + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        roi = blurred_image[y0:y1, x0:x1]
        blurred_image[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (kernel, kernel), 0)

    return blurred_image


def _draw_raw_pose(
    ax,
    raw_kps: list,
    track_id: str,
    color,
    img_scale_x: float,
    img_scale_y: float,
    conf_thresh: float,
) -> None:
    scaled_points = []
    for kp_idx in range(len(KEYPOINT_NAMES)):
        xy = _valid_xy_from_raw(raw_kps, kp_idx, conf_thresh)
        if xy is None:
            scaled_points.append(None)
        else:
            scaled_points.append((xy[0] * img_scale_x, xy[1] * img_scale_y))

    for i0, i1 in SKELETON:
        p0 = scaled_points[i0]
        p1 = scaled_points[i1]
        if p0 is None or p1 is None:
            continue
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=color,
            linewidth=1.2,
            alpha=0.9,
            zorder=2,
        )

    xs = [p[0] for p in scaled_points if p is not None]
    ys = [p[1] for p in scaled_points if p is not None]
    if xs and ys:
        ax.scatter(
            xs,
            ys,
            s=16,
            color=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    head_xy = scaled_points[0]
    neck_xy = scaled_points[2]
    label_xy = head_xy or neck_xy
    if label_xy is not None:
        ax.text(
            label_xy[0],
            label_xy[1] - 10.0 * img_scale_y,
            track_id,
            color="yellow",
            fontsize=8,
            ha="center",
            va="bottom",
            zorder=4,
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.55, pad=1.0),
        )


def _draw_world_pose(
    ax,
    kp_world: list,
    track_id: str,
    color,
) -> None:
    points = []
    for kp_idx in range(len(KEYPOINT_NAMES)):
        points.append(_valid_xy_from_world(kp_world, kp_idx))

    for i0, i1 in SKELETON:
        p0 = points[i0]
        p1 = points[i1]
        if p0 is None or p1 is None:
            continue
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=color,
            linewidth=1.2,
            alpha=0.9,
            zorder=2,
        )

    xs = [p[0] for p in points if p is not None]
    ys = [p[1] for p in points if p is not None]
    if xs and ys:
        ax.scatter(
            xs,
            ys,
            s=16,
            color=color,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )

    hips = [points[9], points[12]]
    hips = [p for p in hips if p is not None]
    if hips:
        hx = float(np.mean([p[0] for p in hips]))
        hy = float(np.mean([p[1] for p in hips]))
        ax.text(
            hx + 0.08,
            hy + 0.08,
            track_id,
            color="black",
            fontsize=8,
            zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0),
        )


def plot_blurred_keypoints(
    frame_id: str,
    raw_by_track: dict[str, list],
    world_by_track: dict[str, list],
    image_path: Path | None,
    out_path: Path,
    source_tag: str,
    world_bounds: tuple[float, float, float, float] | None = None,
    time_str: str = "",
    seg_info: str = "",
    conf_thresh: float = CONF_THRESHOLD,
    blur_radius_px: int = 24,
    blur_kernel_size: int = 31,
    dpi: int = 100,
    keypoint_image_size: tuple[int, int] = KEYPOINT_IMAGE_SIZE,
) -> Path:
    fig, (ax_img, ax_world) = plt.subplots(1, 2, figsize=(16.0, 8.0), dpi=dpi)

    img = _load_image_rgb(image_path)
    img_scale_x = 1.0
    img_scale_y = 1.0
    if img is not None:
        img_h, img_w = img.shape[:2]
        kp_w, kp_h = keypoint_image_size
        if kp_w > 0 and kp_h > 0:
            img_scale_x = img_w / float(kp_w)
            img_scale_y = img_h / float(kp_h)
        img = _blur_head_regions(
            img,
            raw_by_track=raw_by_track,
            img_scale_x=img_scale_x,
            img_scale_y=img_scale_y,
            conf_thresh=conf_thresh,
            radius_px=blur_radius_px,
            kernel_size=blur_kernel_size,
        )
        ax_img.imshow(img)
        ax_img.set_aspect("equal", adjustable="box")
    else:
        ax_img.set_facecolor("#222222")
        msg = "Frame image not found"
        if image_path is not None:
            msg = f"Frame image not found:\n{image_path}"
        ax_img.text(
            0.5,
            0.5,
            msg,
            color="white",
            ha="center",
            va="center",
            transform=ax_img.transAxes,
            fontsize=10,
        )

    track_ids = sorted(raw_by_track.keys(), key=numeric_sort_key)
    legend_handles = []
    for idx, track_id in enumerate(track_ids):
        color = _person_color(idx)
        _draw_raw_pose(
            ax_img,
            raw_by_track[track_id],
            track_id=track_id,
            color=color,
            img_scale_x=img_scale_x,
            img_scale_y=img_scale_y,
            conf_thresh=conf_thresh,
        )
        _draw_world_pose(
            ax_world,
            world_by_track.get(track_id, [None] * len(KEYPOINT_NAMES)),
            track_id=track_id,
            color=color,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                markersize=5,
                linewidth=1.4,
                label=str(track_id),
            )
        )

    ax_img.set_title("Blurred frame with keypoints")
    ax_img.set_xlabel("x (px)")
    ax_img.set_ylabel("y (px)")

    if world_bounds is not None:
        xmin, xmax, ymin, ymax = world_bounds
        ax_world.set_xlim(xmin, xmax)
        ax_world.set_ylim(ymin, ymax)
    ax_world.set_aspect("equal", adjustable="box")
    ax_world.grid(True, linestyle=":", alpha=0.5)
    ax_world.set_title("Projected keypoints (bird's-eye view)")
    ax_world.set_xlabel("X (m)")
    ax_world.set_ylabel("Y (m)")

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(8, max(1, len(legend_handles))),
            frameon=True,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.01),
            title="Track id",
            title_fontsize=9,
        )

    extra = []
    if time_str:
        extra.append(time_str)
    if seg_info:
        extra.append(seg_info)
    extra_part = "  |  " + "  |  ".join(extra) if extra else ""
    fig.suptitle(
        f"{source_tag}  |  frame {frame_id}{extra_part}",
        fontsize=11,
        y=0.98,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _project_frame_keypoints(
    keypoints_by_track: dict,
    camera_params: dict,
    body_height: float,
    conf_thresh: float,
) -> dict[str, list]:
    rvec = camera_params["rvec"]
    tvec = camera_params["tvec"]
    camera_model = camera_params.get("model")
    K = camera_params["K"]
    D = camera_params["D"]
    R, _ = cv2.Rodrigues(rvec)

    world_by_track = {}
    for track_id, raw_kps in keypoints_by_track.items():
        world_by_track[str(track_id)] = project_person_keypoints_to_world(
            raw_kps,
            K=K,
            D=D,
            R=R,
            tvec=tvec,
            body_height=body_height,
            conf_thresh=conf_thresh,
            model=camera_model,
        )
    return world_by_track


def process_results_directory(
    results_dir: str | Path,
    plot_dir: str | Path,
    camera_params_root: str | Path,
    frames_root: str | Path | None,
    camera_numbers=None,
    plot_frame_interval: int = PLOT_FRAME_INTERVAL,
    body_height: float = BODY_HEIGHT,
    conf_thresh: float = CONF_THRESHOLD,
    blur_radius_px: int = 24,
    blur_kernel_size: int = 31,
) -> None:
    results_dir = Path(results_dir)
    plot_dir = Path(plot_dir)
    frames_root = Path(frames_root) if frames_root is not None else None
    selected_camera_numbers = parse_camera_numbers(camera_numbers)

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

        batch_number = batch_number_from_name(cam_name)
        cam_tag = f"cam{cam_number}"
        cam_plot_dir = plot_dir / cam_tag
        cam_plot_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Drawing {json_file.relative_to(results_dir)}")
        with open(json_file) as f:
            data = json.load(f)

        camera_params = load_camera_params(
            cam_number,
            camera_params_root=camera_params_root,
        )
        cam_world_xy = _camera_center_world_xy(
            camera_params["rvec"],
            camera_params["tvec"],
        )
        world_bounds = _bev_bounds_around(cam_world_xy)
        print(
            f"    [plot] bird's-eye window centred on camera "
            f"(X={cam_world_xy[0]:.2f} m, Y={cam_world_xy[1]:.2f} m)"
        )

        annotations = data.get("annotations", {})
        sorted_frame_ids = sorted(annotations.keys(), key=numeric_sort_key)
        selected = sorted_frame_ids[::plot_frame_interval]
        print(f"    [plot] writing {len(selected)} blurred keypoint figures")

        for frame_id in selected:
            frame_data = annotations[frame_id]
            keypoints_by_track = frame_data.get("keypoints", {})
            raw_by_track = {
                str(track_id): keypoints_by_track[track_id]
                for track_id in sorted(keypoints_by_track.keys(), key=numeric_sort_key)
            }
            world_by_track = _project_frame_keypoints(
                raw_by_track,
                camera_params=camera_params,
                body_height=body_height,
                conf_thresh=conf_thresh,
            )

            try:
                local_frame = int(frame_id)
            except (TypeError, ValueError):
                local_frame = 0
            global_frame = _global_frame_index(batch_number, local_frame)
            seg_num, seg_local = _seg_info_for_global_frame(global_frame)
            image_path = _seg_frame_image_path(frames_root, cam_number, seg_num)

            batch_part = (
                f"batch{batch_number:02d}" if batch_number is not None else "batch??"
            )
            source_tag = f"{cam_tag}  |  {batch_part}  |  global frame {global_frame}"
            out_path = cam_plot_dir / f"{cam_tag}__keypoints_frame_{global_frame:07d}.png"
            plot_blurred_keypoints(
                frame_id=str(frame_id),
                raw_by_track=raw_by_track,
                world_by_track=world_by_track,
                image_path=image_path,
                out_path=out_path,
                source_tag=source_tag,
                world_bounds=world_bounds,
                seg_info=f"seg{seg_num:03d} (offset {seg_local} frames)",
                conf_thresh=conf_thresh,
                blur_radius_px=blur_radius_px,
                blur_kernel_size=blur_kernel_size,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw blurred ViTPose keypoint figures without writing dataframes."
    )
    parser.add_argument(
        "results_dir",
        help="Root directory containing cam*/vitpose_keypoints.json files.",
    )
    parser.add_argument(
        "--plot_dir",
        required=True,
        help="Directory where blurred keypoint figures are written.",
    )
    parser.add_argument(
        "--camera_params_root",
        required=True,
        help="Root containing camera_XX/intrinsic.json and extrinsic.json.",
    )
    parser.add_argument(
        "--frames_root",
        default=str(FRAMES_ROOT_DEFAULT),
        help="Root containing cam<XX>/cam<XX>_seg<YYY>_frame0.jpg images.",
    )
    parser.add_argument(
        "--camera_numbers",
        default=None,
        help="Optional comma-separated camera numbers, e.g. '06,08,10'.",
    )
    parser.add_argument(
        "--plot_frame_interval",
        type=int,
        default=PLOT_FRAME_INTERVAL,
        help=f"Frames between plots (default: {PLOT_FRAME_INTERVAL}).",
    )
    parser.add_argument(
        "--body_height",
        type=float,
        default=BODY_HEIGHT,
        help="Assumed body height in meters for back-projection.",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=CONF_THRESHOLD,
        help="Minimum keypoint confidence for drawing/blurring.",
    )
    parser.add_argument(
        "--blur_radius_px",
        type=int,
        default=24,
        help=(
            "Half-width of the blur region in ViTPose keypoint pixels before "
            "scaling to the source image resolution."
        ),
    )
    parser.add_argument(
        "--blur_kernel_size",
        type=int,
        default=31,
        help="Gaussian blur kernel size. Even values are rounded up to odd.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_frame_interval <= 0:
        raise ValueError("--plot_frame_interval must be positive")
    process_results_directory(
        results_dir=args.results_dir,
        plot_dir=args.plot_dir,
        camera_params_root=args.camera_params_root,
        frames_root=args.frames_root,
        camera_numbers=args.camera_numbers,
        plot_frame_interval=args.plot_frame_interval,
        body_height=args.body_height,
        conf_thresh=args.conf_thresh,
        blur_radius_px=args.blur_radius_px,
        blur_kernel_size=args.blur_kernel_size,
    )


if __name__ == "__main__":
    main()
