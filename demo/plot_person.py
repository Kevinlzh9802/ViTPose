"""
plot_positions.py

Render a top-down (bird's-eye) plot of each person's world-floor position and
the *four* orientation estimates (head / shoulder / hip / foot) stored in a
vitpose dataframe.

Each figure shows:
  * a circle marking each person's (x, y) position taken from the anchor
    segment (default: ``hip``), colored per-person so a given track keeps the
    same color across frames,
  * up to four arrows emanating from that circle – one per body segment –
    color-coded by segment so head / shoulder / hip / foot orientations can
    be compared at a glance,
  * a text label with the person id next to the circle,
  * a legend explaining the segment → arrow-color mapping.

Segments with a missing (NaN) orientation for a given person are silently
skipped for that person; the remaining arrows are still drawn.

The dataframe is expected to follow the schema produced by
``vitpose_to_dataframe.py``:

    index:  frame id (timestamp)
    spaceFeat: dict with keys {head, shoulder, hip, foot}; each value is an
               (n_people, 4) object array with columns
               [person_id, x, y, orientation]

Usage as a module (from transfer_vitpose_data.py)::

    from plot_positions import plot_dataframe_positions
    plot_dataframe_positions(df, source_tag="cam06_batch01",
                             output_dir="/.../dante_plotting",
                             frame_interval=1200)

Usage from the command line::

    python plot_positions.py /path/to/vitpose_dataframe.pkl \\
        --source-tag cam06_batch01 --output-dir /.../dante_plotting
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

DEFAULT_SEGMENT = "hip"
DEFAULT_FRAME_INTERVAL = 1200
# 8 cm diameter circle per person (4 cm radius). Arrows are scaled to match.
CIRCLE_RADIUS_M = 0.04
ARROW_LENGTH_M = 0.12
ARROW_HEAD_WIDTH_M = 0.03
ARROW_HEAD_LENGTH_M = 0.04
PLOT_PADDING_M = 1.5

SEGMENT_ORDER = ("head", "shoulder", "hip", "foot")
SEGMENT_COLORS = {
    "head":     "#d62728",  # red
    "shoulder": "#1f77b4",  # blue
    "hip":      "#2ca02c",  # green
    "foot":     "#9467bd",  # purple
}

GROUP_COLORS = [
    "#ff7f0e",  # orange
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#e377c2",  # pink
    "#8c564b",  # brown
    "#7f7f7f",  # gray
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
]


def _row_xy_theta(row: np.ndarray) -> tuple[float, float, float]:
    """Extract (x, y, theta) from a single spaceFeat row, coercing to float."""
    return float(row[1]), float(row[2]), float(row[3])


def _segment_map(arr: np.ndarray | None) -> dict[str, tuple[float, float, float]]:
    """Return ``{person_id: (x, y, theta)}`` for a segment array (may be None/empty)."""
    result: dict[str, tuple[float, float, float]] = {}
    if arr is None or len(arr) == 0:
        return result
    for row in arr:
        pid = str(row[0])
        x, y, theta = _row_xy_theta(row)
        result[pid] = (x, y, theta)
    return result


def compute_plot_bounds(
    df: pd.DataFrame,
    segment: str = DEFAULT_SEGMENT,
    padding: float = PLOT_PADDING_M,
) -> tuple[float, float, float, float] | None:
    """Compute global (xmin, xmax, ymin, ymax) across the full dataframe.

    Bounds are based on the anchor segment used for circle placement.
    """
    xs: list[float] = []
    ys: list[float] = []
    for row in df["spaceFeat"]:
        if not isinstance(row, dict):
            continue
        arr = row.get(segment)
        if arr is None or len(arr) == 0:
            continue
        for r in arr:
            x = float(r[1])
            y = float(r[2])
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)

    if not xs or not ys:
        return None

    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    return xmin - padding, xmax + padding, ymin - padding, ymax + padding


def _sanitize(tag: str) -> str:
    """Make *tag* safe for use inside a filename."""
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(tag))


def _person_color(idx: int) -> tuple[float, float, float, float]:
    """Pick a stable color for the *idx*-th person in a frame."""
    cmap = plt.get_cmap("tab20")
    return cmap(idx % cmap.N)


def _draw_gt_group_polygons(
    ax,
    groups: list[set[int]],
    anchor_map: dict[str, tuple[float, float, float]],
) -> None:
    """Draw a semi-transparent polygon for each GT conversational group.

    Each polygon connects the anchor positions of the group members. Groups
    with only one member are drawn as a larger ring; pairs as a thick line.
    """
    for gi, group in enumerate(groups):
        color = GROUP_COLORS[gi % len(GROUP_COLORS)]
        # Collect (x, y) for members present in anchor_map
        pts: list[tuple[float, float]] = []
        for member_id in sorted(group):
            entry = anchor_map.get(str(member_id))
            if entry is not None and math.isfinite(entry[0]) and math.isfinite(entry[1]):
                pts.append((entry[0], entry[1]))

        if len(pts) == 0:
            continue
        elif len(pts) == 1:
            # Single-member group: draw a larger circle
            ax.add_patch(plt.Circle(
                pts[0], radius=CIRCLE_RADIUS_M * 2.0,
                facecolor="none", edgecolor=color, linewidth=2.0,
                linestyle="--", alpha=0.7, zorder=1,
            ))
        elif len(pts) == 2:
            # Pair: draw a thick line between them
            ax.plot(
                [pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
                color=color, linewidth=2.5, linestyle="-", alpha=0.5, zorder=1,
            )
        else:
            # 3+ members: convex-hull polygon
            arr = np.array(pts)
            # Sort by angle from centroid for a proper polygon
            cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
            angles = np.arctan2(arr[:, 1] - cy, arr[:, 0] - cx)
            order = np.argsort(angles)
            polygon = Polygon(
                arr[order], closed=True,
                facecolor=color, edgecolor=color,
                alpha=0.18, linewidth=2.0, zorder=1,
            )
            ax.add_patch(polygon)


def plot_single_frame(
    frame_id: str,
    spacefeat: dict,
    source_tag: str,
    output_dir: Path,
    anchor_segment: str = DEFAULT_SEGMENT,
    bounds: tuple[float, float, float, float] | None = None,
    figsize: tuple[float, float] = (8.0, 8.0),
    dpi: int = 100,
    groups: list[set[int]] | None = None,
    time_str: str = "",
    out_path: Path | None = None,
    title_override: str | None = None,
) -> Path | None:
    """Render a single frame and return the output path (or None if empty).

    Circles are placed using ``anchor_segment`` coordinates; arrows for all
    four segments are drawn from that anchor, color-coded per segment.
    If *groups* are provided, a polygon is drawn for each conversational group.
    """
    if not isinstance(spacefeat, dict):
        return None

    segment_maps: dict[str, dict[str, tuple[float, float, float]]] = {
        seg: _segment_map(spacefeat.get(seg)) for seg in SEGMENT_ORDER
    }
    anchor_map = segment_maps.get(anchor_segment, {})
    anchor_people = [
        (pid, xy_theta) for pid, xy_theta in anchor_map.items()
        if math.isfinite(xy_theta[0]) and math.isfinite(xy_theta[1])
    ]
    if not anchor_people:
        return None

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    xs = np.array([xt[0] for _, xt in anchor_people], dtype=np.float64)
    ys = np.array([xt[1] for _, xt in anchor_people], dtype=np.float64)

    if bounds is None:
        pad = PLOT_PADDING_M
        xmin, xmax = float(np.min(xs)) - pad, float(np.max(xs)) + pad
        ymin, ymax = float(np.min(ys)) - pad, float(np.max(ys)) + pad
    else:
        xmin, xmax, ymin, ymax = bounds

    if xmax - xmin < 1.0:
        xmid = 0.5 * (xmin + xmax)
        xmin, xmax = xmid - 1.0, xmid + 1.0
    if ymax - ymin < 1.0:
        ymid = 0.5 * (ymin + ymax)
        ymin, ymax = ymid - 1.0, ymid + 1.0

    for i, (pid, (x, y, _theta_anchor)) in enumerate(anchor_people):
        person_color = _person_color(i)

        circle = plt.Circle(
            (x, y),
            radius=CIRCLE_RADIUS_M,
            facecolor=person_color,
            edgecolor="black",
            linewidth=1.0,
            alpha=0.8,
            zorder=3,
        )
        ax.add_patch(circle)

        for segment in SEGMENT_ORDER:
            seg_entry = segment_maps[segment].get(pid)
            if seg_entry is None:
                continue
            _sx, _sy, theta = seg_entry
            if not math.isfinite(theta):
                continue
            seg_color = SEGMENT_COLORS[segment]
            dx = ARROW_LENGTH_M * math.cos(theta)
            dy = ARROW_LENGTH_M * math.sin(theta)
            ax.arrow(
                x,
                y,
                dx,
                dy,
                head_width=ARROW_HEAD_WIDTH_M,
                head_length=ARROW_HEAD_LENGTH_M,
                fc=seg_color,
                ec=seg_color,
                linewidth=1.2,
                length_includes_head=True,
                alpha=0.9,
                zorder=2,
            )

        ax.text(
            x + CIRCLE_RADIUS_M * 1.3,
            y + CIRCLE_RADIUS_M * 1.3,
            pid,
            fontsize=9,
            color="black",
            zorder=4,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
                pad=1.0,
            ),
        )

    legend_handles = [
        Line2D(
            [0], [0],
            color=SEGMENT_COLORS[seg],
            marker=">",
            markersize=8,
            linewidth=2,
            label=seg,
        )
        for seg in SEGMENT_ORDER
    ]

    # Draw GT group polygons
    if groups:
        _draw_gt_group_polygons(ax, groups, anchor_map)
        for gi, group in enumerate(groups):
            color = GROUP_COLORS[gi % len(GROUP_COLORS)]
            members_str = ",".join(str(m) for m in sorted(group))
            legend_handles.append(
                Line2D(
                    [0], [0],
                    color=color,
                    linewidth=3,
                    alpha=0.6,
                    label=f"GT group {{{members_str}}}",
                )
            )

    ax.legend(
        handles=legend_handles,
        title="Orientation source",
        loc="upper right",
        framealpha=0.85,
        fontsize=8,
        title_fontsize=9,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    if title_override is not None:
        ax.set_title(title_override)
    else:
        time_part = f"  |  {time_str}" if time_str else ""
        ax.set_title(
            f"{source_tag}  |  frame {frame_id}{time_part}  |  anchor={anchor_segment}  |  n={len(anchor_people)}"
        )

    if out_path is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_tag = _sanitize(source_tag)
        safe_frame = _sanitize(frame_id)
        out_path = output_dir / f"{safe_tag}__frame_{safe_frame}.png"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# Conflab-17 keypoint pairs for the four body segments we visualize.
KEYPOINT_PAIRS = {
    "head":     (0, 1),    # head, nose
    "shoulder": (6, 3),    # left_shoulder, right_shoulder
    "hip":      (12, 9),   # left_hip, right_hip
    "foot":     (16, 15),  # left_foot, right_foot
}


def _pixel_xy(raw_kps, kp_idx: int, conf_thresh: float = 0.0):
    """Return ``(x, y)`` for a raw Conflab keypoint if its confidence passes."""
    if raw_kps is None or len(raw_kps) <= kp_idx:
        return None
    kp = raw_kps[kp_idx]
    if len(kp) < 3 or float(kp[2]) < conf_thresh:
        return None
    x, y = float(kp[0]), float(kp[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _world_xy(kp_world, kp_idx: int):
    """Return ``(x, y)`` for a projected-world keypoint if present."""
    if kp_world is None or len(kp_world) <= kp_idx:
        return None
    kp = kp_world[kp_idx]
    if kp is None:
        return None
    x, y = float(kp[0]), float(kp[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _numeric_sort_key(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def plot_keypoints_subplots(
    frame_id: str,
    raw_by_track: dict,
    world_by_track: dict,
    image_path: Path | None,
    out_path: Path,
    source_tag: str,
    world_bounds: tuple[float, float, float, float] | None = None,
    time_str: str = "",
    seg_info: str = "",
    conf_thresh: float = 0.0,
    marker_size_px: float = 5.0,
    marker_size_world: float = 6.0,
    dpi: int = 100,
    keypoint_image_size: tuple[int, int] | None = None,
) -> Path:
    """Render a two-subplot figure combining pixel keypoints and the bird's-eye view.

    Left subplot: background frame image (if available) with head / shoulder /
    hip / foot left+right keypoints drawn on their raw pixel positions,
    connected by a thin line per pair. When ``keypoint_image_size`` is set to
    the (width, height) the raw keypoints were produced at, and the background
    image is at a different resolution, the keypoints are scaled so they line
    up with the image pixels.

    Right subplot: the same keypoints after world back-projection, drawn as a
    top-down (bird's-eye) view.
    """
    fig, (ax_img, ax_world) = plt.subplots(1, 2, figsize=(16.0, 8.0), dpi=dpi)

    img_loaded = False
    img_scale_x = 1.0
    img_scale_y = 1.0
    if image_path is not None:
        image_path = Path(image_path)
        if image_path.is_file():
            try:
                img = plt.imread(str(image_path))
                ax_img.imshow(img)
                img_loaded = True
                if keypoint_image_size is not None and img.ndim >= 2:
                    img_h, img_w = img.shape[0], img.shape[1]
                    kp_w, kp_h = keypoint_image_size
                    if kp_w > 0 and kp_h > 0:
                        img_scale_x = img_w / float(kp_w)
                        img_scale_y = img_h / float(kp_h)
            except Exception as exc:  # noqa: BLE001
                ax_img.text(
                    0.5, 0.5, f"Failed to read:\n{image_path}\n({exc})",
                    color="white", ha="center", va="center",
                    transform=ax_img.transAxes, fontsize=9,
                )
    if not img_loaded:
        ax_img.set_facecolor("#222222")
        msg = "Frame image not found"
        if image_path is not None:
            msg = f"Frame image not found:\n{image_path}"
        ax_img.text(
            0.5, 0.5, msg, color="white", ha="center", va="center",
            transform=ax_img.transAxes, fontsize=10,
        )

    track_ids = sorted(raw_by_track.keys(), key=_numeric_sort_key)

    for track_id in track_ids:
        raw = raw_by_track.get(track_id)
        world = world_by_track.get(track_id, [None] * 17)

        for seg_name, (l_idx, r_idx) in KEYPOINT_PAIRS.items():
            color = SEGMENT_COLORS[seg_name]

            l_px = _pixel_xy(raw, l_idx, conf_thresh)
            r_px = _pixel_xy(raw, r_idx, conf_thresh)
            l_px_s = (l_px[0] * img_scale_x, l_px[1] * img_scale_y) if l_px is not None else None
            r_px_s = (r_px[0] * img_scale_x, r_px[1] * img_scale_y) if r_px is not None else None
            for px in (l_px_s, r_px_s):
                if px is not None:
                    ax_img.plot(
                        px[0], px[1], marker="o", color=color,
                        markersize=marker_size_px, markeredgecolor="white",
                        markeredgewidth=0.6, linestyle="none", zorder=3,
                    )
            if l_px_s is not None and r_px_s is not None:
                ax_img.plot(
                    [l_px_s[0], r_px_s[0]], [l_px_s[1], r_px_s[1]],
                    color=color, linewidth=1.1, alpha=0.85, zorder=2,
                )

            l_w = _world_xy(world, l_idx)
            r_w = _world_xy(world, r_idx)
            for wp in (l_w, r_w):
                if wp is not None:
                    ax_world.plot(
                        wp[0], wp[1], marker="o", color=color,
                        markersize=marker_size_world, markeredgecolor="black",
                        markeredgewidth=0.5, linestyle="none", zorder=3,
                    )
            if l_w is not None and r_w is not None:
                ax_world.plot(
                    [l_w[0], r_w[0]], [l_w[1], r_w[1]],
                    color=color, linewidth=1.1, alpha=0.85, zorder=2,
                )

        sh_l = _pixel_xy(raw, 5, conf_thresh)
        sh_r = _pixel_xy(raw, 6, conf_thresh)
        if sh_l is not None and sh_r is not None:
            cx = (sh_l[0] + sh_r[0]) / 2.0 * img_scale_x
            cy = (sh_l[1] + sh_r[1]) / 2.0 * img_scale_y
            ax_img.text(
                cx, cy - 12.0 * img_scale_y, track_id, color="yellow",
                fontsize=9, ha="center", zorder=4,
                bbox=dict(facecolor="black", edgecolor="none",
                          alpha=0.55, pad=1.0),
            )

        hip_l = _world_xy(world, 11)
        hip_r = _world_xy(world, 12)
        if hip_l is not None and hip_r is not None:
            hx = (hip_l[0] + hip_r[0]) / 2.0
            hy = (hip_l[1] + hip_r[1]) / 2.0
            ax_world.text(
                hx + 0.08, hy + 0.08, track_id, color="black", fontsize=9,
                zorder=4,
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.7, pad=1.0),
            )

    ax_img.set_title("Keypoints on frame (pixel coords)")
    ax_img.set_xlabel("x (px)")
    ax_img.set_ylabel("y (px)")
    if img_loaded:
        ax_img.set_aspect("equal", adjustable="box")

    if world_bounds is not None:
        xmin, xmax, ymin, ymax = world_bounds
        ax_world.set_xlim(xmin, xmax)
        ax_world.set_ylim(ymin, ymax)
    ax_world.set_aspect("equal", adjustable="box")
    ax_world.grid(True, linestyle=":", alpha=0.5)
    ax_world.set_title("Projected keypoints (bird's-eye view)")
    ax_world.set_xlabel("X (m)")
    ax_world.set_ylabel("Y (m)")

    legend_handles = [
        Line2D(
            [0], [0], color=SEGMENT_COLORS[seg], marker="o",
            markersize=7, linestyle="-", linewidth=1.5, label=seg,
        )
        for seg in SEGMENT_ORDER
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4,
        frameon=True, fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )

    extra = []
    if time_str:
        extra.append(time_str)
    if seg_info:
        extra.append(seg_info)
    extra_part = "  |  " + "  |  ".join(extra) if extra else ""
    fig.suptitle(
        f"{source_tag}  |  frame {frame_id}{extra_part}",
        fontsize=11, y=0.98,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_dataframe_positions(
    df: pd.DataFrame,
    source_tag: str,
    output_dir: str | Path,
    frame_interval: int = DEFAULT_FRAME_INTERVAL,
    segment: str = DEFAULT_SEGMENT,
    shared_bounds: bool = True,
) -> int:
    """Plot every *frame_interval*-th row of *df* and return the number of plots written.

    Args:
        df: dataframe with a ``spaceFeat`` column (see module docstring).
        source_tag: tag used in filenames (e.g. ``cam06_batch01``).
        output_dir: directory where png files are written (created if missing).
        frame_interval: take one plot every N rows (positional, not by frame id).
        segment: anchor segment used to place each person's circle and compute
            axis bounds. All four segment orientations are always drawn when
            available. One of {``head``, ``shoulder``, ``hip``, ``foot``}.
        shared_bounds: if True, use the same axis limits across all frames in
            this dataframe so motion between frames is easy to compare.
    """
    if frame_interval <= 0:
        raise ValueError("frame_interval must be positive")
    if len(df) == 0:
        print(f"  [plot] {source_tag}: empty dataframe, skipping.")
        return 0

    output_dir = Path(output_dir)
    bounds = compute_plot_bounds(df, segment=segment) if shared_bounds else None

    selected = df.iloc[::frame_interval]
    print(f"  [plot] {source_tag}: writing {len(selected)} plots to {output_dir}")

    written = 0
    for frame_id, row in selected.iterrows():
        groups = row.get("groups") if "groups" in row.index else None
        time_str = row.get("time", "") if "time" in row.index else ""
        out_path = plot_single_frame(
            frame_id=str(frame_id),
            spacefeat=row["spaceFeat"],
            source_tag=source_tag,
            output_dir=output_dir,
            anchor_segment=segment,
            bounds=bounds,
            groups=groups if isinstance(groups, list) and groups else None,
            time_str=str(time_str) if time_str else "",
        )
        if out_path is not None:
            written += 1

    print(f"  [plot] {source_tag}: wrote {written} / {len(selected)} plots "
          f"(skipped empty frames).")
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-frame positions plus all four segment-orientation arrows "
            "(head/shoulder/hip/foot) from a vitpose dataframe."
        ),
    )
    parser.add_argument("input_pkl", help="Path to a vitpose_dataframe.pkl file.")
    parser.add_argument(
        "--source-tag",
        default=None,
        help="Tag used in output filenames. Defaults to the parent folder name of the pkl.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where png files are written.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=DEFAULT_FRAME_INTERVAL,
        help="Plot every N-th row of the dataframe (default: %(default)s).",
    )
    parser.add_argument(
        "--segment",
        default=DEFAULT_SEGMENT,
        choices=list(SEGMENT_ORDER),
        help=(
            "Anchor segment used for circle positions and axis bounds "
            "(all four segment orientations are always drawn as arrows). "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--no-shared-bounds",
        action="store_true",
        help="Use per-frame axis bounds instead of one shared bound across the pkl.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_pkl = Path(args.input_pkl)
    source_tag = args.source_tag or input_pkl.parent.name
    df = pd.read_pickle(input_pkl)
    plot_dataframe_positions(
        df,
        source_tag=source_tag,
        output_dir=args.output_dir,
        frame_interval=args.frame_interval,
        segment=args.segment,
        shared_bounds=not args.no_shared_bounds,
    )


if __name__ == "__main__":
    main()
