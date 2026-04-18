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
CIRCLE_RADIUS_M = 0.2
ARROW_LENGTH_M = 0.6
ARROW_HEAD_WIDTH_M = 0.15
ARROW_HEAD_LENGTH_M = 0.2
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
    time_part = f"  |  {time_str}" if time_str else ""
    ax.set_title(
        f"{source_tag}  |  frame {frame_id}{time_part}  |  anchor={anchor_segment}  |  n={len(anchor_people)}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = _sanitize(source_tag)
    safe_frame = _sanitize(frame_id)
    out_path = output_dir / f"{safe_tag}__frame_{safe_frame}.png"
    fig.tight_layout()
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
