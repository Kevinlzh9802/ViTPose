"""
vitpose_to_gcff.py

Convert per-batch ViTPose dataframe pickles into a single GCFF-compatible
DataFrame and save it as data.pkl.

Batch numbering convention: <Cam><Vid><Seg>  (3 digits)
  228  →  Cam=2, Vid=2, Seg=8
  431  →  Cam=4, Vid=3, Seg=1

Expected layout per camera:
  cam ∈ {2, 4, 6, 8}
  vid ∈ {2, 3}
  seg: 8–9 (vid=2),  1–6 (vid=3)

Input pkl schema (one per batch, from vitpose_to_dataframe.py):
  index:    frame_id (int-valued string)
  columns:  timestamp, time, spaceFeat, pixelFeat, groups, group_ids
  spaceFeat per row: {clue: (N, 4) object array  [str_person_id, x_cm, y_cm, alpha]}
  pixelFeat per row: {clue: (N, 4) object array  [str_person_id, u, v, alpha]}
                     u/v in absolute pixels at intrinsic resolution (1920×1080)

Output pkl schema (GCFF-compatible):
  row_id, Cam, Vid, Seg, Timestamp, concat_ts,
  pixelCoords, spaceCoords, pixelFeat, spaceFeat, GT

  spaceFeat per row: {clue: (N, 4) float64  [person_id, x_m, y_m, alpha]}
  pixelFeat per row: {clue: (N, 4) float64  [person_id, u_rel, v_rel, alpha]}
                     u_rel = u/1920, v_rel = v/1080  (both in [0, 1])
  concat_ts:  continuous 0-based timeline per (Cam, Vid) pair, +1 gap at
              segment boundaries, sorted by ascending Seg within each pair.

Usage
-----
# All batches (default):
python demo/vitpose_to_gcff.py

# Specific batches:
python demo/vitpose_to_gcff.py --batch=228,229,431

# Keep world coords in cm (skip cm→m scaling):
python demo/vitpose_to_gcff.py --world_scale=1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NEON = "/tudelft.net/staff-umbrella/neon"
DEFAULT_VITPOSE_ROOT = f"{NEON}/zonghuan/data/conflab/vitpose_dataframe"
DEFAULT_OUTPUT = f"{NEON}/zonghuan/data/conflab/GCFF/data.pkl"

CLUES = ["head", "shoulder", "hip", "foot"]


# ---------------------------------------------------------------------------
# spaceFeat normalisation
# ---------------------------------------------------------------------------

def _normalise_spacefeat(sf: dict, world_scale: float) -> dict:
    """Convert object-dtype per-clue arrays to float64 and scale x/y.

    ViTPose stores person_id as a string inside a dtype=object array.
    GCFF's graph_cut() calls feat[:, 0].astype(int), which works on
    float64 but not reliably on object strings — convert here.

    Columns: [person_id, x, y, alpha]
    """
    out = {}
    for clue in CLUES:
        arr = sf.get(clue)
        if arr is None or (hasattr(arr, "__len__") and len(arr) == 0):
            out[clue] = np.empty((0, 4), dtype=np.float64)
            continue
        arr_f = np.asarray(arr, dtype=np.float64)
        if world_scale != 1.0 and arr_f.shape[0] > 0:
            arr_f[:, 1] *= world_scale  # x
            arr_f[:, 2] *= world_scale  # y
        out[clue] = arr_f
    return out


def _normalise_kp_pairs_space(kp_pairs: dict, world_scale: float) -> dict:
    """Normalize spaceKpPairs: convert object dtype to float64 and apply world_scale to x/y."""
    out = {}
    for clue in CLUES:
        arr = kp_pairs.get(clue) if isinstance(kp_pairs, dict) else None
        if arr is None or (hasattr(arr, "__len__") and len(arr) == 0):
            out[clue] = np.empty((0, 5), dtype=np.float64)
            continue
        a = np.asarray(arr, dtype=np.float64)  # (N, 5): [pid, x1, y1, x2, y2]
        if world_scale != 1.0 and a.shape[0] > 0:
            a[:, 1] *= world_scale; a[:, 2] *= world_scale
            a[:, 3] *= world_scale; a[:, 4] *= world_scale
        out[clue] = a
    return out


def _normalise_kp_pairs_pixel(kp_pairs: dict) -> dict:
    """Normalize pixelKpPairs: convert to float64 and divide u/v by 1920/1080."""
    out = {}
    for clue in CLUES:
        arr = kp_pairs.get(clue) if isinstance(kp_pairs, dict) else None
        if arr is None or (hasattr(arr, "__len__") and len(arr) == 0):
            out[clue] = np.empty((0, 5), dtype=np.float64)
            continue
        a = np.asarray(arr, dtype=np.float64)  # (N, 5): [pid, u1, v1, u2, v2]
        if a.shape[0] > 0:
            a[:, 1] /= 1920.0; a[:, 2] /= 1080.0
            a[:, 3] /= 1920.0; a[:, 4] /= 1080.0
        out[clue] = a
    return out


def _fix_orientation_outlier(sf: dict) -> dict:
    """Flip the one segment orientation that is opposite to the other three.

    For each person, collect the 4 orientations (one per clue in CLUES order).
    Convert each to a 2-D unit vector (cos θ, sin θ) and compute all pairwise
    dot products.  Exactly one orientation qualifies as the "odd one out" when:

        • its dot product with every other orientation is negative  (> 90° apart)
        • every pair among the remaining three has a positive dot product  (< 90°)

    In that case the orientation is reversed by adding π.  All other
    configurations (0, 2, 3, or 4 outliers) are left unchanged.
    """
    n_clues = len(CLUES)
    sf = {k: v.copy() for k, v in sf.items()}
    n_people = sf[CLUES[0]].shape[0] if CLUES[0] in sf else 0

    for i in range(n_people):
        thetas = np.array([sf[clue][i, 3] for clue in CLUES])
        if np.isnan(thetas).any():
            continue

        vecs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)  # (4, 2)
        dots = vecs @ vecs.T  # (4, 4)

        odd = None
        for j in range(n_clues):
            others = [k for k in range(n_clues) if k != j]
            if (all(dots[j, k] < 0 for k in others) and
                    all(dots[k, l] > 0 for k in others for l in others if k != l)):
                odd = j
                break

        if odd is not None:
            sf[CLUES[odd]][i, 3] += np.pi

    return sf


def _normalise_pixelcoords(pc: dict | None) -> dict | None:
    """Convert object-dtype per-clue pixel arrays to float64 with relative coords.

    u and v are divided by the intrinsic image width (1920) and height (1080),
    yielding values in [0, 1] that are resolution-agnostic.

    Columns: [person_id, u_rel, v_rel, orientation]
    """
    if pc is None:
        return None
    out = {}
    for clue in CLUES:
        arr = pc.get(clue)
        if arr is None or (hasattr(arr, "__len__") and len(arr) == 0):
            out[clue] = np.empty((0, 4), dtype=np.float64)
            continue
        arr_f = np.asarray(arr, dtype=np.float64)
        if arr_f.shape[0] > 0 and arr_f.shape[1] >= 3:
            arr_f[:, 1] /= 1920.0  # u → [0, 1]
            arr_f[:, 2] /= 1080.0  # v → [0, 1]
        out[clue] = arr_f
    return out


# ---------------------------------------------------------------------------
# concat_ts construction
# ---------------------------------------------------------------------------

def _build_concat_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``concat_ts`` column: continuous 0-based timeline per (Cam, Vid).

    Segments within each (Cam, Vid) group are chained in ascending Seg order.
    A gap of +1 is inserted between consecutive segments so that no two rows
    share the same concat_ts value.
    """
    df = df.copy()
    df["concat_ts"] = np.nan

    for (cam, vid), group_idx in df.groupby(["Cam", "Vid"]).groups.items():
        sub = df.loc[group_idx]
        seg_order = sorted(sub["Seg"].unique())
        offset = 0.0
        for seg in seg_order:
            seg_row_idx = sub.index[sub["Seg"] == seg]
            ts = df.loc[seg_row_idx, "Timestamp"].to_numpy(dtype=float)
            df.loc[seg_row_idx, "concat_ts"] = offset + ts
            offset = float(df.loc[seg_row_idx, "concat_ts"].iloc[-1]) + 1.0

    return df


# ---------------------------------------------------------------------------
# Batch discovery
# ---------------------------------------------------------------------------

def _resolve_batches(vitpose_root: Path, batches_raw: str) -> list[int]:
    if batches_raw.strip().lower() == "all":
        batches = []
        for d in sorted(vitpose_root.iterdir()):
            if d.is_dir() and d.name.isdigit() and len(d.name) == 3:
                pkl = d / "vitpose_dataframe.pkl"
                if pkl.is_file():
                    batches.append(int(d.name))
        if not batches:
            raise FileNotFoundError(
                f"No <3-digit-batch>/vitpose_dataframe.pkl found under {vitpose_root}"
            )
        return batches

    batches = []
    for b in batches_raw.split(","):
        b = b.strip()
        if not b:
            continue
        pkl = vitpose_root / b / "vitpose_dataframe.pkl"
        if not pkl.is_file():
            print(f"  Warning: pkl not found for batch {b}: {pkl}")
        else:
            batches.append(int(b))
    return batches


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    vitpose_root: Path,
    output_path: Path,
    batches_raw: str = "all",
    world_scale: float = 0.01,
) -> None:
    batches = _resolve_batches(vitpose_root, batches_raw)
    if not batches:
        raise SystemExit("No valid batches found.")

    print(f"Converting {len(batches)} batch(es): {batches}")
    print(f"  world_scale={world_scale}  (x/y multiplied before saving)")

    all_dfs: list[pd.DataFrame] = []

    for batch in sorted(batches):
        pkl_path = vitpose_root / str(batch) / "vitpose_dataframe.pkl"
        print(f"  [{batch}] loading {pkl_path}")
        try:
            df = pd.read_pickle(pkl_path)
        except Exception as exc:
            print(f"  Warning: failed to load batch {batch}: {exc}. Skipping.")
            continue

        if "spaceFeat" not in df.columns:
            print(
                f"  Warning: [{batch}] pkl has no spaceFeat column "
                f"(columns={list(df.columns)}). Skipping."
            )
            continue

        cam = int(str(batch)[0])
        vid = int(str(batch)[1])
        seg = int(str(batch)[2])

        has_pixel = "pixelFeat" in df.columns
        has_kp_pairs = "spaceCoords" in df.columns
        rows_list = list(df.iterrows())
        new_df = pd.DataFrame({
            "Cam":        [cam] * len(df),
            "Vid":        [vid] * len(df),
            "Seg":        [seg] * len(df),
            "Timestamp":  list(range(len(df))),
            "spaceFeat":  [
                _fix_orientation_outlier(_normalise_spacefeat(row["spaceFeat"], world_scale))
                for _, row in rows_list
            ],
            "pixelFeat":  [
                _normalise_pixelcoords(row["pixelFeat"]) if has_pixel else None
                for _, row in rows_list
            ],
            "spaceCoords": [
                _normalise_kp_pairs_space(row["spaceCoords"], world_scale)
                if has_kp_pairs else {c: np.empty((0, 5), dtype=np.float64) for c in CLUES}
                for _, row in rows_list
            ],
            "pixelCoords": [
                _normalise_kp_pairs_pixel(row["pixelCoords"])
                if has_kp_pairs else {c: np.empty((0, 5), dtype=np.float64) for c in CLUES}
                for _, row in rows_list
            ],
            "GT":          [[] for _ in range(len(df))],
        })

        all_dfs.append(new_df)
        print(f"    {len(new_df)} rows  (Cam={cam}, Vid={vid}, Seg={seg})")

    if not all_dfs:
        raise SystemExit("No batches were converted successfully.")

    merged = pd.concat(all_dfs, ignore_index=True)
    merged["row_id"] = list(range(len(merged)))

    print(f"\nBuilding concat_ts for {merged.groupby(['Cam','Vid']).ngroups} (Cam, Vid) pair(s)...")
    merged = _build_concat_ts(merged)

    # Final column order matching GCFF's expected schema
    cols = [
        "row_id", "Cam", "Vid", "Seg", "Timestamp", "concat_ts",
        "pixelCoords", "spaceCoords", "pixelFeat", "spaceFeat", "GT",
    ]
    merged = merged[cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_pickle(output_path)

    print(f"\nDone.")
    print(f"  Total rows : {len(merged)}")
    print(f"  Output     : {output_path}")
    print("\nconcat_ts ranges per (Cam, Vid):")
    for (cam, vid), grp in merged.groupby(["Cam", "Vid"]):
        print(
            f"  Cam={cam} Vid={vid}: "
            f"concat_ts [{grp['concat_ts'].min():.0f}, {grp['concat_ts'].max():.0f}]  "
            f"({grp.groupby('Seg').size().to_dict()})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert per-batch ViTPose pkls into a single GCFF-compatible pkl."
    )
    parser.add_argument(
        "--vitpose_root",
        default=DEFAULT_VITPOSE_ROOT,
        help=(
            "Root directory containing <batch>/vitpose_dataframe.pkl subfolders "
            f"(default: {DEFAULT_VITPOSE_ROOT})."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output path for data.pkl (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--batch",
        default="all",
        metavar="BATCHES",
        help="Comma-separated 3-digit batch numbers (e.g. 228,431) or 'all' (default).",
    )
    parser.add_argument(
        "--world_scale",
        type=float,
        default=0.01,
        help=(
            "Multiply spaceFeat x/y by this factor before saving. "
            "Default 0.01 converts cm → metres to match GCFF's metre-scale params. "
            "Use 1.0 to keep cm units."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert(
        vitpose_root=Path(args.vitpose_root),
        output_path=Path(args.output),
        batches_raw=args.batch,
        world_scale=args.world_scale,
    )
