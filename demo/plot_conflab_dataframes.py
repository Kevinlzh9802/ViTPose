"""
plot_conflab_dataframes.py

Load per-batch vitpose_dataframe.pkl files produced by demo/vitpose_to_dataframe.py
for the conflab dataset and write top-view BEV position plots at a fixed frame
interval into each batch's own output folder.

Conflab pkls store world coordinates in centimetres (extrinsics in cm,
body_height=170).  plot_person assumes metres, so x/y are scaled by 0.01 before
plotting.

Usage
-----
# specific batches
python demo/plot_conflab_dataframes.py <results_dir> --batch=228,229

# all batches found under results_dir
python demo/plot_conflab_dataframes.py <results_dir> --batch=all

# custom interval (default 300 = every 5 s at 60 fps)
python demo/plot_conflab_dataframes.py <results_dir> --batch=all --frame_interval=600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from demo.plot_person import plot_dataframe_positions

# Conflab world coordinates are in centimetres; plot_person expects metres.
_CM_TO_M = 0.01


def _scale_spacefeat_df(df: pd.DataFrame, factor: float = _CM_TO_M) -> pd.DataFrame:
    """Return a copy of *df* with spaceFeat x/y values multiplied by *factor*.

    Each spaceFeat entry is a dict mapping segment names to (n_people, 4)
    object arrays whose columns are [person_id, x, y, orientation].  Only
    columns 1 (x) and 2 (y) are scaled; person_id and orientation are left
    unchanged.  Empty arrays are passed through as-is.
    """
    def _scale_sf(sf: dict) -> dict:
        out = {}
        for seg, arr in sf.items():
            if arr is None or len(arr) == 0:
                out[seg] = arr
                continue
            arr2 = arr.copy()
            arr2[:, 1] = (arr[:, 1].astype(float) * factor).astype(object)
            arr2[:, 2] = (arr[:, 2].astype(float) * factor).astype(object)
            out[seg] = arr2
        return out

    df2 = df.copy()
    df2["spaceFeat"] = df["spaceFeat"].map(_scale_sf)
    return df2


def _resolve_batch_dirs(
    results_dir: Path, batches_raw: str
) -> list[tuple[str, Path]]:
    """Return ``[(batch_folder_name, pkl_path), ...]`` for *batches_raw*.

    *batches_raw* is either ``"all"`` or a comma-separated list of 3-digit
    batch numbers.  Folders are matched by the suffix ``*_batch<BBB>``.
    A warning is printed for any requested batch with no matching pkl.
    """
    if batches_raw.strip().lower() == "all":
        pkls = sorted(results_dir.glob("*/vitpose_dataframe.pkl"))
        if not pkls:
            raise FileNotFoundError(
                f"No vitpose_dataframe.pkl found under {results_dir}"
            )
        return [(pkl.parent.name, pkl) for pkl in pkls]

    batch_numbers = [b.strip() for b in batches_raw.split(",") if b.strip()]
    results: list[tuple[str, Path]] = []
    for batch in batch_numbers:
        matches = sorted(results_dir.glob(f"*_batch{batch}/vitpose_dataframe.pkl"))
        if not matches:
            print(f"  Warning: no vitpose_dataframe.pkl for batch {batch} under {results_dir}")
            continue
        for pkl in matches:
            results.append((pkl.parent.name, pkl))
    return results


def process(
    results_dir: Path,
    batches_raw: str,
    frame_interval: int,
    output_dir: Path | None = None,
) -> None:
    output_root = output_dir if output_dir is not None else results_dir
    batch_dirs = _resolve_batch_dirs(results_dir, batches_raw)

    print(f"Plotting conflab BEV positions")
    print(f"  results_dir={results_dir}")
    print(f"  output_root={output_root}")
    print(f"  batches found: {len(batch_dirs)}")
    print(f"  frame_interval={frame_interval}")

    total_written = 0
    for batch_name, pkl_path in batch_dirs:
        print(f"\n  [{batch_name}] loading {pkl_path}")
        df = pd.read_pickle(pkl_path)
        df_m = _scale_spacefeat_df(df, factor=_CM_TO_M)

        batch_out_dir = output_root / batch_name
        batch_out_dir.mkdir(parents=True, exist_ok=True)

        n = plot_dataframe_positions(
            df_m,
            source_tag=batch_name,
            output_dir=batch_out_dir,
            frame_interval=frame_interval,
        )
        total_written += n

    print(f"\nDone. Total plots written: {total_written}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot top-view BEV positions from conflab vitpose_dataframe pkl files. "
            "Images are written alongside each pkl in the same batch folder."
        )
    )
    parser.add_argument(
        "results_dir",
        help="Root containing cam<NN>_batch<BBB>/vitpose_dataframe.pkl subfolders.",
    )
    parser.add_argument(
        "--batch",
        required=True,
        metavar="BATCHES",
        help="Comma-separated 3-digit batch numbers (e.g. 228,229) or 'all'.",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=300,
        help="Write one plot every N rows of the dataframe (default: 300 = 5 s at 60 fps).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Root for output images; plots go to <output_dir>/<batch_name>/. "
            "Defaults to results_dir (images stored alongside each pkl)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process(
        results_dir=Path(args.results_dir),
        batches_raw=args.batch,
        frame_interval=args.frame_interval,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
