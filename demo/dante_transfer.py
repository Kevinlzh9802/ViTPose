import argparse
import json
import os
import pickle
import re
import shutil
import sys

import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(DATASETS_DIR)
if DATASETS_DIR not in sys.path:
    sys.path.append(DATASETS_DIR)

from build_dataset import build_X, build_Y, split_test_train_val
from reformat_data import add_time, affinities_and_timechanges, compute_data_shift


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an InGroup dataframe into the repository's dataset format."
    )
    parser.add_argument("--input", required=True, help="Path to a pickled dataframe, CSV, or parquet file.")
    parser.add_argument(
        "--dataset-name",
        default="ingroup",
        help="Output dataset folder under datasets/; generated files are written to datasets/<dataset-name>/data.",
    )
    parser.add_argument("--spacefeat-col", default="spaceFeat", help="Column containing the per-frame feature dict.")
    parser.add_argument("--head-key", default="head", help="Key inside spaceFeat containing the n x 4 head array.")
    parser.add_argument("--time-col", default=None, help="Optional frame/timestamp column. Defaults to dataframe index.")
    parser.add_argument(
        "--group-col",
        default=None,
        help="Column containing frame groups as an iterable of iterables of person ids.",
    )
    parser.add_argument(
        "--group-ids-col",
        default=None,
        help="Column containing one group id per detected person in the same order as the head array.",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=0.01,
        help="Scale factor applied to x/y coordinates. Use 0.01 to convert cm to meters.",
    )
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds to generate.")
    parser.add_argument(
        "--singleton-values",
        nargs="*",
        default=["-1", "None", "nan"],
        help="Group-id values that should be treated as singleton people when using --group-ids-col.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing datasets/<dataset-name>/data directory.",
    )
    return parser.parse_args()


# def load_tabular_input(path):
#     _, ext = os.path.splitext(path)
#     ext = ext.lower()
#     with open(path, "rb") as f:
#         return pickle.load(f)

def canonical_person_id(idx):
    return "ID_{:03d}".format(idx)


def stringify_timestamp(value):
    return str(value)


def numeric_sort_key(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def normalize_head_array(spacefeat_value, head_key):
    if isinstance(spacefeat_value, dict):
        head_value = spacefeat_value[head_key]
    else:
        head_value = spacefeat_value

    head = np.asarray(head_value)
    if head.ndim != 2 or head.shape[1] != 4:
        raise ValueError("Expected a head array with shape (n_people, 4); got {}".format(head.shape))
    return head


def normalize_singleton_values(values):
    normalized = set()
    for value in values:
        normalized.add(str(value))
    return normalized

def singleton_to_list(group):
    if isinstance(group, list):
        return group
    return [group]

def normalize_groups(frame_value, person_ids, group_col=None, group_ids_col=None, singleton_values=None):
    if group_col:
        raw_groups = frame_value[group_col] # raw_groups is list of lists of person ids
        raw_groups = [singleton_to_list(group) for group in raw_groups]
        
        groups = []
        for group in raw_groups:
            members = [member for member in list(group) if member in person_ids]
            if members:
                groups.append(members)
        return fill_missing_singletons(groups, person_ids)

    if group_ids_col:
        raw_group_ids = frame_value[group_ids_col]
        groups_by_label = {}
        singleton_values = singleton_values or set()

        if isinstance(raw_group_ids, dict):
            group_id_sequence = [raw_group_ids.get(person_id) for person_id in person_ids]
        else:
            group_id_sequence = list(raw_group_ids)

        if len(group_id_sequence) != len(person_ids):
            raise ValueError(
                "group id sequence length ({}) does not match detected people ({})".format(
                    len(group_id_sequence), len(person_ids)
                )
            )

        for person_id, group_id in zip(person_ids, group_id_sequence):
            if group_id is None:
                groups_by_label[("__singleton__", person_id)] = [person_id]
                continue

            group_label = str(group_id)
            if group_label in singleton_values:
                groups_by_label[("__singleton__", person_id)] = [person_id]
                continue

            groups_by_label.setdefault(group_label, []).append(person_id)

        return fill_missing_singletons(list(groups_by_label.values()), person_ids)

    raise ValueError("Either --group-col or --group-ids-col must be provided.")


def fill_missing_singletons(groups, present_person_ids):
    covered = set()
    normalized = []
    for group in groups:
        deduped = []
        for person_id in group:
            if person_id in covered:
                continue
            covered.add(person_id)
            deduped.append(person_id)
        if deduped:
            normalized.append(deduped)

    for person_id in present_person_ids:
        if person_id not in covered:
            normalized.append([person_id])
    return normalized


def build_global_person_map(frame_iterable, spacefeat_col, head_key):
    seen_ids = set()
    for _, frame_value in frame_iterable:
        head = normalize_head_array(frame_value[spacefeat_col], head_key)
        for person_id in head[:, 0].tolist():
            seen_ids.add(person_id)

    sorted_ids = sorted(seen_ids, key=numeric_sort_key)
    return {person_id: idx + 1 for idx, person_id in enumerate(sorted_ids)}


def build_feature_line(timestamp, head, dense_person_map, n_people, position_scale):
    slots = [["fake", "fake", "fake", "fake"] for _ in range(n_people)]
    for row in head:
        external_person_id = row[0]
        dense_person_id = dense_person_map[external_person_id]
        x = float(row[1]) * position_scale
        y = float(row[2]) * position_scale
        theta = float(row[3])
        slots[dense_person_id - 1] = [
            canonical_person_id(dense_person_id),
            "{:.8f}".format(x),
            "{:.8f}".format(y),
            "{:.8f}".format(theta),
        ]

    line = [timestamp]
    for slot in slots:
        line.extend(slot)
    return " ".join(line)


def build_group_line(timestamp, groups, dense_person_map):
    group_strings = []
    for group in groups:
        dense_ids = sorted([dense_person_map[person_id] for person_id in group])
        group_strings.append("< {} >".format(" ".join(canonical_person_id(person_id) for person_id in dense_ids)))
    return "{} {}".format(timestamp, " ".join(group_strings)).strip()


def write_lines(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def dump_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def compute_time_splits(timechanges, total_len, fold_index, num_folds):
    time_boundaries = list(timechanges) + [total_len]
    num_times = len(time_boundaries) - 1

    start_test_idx = int(num_times / num_folds * fold_index)
    end_test_idx = int(num_times / num_folds * (fold_index + 1))
    val_fold_idx_diff = int(num_times / num_folds / 2)

    start_test = time_boundaries[start_test_idx]
    end_test = time_boundaries[end_test_idx]

    if start_test == 0:
        val_start = end_test
        val_end_idx = end_test_idx + 2 * val_fold_idx_diff
        val_end = time_boundaries[val_end_idx]
        val_start_ends = [(val_start, val_end)]
    elif fold_index == num_folds - 1:
        val_start_idx = start_test_idx - 2 * val_fold_idx_diff
        val_start = time_boundaries[val_start_idx]
        val_end = start_test
        val_start_ends = [(val_start, val_end)]
    else:
        val_start_idx = start_test_idx - val_fold_idx_diff
        val_start = time_boundaries[val_start_idx]
        val_end = start_test
        val_start_ends = [(val_start, val_end)]

        val_start = end_test
        val_end_idx = end_test_idx + val_fold_idx_diff
        val_end = time_boundaries[val_end_idx]
        val_start_ends.append((val_start, val_end))

    return start_test, end_test, val_start_ends


def prepare_from_dataframe(
    df,
    dataset_name="ingroup",
    spacefeat_col="spaceFeat",
    head_key="head",
    time_col=None,
    group_col=None,
    group_ids_col=None,
    position_scale=0.01,
    num_folds=5,
    singleton_values=None,
    overwrite=False,
):
    if group_col is None and group_ids_col is None:
        raise ValueError("A ground-truth grouping source is required.")

    dataset_root = os.path.join(DATASETS_DIR, dataset_name)
    dataset_dir = os.path.join(dataset_root, "data")
    ds_utils_dir = os.path.join(dataset_dir, "DS_utils")

    if os.path.isdir(dataset_dir):
        if not overwrite:
            raise ValueError(
                "Dataset directory '{}' already exists. Re-run with --overwrite to replace it.".format(dataset_dir)
            )
        shutil.rmtree(dataset_dir)

    os.makedirs(ds_utils_dir)

    records = list(df.iterrows())
    dense_person_map = build_global_person_map(records, spacefeat_col, head_key)
    n_people = len(dense_person_map)
    singleton_values = normalize_singleton_values(singleton_values or [])

    feature_lines = []
    group_lines = []
    valid_times = []

    for row_index, frame_value in records:
        timestamp = stringify_timestamp(frame_value[time_col] if time_col else row_index)
        head = normalize_head_array(frame_value[spacefeat_col], head_key)
        person_ids = head[:, 0].tolist()
        groups = normalize_groups(
            frame_value,
            person_ids,
            group_col=group_col,
            group_ids_col=group_ids_col,
            singleton_values=singleton_values,
        )

        feature_lines.append(
            build_feature_line(timestamp, head, dense_person_map, n_people=n_people, position_scale=position_scale)
        )
        group_lines.append(build_group_line(timestamp, groups, dense_person_map))
        valid_times.append(timestamp)

    write_lines(os.path.join(ds_utils_dir, "features.txt"), feature_lines)
    write_lines(os.path.join(ds_utils_dir, "group_names.txt"), group_lines)

    positions = np.array([line.split() for line in feature_lines], dtype=str)
    groups_at_time = add_time(np.array(group_lines, dtype=str))

    shifted_coordinates = []
    for timestamp in valid_times:
        frame_shifted_coordinates = compute_data_shift(positions, timestamp, n_people, augment_flipped_data=True)
        if len(frame_shifted_coordinates) == 0:
            continue
        if len(shifted_coordinates) == 0:
            shifted_coordinates = frame_shifted_coordinates
        else:
            shifted_coordinates = np.concatenate((shifted_coordinates, frame_shifted_coordinates), axis=0)

    if len(shifted_coordinates) == 0:
        raise ValueError("No pair examples were generated. Each frame needs at least two valid people.")

    affinities, timechanges = affinities_and_timechanges(shifted_coordinates, groups_at_time)

    coordinate_lines = [" ".join([str(item) for item in row]) for row in shifted_coordinates]
    affinity_lines = [" ".join([str(item) for item in row]) for row in affinities]

    write_lines(os.path.join(dataset_dir, "coordinates.txt"), coordinate_lines)
    write_lines(os.path.join(dataset_dir, "affinities.txt"), affinity_lines)
    write_lines(os.path.join(dataset_dir, "timechanges.txt"), [str(change) for change in timechanges])

    d = 4
    max_people = n_people - 2
    X_group, X_pairs = build_X(coordinate_lines, max_people=max_people, d=d)
    Y, timestamps = build_Y(affinity_lines)

    for fold_index in range(num_folds):
        start_test, end_test, val_start_ends = compute_time_splits(timechanges, len(affinity_lines), fold_index, num_folds)

        X_group_test, X_group_train, X_group_val = split_test_train_val(start_test, end_test, val_start_ends, X_group)
        X_pairs_test, X_pairs_train, X_pairs_val = split_test_train_val(start_test, end_test, val_start_ends, X_pairs)
        Y_test, Y_train, Y_val = split_test_train_val(start_test, end_test, val_start_ends, Y)
        timestamps_test, timestamps_train, timestamps_val = split_test_train_val(
            start_test, end_test, val_start_ends, np.array(timestamps)
        )

        fold_dir = os.path.join(dataset_dir, "fold_{}".format(fold_index))
        os.makedirs(fold_dir)
        dump_pickle(os.path.join(fold_dir, "test.p"), ([X_group_test, X_pairs_test], Y_test, timestamps_test))
        dump_pickle(os.path.join(fold_dir, "train.p"), ([X_group_train, X_pairs_train], Y_train, timestamps_train))
        dump_pickle(os.path.join(fold_dir, "val.p"), ([X_group_val, X_pairs_val], Y_val, timestamps_val))

    with open(os.path.join(dataset_dir, "dataset_config.json"), "w") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "n_people": int(n_people),
                "n_features": 4,
                "position_scale": position_scale,
                "num_folds": int(num_folds),
                "spacefeat_col": spacefeat_col,
                "head_key": head_key,
                "time_col": time_col,
                "group_col": group_col,
                "group_ids_col": group_ids_col,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    with open(os.path.join(dataset_dir, "person_id_map.json"), "w") as f:
        json.dump({str(key): value for key, value in dense_person_map.items()}, f, indent=2, sort_keys=True)

    return dataset_dir


if __name__ == "__main__":
    args = parse_args()
    with open(args.input, "rb") as f:
        dataframe = pickle.load(f)
    dataframe = dataframe[2000:3000]
    output_dir = prepare_from_dataframe(
        dataframe,
        dataset_name=args.dataset_name,
        spacefeat_col=args.spacefeat_col,
        head_key=args.head_key,
        time_col=args.time_col,
        group_col=args.group_col,
        group_ids_col=args.group_ids_col,
        position_scale=args.position_scale,
        num_folds=args.num_folds,
        singleton_values=args.singleton_values,
        overwrite=args.overwrite,
    )
    print("Prepared dataset at {}".format(output_dir))
