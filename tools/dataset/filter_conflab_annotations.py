"""Filter low-quality annotations from a COCO-format keypoint JSON.

For each annotation the script checks four criteria and removes those that
fail any of them:
  1) Minimum number of valid keypoints (visibility > 0).
  2) Bounding-box area within a reasonable range (as fraction of image area).
  3) Bounding-box height/width ratio within a reasonable range.
  4) Keypoints not clustered too close to any single image edge.

Images that end up with zero remaining annotations are also dropped.
The filtered JSON is written next to the original with a '_filtered' suffix
(or to a custom path via --out-file).

Usage (from repo root):
    python tools/dataset/filter_conflab_annotations.py \
        --ann-file data/conflab/keypoints_and_bboxes_train.json \
        --out-file data/conflab/keypoints_and_bboxes_train_filtered.json
"""

import argparse
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Filter COCO-format keypoint annotations by quality")
    p.add_argument("--ann-file", required=True,
                   help="Input COCO-format annotation JSON")
    p.add_argument("--out-file", default=None,
                   help="Output path for filtered JSON "
                        "(default: <ann-file>_filtered.json)")
    # --- filtering thresholds ---
    p.add_argument("--min-valid-kps", type=int, default=12,
                   help="Min number of valid keypoints (vis > 0) per person "
                        "(default: 12 out of 17)")
    p.add_argument("--min-area-ratio", type=float, default=0.002,
                   help="Min bbox area / image area (default: 0.002)")
    p.add_argument("--max-area-ratio", type=float, default=0.15,
                   help="Max bbox area / image area (default: 0.15)")
    p.add_argument("--min-hw-ratio", type=float, default=1 / 6,
                   help="Min bbox height/width ratio (default: 1/6)")
    p.add_argument("--max-hw-ratio", type=float, default=6.0,
                   help="Max bbox height/width ratio (default: 6)")
    p.add_argument("--edge-sum-thresh", type=float, default=0.3,
                   help="Normalised-distance sum threshold for the edge "
                        "check (default: 0.3)")
    return p.parse_args()


def is_annotation_valid(
    ann: dict,
    img_width: int,
    img_height: int,
    min_valid_kps: int,
    min_area_ratio: float,
    max_area_ratio: float,
    min_hw_ratio: float,
    max_hw_ratio: float,
    edge_sum_thresh: float,
) -> bool:
    """Return True if the annotation passes all quality filters."""

    img_area = img_width * img_height

    # ---- bbox sanity ----
    bbox = ann.get("bbox", None)
    if bbox is None or len(bbox) != 4:
        return False
    x, y, w, h = bbox
    if not (np.isfinite(x) and np.isfinite(y)
            and np.isfinite(w) and np.isfinite(h)):
        return False
    if w <= 0 or h <= 0:
        return False

    # ---- 1) minimum valid keypoints ----
    kps = np.array(ann["keypoints"], dtype=float).reshape(-1, 3)  # (17, 3)
    vis = kps[:, 2]            # visibility flag
    valid = vis > 0            # 1 (occluded) or 2 (visible)
    num_valid = int(valid.sum())
    if num_valid < min_valid_kps:
        return False

    # ---- 2) bbox area ratio ----
    area = w * h
    area_ratio = area / img_area
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False

    # ---- 3) bbox height / width ratio ----
    hw_ratio = h / w
    if hw_ratio < min_hw_ratio or hw_ratio > max_hw_ratio:
        return False

    # ---- 4) keypoints near image edge ----
    #   For every valid keypoint we compute its normalised distance to each of
    #   the four edges.  If the *sum* of those distances (across all valid kps)
    #   for any single edge is below edge_sum_thresh, it means the majority of
    #   keypoints are hugging that edge -> reject.
    kp_xy = kps[valid, :2]  # (n_valid, 2) in pixel coords
    x_norm = kp_xy[:, 0] / img_width
    y_norm = kp_xy[:, 1] / img_height

    if x_norm.size > 0:
        left_sum = float(np.sum(x_norm))          # distance from left
        right_sum = float(np.sum(1.0 - x_norm))   # distance from right
        top_sum = float(np.sum(y_norm))            # distance from top
        bottom_sum = float(np.sum(1.0 - y_norm))   # distance from bottom

        if (left_sum < edge_sum_thresh or right_sum < edge_sum_thresh
                or top_sum < edge_sum_thresh or bottom_sum < edge_sum_thresh):
            return False

    return True


def main():
    args = parse_args()

    # --- load ---
    with open(args.ann_file, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    total_anns = len(coco["annotations"])

    # --- filter annotations ---
    kept_anns = []
    removed_reasons = {
        "min_valid_kps": 0,
        "area_ratio": 0,
        "hw_ratio": 0,
        "edge": 0,
        "other": 0,
    }

    for ann in coco["annotations"]:
        img = images.get(ann["image_id"])
        if img is None:
            removed_reasons["other"] += 1
            continue

        img_w = img["width"]
        img_h = img["height"]

        # Run individual checks for statistics (slightly redundant, but clear)
        kps = np.array(ann["keypoints"], dtype=float).reshape(-1, 3)
        vis = kps[:, 2]
        num_valid = int((vis > 0).sum())
        bbox = ann.get("bbox", [0, 0, 0, 0])
        _, _, bw, bh = bbox
        img_area = img_w * img_h
        area = bw * bh

        if num_valid < args.min_valid_kps:
            removed_reasons["min_valid_kps"] += 1
            continue
        if bw <= 0 or bh <= 0 or area <= 0:
            removed_reasons["other"] += 1
            continue
        area_ratio = area / img_area
        if area_ratio < args.min_area_ratio or area_ratio > args.max_area_ratio:
            removed_reasons["area_ratio"] += 1
            continue
        hw_ratio = bh / bw
        if hw_ratio < args.min_hw_ratio or hw_ratio > args.max_hw_ratio:
            removed_reasons["hw_ratio"] += 1
            continue

        valid_mask = vis > 0
        kp_xy = kps[valid_mask, :2]
        x_norm = kp_xy[:, 0] / img_w
        y_norm = kp_xy[:, 1] / img_h
        if x_norm.size > 0:
            if (float(np.sum(x_norm)) < args.edge_sum_thresh
                    or float(np.sum(1.0 - x_norm)) < args.edge_sum_thresh
                    or float(np.sum(y_norm)) < args.edge_sum_thresh
                    or float(np.sum(1.0 - y_norm)) < args.edge_sum_thresh):
                removed_reasons["edge"] += 1
                continue

        kept_anns.append(ann)

    # --- pad bboxes: 2% + 5px on each side, clamped to image bounds ---
    for ann in kept_anns:
        img = images[ann["image_id"]]
        img_w, img_h = img["width"], img["height"]

        x, y, w, h = ann["bbox"]
        pad_w = w * 0.02 + 5
        pad_h = h * 0.02 + 5

        x_new = max(0, x - pad_w)
        y_new = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        ann["bbox"] = [x_new, y_new, x2 - x_new, y2 - y_new]

    # --- drop images that have no remaining annotations ---
    kept_img_ids = {ann["image_id"] for ann in kept_anns}
    kept_images = [img for img in coco["images"] if img["id"] in kept_img_ids]
    removed_imgs = len(coco["images"]) - len(kept_images)

    # --- build output ---
    coco_filtered = {
        "images": kept_images,
        "annotations": kept_anns,
    }
    if "categories" in coco:
        coco_filtered["categories"] = coco["categories"]

    # --- save ---
    if args.out_file is None:
        base, ext = os.path.splitext(args.ann_file)
        out_path = f"{base}_filtered{ext}"
    else:
        out_path = args.out_file

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco_filtered, f)

    # --- report ---
    kept = len(kept_anns)
    removed = total_anns - kept
    print(f"Annotations: {total_anns} total -> {kept} kept, {removed} removed")
    print(f"  removed by min_valid_kps ({args.min_valid_kps}): "
          f"{removed_reasons['min_valid_kps']}")
    print(f"  removed by area_ratio [{args.min_area_ratio}, "
          f"{args.max_area_ratio}]: {removed_reasons['area_ratio']}")
    print(f"  removed by hw_ratio [{args.min_hw_ratio:.3f}, "
          f"{args.max_hw_ratio:.1f}]: {removed_reasons['hw_ratio']}")
    print(f"  removed by edge_sum_thresh ({args.edge_sum_thresh}): "
          f"{removed_reasons['edge']}")
    if removed_reasons["other"]:
        print(f"  removed by other (bad bbox / missing image): "
              f"{removed_reasons['other']}")
    print(f"Images: {len(coco['images'])} total -> {len(kept_images)} kept, "
          f"{removed_imgs} removed (no annotations left)")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
