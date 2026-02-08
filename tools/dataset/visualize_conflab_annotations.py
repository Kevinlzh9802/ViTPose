"""Visualize bounding boxes and keypoints from a COCO-format annotation file.

Draws bboxes (green) and keypoints with skeleton links on the source images,
sampling every N images. No mmpose/mmcv dependencies required -- only
cv2, numpy and the standard library.

Usage (from repo root):
    python tools/dataset/visualize_conflab_annotations.py \
        --ann-file data/conflab/keypoints_and_bboxes_train.json \
        --img-root data/conflab/images_train \
        --out-dir  data/conflab/vis_train \
        --every-n  100
"""

import argparse
import json
import os

import cv2
import numpy as np

# ---------- ConfLab skeleton definition (17 keypoints) ----------
KEYPOINT_NAMES = [
    "head", "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_foot", "left_foot",
]

# (src_idx, dst_idx) pairs that form the skeleton
SKELETON = [
    (14, 12),  # left_ankle  - left_knee
    (13, 12),  # left_knee   - left_hip
    (11,  9),  # right_ankle - right_knee
    (10,  9),  # right_knee  - right_hip
    (12,  9),  # left_hip    - right_hip  (note: indices for hip pair)
    ( 6, 12),  # left_shoulder  - left_hip
    ( 3,  9),  # right_shoulder - right_hip
    ( 6,  7),  # left_shoulder  - left_elbow
    ( 3,  4),  # right_shoulder - right_elbow
    ( 7,  8),  # left_elbow  - left_wrist
    ( 4,  5),  # right_elbow - right_wrist
    (16, 14),  # left_foot   - left_ankle
    (15, 11),  # right_foot  - right_ankle
    ( 1,  0),  # nose - head
    ( 2,  0),  # neck - head
    ( 6,  2),  # left_shoulder  - neck
    ( 3,  2),  # right_shoulder - neck
]

# Colours: right side orange, left side green, central blue
SKELETON_COLORS = [
    (0, 255, 0),      # left_ankle  - left_knee
    (0, 255, 0),      # left_knee   - left_hip
    (0, 128, 255),    # right_ankle - right_knee
    (0, 128, 255),    # right_knee  - right_hip
    (255, 153, 51),   # left_hip    - right_hip
    (255, 153, 51),   # left_shoulder  - left_hip
    (255, 153, 51),   # right_shoulder - right_hip
    (0, 255, 0),      # left_shoulder  - left_elbow
    (0, 128, 255),    # right_shoulder - right_elbow
    (0, 255, 0),      # left_elbow  - left_wrist
    (0, 128, 255),    # right_elbow - right_wrist
    (0, 255, 0),      # left_foot   - left_ankle
    (0, 128, 255),    # right_foot  - right_ankle
    (255, 153, 51),   # nose - head
    (255, 153, 51),   # neck - head
    (255, 153, 51),   # left_shoulder  - neck
    (255, 153, 51),   # right_shoulder - neck
]

KPT_COLOR_MAP = {
    "head": (255, 153, 51), "nose": (255, 153, 51), "neck": (255, 153, 51),
    "right_shoulder": (0, 128, 255), "right_elbow": (0, 128, 255),
    "right_wrist": (0, 128, 255), "left_shoulder": (0, 255, 0),
    "left_elbow": (0, 255, 0), "left_wrist": (0, 255, 0),
    "right_hip": (0, 128, 255), "right_knee": (0, 128, 255),
    "right_ankle": (0, 128, 255), "left_hip": (0, 255, 0),
    "left_knee": (0, 255, 0), "left_ankle": (0, 255, 0),
    "right_foot": (0, 128, 255), "left_foot": (0, 255, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize COCO-format bbox + keypoint annotations")
    parser.add_argument("--ann-file", required=True,
                        help="Path to COCO-format annotation JSON")
    parser.add_argument("--img-root", required=True,
                        help="Root directory of the images")
    parser.add_argument("--out-dir", default="data/conflab/vis_train",
                        help="Directory to save visualized images")
    parser.add_argument("--every-n", type=int, default=100,
                        help="Visualize every N-th image (default: 100)")
    parser.add_argument("--radius", type=int, default=4,
                        help="Keypoint circle radius")
    parser.add_argument("--thickness", type=int, default=2,
                        help="Skeleton link thickness")
    parser.add_argument("--bbox-thickness", type=int, default=2,
                        help="Bounding box line thickness")
    return parser.parse_args()


def draw_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    """Draw a single xywh bounding box on *img*."""
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)


def draw_keypoints_and_skeleton(img, keypoints, radius=4, thickness=2):
    """Draw keypoints and skeleton links.

    Args:
        keypoints: flat list [x1,y1,v1, x2,y2,v2, ...] or (N,3) array.
    """
    kps = np.array(keypoints).reshape(-1, 3)
    num_kps = kps.shape[0]

    # Draw skeleton links first (behind the keypoint dots)
    for idx, (src, dst) in enumerate(SKELETON):
        if src >= num_kps or dst >= num_kps:
            continue
        xs, ys, vs = kps[src]
        xd, yd, vd = kps[dst]
        if vs > 0 and vd > 0:  # both visible / labelled
            color = SKELETON_COLORS[idx] if idx < len(SKELETON_COLORS) else (200, 200, 200)
            cv2.line(img, (int(xs), int(ys)), (int(xd), int(yd)),
                     color, thickness, cv2.LINE_AA)

    # Draw keypoint circles
    for i, (x, y, v) in enumerate(kps):
        if v == 0:
            continue  # not labelled
        name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else "unknown"
        color = KPT_COLOR_MAP.get(name, (255, 255, 255))
        # Filled circle for visible (v==2), hollow for occluded (v==1)
        if v == 2:
            cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (int(x), int(y)), radius, color, 1, cv2.LINE_AA)


def main():
    args = parse_args()

    # Load annotations
    with open(args.ann_file, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    os.makedirs(args.out_dir, exist_ok=True)

    img_ids = sorted(images.keys())
    sampled = img_ids[::args.every_n]

    print(f"Total images: {len(img_ids)}, sampling every {args.every_n} "
          f"-> {len(sampled)} images to visualize")

    for count, img_id in enumerate(sampled):
        img_info = images[img_id]
        img_path = os.path.join(args.img_root, img_info["file_name"])

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read {img_path}, skipping.")
            continue

        anns = ann_by_img.get(img_id, [])
        for ann in anns:
            # Draw bounding box (COCO format: [x, y, w, h])
            if "bbox" in ann:
                draw_bbox(img, ann["bbox"],
                          color=(0, 255, 0), thickness=args.bbox_thickness)

            # Draw keypoints + skeleton
            if "keypoints" in ann:
                draw_keypoints_and_skeleton(
                    img, ann["keypoints"],
                    radius=args.radius, thickness=args.thickness)

        out_name = f"vis_{img_id:06d}.jpg"
        out_path = os.path.join(args.out_dir, out_name)
        cv2.imwrite(out_path, img)

        if (count + 1) % 10 == 0 or (count + 1) == len(sampled):
            print(f"  [{count + 1}/{len(sampled)}] saved {out_path}")

    print(f"Done. {len(sampled)} images saved to {args.out_dir}")


if __name__ == "__main__":
    main()
