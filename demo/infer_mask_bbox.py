"""Run ViTPose inference on images with pre-computed per-person bounding boxes.

Expected folder layout under <data-root>:
    <data-root>/
        images/
            000000.jpg
            000001.jpg
            ...
        mask_bbox.json

Input JSON format (mask_bbox.json):
    {
        "annotations": {
            "1": {
                "bbox": {
                    "1": [x, y, w, h],
                    "3": [x, y, w, h]
                }
            },
            "2": {
                "bbox": { ... }
            }
        }
    }
    where top-level keys in "annotations" are image_id strings,
    and keys inside "bbox" are person_id strings.

Outputs (saved to <out-dir>, defaults to <data-root>):
    - vitpose_keypoints.json   (or custom name via --out-json)
    - vitpose_kp.mp4           (optional, enabled with --save-video)

Output JSON format:
    {
        "annotations": {
            "1": {
                "bbox": {
                    "1": [x, y, w, h],
                    "3": [x, y, w, h]
                },
                "keypoints": {
                    "1": [[x, y, score], [x, y, score], ...],
                    "3": [[x, y, score], [x, y, score], ...]
                }
            },
            ...
        }
    }

Usage:
    python demo/infer_mask_bbox.py \
        <pose_checkpoint> \
        --data-root /path/to/data_folder \
        --save-video \
        --device cuda:0

    pose_config defaults to configs/ViTPose_coco_plus_conflab_w_bg_256x192.py
"""

import json
import os
import warnings
from argparse import ArgumentParser

import cv2
import numpy as np
from tqdm import tqdm

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def parse_args():
    parser = ArgumentParser(
        description='ViTPose inference with per-person mask_bbox.json')
    parser.add_argument('pose_config', nargs='?',
                        default='configs/ViTPose_coco_plus_conflab_w_bg_256x192.py',
                        help='Config file for pose model '
                             '(default: configs/ViTPose_coco_plus_conflab_w_bg_256x192.py)')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose model')
    parser.add_argument('--data-root', required=True,
                        help='Root folder containing mask_bbox.json')
    parser.add_argument('--img-dir', default=None,
                        help='Folder containing images (default: <data-root>/images)')
    parser.add_argument('--out-dir', default=None,
                        help='Directory to write output files (default: same as --data-root)')
    parser.add_argument('--bbox-file', default='mask_bbox.json',
                        help='Name of the input bbox JSON (default: mask_bbox.json)')
    parser.add_argument('--out-json', default='vitpose_keypoints.json',
                        help='Output JSON filename (default: vitpose_keypoints.json)')
    parser.add_argument('--save-video', action='store_true',
                        help='Save a visualisation video (vitpose_kp.mp4)')
    parser.add_argument('--video-fps', type=int, default=60,
                        help='FPS for the output video (default: 60)')
    parser.add_argument('--device', default='cuda:0',
                        help='Device used for inference')
    parser.add_argument('--kpt-thr', type=float, default=0.3,
                        help='Keypoint score threshold for visualisation')
    parser.add_argument('--radius', type=int, default=4,
                        help='Keypoint radius for visualisation')
    parser.add_argument('--thickness', type=int, default=2,
                        help='Link thickness for visualisation')
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = args.out_dir if args.out_dir is not None else args.data_root
    os.makedirs(out_dir, exist_ok=True)

    img_dir = args.img_dir if args.img_dir is not None else os.path.join(args.data_root, 'images')
    bbox_path = os.path.join(args.data_root, args.bbox_file)
    out_json_path = os.path.join(out_dir, args.out_json)
    out_video_path = os.path.join(out_dir, 'vitpose_kp.mp4')

    # --- Load bbox JSON ---
    with open(bbox_path, 'r') as f:
        bbox_data = json.load(f)
    annotations = bbox_data['annotations']  # dict: {image_id_str: {"bbox": {...}}}

    # --- Build pose model ---
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config. '
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # --- Temp video path (write locally, then move to avoid NFS corruption) ---
    # Use PID to avoid collisions when multiple jobs share the same /tmp
    tmp_video_path = os.path.join('/tmp', f'vitpose_kp_tmp_{os.getpid()}.mp4')

    # --- Video writer (lazy init on first frame) ---
    video_writer = None

    # --- Sort by image_id (numeric) for deterministic ordering ---
    sorted_image_ids = sorted(annotations.keys(), key=lambda k: int(k))

    # --- Inference loop ---
    output_annotations = {}

    try:
        for image_id_str in tqdm(sorted_image_ids, desc='Inferring'):
            image_id = int(image_id_str)
            ann = annotations[image_id_str]
            bbox_dict = ann['bbox']  # {person_id_str: [x, y, w, h], ...}

            # Resolve image path
            img_path = os.path.join(img_dir, f'{image_id:08d}.jpg')
            if not os.path.isfile(img_path):
                print(f'[WARN] No image found at {img_path}, skipping.')
                continue

            # Build ordered person_results list, keeping track of person_ids
            person_ids = list(bbox_dict.keys())
            person_results = []
            for pid in person_ids:
                person_results.append({'bbox': np.array(bbox_dict[pid], dtype=np.float32)})

            # Run inference
            if len(person_results) > 0:
                pose_results, _ = inference_top_down_pose_model(
                    pose_model,
                    img_path,
                    person_results,
                    bbox_thr=None,
                    format='xywh',
                    dataset=dataset,
                    dataset_info=dataset_info,
                    return_heatmap=False,
                    outputs=None)
            else:
                pose_results = []

            # Map results back to person_ids
            keypoints_dict = {}
            for pid, pose_res in zip(person_ids, pose_results):
                # pose_res['keypoints'] is ndarray (K, 3): x, y, score
                kps = pose_res['keypoints']
                keypoints_dict[pid] = kps.tolist()

            # Build output annotation
            output_annotations[image_id_str] = {
                'bbox': bbox_dict,
                'keypoints': keypoints_dict,
            }

            # --- Optional: render visualisation frame ---
            if args.save_video:
                if len(pose_results) > 0:
                    vis_img = vis_pose_result(
                        pose_model,
                        img_path,
                        pose_results,
                        dataset=dataset,
                        dataset_info=dataset_info,
                        kpt_score_thr=args.kpt_thr,
                        radius=args.radius,
                        thickness=args.thickness,
                        show=False)
                else:
                    vis_img = cv2.imread(img_path)

                if video_writer is None:
                    h, w = vis_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        tmp_video_path, fourcc, args.video_fps, (w, h))

                video_writer.write(vis_img)

    finally:
        # Always finalise the video writer, even if inference crashes mid-run
        if video_writer is not None:
            video_writer.release()
            # Keep the raw OpenCV mp4v output and copy it to out-dir.
            import shutil
            shutil.copyfile(tmp_video_path, out_video_path)
            os.remove(tmp_video_path)
            print(f'Saved raw video to {out_video_path}')

    # --- Save output JSON ---
    output_data = {'annotations': output_annotations}
    with open(out_json_path, 'w') as f:
        json.dump(output_data, f)
    print(f'Saved keypoints to {out_json_path}')


if __name__ == '__main__':
    main()