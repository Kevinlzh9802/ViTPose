# ViTPose Dataframe Output Format

`demo/vitpose_to_dataframe.py` converts per-camera `vitpose_keypoints.json` files into pickled pandas dataframes for `demo/dante_transfer.py`.

## Input Layout

The converter scans one level of camera folders under `results_dir`:

```text
<results_dir>/
  cam06_batch01/
    vitpose_keypoints.json
  cam08_batch01/
    vitpose_keypoints.json
```

The camera number is parsed from the folder name, for example `cam06_batch01 -> 06`. Calibration is then loaded from:

```text
<camera_params_root>/
  camera_06/
    intrinsic.json
    extrinsic.json
```

## Output Layout

By default, each camera folder receives one pickle:

```text
<results_dir>/
  cam06_batch01/
    vitpose_keypoints.json
    vitpose_dataframe.pkl
```

If `--output_dir` is provided, output is written under the same camera subfolder name:

```text
<output_dir>/
  cam06_batch01/
    vitpose_dataframe.pkl
```

## Dataframe Schema

Each dataframe row is one frame/timestamp. The dataframe index is also the frame timestamp.

| Column | Type | Description |
| --- | --- | --- |
| `timestamp` | `str` | Frame id from `vitpose_keypoints.json`. |
| `spaceFeat` | `dict[str, np.ndarray]` | Per-body-part spatial features for DANTE. |
| `groups` | `list` | Empty placeholder for group annotations. |
| `group_ids` | `list` | Empty placeholder for group id annotations. |

`spaceFeat` always has four keys:

```python
{
    "head": np.ndarray,
    "shoulder": np.ndarray,
    "hip": np.ndarray,
    "foot": np.ndarray,
}
```

Each value is an `n_people x 4` object array:

```text
[
  [person_id, x, y, orientation],
  ...
]
```

The four columns mean:

| Column | Meaning |
| --- | --- |
| `person_id` | Track/person id copied from `vitpose_keypoints.json`, stored as a string. |
| `x` | World-floor x coordinate after camera calibration and back-projection. |
| `y` | World-floor y coordinate after camera calibration and back-projection. |
| `orientation` | Body-part orientation in radians. For lateral body parts this is computed from the left/right keypoint pair; for head it uses the Conflab `head -> nose` vector. |

If a body part cannot be projected, `x`, `y`, or `orientation` may be `NaN`.

## Body Part Definitions

The body-part rows are built from Conflab-17 keypoints, matching
`configs/_base_/datasets/conflab.py`:

| `spaceFeat` key | Main keypoint pair |
| --- | --- |
| `head` | head, nose |
| `shoulder` | left shoulder, right shoulder |
| `hip` | left hip, right hip |
| `foot` | left foot, right foot |

When both keypoints in the pair are visible, `x/y` is their midpoint and `orientation` is computed from the pair. If only fallback keypoints are available, `x/y` is the mean of visible fallback points and `orientation` is `NaN`.

## Example

```python
import pandas as pd

df = pd.read_pickle("cam06_batch01/vitpose_dataframe.pkl")
row = df.iloc[0]

print(row["timestamp"])
print(row["spaceFeat"]["head"])
```

Example `spaceFeat["head"]` value:

```text
[
  ["1", 0.124, -1.532, 1.571],
  ["3", 0.812, -0.944, 1.228],
]
```

## DANTE Transfer

`demo/dante_transfer.py` reads the dataframe with:

```bash
python demo/dante_transfer.py \
  --input /path/to/vitpose_dataframe.pkl \
  --spacefeat-col spaceFeat \
  --head-key head \
  --group-col groups
```

`dante_transfer.py` currently builds features from the `head` array by default. The empty `groups` list causes people to be treated as singleton groups unless group annotations are added later.

## Common Commands

Convert all cameras:

```bash
python demo/vitpose_to_dataframe.py \
  /path/to/vitpose_results \
  --output_dir /path/to/vitpose_dataframe \
  --camera_params_root /path/to/camera_params
```

Convert selected cameras:

```bash
python demo/vitpose_to_dataframe.py \
  /path/to/vitpose_results \
  --output_dir /path/to/vitpose_dataframe \
  --camera_params_root /path/to/camera_params \
  --camera_numbers 06,08,10
```

Submit selected cameras on DAIC:

```bash
sbatch slurm/submit_vitpose_dataframe_daic.sh --cam=06,08,10
```
