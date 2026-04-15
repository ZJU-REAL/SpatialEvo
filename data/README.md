# Dataset Processing

The `data/` directory uses one shared goal: normalize different datasets into a ScanNet-like scene layout, then extract `frame_processed` and `metadata`.

The target scene format is:

```text
/mnt/jfs/lidingm/data/dataset/ScanNet/train/scene0000_01
├── color/*.jpg
├── depth/*.png
├── pose/*.txt
├── frame_processed/*.jpg
├── intrinsic_color.txt
├── intrinsic_depth.txt
├── extrinsic_color.txt
├── extrinsic_depth.txt
├── scene0000_01.txt
├── scene0000_01_vh_clean.ply
├── scene0000_01_vh_clean.segs.json
└── scene0000_01_vh_clean.aggregation.json
```

The matching metadata format is:

```text
/mnt/jfs/lidingm/data/dataset/<dataset_name>/metadata/<scene_id>
├── <scene_id>.json
└── frame_processed/*.json
```

---

## 1. ScanNet

### Inputs

- Scene root: `/mnt/jfs/lidingm/data/dataset/ScanNet`
- Scenes are already close to the target format, so no structural conversion is needed

### Scripts

- `scannet_process/scannet_select_frames/scannet_select_frames_process.py`
  - Converts raw frame selections into a `scene_id -> frame_ids` mapping
- `scannet_process/data_scripts/materialize_selected_frames.py`
  - Copies selected RGB frames into each scene's `frame_processed/`
- `scannet_process/data_scripts/metadata_extractor.py`
  - Extracts scene-level and frame-level metadata from `frame_processed/`, mesh, segmentation, and pose data

### Recommended flow

If you already have a `scene_id -> frame_ids` JSON:

```bash
python data/scannet_process/data_scripts/materialize_selected_frames.py \
  --dataset-root /mnt/jfs/lidingm/data/dataset/ScanNet \
  --split train \
  --manifest /path/to/scannet_scene_to_frames.json
```

Then extract metadata:

```bash
python data/scannet_process/data_scripts/metadata_extractor.py \
  --dataset-root /mnt/jfs/lidingm/data/dataset/ScanNet \
  --split train \
  --frame-type frame_processed
```

### Conversion summary

- Input: native ScanNet scenes
- Output: the same scenes with `frame_processed/`
- Final output: `/mnt/jfs/lidingm/data/dataset/ScanNet/metadata/<scene_id>`

---

## 2. ScanNet++

### Inputs

- Raw data: `/mnt/jfs/lidingm/data/dataset/scannetpp_raw`
- Depth cache: `/mnt/jfs/lidingm/data/dataset/scannetpp/processed`
- Normalized output: `/mnt/jfs/lidingm/data/dataset/scannetpp`

### Scripts

- `scannetpp_process/data_scripts/scannetpp_pipeline.py`
  - Converts `scannetpp_raw/<scene_id>` into `train/scene<scene_id>_00`
  - Extracts metadata from the normalized scene
- `scannetpp_process/scannetpp_as_scannet/convert_scannetpp_scene.py`
  - Single-scene conversion wrapper
- `scannetpp_process/scannetpp_as_scannet/verify_scannet_scene.py`
  - Validates the converted ScanNet-like scene structure

### Recommended flow

Normalize first:

```bash
python data/scannetpp_process/data_scripts/scannetpp_pipeline.py \
  --mode convert \
  --raw-root /mnt/jfs/lidingm/data/dataset/scannetpp_raw \
  --depth-root /mnt/jfs/lidingm/data/dataset/scannetpp/processed \
  --output-root /mnt/jfs/lidingm/data/dataset/scannetpp \
  --split train
```

Then extract metadata:

```bash
python data/scannetpp_process/data_scripts/scannetpp_pipeline.py \
  --mode extract-metadata \
  --output-root /mnt/jfs/lidingm/data/dataset/scannetpp \
  --split train
```

### Conversion summary

- Input: `scannetpp_raw/<scene_id>/dslr + scans`
- Intermediate output: `scannetpp/train/scene<scene_id>_00`
- Final output: `scannetpp/metadata/scene<scene_id>_00`

### Notes

- The pipeline is fully local and no longer depends on S3 or AWS.
- It assumes `scannetpp_raw` is already downloaded.
- Depth is read from `undistorted_depths` first and `render_depth` second.

---

## 3. ARKitScenes

### Inputs

- Raw data: `/mnt/jfs/lidingm/data/dataset/ARKitScenes_full`
- Normalized output: `/mnt/jfs/lidingm/data/dataset/ARKitScenes_scannet_like`

### Scripts

- `arkitscene_process/export_scannet_like.py`
  - Converts raw ARKitScenes assets into ScanNet-like scene folders
  - Exports `frame_processed/` at the same time
- `arkitscene_process/generate_scannet_like_metadata.py`
  - Extracts metadata from the normalized scenes
- `arkitscene_process/rectify_existing_scannet_like_train.py`
  - Fixes structure issues in existing training exports
- `arkitscene_process/run_metadata_scene_list.sh`
  - Runs metadata extraction for a selected scene list

### Recommended flow

Normalize first:

```bash
python data/arkitscene_process/export_scannet_like.py \
  --download-root /mnt/jfs/lidingm/data/dataset/ARKitScenes_full \
  --export-root /mnt/jfs/lidingm/data/dataset/ARKitScenes_scannet_like \
  --selected-frame-count 32
```

Then extract metadata:

```bash
python data/arkitscene_process/generate_scannet_like_metadata.py \
  --export-root /mnt/jfs/lidingm/data/dataset/ARKitScenes_scannet_like
```

### Conversion summary

- Input: `ARKitScenes_full`
- Intermediate output: `ARKitScenes_scannet_like/train/scene<video_id>_00`
- Final output: `ARKitScenes_scannet_like/metadata/scene<video_id>_00`

---

## Summary

All three datasets follow the same two-stage workflow:

1. Normalize raw scenes into the `ScanNet/train/scene0000_01` layout
2. Extract `frame_processed/` and `metadata/<scene_id>`

This keeps the SpatialEvo simulator, task generation, and training pipeline on one shared scene interface.
