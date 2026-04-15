# ScanNet++ Processing

This pipeline is fully local and no longer uses S3 or AWS.

Default assumptions:

- Raw data is already available at `/mnt/jfs/lidingm/data/dataset/scannetpp_raw`
- Depth cache is available at `/mnt/jfs/lidingm/data/dataset/scannetpp/processed`
- Normalized output is written to `/mnt/jfs/lidingm/data/dataset/scannetpp`

## Raw layout

Each scene is expected to look like this:

```text
/mnt/jfs/lidingm/data/dataset/scannetpp_raw/<scene_id>
├── dslr
│   ├── nerfstudio/transforms_undistorted.json
│   ├── train_test_lists.json
│   └── undistorted_images/
└── scans
    ├── mesh_aligned_0.05.ply
    ├── mesh_aligned_0.05_semantic.ply
    ├── segments.json
    └── segments_anno.json
```

Depth is read from the first available directory below:

```text
/mnt/jfs/lidingm/data/dataset/scannetpp/processed/<scene_id>/dslr/undistorted_depths
/mnt/jfs/lidingm/data/dataset/scannetpp/processed/<scene_id>/dslr/render_depth
```

## Target layout

Each converted scene is written as:

```text
/mnt/jfs/lidingm/data/dataset/scannetpp/train/scene<scene_id>_00
├── color/*.jpg
├── depth/*.png
├── pose/*.txt
├── frame_processed/*.jpg
├── intrinsic_color.txt
├── intrinsic_depth.txt
├── extrinsic_color.txt
├── extrinsic_depth.txt
├── scene<scene_id>_00.txt
├── scene<scene_id>_00_vh_clean.ply
├── scene<scene_id>_00_vh_clean.segs.json
├── scene<scene_id>_00_vh_clean.aggregation.json
├── asset_manifest.json
├── frame_index.json
└── scannetpp_source.json
```

Object-level and frame-level metadata is written to:

```text
/mnt/jfs/lidingm/data/dataset/scannetpp/metadata/scene<scene_id>_00
├── scene<scene_id>_00.json
└── frame_processed/*.json
```

## Recommended flow

### 1. List local scenes

```bash
python data/scannetpp_process/data_scripts/scannetpp_pipeline.py \
  --mode list-scenes \
  --raw-root /mnt/jfs/lidingm/data/dataset/scannetpp_raw
```

### 2. Normalize raw scenes

```bash
python data/scannetpp_process/data_scripts/scannetpp_pipeline.py \
  --mode convert \
  --raw-root /mnt/jfs/lidingm/data/dataset/scannetpp_raw \
  --depth-root /mnt/jfs/lidingm/data/dataset/scannetpp/processed \
  --output-root /mnt/jfs/lidingm/data/dataset/scannetpp \
  --split train \
  --scene-ids 036bce3393,1ada7a0617 \
  --frame-processed-max 32
```

### 3. Extract metadata

```bash
python data/scannetpp_process/data_scripts/scannetpp_pipeline.py \
  --mode extract-metadata \
  --output-root /mnt/jfs/lidingm/data/dataset/scannetpp \
  --split train \
  --scene-ids 036bce3393,1ada7a0617
```

### 4. Run both stages

```bash
python data/scannetpp_process/data_scripts/scannetpp_pipeline.py \
  --mode convert-and-extract \
  --raw-root /mnt/jfs/lidingm/data/dataset/scannetpp_raw \
  --depth-root /mnt/jfs/lidingm/data/dataset/scannetpp/processed \
  --output-root /mnt/jfs/lidingm/data/dataset/scannetpp \
  --split train \
  --max-scenes 10
```

### 5. Convert one scene

```bash
python data/scannetpp_process/scannetpp_as_scannet/convert_scannetpp_scene.py \
  --scene-id 036bce3393 \
  --raw-root /mnt/jfs/lidingm/data/dataset/scannetpp_raw \
  --depth-root /mnt/jfs/lidingm/data/dataset/scannetpp/processed \
  --output-root /mnt/jfs/lidingm/data/dataset/scannetpp \
  --split train
```

## Notes

- `frame_processed` is sampled evenly from all valid frames, with 32 frames kept by default.
- `transforms_undistorted.json` is preferred over `transforms.json`.
- If `mesh`, `segments`, or `segments_anno` is missing, the pipeline can still build a camera-only scene layout. Metadata extraction then falls back to `camera_only`.
- `frame_index.json` records the mapping from exported frame ids to original DSLR filenames.
- `asset_manifest.json` and `scannetpp_source.json` keep the original asset paths for later debugging or reprocessing.
