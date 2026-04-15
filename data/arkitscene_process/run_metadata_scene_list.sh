#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/i-lidingming/miniforge3/envs/ldm/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR="${SCRIPT_DIR}/generate_scannet_like_metadata.py"
EXPORT_ROOT="${EXPORT_ROOT:-/mnt/jfs/lidingm/data/dataset/ARKitScenes_scannet_like}"
SCENE_LIST_PATH="${SCENE_LIST_PATH:-}"
SCENE_JOBS="${SCENE_JOBS:-16}"
FRAME_JOBS="${FRAME_JOBS:-2}"
VISUALIZE="${VISUALIZE:-1}"
OVERWRITE="${OVERWRITE:-1}"

usage() {
    cat <<'EOF'
Usage:
  run_metadata_scene_list.sh --scene-list PATH [--export-root PATH] [--scene-jobs N] [--frame-jobs N] [--no-visualize] [--no-overwrite]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene-list)
            SCENE_LIST_PATH="$2"
            shift 2
            ;;
        --export-root)
            EXPORT_ROOT="$2"
            shift 2
            ;;
        --scene-jobs)
            SCENE_JOBS="$2"
            shift 2
            ;;
        --frame-jobs)
            FRAME_JOBS="$2"
            shift 2
            ;;
        --no-visualize)
            VISUALIZE=0
            shift
            ;;
        --no-overwrite)
            OVERWRITE=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${SCENE_LIST_PATH}" ]]; then
    echo "Missing --scene-list" >&2
    exit 2
fi

if [[ ! -f "${SCENE_LIST_PATH}" ]]; then
    echo "Scene list not found: ${SCENE_LIST_PATH}" >&2
    exit 1
fi

TOTAL_SCENES="$(wc -l < "${SCENE_LIST_PATH}")"
echo "[start] $(date -Is) total_scenes=${TOTAL_SCENES} scene_jobs=${SCENE_JOBS} frame_jobs=${FRAME_JOBS} list=${SCENE_LIST_PATH} root=${EXPORT_ROOT}"

xargs -P "${SCENE_JOBS}" -I{} bash -lc '
scene="$1"
python_bin="$2"
generator="$3"
export_root="$4"
frame_jobs="$5"
visualize="$6"
overwrite="$7"
echo "[scene_start] $(date -Is) ${scene}"
cmd=( "$python_bin" "$generator" --export-root "$export_root" --scene-id "$scene" --jobs "$frame_jobs" )
if [[ "$visualize" == "1" ]]; then
    cmd+=( --visualize )
fi
if [[ "$overwrite" == "1" ]]; then
    cmd+=( --overwrite )
fi
"${cmd[@]}"
echo "[scene_done] $(date -Is) ${scene}"
' _ {} "${PYTHON_BIN}" "${GENERATOR}" "${EXPORT_ROOT}" "${FRAME_JOBS}" "${VISUALIZE}" "${OVERWRITE}" < "${SCENE_LIST_PATH}"

echo "[done] $(date -Is) total_scenes=${TOTAL_SCENES}"
