#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/jfs/lidingm/data/dataset/ARKitScenes_scannet_like"
PYTHON_BIN="/home/i-lidingming/miniforge3/envs/ldm/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR="${SCRIPT_DIR}/generate_scannet_like_metadata.py"
SCENE_JOBS="${SCENE_JOBS:-6}"
FRAME_JOBS="${FRAME_JOBS:-4}"
CLEAN_METADATA="${CLEAN_METADATA:-0}"
VISUALIZE="${VISUALIZE:-1}"

usage() {
    cat <<'EOF'
Usage:
  batch_generate_scannet_like_metadata.sh [--clean] [--scene-jobs N] [--frame-jobs N] [--root PATH] [--python PATH] [--no-visualize]

Environment variables:
  SCENE_JOBS      Number of scenes processed in parallel. Default: 6
  FRAME_JOBS      Number of frame workers per scene. Default: 4
  CLEAN_METADATA  Set to 1 to remove ROOT/metadata before processing. Default: 0
  VISUALIZE       Set to 0 to disable frame/3D visualization export. Default: 1
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN_METADATA=1
            shift
            ;;
        --scene-jobs)
            SCENE_JOBS="$2"
            shift 2
            ;;
        --frame-jobs)
            FRAME_JOBS="$2"
            shift 2
            ;;
        --root)
            ROOT="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --no-visualize)
            VISUALIZE=0
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

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -f "${GENERATOR}" ]]; then
    echo "Missing generator script: ${GENERATOR}" >&2
    exit 1
fi

RUN_LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${RUN_LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SCENE_LIST="${RUN_LOG_DIR}/metadata_scene_list_${TIMESTAMP}.txt"

find "${ROOT}/train" "${ROOT}/val" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort > "${SCENE_LIST}"
TOTAL_SCENES="$(wc -l < "${SCENE_LIST}")"

if [[ "${CLEAN_METADATA}" == "1" ]]; then
    rm -rf "${ROOT}/metadata"
fi
mkdir -p "${ROOT}/metadata"

echo "[start] $(date -Is) total_scenes=${TOTAL_SCENES} scene_jobs=${SCENE_JOBS} frame_jobs=${FRAME_JOBS} root=${ROOT}"
echo "[scene_list] ${SCENE_LIST}"

xargs -P "${SCENE_JOBS}" -I{} bash -lc '
scene="$1"
python_bin="$2"
generator="$3"
root="$4"
frame_jobs="$5"
visualize="$6"
echo "[scene_start] $(date -Is) ${scene}"
cmd=( "$python_bin" "$generator" --export-root "$root" --scene-id "$scene" --jobs "$frame_jobs" )
if [[ "$visualize" == "1" ]]; then
    cmd+=( --visualize )
fi
"${cmd[@]}"
echo "[scene_done] $(date -Is) ${scene}"
' _ {} "${PYTHON_BIN}" "${GENERATOR}" "${ROOT}" "${FRAME_JOBS}" "${VISUALIZE}" < "${SCENE_LIST}"

echo "[done] $(date -Is) total_scenes=${TOTAL_SCENES}"
