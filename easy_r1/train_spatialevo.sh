set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # GPUs to use
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
export SWANLAB_MODE=cloud # Set to `local` for offline logging.

MODEL_PATH=Qwen2。5-VL-3B-Instruct
#export VLLM_ATTENTION_BACKEND=XFORMERS  # Alternative backend.
export VLLM_ATTENTION_BACKEND=FLEX_ATTENTION
# Skip the Triton CUDA check for this setup.
export FLASH_ATTENTION_SKIP_CUDA_CHECK=1
export VLLM_USE_V1=1
python3 -m verl.trainer.main \
    config=examples/config_spatialevo.yaml \
    data.train_files=./dataset/spatialevo_train.jsonl \
    worker.actor.model.model_path="${MODEL_PATH}" \
    trainer.project_name=SpatialEvo \
    trainer.experiment_name=SpatialEvo_qwen3_4b_0410\
    trainer.save_checkpoint_path=./checkpoints/spatialevo_qwen3_4b_0410 \
    trainer.logger=['file','swanlab'] \
    trainer.n_gpus_per_node=8 \
