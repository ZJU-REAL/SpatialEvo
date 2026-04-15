
export VLLM_DISABLE_COMPILE_CACHE=1
CUDA_VISIBLE_DEVICES=1 python start_vllm_server.py --port 5000 --model_path /home/chengxinrui/model/Qwen/Qwen2.5-0.5B-Instruct
# CUDA_VISIBLE_DEVICES=5 python start_vllm_server.py --port 6001 --model_path $model_path &
# CUDA_VISIBLE_DEVICES=6 python start_vllm_server.py --port 6002 --model_path $model_path &
# CUDA_VISIBLE_DEVICES=7 python start_vllm_server.py --port 6003 --model_path $model_path &