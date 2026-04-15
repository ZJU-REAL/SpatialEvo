#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Start the vLLM server."""

from flask import Flask, request, jsonify  # Flask app
import vllm  # vLLM runtime
import argparse
import json  # JSON IO
import os
import threading
import time
import torch  # PyTorch GPU ops
from transformers import AutoTokenizer  # Hugging Face tokenizer
from mathruler.grader import extract_boxed_content, grade_answer
import stopit
import base64  # Base64 decoding
import io
from PIL import Image

# CLI arguments.
parser = argparse.ArgumentParser(description="vLLM multimodal VQA inference server")
parser.add_argument(
    '--port',
    type=str,
    default='5000',
    help='Server port (default: 5000)'
)
parser.add_argument(
    '--model_path',
    type=str,
    default='Qwen/Qwen3-4B-Base',
    help='vLLM model path (default: Qwen/Qwen3-4B-Base)'
)
parser.add_argument(
    '--gpu_mem_util',
    type=float,
    default=0.8,
    help='Max GPU memory utilization (0.0-1.0, default: 0.8)'
)
args = parser.parse_args()

print('[init] Loading model...')

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# Initialize the vLLM engine.
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,  # Fraction of GPU memory to use.
)

# Default sampling setup for candidate generation.
sample_params = vllm.SamplingParams(
    max_tokens=4096,  # Max generated tokens.
    temperature=1.0,  # Full-temperature sampling.
    top_p=1.0,  # No nucleus truncation.
    top_k=40,  # Top-k sampling cutoff.
    stop_token_ids=[tokenizer.eos_token_id],  # Stop at EOS.
    n=10,  # Number of candidates per prompt.
)

# Keep the GPU warm between requests.
# `pause_event` is set while the server is actively handling inference.

stop_event = threading.Event()
pause_event = threading.Event()  # Pause the idle GPU worker.

def gpu_idle_worker():
    """Keep the GPU busy while the server is idle."""
    print('[idle_worker] Idle GPU worker started.')
    running = True

    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print('[idle_worker] Paused for an inference request.')
                running = False
            time.sleep(0.1)  # Avoid busy-waiting on the CPU.
            continue
        else:
            if not running:
                print('[idle_worker] Resumed after inference.')
                running = True

        try:
            # Run a small CUDA matmul to keep the GPU active.
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()  # Wait for the CUDA work to finish.
        except RuntimeError as e:
            print(f'[idle_worker] Caught RuntimeError: {e}. Retry in 1 second...')
            time.sleep(1)

    print('[idle_worker] Idle GPU worker stopped.')

idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
idle_thread.start()

# Wrap `grade_answer` to avoid hanging a Flask worker thread.
# The underlying grader may trigger slow symbolic checks.

@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """Run the answer grader with a timeout."""
    return grade_answer(res1, res2)

# Flask app.
app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    """Run batched VQA inference for a task JSON file."""

    # Pause the idle worker before serving the request.
    pause_event.set()
    torch.cuda.synchronize()  # Wait for any warm-up work to finish.

    # Load the task file path from the query string.
    name = request.args.get('name', 'None')
    print(f'[server] Received task file request: {name}')

    # Expected format:
    # [{"question": "...", "answer": "...", "types": "...", "image": "..."}, ...]
    with open(name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.remove(name)

    questions = [item.get('question', '') for item in data]
    answers   = [item.get('answer',   '') for item in data]
    types     = [item.get('types',    '') for item in data]  # "counting", "geometry"
    image     = [item.get('image',    '') for item in data]  # Base64 image payloads.

    # Decode Base64 images.

    def base64_to_pil(b64_string):
        """Decode a Base64 image into an RGB PIL image."""
        # Strip an optional data URI prefix.
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        # Decode the Base64 payload.
        image_data = base64.b64decode(b64_string)

        # Convert to RGB for the model input.
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    pil_images = []
    for img_b64 in image:
        if img_b64:
            try:
                pil_images.append(base64_to_pil(img_b64))
            except Exception as e:
                # Keep invalid images as None.
                print(f"[warning] Failed to decode image: {e}")
                pil_images.append(None)
        else:
            pil_images.append(None)

    # Build Qwen-style multimodal prompts.
    valid_chats = []  # vLLM prompt list.

    for i, (q, a, t, img) in enumerate(zip(questions, answers, types, pil_images)):
        if q and a and t and img:
            # System message + user message + assistant prefix.
            prompt = (
                "<|im_start|>system\n"
                "You are an AI visual question answering assistant. "
                "Answer questions based only on the visual content provided. "
                "You **must only output your final answer inside \\boxed{}**. "
                "Do not write explanations or any other text.\n"
                "<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"This is a question: {q}. The question type is: {t}.\n"
                "**IMPORTANT:** Only output your answer in the form \\boxed{{answer}}.Do NOT include any units; provide only the numeric value or option.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            # vLLM expects {"prompt": ..., "multi_modal_data": {"image": PIL.Image}}.
            valid_chats.append({
                "prompt": prompt,
                "multi_modal_data": {"image": img}
            })

    print(f'[server] Prepared {len(valid_chats)} valid prompts.')

    # Run batched generation with vLLM.
    print('[server] Generating candidate answers with vLLM...')
    responses = model.generate(valid_chats, sampling_params=sample_params, use_tqdm=True)
    print('[server] vLLM generation finished.')

    # Aggregate answers for each question.

    def process_single(question, golden_answer, response):
        """Aggregate candidate answers for one question."""
        # Extract boxed answers from all sampled completions.
        results = [extract_boxed_content(out.text) for out in response.outputs]

        answer_counts = {}

        for res in results:
            if not res:
                continue

            matched = False

            for exist_ans in list(answer_counts.keys()):
                # Fast path for exact matches and common "no solution" variants.
                if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                    answer_counts[exist_ans] += 1
                    matched = True
                    break

                # Fall back to symbolic equivalence checking.
                try:
                    is_match = False

                    # Compare both directions because the grader may be asymmetric.
                    match_result_1 = grade_answer_with_timeout(res, exist_ans, timeout=10)
                    if match_result_1 == 'TIMED_OUT':
                        print(f"      [grader] Timeout comparing '{res[:30]}...' and '{exist_ans[:30]}...'")
                    elif match_result_1:
                        is_match = True

                    if not is_match:
                        match_result_2 = grade_answer_with_timeout(exist_ans, res, timeout=10)
                        if match_result_2 == 'TIMED_OUT':
                            print(
                                f"      [grader] Timeout comparing '{exist_ans[:30]}...' and "
                                f"'{res[:30]}...'. Skip this pair."
                            )
                        elif match_result_2:
                            is_match = True

                    if is_match:
                        answer_counts[exist_ans] += 1
                        matched = True
                        break

                except Exception as e:
                    print(
                        f"      [grader] Error comparing '{res[:30]}...' and "
                        f"'{exist_ans[:30]}...': {e}. Skip."
                    )
                    continue

            if not matched:
                answer_counts[res] = 1

        if not answer_counts:
            majority_ans, max_count = '', 0
        else:
            majority_ans = max(answer_counts, key=answer_counts.get)
            max_count = answer_counts[majority_ans]

        # Majority vote ratio across all sampled answers.
        score = max_count / len(results) if results else 0.0

        return {
            'question': question,
            'answer':   majority_ans,
            'score':    score,  # Confidence in [0.0, 1.0].
            'results':  results
        }

    # Map generated responses back to the original question list.
    results_all = []
    response_idx = 0  # Index over `responses` for valid prompts.

    for q, a in zip(questions, answers):
        try:
            if q and a:
                response = responses[response_idx]
                response_idx += 1
                item = process_single(q, a, response)
                results_all.append(item)
            else:
                results_all.append({
                    'question': q,
                    'answer': a,
                    'score': -1,  # Missing input.
                    'results': []
                })
        except Exception as e:
            print(f'[server] Fatal error while processing question: {q}')
            print(f'[server] Error details: {e}')
            results_all.append({
                'question': q,
                'answer':   a,
                'score':    -1,
                'results':  [],
                'error':    f'Unhandled exception in process_single: {str(e)}'
            })

    print(f'[server] Finished processing all {len(results_all)} questions.')

    # Save results next to the input JSON file.
    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results_all, f, indent=4, ensure_ascii=False)

    # Resume the idle worker after the request is done.
    pause_event.clear()
    print(f'[server] Processed {name}; saved results to {out_path}. Resuming idle GPU worker.')

    return jsonify({'message': f'Processed {name}; saved results to {out_path}.'})

@app.route('/evaluate_quality', methods=['POST'])
def evaluate_quality():
    """Score question quality with vLLM."""

    # Pause the idle worker before serving the request.
    pause_event.set()
    torch.cuda.synchronize()

    try:
        # Validate the request payload.
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body must contain JSON data'}), 400

        questions = data.get('questions', [])
        multi_modal_data_list = data.get('multi_modal_data', [])

        if not questions:
            return jsonify({'error': '`questions` must not be empty'}), 400

        if len(multi_modal_data_list) != len(questions):
            return jsonify({'error': '`multi_modal_data` must match `questions` in length'}), 400

        print(f'[evaluate_quality] Received {len(questions)} question-quality requests')

        # Decode images.
        def base64_to_pil(b64_string):
            """Decode a Base64 image into an RGB PIL image."""
            if "," in b64_string:
                b64_string = b64_string.split(",")[1]
            image_data = base64.b64decode(b64_string)
            return Image.open(io.BytesIO(image_data)).convert("RGB")

        pil_images = []
        for mm_data in multi_modal_data_list:
            if mm_data and "images" in mm_data and mm_data["images"]:
                img_b64 = mm_data["images"][0]
                try:
                    pil_images.append(base64_to_pil(img_b64))
                except Exception as e:
                    print(f"[evaluate_quality] Failed to decode image: {e}")
                    pil_images.append(None)
            else:
                pil_images.append(None)

        # Build evaluation prompts.
        valid_chats = []

        for question, img in zip(questions, pil_images):
            if not question:
                continue

            # Qwen chat prompt.
            evaluation_prompt = (
                "<|im_start|>system\n"
                "You are an AI assistant that evaluates the quality of visual questions. "
                "Evaluate questions based on the following dimensions:\n"
                "1. Relevance: Does the question relate to the image content?\n"
                "2. Clarity: Is the question clearly expressed?\n"
                "3. Challenge: Does the question have appropriate difficulty?\n"
                "4. Answerability: Can the question be reasonably answered?\n"
                "Output only an integer score from 0 to 10, without any explanation.\n"
                "<|im_end|>\n"
                f"<|im_start|>user\n"
            )

            if img:
                evaluation_prompt += "<|vision_start|><|image_pad|><|vision_end|>"

            evaluation_prompt += (
                f"Question: {question}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            chat_item = {"prompt": evaluation_prompt}
            if img:
                chat_item["multi_modal_data"] = {"image": img}

            valid_chats.append(chat_item)

        if not valid_chats:
            return jsonify({'error': 'No valid questions to evaluate'}), 400

        print(f'[evaluate_quality] Prepared {len(valid_chats)} evaluation prompts')

        # Deterministic decoding for a 0-10 score.
        quality_sampling_params = vllm.SamplingParams(
            temperature=0.0,
            max_tokens=10,
            stop=["\n", ".", "。", "<|im_end|>"]
        )

        responses = model.generate(valid_chats, sampling_params=quality_sampling_params, use_tqdm=False)

        # Parse scores.
        scores = []
        import re

        for response in responses:
            generated_text = response.outputs[0].text.strip()

            numbers = re.findall(r'\d+', generated_text)
            if numbers:
                score = float(numbers[0])
                # Normalize to [0, 1].
                score = max(0.0, min(10.0, score)) / 10.0
                scores.append(score)
            else:
                print(f"[evaluate_quality] Failed to parse score: {generated_text}; using 0.5")
                scores.append(0.5)

        while len(scores) < len(questions):
            scores.append(0.5)

        print(f'[evaluate_quality] Evaluation complete; returning {len(scores)} scores')

        return jsonify({'scores': scores})

    except Exception as e:
        print(f'[evaluate_quality] Error: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Evaluation failed: {str(e)}'}), 500

    finally:
        # Resume the idle worker after the request.
        pause_event.clear()

if __name__ == '__main__':
    """Start the local Flask server and stop the idle worker on exit."""
    try:
        print(f'[main] Starting Flask server on port {args.port}...')
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
        print('[main] Shutting down server...')
        stop_event.set()  # Stop the idle GPU worker.
        idle_thread.join()
        print('[main] Shutdown complete.')
