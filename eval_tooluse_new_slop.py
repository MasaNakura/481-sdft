"""
Tool-use eval for Qwen3.5+ checkpoints (transformers>=5 tokenizer + recent vLLM).

Same infrastructure as eval_tooluse_new.py (vLLM loading, prompts, generation).
Scoring matches Tybuu/sft_slop eval.py for tooluse:
  - looser extract_action_inputs parsing
  - empty-string / None values dropped before input dict comparison
  - response truncated at Question:/User: before scoring

Use requirements_new.txt.
"""

import argparse
import json
import os
import re
from collections import Counter

# CHANGE: must precede `from vllm import ...` so vLLM picks up the in-process EngineCore mode.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import numpy as np
from datasets import load_from_disk
from vllm import SamplingParams

from qwen_compat_new import apply_chat_template_compat, build_vllm_llm, load_tokenizer


def parse_args():
    # DIFF FROM eval_tooluse_new.py: description notes sft_slop scoring variant.
    parser = argparse.ArgumentParser(
        description="Evaluate a model on tooluse test set (new stack, sft_slop scoring)"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling cumulative probability")
    parser.add_argument("--top_k", type=int, default=-1, help="Top-k filter; -1 disables")
    parser.add_argument("--min_p", type=float, default=0.0, help="Min-p filter; 0 disables")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="vLLM presence_penalty (0-2)")
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="vLLM repetition_penalty (1 = no penalty)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed for reproducibility")
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable Qwen3.5 thinking template (default: off for deterministic eval)",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=None,
        help="Original hub model id if model_path lacks config.json",
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    model_path, gpu_memory_utilization=0.8, enable_thinking: bool = False, base_model_name=None
):
    from eval_checkpoint import resolve_pretrained_dirs

    vllm_path, tokenizer_path = resolve_pretrained_dirs(model_path, base_model_name)
    print(f"Loading model for vLLM from {vllm_path}")
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path, padding_side="left")
    llm = build_vllm_llm(
        vllm_path,
        gpu_memory_utilization=gpu_memory_utilization,
        default_chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    return llm, tokenizer


def load_test_data(tokenizer, enable_thinking: bool):
    data_dir = "data/tooluse_data/eval_data"
    data = load_from_disk(data_dir).to_list()

    for example in data:
        example["prompt"] = apply_chat_template_compat(
            tokenizer,
            [{"role": "user", "content": example["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    return data


def _collect_stop_token_ids(tokenizer) -> list[int]:
    if tokenizer is None:
        return None
    stops: list[int] = []
    if tokenizer.eos_token_id is not None:
        stops.append(int(tokenizer.eos_token_id))
    for marker in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(marker)
        except Exception:
            tid = None
        if isinstance(tid, int) and tid >= 0:
            stops.append(tid)
    seen: set[int] = set()
    out: list[int] = []
    for tid in stops:
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out or None


def generate_responses(
    llm,
    prompts,
    max_new_tokens=1024,
    temperature=0.0,
    tokenizer=None,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
):
    if prompts:
        tail = prompts[0][-200:]
        print("=" * 60)
        print("Sample rendered prompt tail (last 200 chars):")
        print(repr(tail))
        print("=" * 60)
    sampling_kwargs = dict(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop_token_ids=_collect_stop_token_ids(tokenizer),
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    if seed is not None:
        sampling_kwargs["seed"] = seed
    sampling_params = SamplingParams(**sampling_kwargs)

    print(f"Generating responses for {len(prompts)} prompts...")
    print(
        f"  sampling: temperature={temperature} top_p={top_p} top_k={top_k} min_p={min_p}"
        f" presence_penalty={presence_penalty} repetition_penalty={repetition_penalty}"
    )
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def extract_actions(text):
    return re.findall(r"Action:\s*(\w+)", text)


# DIFF FROM eval_tooluse_new.py: sft_slop uses a looser regex and JSON recovery heuristics
# instead of strict `Action Input:\s*({.*?})` + json.loads only.
# Source: https://github.com/Tybuu/sft_slop/blob/master/eval.py
def extract_action_inputs(text):
    matches = re.findall(r"Action Input:\s*(.*?)(?=Action:|$)", text, re.DOTALL)
    merged_inputs = {}
    for content in matches:
        content = content.strip()
        if not content:
            continue
        # Heuristic: if it's missing brackets, add them.
        if not content.startswith("{") and ":" in content:
            content = "{" + content + "}"
        try:
            json_match = re.search(r"({.*?})", content, re.DOTALL)
            if json_match:
                merged_inputs.update(json.loads(json_match.group(1)))
        except json.JSONDecodeError:
            pass
    return merged_inputs


# DIFF FROM eval_tooluse_new.py: sft_slop drops empty-string / None values from both
# prediction and golden inputs before comparing dicts.
def _filter_nonempty_inputs(inputs: dict) -> dict:
    return {k: v for k, v in inputs.items() if v != "" and v is not None}


# DIFF FROM eval_tooluse_new.py: sft_slop truncates at Question:/User: before scoring.
def _clean_prediction_for_scoring(response: str) -> str:
    return response.split("Question:")[0].split("User:")[0].strip()


def evaluate_correctness(responses, golden_answers):
    results = []

    for response, golden_answer in zip(responses, golden_answers):
        # DIFF FROM eval_tooluse_new.py: clean loop artifacts before extraction.
        clean_response = _clean_prediction_for_scoring(response)
        pred_actions = extract_actions(clean_response)
        pred_inputs = extract_action_inputs(clean_response)

        gt_actions = [item["Action"] for item in golden_answer]
        gt_inputs = {}
        for item in golden_answer:
            try:
                loaded = json.loads(item["Action_Input"])
                # DIFF FROM eval_tooluse_new.py: golden omits "" / None keys (sft_slop).
                gt_inputs.update(_filter_nonempty_inputs(loaded))
            except json.JSONDecodeError:
                pass

        # DIFF FROM eval_tooluse_new.py: prediction also filtered before compare.
        filtered_pred_inputs = _filter_nonempty_inputs(pred_inputs)

        actions_match = Counter(pred_actions) == Counter(gt_actions)
        inputs_match = filtered_pred_inputs == gt_inputs

        results.append(1 if (actions_match and inputs_match) else 0)

    return results


def main():
    args = parse_args()

    llm, tokenizer = load_model_and_tokenizer(
        args.model_path,
        enable_thinking=args.enable_thinking,
        base_model_name=args.base_model_name,
    )
    test_data = load_test_data(tokenizer, enable_thinking=args.enable_thinking)

    prompts = [example["prompt"] for example in test_data]
    golden_answers = [example["golden_answer"] for example in test_data]

    responses = generate_responses(
        llm,
        prompts,
        args.max_new_tokens,
        args.temperature,
        tokenizer=tokenizer,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    print("\nEvaluating responses (sft_slop scoring)...")
    scores = evaluate_correctness(responses, golden_answers)
    accuracy = np.mean(scores)

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  Total samples: {len(scores)}")
    print(f"  Correct: {sum(scores)}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 60)

    output_dir = args.output_dir if args.output_dir else args.model_path
    os.makedirs(output_dir, exist_ok=True)

    results_to_save = {
        "accuracy": float(accuracy),
        "num_correct": int(sum(scores)),
        "num_total": len(scores),
        "per_sample_scores": scores,
        "config": {
            "model_path": args.model_path,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "enable_thinking": args.enable_thinking,
            # DIFF FROM eval_tooluse_new.py: records which scoring variant was used.
            "scoring": "sft_slop",
        },
    }

    output_path = os.path.join(output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nSaved results to {output_path}")

    responses_path = os.path.join(output_dir, "eval_responses.json")
    with open(responses_path, "w") as f:
        json.dump(
            [
                {
                    "prompt": test_data[i]["prompt"],
                    "response": responses[i],
                    "golden_answer": golden_answers[i],
                    "correct": bool(scores[i]),
                }
                for i in range(len(responses))
            ],
            f,
            indent=2,
        )
    print(f"Saved responses to {responses_path}")


if __name__ == "__main__":
    main()
