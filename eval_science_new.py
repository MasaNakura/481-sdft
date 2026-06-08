"""
Science eval for Qwen3.5+ checkpoints (transformers>=5 tokenizer + recent vLLM).

Use requirements_new.txt.
"""

import argparse
import json
import os

# CHANGE: must precede `from vllm import ...` so vLLM picks up the in-process EngineCore mode.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import numpy as np
import torch
from datasets import Dataset
from vllm import SamplingParams

from qwen_compat_new import apply_chat_template_compat, build_vllm_llm, load_tokenizer


def _debug_print_loaded_weight_paths(vllm_path: str) -> None:
    """Resolve and print every safetensors / pytorch_model.* file under vllm_path.

    # CHANGE: when the eval reports identical scores between an SFT checkpoint and the
    # base model, it's almost always because the staging-dir symlink chain ended up
    # pointing at the wrong weights. Printing the realpath (size + hash-prefix) of
    # every weight file makes that immediately visible. Cheap (1 stat + 4kB read per
    # shard) and only runs once per eval.
    """
    import glob
    import hashlib
    from pathlib import Path

    if not os.path.isdir(vllm_path):
        print(f"[debug] vllm_path is not a local directory: {vllm_path}")
        return

    patterns = ("model.safetensors", "model-*.safetensors", "pytorch_model.bin", "pytorch_model-*.bin")
    shards: list[str] = []
    for pat in patterns:
        shards.extend(glob.glob(os.path.join(vllm_path, pat)))

    if not shards:
        print(f"[debug] no weight shards found directly under {vllm_path}")
        return

    print("[debug] weights vLLM will load (realpath, size, sha256[:16]):")
    for shard in sorted(shards):
        real = os.path.realpath(shard)
        try:
            size = os.path.getsize(real)
            # cheap fingerprint: hash the first 1 MiB. Full hash on a 4GB shard is overkill
            # for "are these two files the same blob?" — collisions in 1MiB are negligible.
            h = hashlib.sha256()
            with open(real, "rb") as f:
                h.update(f.read(1024 * 1024))
            fp = h.hexdigest()[:16]
        except OSError as exc:
            size, fp = -1, f"<unreadable: {exc}>"
        link_arrow = f" -> {real}" if real != shard else ""
        print(f"[debug]   {shard}{link_arrow}  size={size}  sha256[:16]={fp}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on science test set (new stack)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of tokens to generate")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    # CHANGE: expose the full Qwen3.5 recommended sampling surface so users can match the
    # model card recipe without editing source. Defaults below mirror vLLM SamplingParams
    # defaults (neutral, no behavior change for existing commands). For Qwen3.5-2B
    # non-thinking text tasks the model card recommends:
    #   --temperature 1.0 --top_p 1.0 --top_k 20 --presence_penalty 2.0
    # For Qwen3.5 thinking text tasks:
    #   --temperature 1.0 --top_p 0.95 --top_k 20 --presence_penalty 1.5
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

    # CHANGE: diagnostic — print the *real* path of every safetensors shard vLLM will load
    # (resolving symlinks) so we can confirm the staging dir is pointing at the fine-tuned
    # weights and not silently backfilling from the base model snapshot.
    _debug_print_loaded_weight_paths(vllm_path)

    tokenizer = load_tokenizer(tokenizer_path, padding_side="left")
    # CHANGE: for multimodal checkpoints missing preprocessor_config.json the resolver
    # returns a staging dir (same path twice) that contains both the checkpoint files
    # AND the base-model processor assets — so no separate tokenizer= kwarg is needed.
    llm = build_vllm_llm(
        vllm_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
        default_chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    return llm, tokenizer


def load_test_data():
    path = "data/science_data/eval_data"
    print(f"Loading science test dataset from {path}")
    return Dataset.load_from_disk(path)


def _collect_stop_token_ids(tokenizer) -> list[int]:
    """Collect every plausible end-of-turn token id for the tokenizer.

    # CHANGE: Qwen3.5 generation_config.json lists multiple eos tokens (typically
    # [<|im_end|>=151645, <|endoftext|>=151643]); previous eval used only
    # `tokenizer.eos_token_id`, missing the second one. Together with the
    # apply_chat_template_compat path bug, missing stop tokens contributed to the
    # rambling / never-terminating outputs reported in the field.
    """
    stops: list[int] = []
    if tokenizer.eos_token_id is not None:
        stops.append(int(tokenizer.eos_token_id))
    # gather any extra eos ids from the tokenizer/model generation_config if available
    gen_eos = None
    if hasattr(tokenizer, "init_kwargs"):
        gen_eos = tokenizer.init_kwargs.get("eos_token_id")
    if gen_eos is None:
        gen_eos = getattr(tokenizer, "added_tokens_encoder", {}).get("<|im_end|>")
    if isinstance(gen_eos, int):
        stops.append(gen_eos)
    elif isinstance(gen_eos, list):
        stops.extend(int(x) for x in gen_eos if isinstance(x, int))
    # explicit lookup for Qwen-family end markers in case they aren't surfaced above
    for marker in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(marker)
        except Exception:
            tid = None
        if isinstance(tid, int) and tid >= 0:
            stops.append(tid)
    # de-dup, preserve order
    seen: set[int] = set()
    out: list[int] = []
    for tid in stops:
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out or None


def generate_responses(
    llm,
    tokenizer,
    prompts,
    max_new_tokens=2048,
    temperature=0.0,
    enable_thinking=False,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
):
    formatted_prompts = []
    for prompt in prompts:
        formatted_prompt = apply_chat_template_compat(
            tokenizer,
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        formatted_prompts.append(formatted_prompt)

    # Sanity log: print the first rendered prompt's tail so the user can verify the
    # train/eval chat-template format matches (i.e. with enable_thinking=False the prompt
    # ends with `<|im_start|>assistant\n<think>\n\n</think>\n\n`).
    if formatted_prompts:
        tail = formatted_prompts[0][-200:]
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

    print(f"Generating responses for {len(formatted_prompts)} prompts...")
    print(
        f"  sampling: temperature={temperature} top_p={top_p} top_k={top_k} min_p={min_p}"
        f" presence_penalty={presence_penalty} repetition_penalty={repetition_penalty}"
    )
    outputs = llm.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def evaluate_correctness(responses, answers):
    results = []
    for response, answer in zip(responses, answers):
        extracted = extract_xml_answer(response)
        results.append(1 if extracted == answer else 0)
    return results


def main():
    args = parse_args()

    llm, tokenizer = load_model_and_tokenizer(
        args.model_path,
        enable_thinking=args.enable_thinking,
        base_model_name=args.base_model_name,
    )
    test_data = load_test_data()

    prompts = [example["prompt"] for example in test_data]
    answers = [example["answer"] for example in test_data]

    responses = generate_responses(
        llm,
        tokenizer,
        prompts,
        args.max_new_tokens,
        args.temperature,
        enable_thinking=args.enable_thinking,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    print("\nEvaluating responses...")
    scores = evaluate_correctness(responses, answers)
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
                    "prompt": prompts[i],
                    "response": responses[i],
                    "answer": answers[i],
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
