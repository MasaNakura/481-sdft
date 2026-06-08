"""
SFT baseline for Qwen3.5+ with transformers>=5.

Use requirements_new.txt. Original sft.py remains on the pinned Qwen2.5 stack.
"""

# CHANGE: set before any (transitive) vLLM import; harmless for pure-HF SFT but keeps the
# same launch environment as main_new.py / eval_*_new.py.
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import json
from typing import Dict, List

import torch
from datasets import Dataset, load_from_disk
from transformers import LogitsProcessor, LogitsProcessorList, Trainer, TrainerCallback, TrainingArguments

from qwen_compat_new import apply_chat_template_compat, load_causal_lm, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="SFT baseline trainer (Qwen3.5 / new stack)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Effective batch size")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3.5-9B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tooluse",
        choices=["tooluse", "science"],
        help="Dataset name",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Use Qwen3.5 thinking chat template (default off for stable SFT on golden text)",
    )
    # CHANGE: in-training eval, mirrors main_new.py's SDFTEvalCallback CLI surface.
    # SFT has no colocated vLLM so the callback uses HF model.generate().
    # - temperature / top_p / top_k / repetition_penalty / seed: native HF generate kwargs.
    # - min_p: native HF generate kwarg (requires transformers >= 4.40, satisfied by requirements_new.txt).
    # - presence_penalty: not native in HF — implemented below via _PresencePenaltyLogitsProcessor
    #   to match vLLM's behavior (penalize tokens that have appeared in the generated
    #   suffix, NOT in the prompt — same as vLLM SamplingParams).
    parser.add_argument("--eval_max_new_tokens", type=int, default=512, help="Max new tokens for in-training eval.")
    parser.add_argument("--eval_temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy).")
    parser.add_argument("--eval_top_p", type=float, default=1.0, help="Nucleus sampling.")
    parser.add_argument("--eval_top_k", type=int, default=-1, help="Top-k; -1 disables.")
    parser.add_argument("--eval_min_p", type=float, default=0.0, help="Min-p; 0 disables.")
    parser.add_argument("--eval_presence_penalty", type=float, default=0.0, help="vLLM-style presence_penalty (0 disables).")
    parser.add_argument("--eval_repetition_penalty", type=float, default=1.0, help="repetition_penalty (1 = off).")
    parser.add_argument("--eval_seed", type=int, default=None, help="Sampling seed for reproducibility.")
    parser.add_argument("--skip_eval", action="store_true", help="Skip in-training eval entirely.")
    return parser.parse_args()


def load_tooluse_dataset(seed=42) -> Dataset:
    train_dir = "data/tooluse_data/train_data"
    dataset = load_from_disk(train_dir)

    def format_example(example):
        return {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "response": "\n".join(example["golden_response"]),
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    return dataset


def load_science_dataset(seed=42) -> Dataset:
    path = "data/science_data/train_data"
    dataset = load_from_disk(path)

    def format_example(example):
        return {
            "prompt": example["messages"],
            "response": example["output_text"],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    return dataset


def build_tokenized_dataset(dataset: Dataset, tokenizer, max_length: int, enable_thinking: bool) -> Dataset:
    def tokenize_row(example: Dict) -> Dict[str, List[int]]:
        # CHANGE: tokenize prompt and response SEPARATELY rather than diffing two chat-template
        # renderings. The original `labels[:len(prompt_ids)] = -100` pattern only works when
        # prompt_ids is a strict prefix of full_ids. Qwen2.5 satisfies that; Qwen3 / Qwen3.5
        # with enable_thinking=False does NOT — the chat template inserts a
        # `<think>\n\n</think>\n\n` block after `<|im_start|>assistant\n` ONLY when
        # add_generation_prompt=True, so prompt_ids is several tokens LONGER at the divergence
        # point than the corresponding region of full_ids. The old code then masked the first
        # few real response tokens (and trained the model on a train↔eval distribution that
        # never matched), tanking accuracy.
        #
        # Composing the full sequence as `prompt_ids + response_ids + eos` makes label masking
        # correct by construction for both Qwen2.5 and Qwen3.5, regardless of enable_thinking.
        prompt_text = apply_chat_template_compat(
            tokenizer,
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(example["response"], add_special_tokens=False)["input_ids"]

        # End-of-turn token. Qwen-family tokenizers use <|im_end|> as eos_token; that exactly
        # mirrors how the chat template terminates an assistant message
        # (`<|im_start|>assistant\n{content}<|im_end|>`).
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            response_ids = list(response_ids) + [eos_id]

        full_ids = list(prompt_ids) + list(response_ids)
        labels = [-100] * len(prompt_ids) + list(response_ids)

        # CHANGE: previously we truncated full_ids/labels at max_length, which silently
        # dropped the final `</answer>` and EOS token from long-response examples. The
        # model would then be supervised on a prefix that never terminates, learn to keep
        # generating, and at eval ramble for max_new_tokens without ever closing the
        # answer block. To preserve learn-to-stop signal we keep the trailing EOS by
        # truncating the *response* (not the suffix), so the eos_id always lands in
        # labels. The system prompt itself fits well under max_length, so we leave
        # prompt_ids untouched and only carve into response_ids.
        if len(full_ids) > max_length:
            keep_response = max(0, max_length - len(prompt_ids))
            if keep_response <= 1:
                # Pathological: prompt alone already exceeds max_length. Drop the example
                # by emitting an empty label sequence so the trainer skips its loss
                # contribution (all labels == -100). full_ids still needs to be valid.
                full_ids = list(prompt_ids[:max_length])
                labels = [-100] * len(full_ids)
            else:
                # Reserve last slot for eos; truncate response head-only to keep the
                # terminator intact.
                trunc_response = list(response_ids[: keep_response - 1])
                if eos_id is not None:
                    trunc_response.append(eos_id)
                full_ids = list(prompt_ids) + trunc_response
                labels = [-100] * len(prompt_ids) + list(trunc_response)

        attention_mask = [1] * len(full_ids)
        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized = dataset.map(tokenize_row, remove_columns=dataset.column_names)
    return tokenized


class SFTDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            seq_len = len(feature["input_ids"])
            pad_len = max_len - seq_len
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# CHANGE: HF generate has no `presence_penalty` (vLLM-only kwarg). Implement it as a
# LogitsProcessor so the SFT eval CLI mirrors main_new.py / eval_*_new.py exactly.
# Semantics match vLLM SamplingParams.presence_penalty: subtract `penalty` from the
# logit of every token that has appeared in the GENERATED suffix (after prompt_len),
# NOT in the prompt itself.
class _PresencePenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float, prompt_len: int):
        self.penalty = float(penalty)
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] <= self.prompt_len:
            return scores
        for i in range(input_ids.shape[0]):
            generated = input_ids[i, self.prompt_len:]
            if generated.numel() == 0:
                continue
            unique = torch.unique(generated)
            scores[i, unique] = scores[i, unique] - self.penalty
        return scores


# CHANGE: mirror eval_*_new.py's stop-token logic so HF generate stops on the same
# tokens vLLM would. Qwen3.5 has multiple eos ids ([<|im_end|>, <|endoftext|>]) and
# using only tokenizer.eos_token_id can cause runaway generation.
def _collect_stop_token_ids(tokenizer):
    stops = []
    if tokenizer.eos_token_id is not None:
        stops.append(int(tokenizer.eos_token_id))
    for marker in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(marker)
        except Exception:
            tid = None
        if isinstance(tid, int) and tid >= 0:
            stops.append(tid)
    seen = set()
    out = []
    for tid in stops:
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out or None


# CHANGE: in-training eval callback for SFT, analogous to main_new.py's SDFTEvalCallback.
# Key difference: SFT has no colocated vLLM, so we generate via HF model.generate() on the
# in-memory model. Slower than vLLM but no extra model load and no OOM risk. Writes only
# eval_results.json + eval_responses.json — no model state — into checkpoint-<step>/ at
# each epoch, and into args.output_dir at training end (alongside the final saved model).
class SFTEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset_name, enable_thinking, eval_max_new_tokens,
                 temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty, seed):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.enable_thinking = enable_thinking
        self.eval_max_new_tokens = eval_max_new_tokens
        self.sampling = dict(
            temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
            presence_penalty=presence_penalty, repetition_penalty=repetition_penalty, seed=seed,
        )
        self.trainer = None  # set after trainer construction

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None or state.global_step == 0:
            return
        out_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self._run_eval(out_dir, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return
        self._run_eval(args.output_dir, state.global_step)

    def _run_eval(self, out_dir, step):
        model = self.trainer.model
        model.eval()
        try:
            if self.dataset_name == "science":
                from eval_science_new import load_test_data, evaluate_correctness
                test_data = load_test_data()
                msgs = [ex["prompt"] for ex in test_data]
                answers = [ex["answer"] for ex in test_data]
                prompts = [
                    apply_chat_template_compat(self.tokenizer, m, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=self.enable_thinking)
                    for m in msgs
                ]
                responses = self._generate(model, prompts)
                scores = evaluate_correctness(responses, answers)
                rows = [{"prompt": prompts[i], "response": responses[i],
                         "answer": answers[i], "correct": bool(scores[i])}
                        for i in range(len(responses))]
            else:  # tooluse
                from eval_tooluse_new import load_test_data, evaluate_correctness
                test_data = load_test_data(self.tokenizer, enable_thinking=self.enable_thinking)
                prompts = [ex["prompt"] for ex in test_data]
                golden = [ex["golden_answer"] for ex in test_data]
                responses = self._generate(model, prompts)
                scores = evaluate_correctness(responses, golden)
                rows = [{"prompt": prompts[i], "response": responses[i],
                         "golden_answer": golden[i], "correct": bool(scores[i])}
                        for i in range(len(responses))]

            os.makedirs(out_dir, exist_ok=True)
            acc = float(sum(scores)) / len(scores) if scores else 0.0
            with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
                json.dump({"step": step, "accuracy": acc,
                           "num_correct": int(sum(scores)), "num_total": len(scores)}, f, indent=2)
            with open(os.path.join(out_dir, "eval_responses.json"), "w") as f:
                json.dump(rows, f, indent=2)
            print(f"[SFTEvalCallback] step={step} accuracy={acc * 100:.2f}% -> {out_dir}")
        finally:
            model.train()

    def _generate(self, model, prompts):
        # batch_size=1 keeps the implementation tiny: no need to swap the tokenizer's
        # padding_side (sft uses right padding for training, but generate needs left).
        # SFT eval over 97 (tooluse) or ~500 (science) samples at a few seconds each is
        # tolerable next to per-epoch training cost.
        device = next(model.parameters()).device
        stop_ids = _collect_stop_token_ids(self.tokenizer)
        s = self.sampling
        do_sample = (s.get("temperature") or 0.0) > 0
        if s.get("seed") is not None:
            torch.manual_seed(int(s["seed"]))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(s["seed"]))
        gen_kwargs = dict(
            max_new_tokens=self.eval_max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=stop_ids,
            do_sample=do_sample,
            repetition_penalty=s.get("repetition_penalty", 1.0),
        )
        if do_sample:
            gen_kwargs["temperature"] = s["temperature"]
            if s.get("top_p") is not None and s["top_p"] < 1.0:
                gen_kwargs["top_p"] = s["top_p"]
            if s.get("top_k") is not None and s["top_k"] > 0:
                gen_kwargs["top_k"] = s["top_k"]
            # min_p is a native HF generate kwarg (transformers>=4.40); only forward when > 0.
            if s.get("min_p") is not None and s["min_p"] > 0:
                gen_kwargs["min_p"] = s["min_p"]
        # presence_penalty handled below per-call (needs prompt_len for vLLM-matching semantics).
        presence_penalty = s.get("presence_penalty") or 0.0

        print(f"[SFTEvalCallback] generating {len(prompts)} responses "
              f"(HF generate, max_new_tokens={self.eval_max_new_tokens})")
        responses = []
        with torch.no_grad():
            for prompt in prompts:
                enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
                prompt_len = enc["input_ids"].shape[1]
                per_call_kwargs = gen_kwargs
                if presence_penalty != 0.0:
                    per_call_kwargs = dict(gen_kwargs)
                    per_call_kwargs["logits_processor"] = LogitsProcessorList(
                        [_PresencePenaltyLogitsProcessor(presence_penalty, prompt_len)]
                    )
                out = model.generate(**enc, **per_call_kwargs)
                new_tokens = out[0, prompt_len:]
                responses.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return responses


if __name__ == "__main__":
    args = parse_args()

    tokenizer = load_tokenizer(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.dataset_name == "tooluse":
        dataset = load_tooluse_dataset(args.seed)
    elif args.dataset_name == "science":
        dataset = load_science_dataset(args.seed)
    else:
        raise ValueError(f"Invalid dataset_name: {args.dataset_name}")

    train_dataset = build_tokenized_dataset(
        dataset, tokenizer, args.max_length, enable_thinking=args.enable_thinking
    )

    # CHANGE: dtype= via load_causal_lm (transformers>=5).
    model = load_causal_lm(args.model_name)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_prompts_per_batch,
        logging_steps=1,
        # CHANGE: disable HF Trainer's intermediate checkpointing (was save_steps=100).
        # SFTEvalCallback writes eval results per epoch into checkpoint-<step>/; the
        # final model is saved once after train() returns (unchanged from before).
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        fp16=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    eval_cb = None
    if not args.skip_eval:
        eval_cb = SFTEvalCallback(
            tokenizer, args.dataset_name, args.enable_thinking, args.eval_max_new_tokens,
            temperature=args.eval_temperature, top_p=args.eval_top_p, top_k=args.eval_top_k,
            min_p=args.eval_min_p, presence_penalty=args.eval_presence_penalty,
            repetition_penalty=args.eval_repetition_penalty, seed=args.eval_seed,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=SFTDataCollator(tokenizer),
        callbacks=[eval_cb] if eval_cb else None,
    )
    if eval_cb is not None:
        eval_cb.trainer = trainer
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
