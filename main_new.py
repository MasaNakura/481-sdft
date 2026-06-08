# CHANGE: must be set before vLLM is imported (via distil_trainer_new).
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# CHANGE: vLLM 0.21's flashinfer top-k/top-p sampler kernel is mismatched with the
# installed flashinfer-python wrapper on this cluster (missing `top_k_mask_logits`
# in the compiled .so), which dies during _dummy_sampler_run. Fall back to the
# PyTorch/Triton sampler. See https://github.com/vllm-project/vllm/issues/23023
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from distil_trainer_new import DistilTrainer
from distil_config import DistilConfig
from qwen_compat_new import load_causal_lm, load_tokenizer
import torch
from datasets import Dataset, load_from_disk
from string import Template
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    # CHANGE: default checkpoint targets Qwen3.5 dense instruct (override for MoE variants).
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3.5-9B-Instruct",
        help="Model name or path",
    )
    parser.add_argument("--dataset_name", type=str, default="tooluse", help="Dataset name", choices=["tooluse", "science"])
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--enable_thinking", action="store_true", help="Qwen3.5: enable thinking template")
    # CHANGE: tooluse prompts are ~900–1600+ tokens after chat template; 1024 truncates
    # tool docs / question during SDFT rollouts. Science fits in 1024. Defaults are set
    # per dataset below; override here if needed.
    parser.add_argument("--max_prompt_length", type=int, default=None,
                        help="Max prompt tokens for SDFT rollouts (default: 2048 tooluse, 1024 science).")
    parser.add_argument("--max_completion_length", type=int, default=None,
                        help="Max generated tokens per rollout (default: 1024).")
    # CHANGE: in-training eval runs at the end of every epoch (and once more at train end).
    parser.add_argument("--eval_max_new_tokens", type=int, default=1024, help="Max new tokens for in-training eval.")
    # CHANGE: vLLM sampling params forwarded to eval_*_new.generate_responses.
    parser.add_argument("--eval_temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy).")
    parser.add_argument("--eval_top_p", type=float, default=1.0, help="Nucleus sampling.")
    parser.add_argument("--eval_top_k", type=int, default=-1, help="Top-k; -1 disables.")
    parser.add_argument("--eval_min_p", type=float, default=0.0, help="Min-p; 0 disables.")
    parser.add_argument("--eval_presence_penalty", type=float, default=0.0, help="vLLM presence_penalty.")
    parser.add_argument("--eval_repetition_penalty", type=float, default=1.0, help="vLLM repetition_penalty (1 = off).")
    parser.add_argument("--eval_seed", type=int, default=None, help="Sampling seed for reproducibility.")
    return parser.parse_args()

def load_tooluse_dataset(seed=42) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts."""
    train_dir = 'data/tooluse_data/train_data'
    train_dataset = load_from_disk(train_dir) 

    def format_example(example):

        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": example['prompt']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(orig_content=example['prompt'], output_text='\n'.join(example['golden_response']))}],
        }
    
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None


def load_science_dataset(seed=42) -> Dataset:
    """Load and prepare science dataset with formatted prompts."""
    path = 'data/science_data/train_data'
    print(f"Loading science dataset from {path}")
    dataset = load_from_disk(path)

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                example["messages"][0],
                {'role': 'user', 'content': teacher_prompt.substitute(
                    orig_content=example['messages'][1]['content'],
                    output_text=example['output_text']
                )},
            ],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples")
    return dataset, None


# CHANGE: write eval_*.json into checkpoint-<step>/ at every epoch end, using
# the trainer's already-loaded vLLM. HF Trainer also saves full checkpoints there
# when save_strategy="epoch".
class SDFTEvalCallback(TrainerCallback):
    def __init__(self, dataset_name, eval_max_new_tokens,
                 temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty, seed):
        self.dataset_name = dataset_name
        self.eval_max_new_tokens = eval_max_new_tokens
        self.sampling = dict(
            temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
            presence_penalty=presence_penalty, repetition_penalty=repetition_penalty, seed=seed,
        )
        self.trainer = None  # set after trainer construction

    # CHANGE: evaluate after every epoch instead of every N steps.
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None or state.global_step == 0:
            return
        out_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self._run_eval(out_dir, state.global_step)

    # CHANGE: also evaluate the final model after training completes. Writes alongside
    # the final model in args.output_dir (skipped if step coincidentally matches an
    # eval_steps interval that already wrote to checkpoint-<step>/).
    def on_train_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return
        self._run_eval(args.output_dir, state.global_step)

    def _run_eval(self, out_dir, step):
        # Sync the trainer's vLLM with the current student weights so eval reflects this step.
        was_asleep = False
        if self.trainer.vllm_mode == "colocate" and self.trainer.args.vllm_enable_sleep_mode:
            torch.cuda.empty_cache()
            self.trainer.llm.wake_up()
            was_asleep = True
        try:
            self.trainer._move_model_to_vllm()
            self.trainer._last_loaded_step = step

            if self.dataset_name == "science":
                from eval_science_new_flop import load_test_data, generate_responses, evaluate_correctness
                test_data = load_test_data()
                prompts = [ex["prompt"] for ex in test_data]
                answers = [ex["answer"] for ex in test_data]
                responses = generate_responses(
                    self.trainer.llm, self.trainer.processing_class, prompts,
                    max_new_tokens=self.eval_max_new_tokens,
                    enable_thinking=getattr(self.trainer, "enable_thinking", False),
                    **self.sampling,
                )
                scores = evaluate_correctness(responses, answers)
                rows = [{"prompt": prompts[i], "response": responses[i], "answer": answers[i], "correct": bool(scores[i])}
                        for i in range(len(responses))]
            else:  # tooluse
                from eval_tooluse_new import load_test_data, generate_responses, evaluate_correctness
                test_data = load_test_data(self.trainer.processing_class,
                                           enable_thinking=getattr(self.trainer, "enable_thinking", False))
                prompts = [ex["prompt"] for ex in test_data]
                golden = [ex["golden_answer"] for ex in test_data]
                responses = generate_responses(
                    self.trainer.llm, prompts,
                    max_new_tokens=self.eval_max_new_tokens,
                    tokenizer=self.trainer.processing_class,
                    **self.sampling,
                )
                scores = evaluate_correctness(responses, golden)
                rows = [{"prompt": prompts[i], "response": responses[i], "golden_answer": golden[i], "correct": bool(scores[i])}
                        for i in range(len(responses))]

            os.makedirs(out_dir, exist_ok=True)
            acc = float(sum(scores)) / len(scores) if scores else 0.0
            with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
                json.dump({"step": step, "accuracy": acc,
                           "num_correct": int(sum(scores)), "num_total": len(scores)}, f, indent=2)
            with open(os.path.join(out_dir, "eval_responses.json"), "w") as f:
                json.dump(rows, f, indent=2)
            print(f"[SDFTEvalCallback] step={step} accuracy={acc*100:.2f}% -> {out_dir}")
        finally:
            if was_asleep:
                self.trainer.llm.sleep(level=1)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    model = load_causal_lm(args.model_name)
    teacher_model = load_causal_lm(args.model_name)
    tokenizer = load_tokenizer(args.model_name)
    if args.dataset_name == "tooluse":
        dataset, _ = load_tooluse_dataset(args.seed)
    elif args.dataset_name == "science":
        dataset, _ = load_science_dataset(args.seed)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    # CHANGE: dataset-specific context defaults. Tooluse embeds full API docs in every
    # prompt (median ~3k chars ≈ 900–1100 tokens + template); teacher_prompt adds the
    # golden_response on top. Science prompts are much shorter and work at 1024.
    if args.max_prompt_length is None:
        args.max_prompt_length = 2048 if args.dataset_name == "tooluse" else 1024
    if args.max_completion_length is None:
        args.max_completion_length = 1024
    print(f"SDFT context: max_prompt_length={args.max_prompt_length}, "
          f"max_completion_length={args.max_completion_length}, "
          f"vLLM max_model_len={args.max_prompt_length + args.max_completion_length}")

    config = DistilConfig(
        seed=args.seed,
        use_vllm = True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1, 
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True, 
        learning_rate = args.learning_rate,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = args.num_prompts_per_batch,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = args.num_train_epochs,
        num_iterations = 1,
        num_generations = 1,
        save_strategy = "epoch",
        save_total_limit = 1,
        max_grad_norm = 1,
        report_to = "wandb",
        output_dir = args.output_dir,
        log_completions = False, # True for debugging
        sync_ref_model = True,
        ref_model_sync_steps = 1,
        ref_model_mixup_alpha = args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction = True,
        num_loss_tokens_to_skip = 3,
        enable_thinking = args.enable_thinking,
    )
    eval_cb = SDFTEvalCallback(
        args.dataset_name, args.eval_max_new_tokens,
        temperature=args.eval_temperature, top_p=args.eval_top_p, top_k=args.eval_top_k,
        min_p=args.eval_min_p, presence_penalty=args.eval_presence_penalty,
        repetition_penalty=args.eval_repetition_penalty, seed=args.eval_seed,
    )
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[eval_cb],
    )
    eval_cb.trainer = trainer

    # Resume only if output_dir contains a valid HF checkpoint (checkpoint-N/
    # with trainer_state.json). Eval-only dirs are ignored.
    resume_from_checkpoint = None
    if args.output_dir and os.path.isdir(args.output_dir):
        resume_from_checkpoint = get_last_checkpoint(args.output_dir)
    if resume_from_checkpoint:
        print(f"Resuming training from {resume_from_checkpoint}")
    else:
        print("No checkpoint found; starting training from scratch.")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Final save -- model weights + tokenizer in output_dir root (alongside last epoch eval).
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
