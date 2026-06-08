"""
SDFT training entrypoint for Qwen3.5+ with transformers>=5 and recent vLLM.

Use requirements_new.txt. Original main.py remains on the pinned Qwen2.5 stack.
"""

# CHANGE: must run before vLLM (transitively imported via distil_trainer_new -> trl_compat_new)
# is loaded. Keeps the EngineCore in-process so weight sync over collective_rpc actually works.
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from distil_trainer_new import DistilTrainer
from distil_config import DistilConfig
from qwen_compat_new import load_causal_lm, load_tokenizer
from datasets import Dataset, load_from_disk
from string import Template
from transformers.trainer_utils import get_last_checkpoint
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer (Qwen3.5 / new stack)")
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
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Qwen3.5 only: enable thinking mode in chat template and vLLM (default: off)",
    )
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


if __name__ == "__main__":
    args = parse_args()
    # CHANGE: load_causal_lm uses dtype= + trust_remote_code (not torch_dtype).
    model = load_causal_lm(args.model_name)
    teacher_model = load_causal_lm(args.model_name)
    tokenizer = load_tokenizer(args.model_name)
    if args.dataset_name == "tooluse":
        dataset, _ = load_tooluse_dataset(args.seed)
    elif args.dataset_name == "science":
        dataset, _ = load_science_dataset(args.seed)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

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
        max_prompt_length = 1024,
        max_completion_length = 1024,
        num_train_epochs = args.num_train_epochs,
        num_iterations = 1,
        num_generations = 1,
        save_strategy = "epoch",
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
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # CHANGE: resume only if output_dir contains a valid HF checkpoint (checkpoint-N/
    # with trainer_state.json). Eval-only dirs (eval_results.json only) are ignored.
    resume_from_checkpoint = None
    if args.output_dir and os.path.isdir(args.output_dir):
        resume_from_checkpoint = get_last_checkpoint(args.output_dir)
    if resume_from_checkpoint:
        print(f"Resuming training from {resume_from_checkpoint}")
    else:
        print("No checkpoint found; starting training from scratch.")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
