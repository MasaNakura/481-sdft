import argparse
from typing import Dict, List

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT baseline trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Effective batch size")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="tooluse",
        choices=["tooluse", "science"],
        help="Dataset name",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
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


def build_tokenized_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def tokenize_row(example: Dict) -> Dict[str, List[int]]:
        prompt_text = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_messages = example["prompt"] + [{"role": "assistant", "content": example["response"]}]
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]

        labels = full_ids.copy()
        supervised_start = min(len(prompt_ids), len(labels))
        labels[:supervised_start] = [-100] * supervised_start

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


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.dataset_name == "tooluse":
        dataset = load_tooluse_dataset(args.seed)
    elif args.dataset_name == "science":
        dataset = load_science_dataset(args.seed)
    else:
        raise ValueError(f"Invalid dataset_name: {args.dataset_name}")

    train_dataset = build_tokenized_dataset(dataset, tokenizer, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_prompts_per_batch,
        logging_steps=1,
        save_steps=100,
        bf16=torch.cuda.is_available(),
        fp16=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=SFTDataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
