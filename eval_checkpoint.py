"""
Resolve Hugging Face / vLLM load paths for Trainer output directories.

SFT/SDFT runs often leave weights under checkpoint-* without a valid config.json
in the training output root. Eval scripts use this helper before loading.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional


_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "adapter_model.safetensors",
)


def _has_config(path: Path) -> bool:
    config = path / "config.json"
    if not config.is_file():
        return False
    try:
        with config.open() as f:
            data = json.load(f)
        return bool(data.get("model_type"))
    except (json.JSONDecodeError, OSError):
        return False


def _has_weights(path: Path) -> bool:
    return any((path / name).is_file() for name in _WEIGHT_FILES)


# CHANGE: vLLM treats Qwen3.5 as multimodal (Qwen3_5ForConditionalGeneration) and tries to
# load an image processor from the tokenizer dir. HF Trainer.save() only persists the
# tokenizer, so checkpoint dirs are missing preprocessor_config.json and vLLM crashes.
def _has_preprocessor(path: Path) -> bool:
    return (path / "preprocessor_config.json").is_file()


_MULTIMODAL_ARCH_HINTS = (
    "ForConditionalGeneration",
    "VLForConditionalGeneration",
    "VLForCausalLM",
)


def _checkpoint_is_multimodal(path: Path) -> bool:
    """Heuristic: does the checkpoint's config.json declare a multimodal architecture?"""
    config = path / "config.json"
    if not config.is_file():
        return False
    try:
        with config.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    archs = data.get("architectures") or []
    return any(any(hint in a for hint in _MULTIMODAL_ARCH_HINTS) for a in archs)


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def _list_checkpoints(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=_checkpoint_step,
    )


# CHANGE: vLLM's `cached_processor_from_config` reads `model_config.model` (the model
# path) when loading the image/audio processor and IGNORES the `tokenizer=` kwarg
# (vllm bug class tracked in vllm-project/vllm#18016). So redirecting only the
# tokenizer doesn't help. We instead build a unified staging dir that symlinks the
# checkpoint files plus the missing processor JSONs pulled from the base Hub repo,
# and point vLLM at that single dir.
_PROCESSOR_FILE_PATTERNS = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "chat_template.json",
)


def _build_unified_model_dir(checkpoint_dir: Path, base_model_name: str) -> Path:
    """Stage a temp dir with checkpoint symlinks + base-model processor JSONs.

    Idempotent across calls: HF snapshot_download is cached in the HF hub cache, so the
    only per-call cost is creating a fresh symlink farm. Returns the staging dir path.
    """
    from huggingface_hub import snapshot_download

    try:
        base_local = Path(
            snapshot_download(
                base_model_name,
                allow_patterns=list(_PROCESSOR_FILE_PATTERNS),
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch processor assets for {base_model_name} from the HF Hub "
            "(needed because the checkpoint is multimodal but lacks "
            "preprocessor_config.json). Make sure HF_TOKEN is set if the repo is "
            f"gated, or pre-download the repo into the HF cache. Original error: {exc}"
        ) from exc

    staging = Path(tempfile.mkdtemp(prefix="vllm_eval_staging_"))

    for entry in checkpoint_dir.iterdir():
        (staging / entry.name).symlink_to(entry.resolve())

    # CHANGE: only symlink the explicit processor patterns from base, not the entire
    # snapshot directory. The snapshot dir is shared across `snapshot_download` calls in
    # the same HF cache, so it may contain a previous full download of the base model —
    # including weight shards (`model.safetensors-00001-of-00001.safetensors`,
    # `model.safetensors.index.json`) and tokenizer files (`vocab.json`, `merges.txt`).
    # Earlier code blindly iterated `base_local.iterdir()` and added any file whose name
    # wasn't already in staging. Because the SFT checkpoint saved a single, unsharded
    # `model.safetensors` (different name from the base's `model.safetensors-00001-of-00001.safetensors`),
    # the base weight shards slipped through the `target.exists()` check and ended up in
    # staging. vLLM then preferred the sharded `index.json` over the single file and
    # loaded the BASE WEIGHTS, making any SFT eval byte-identical to the base-model eval.
    added: list[str] = []
    for pattern in _PROCESSOR_FILE_PATTERNS:
        src = base_local / pattern
        if not src.is_file():
            continue
        target = staging / pattern
        if target.exists():
            continue
        target.symlink_to(src.resolve())
        added.append(pattern)

    # Sanity: emit a loud warning if any weight-like file made it into staging via the
    # checkpoint side — usually fine (that's the SFT weights) but worth flagging.
    weight_files = sorted(p.name for p in staging.iterdir() if any(tag in p.name for tag in ("safetensors", "pytorch_model.bin")))
    print(
        f"Staged vLLM model dir at {staging} "
        f"(checkpoint={checkpoint_dir.name}, added from {base_model_name}: {added}, "
        f"weight files in staging: {weight_files})"
    )
    return staging


def _resolve_tokenizer_path(vllm_dir: Path, base_model_name: Optional[str]) -> Path:
    """
    Decide where vLLM should load the model/tokenizer/processor from.

    If the resolved model dir is multimodal but lacks preprocessor_config.json (Trainer
    saves the tokenizer only), build a unified staging dir that combines the checkpoint
    with the base model's processor JSONs, and return that. Otherwise return vllm_dir.

    Returns a Path that callers should pass as BOTH the vLLM model path and the
    tokenizer path; vLLM ignores `tokenizer=` for processor lookups, so they must agree.
    """
    needs_preprocessor = _checkpoint_is_multimodal(vllm_dir)
    has_preprocessor = _has_preprocessor(vllm_dir)
    if needs_preprocessor and not has_preprocessor:
        if not base_model_name:
            raise ValueError(
                f"{vllm_dir} is a multimodal architecture (e.g. Qwen3.5) but has no "
                "preprocessor_config.json — HF Trainer save only persists the tokenizer. "
                "Pass --base_model_name <hub_id> (e.g. Qwen/Qwen3.5-...-Instruct) so the "
                "eval can stage the image processor from the original checkpoint."
            )
        return _build_unified_model_dir(vllm_dir, base_model_name)
    return vllm_dir


def _looks_like_hub_id(model_path: str) -> bool:
    """Heuristic: a Hub id is `org/name` with no path separators that look local.

    HF allows arbitrary characters but in practice user-facing ids are `org/name` or
    `org/name@revision`. We accept anything with exactly one '/' and no leading dot or
    backslash; vLLM / HF will give a clear error if the id is bogus.
    """
    if not model_path or model_path.startswith((".", "/", "\\", "~")):
        return False
    return model_path.count("/") == 1 and "\\" not in model_path


def resolve_pretrained_dirs(
    model_path: str,
    base_model_name: Optional[str] = None,
) -> tuple[str, str]:
    """
    Return (vllm_load_path, tokenizer_load_path).

    Search order:
    1. model_path if it contains a valid config.json
    2. Latest checkpoint-* under model_path with valid config.json
    3. model_path for weights + base_model_name for tokenizer/config (needs base_model_name)
    4. base_model_name only if model_path has no weights (evaluate base model)
    5. model_path treated as a HF Hub id (e.g. "Qwen/Qwen3.5-2B") if it doesn't exist on
       disk and looks like one — lets you `--model_path Qwen/Qwen3.5-2B` to evaluate the
       base model directly without staging a local copy.

    # CHANGE: tokenizer_load_path is independently resolved when the model dir is multimodal
    # but lacks preprocessor_config.json — in that case it falls back to base_model_name so
    # vLLM can find the image-processor assets. vllm_load_path always points at the weights.
    """
    root = Path(model_path).expanduser().resolve()
    if not root.exists():
        # CHANGE: graceful Hub-id fallback. Earlier the function hard-raised here, which
        # forced users to either git clone the base model or maintain a local mirror just
        # to run a baseline eval. vLLM and HF both accept Hub ids directly, so when the
        # local path doesn't exist we hand the id through unchanged.
        if _looks_like_hub_id(model_path):
            print(
                f"model_path `{model_path}` not found locally; treating as Hugging Face "
                "Hub id and loading directly."
            )
            return model_path, model_path
        raise FileNotFoundError(
            f"model_path does not exist: {root}\n"
            "If you meant to evaluate a Hub model (e.g. `Qwen/Qwen3.5-2B`), pass it as "
            "`--model_path Qwen/Qwen3.5-2B` (org/name form) and it will be loaded from "
            "the Hugging Face Hub."
        )

    if _has_config(root):
        unified = _resolve_tokenizer_path(root, base_model_name)
        # CHANGE: vLLM's processor loader keys off `model_config.model`, not the
        # tokenizer arg, so the staging dir (when used) must be both the model and
        # tokenizer path. Otherwise vllm/tokenizer paths are the same as before.
        return str(unified), str(unified)

    checkpoints = _list_checkpoints(root)
    for ckpt in reversed(checkpoints):
        if _has_config(ckpt):
            print(f"Resolved checkpoint for eval: {ckpt}")
            unified = _resolve_tokenizer_path(ckpt, base_model_name)
            return str(unified), str(unified)

    if base_model_name:
        for ckpt in reversed(checkpoints):
            if _has_weights(ckpt) and not _has_config(ckpt):
                raise ValueError(
                    f"{ckpt} contains weights but no valid config.json. "
                    "vLLM cannot load this folder. Re-run SFT (updated sft.py saves config), "
                    "or use a checkpoint that includes config.json."
                )
        if _has_weights(root) and not _has_config(root):
            raise ValueError(
                f"{root} contains weights but no valid config.json in the output root. "
                "Point --model_path at a checkpoint-* subdirectory, or re-run SFT."
            )
        if not _has_weights(root) and not any(_has_weights(c) for c in checkpoints):
            print(f"No fine-tuned weights under {root}; evaluating base model {base_model_name}")
            return base_model_name, base_model_name

    hint = (
        f"No valid config.json under {root} or its checkpoint-* subfolders. "
        "Re-run SFT with the updated script (saves model + tokenizer), pass a checkpoint path "
        "(e.g. .../science/checkpoint-200), or set --base_model_name to the original hub id."
    )
    raise ValueError(hint)
