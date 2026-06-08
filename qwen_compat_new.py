"""
Shared compatibility helpers for Qwen3.x / Qwen3.5 with transformers>=5 and recent vLLM.

Used by *_new.py entrypoints only; original scripts stay on the pinned stack in requirements.txt.
"""

from __future__ import annotations

import os

# CHANGE: vLLM v1 (>=0.10) spawns the EngineCore in a separate process by default and routes
# everything through msgpack RPC. That serialization path silently turns CUDA tensors passed
# as collective_rpc args into plain Python lists on the receiving side (no Tensor type hint
# means dec_hook never reconstructs them), which breaks any weight-sync flow that ships
# (name, tensor) tuples. Forcing in-process mode keeps everything in the trainer process so
# collective_rpc is a direct function call AND vLLM re-exposes `llm_engine.model_executor`
# for v0-style attribute access (see vllm/v1/engine/llm_engine.py: "for v0 compatibility").
# setdefault so users can still override via the environment.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import inspect
from typing import Any, Optional

import torch


def is_qwen3_family(model_id: str) -> bool:
    """Heuristic: Qwen3 / Qwen3.5 / Qwen3.6 checkpoints need newer HF + vLLM paths.

    Used for things keyed by the hub/source model id (e.g. vLLM `reasoning_parser`, dtype
    defaults). For tokenizer/chat-template behavior at eval time, prefer
    `tokenizer_uses_enable_thinking` instead — fine-tuned checkpoints save to arbitrary
    output dirs whose paths don't contain "qwen3".
    """
    mid = (model_id or "").lower()
    return any(tag in mid for tag in ("qwen3", "qwen3.5", "qwen3_5", "qwen3.6", "qwen3_6"))


def tokenizer_uses_enable_thinking(tokenizer) -> bool:
    """Return True iff the tokenizer's chat template references the `enable_thinking` variable.

    # CHANGE: previously we gated `enable_thinking` forwarding on
    # `is_qwen3_family(tokenizer.name_or_path)`. That works during training (path is
    # `Qwen/Qwen3.5-…`) but BREAKS at eval: `load_tokenizer(checkpoint_dir)` sets
    # `name_or_path` to whatever the user saved the SFT/SDFT output under, which almost
    # never contains the literal substring "qwen3". The flag was silently dropped at eval
    # only, and the Qwen3.5 template fell back to its default `enable_thinking=True`, so
    # eval rendered a thinking-on prompt for a model that had been trained on a thinking-off
    # prompt. The result was the rambling, never-terminating completions reported in the
    # field. Inspecting the saved chat-template string is invariant to local paths.

    Handles three template shapes:
      - str (most tokenizers)
      - dict {name: template_str} (multi-template tokenizers, e.g. tool_use vs default)
      - None / missing (older tokenizers without a chat template)
    """
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        return False
    if isinstance(template, dict):
        template = "\n".join(str(v) for v in template.values())
    return "enable_thinking" in str(template)


def filter_ctor_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs that the target __init__ does not accept (API drift across vLLM versions).

    # CHANGE: vLLM SamplingParams is a `msgspec.Struct` subclass whose generated __init__
    # is not always introspectable via `inspect.signature`. Prefer `__struct_fields__` when
    # available; fall back to inspect for plain classes.
    """
    struct_fields = getattr(cls, "__struct_fields__", None)
    if struct_fields is not None:
        return {k: v for k, v in kwargs.items() if k in struct_fields}
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def get_hf_model_init_kwargs(
    model_id: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: Optional[str] = "sdpa",
) -> dict[str, Any]:
    """
    # CHANGE: transformers>=5 prefers `dtype=` (not deprecated `torch_dtype=`).
    # CHANGE: Qwen3.5 registers custom code on the Hub — trust_remote_code is required.
    """
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def load_causal_lm(model_id: str, **extra_kwargs):
    """Load a causal LM with Qwen3.5-safe defaults."""
    from transformers import AutoConfig, AutoModelForCausalLM

    init_kwargs = get_hf_model_init_kwargs(model_id)
    init_kwargs.update(extra_kwargs)
    # CHANGE: explicit config load with trust_remote_code for Qwen3_5* architectures.
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if getattr(config, "architectures", None):
        import transformers

        architecture = getattr(transformers, config.architectures[0])
        return architecture.from_pretrained(model_id, config=config, **init_kwargs)
    return AutoModelForCausalLM.from_pretrained(model_id, **init_kwargs)


def load_tokenizer(model_id: str, **extra_kwargs):
    from transformers import AutoTokenizer

    # CHANGE: trust_remote_code for Qwen3.5 tokenizer / chat template assets.
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, **extra_kwargs)


def chat_template_kwargs(enable_thinking: bool) -> dict[str, Any]:
    """Kwargs for apply_chat_template on Qwen3+ models."""
    return {"enable_thinking": enable_thinking}


def apply_chat_template_compat(tokenizer, messages, *, enable_thinking: bool = False, **kwargs) -> str:
    """
    # CHANGE: Qwen3.5 chat templates accept enable_thinking (see model cards).
    Always forward enable_thinking when the tokenizer's chat template references the
    variable. Using the tokenizer's chat-template content (not its `name_or_path`) makes
    this work for fine-tuned checkpoints whose output dir doesn't contain "qwen3".

    Older revisions of this helper:
      1) Inspected `tokenizer.apply_chat_template` for an explicit `enable_thinking`
         parameter (always False — `apply_chat_template` forwards via **kwargs to the
         Jinja template; see
         https://huggingface.co/docs/transformers/main/en/chat_template_advanced).
      2) Gated on `is_qwen3_family(tokenizer.name_or_path)`. That worked during training
         (`Qwen/Qwen3.5-…`) but at eval `name_or_path` is the user's local SFT/SDFT save
         dir, which almost never contains "qwen3", so the flag was silently dropped at
         eval only. The Qwen3.5 template then fell back to its default
         `enable_thinking=True`, producing a thinking-on prompt for a model that had been
         SFT'd on the thinking-off prompt. The trained model ran in base-model "exploratory
         thinking" mode at eval, rambling past `max_new_tokens` without ever closing
         `</reasoning>` or emitting `<answer>`.

    Templates that don't understand `enable_thinking` silently ignore unknown Jinja
    variables, so we only forward when the template actually references the variable.
    """
    template_kwargs = dict(kwargs)
    if tokenizer_uses_enable_thinking(tokenizer):
        template_kwargs.setdefault("enable_thinking", enable_thinking)
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def build_vllm_llm(model_path: str, **user_kwargs):
    """
    Construct vLLM LLM with version-tolerant kwargs.

    # CHANGE: vLLM>=0.13 uses dtype (not torch_dtype) and adds reasoning_parser for Qwen3*.
    # CHANGE: logprobs_mode may be absent on some vLLM builds — only pass if supported.
    """
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        **user_kwargs,
    }

    if is_qwen3_family(model_path):
        kwargs.setdefault("reasoning_parser", "qwen3")
        # enable_thinking for vLLM chat template is opt-in via default_chat_template_kwargs in the caller.

    if "logprobs_mode" not in kwargs:
        kwargs["logprobs_mode"] = "processed_logprobs"

    kwargs = filter_ctor_kwargs(LLM, kwargs)
    # If this vLLM build dropped logprobs_mode, filter_ctor_kwargs already removed it.
    return LLM(**kwargs)


def model_generate_kwargs() -> dict[str, Any]:
    """
    # CHANGE: transformers>=5 removed disable_compile from GenerationMixin.generate on some models.
    """
    from transformers import GenerationMixin

    sig = inspect.signature(GenerationMixin.generate)
    if "disable_compile" in sig.parameters:
        return {"disable_compile": True}
    return {}


def merge_model_init_kwargs(base: Optional[dict], model_id: str) -> dict:
    """Merge DistilConfig.model_init_kwargs with Qwen3.5-safe defaults."""
    merged = get_hf_model_init_kwargs(model_id)
    if base:
        # Allow explicit overrides from config while normalizing deprecated torch_dtype.
        for key, value in base.items():
            if key == "torch_dtype" and "dtype" not in base:
                merged["dtype"] = value
            else:
                merged[key] = value
    merged.pop("torch_dtype", None)
    return merged
