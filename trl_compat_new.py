"""
TRL / vLLM import compatibility for the *_new.py stack (trl>=0.28, transformers>=5).

TRL reorganized several modules (e.g. VLLMClient moved from trl.extras to trl.generation).
This module centralizes fallbacks so distil_trainer_new.py stays aligned with upstream TRL.

References:
- https://huggingface.co/docs/trl/en/vllm_integration
- https://github.com/huggingface/trl/pull/4928 (VLLMClient -> trl.generation.vllm_client)
"""

from __future__ import annotations

from packaging.version import Version


def _trl_version() -> Version:
    import trl

    return Version(trl.__version__)


# CHANGE: VLLMClient path moved in trl>=0.28 (see PR #4928).
try:
    from trl.generation.vllm_client import VLLMClient
except ImportError:  # trl<0.28
    from trl.extras.vllm_client import VLLMClient  # type: ignore[no-redef]

from trl.data_utils import apply_chat_template, is_conversational, prepare_multimodal_messages

try:
    from trl.data_utils import maybe_apply_chat_template as trl_maybe_apply_chat_template
except ImportError:

    def trl_maybe_apply_chat_template(example: dict, processing_class) -> dict:
        """Fallback when maybe_apply_chat_template is unavailable."""
        if "prompt" in example:
            return {
                **example,
                "prompt": apply_chat_template(example["prompt"], processing_class),
            }
        return example


from trl.extras.profiling import profiling_context, profiling_decorator

try:
    from trl.import_utils import is_liger_kernel_available, is_vllm_available
except ImportError:
    from trl.import_utils import is_vllm_available

    def is_liger_kernel_available() -> bool:
        return False
from trl.models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection

# CHANGE: TRL main renamed BaseTrainer -> _BaseTrainer (still public via this alias).
try:
    from trl.trainer.base_trainer import BaseTrainer
except ImportError:
    from trl.trainer.base_trainer import _BaseTrainer as BaseTrainer  # type: ignore[misc]

from trl.trainer.utils import (
    RepeatSampler,
    disable_dropout_in_model,
    ensure_master_addr_port,
    entropy_from_logits,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)


def prepare_peft_model(model, peft_config, args):
    """
    # CHANGE: prepare_peft_model removed from trl.models exports in trl>=0.29; keep TRL 0.24 logic.
    """
    try:
        from trl.models.utils import prepare_peft_model as _fn

        return _fn(model, peft_config, args)
    except ImportError:
        return _prepare_peft_model_compat(model, peft_config, args)


def _prepare_peft_model_compat(model, peft_config, args):
    """Minimal port of trl v0.24 prepare_peft_model for trl>=0.29."""
    from peft import PeftModel, get_peft_model

    from transformers.utils import is_peft_available

    if not is_peft_available():
        raise ImportError("PEFT is required. Run `pip install peft`.")

    if isinstance(model, PeftModel) and peft_config is not None:
        model = model.merge_and_unload()

    if peft_config is not None:
        model = get_peft_model(model, peft_config)

    if getattr(args, "gradient_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=getattr(args, "gradient_checkpointing_kwargs", None) or {}
        )

    return model


def check_trl_vllm_versions() -> None:
    """Warn when TRL/vLLM versions are outside the documented integration range."""
    import warnings

    trl_v = _trl_version()
    if trl_v < Version("0.28.0"):
        warnings.warn(
            f"trl {_trl_version()} is older than 0.28.0; *_new.py expects trl>=0.28 "
            "(VLLMClient in trl.generation). Use requirements_new.txt.",
            stacklevel=2,
        )

    if is_vllm_available():
        import vllm

        vllm_v = Version(vllm.__version__.split("+")[0])
        # TRL pins 0.12–0.18; Qwen3.5 often needs >=0.17. *_new uses qwen_compat_new.build_vllm_llm directly.
        if vllm_v < Version("0.12.0"):
            warnings.warn(
                f"vLLM {vllm.__version__} is below 0.12.0; vLLM generation may not work.",
                stacklevel=2,
            )
        elif vllm_v > Version("0.18.0"):
            warnings.warn(
                f"vLLM {vllm.__version__} is newer than TRL's tested range (0.12–0.18). "
                "This is expected for Qwen3.5; report issues if colocate generation fails.",
                stacklevel=2,
            )


check_trl_vllm_versions()

__all__ = [
    "VLLMClient",
    "apply_chat_template",
    "is_conversational",
    "prepare_multimodal_messages",
    "trl_maybe_apply_chat_template",
    "profiling_context",
    "profiling_decorator",
    "is_liger_kernel_available",
    "is_vllm_available",
    "prepare_deepspeed",
    "prepare_fsdp",
    "prepare_peft_model",
    "unwrap_model_for_generation",
    "_ForwardRedirection",
    "BaseTrainer",
    "RepeatSampler",
    "disable_dropout_in_model",
    "ensure_master_addr_port",
    "entropy_from_logits",
    "identity",
    "nanmax",
    "nanmin",
    "nanstd",
    "pad",
    "print_prompt_completions_sample",
    "selective_log_softmax",
    "shuffle_sequence_dict",
    "split_pixel_values_by_grid",
    "split_tensor_dict",
    "unsplit_pixel_values_by_grid",
]
