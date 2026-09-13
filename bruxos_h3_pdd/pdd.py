"""Runtime adapter for Alibaba PAI MiniMax-H3 PDD LoRAs.

The original checkpoint is consumed directly.  Diffusers module names are
mapped to ComfyUI's fused MiniMax-H3 implementation at load time; no converted
LoRA is written to disk.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.model_sampling
import comfy.ops
from comfy.weight_adapter.lora import LoRAAdapter


LOGGER = logging.getLogger(__name__)

PDD_NUM_STEPS = 32
PDD_BLOCK_SIZE = 4
PDD_NFE = PDD_NUM_STEPS // PDD_BLOCK_SIZE
PDD_VIDEO_SHIFT = 12.0
PDD_AUDIO_SHIFT = 3.0
EXPECTED_TRANSFORMER_BLOCKS = 50
EXPECTED_REFINER_BLOCKS = 2
EXPECTED_ADAPTERS = EXPECTED_TRANSFORMER_BLOCKS * 7 + EXPECTED_REFINER_BLOCKS * 6
EXPECTED_TENSORS = EXPECTED_ADAPTERS * 2 + 4


def shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def pdd_time_grid(shift: float, num_steps: int = PDD_NUM_STEPS) -> torch.Tensor:
    """Official ascending PDD grid: ``0 = t_0 < ... < t_N = 1``."""
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shifted_sigma(float(shift), sigma)


def pdd_plans(
    shift: float,
    num_steps: int = PDD_NUM_STEPS,
    block_size: int = PDD_BLOCK_SIZE,
) -> torch.Tensor:
    """Return one normalized direction-mixing plan per transformer call."""
    if block_size < 1 or num_steps % block_size:
        raise ValueError("pdd_num_steps must be divisible by pdd_block_size")
    step_sizes = pdd_time_grid(shift, num_steps).diff()
    plans = torch.zeros(num_steps // block_size, num_steps, dtype=torch.float32)
    for index, start in enumerate(range(0, num_steps, block_size)):
        block = step_sizes[start : start + block_size]
        plans[index, start : start + block_size] = (block / block.sum()).float()
    return plans


def pdd_sigma_starts(
    shift: float = PDD_VIDEO_SHIFT,
    num_steps: int = PDD_NUM_STEPS,
    block_size: int = PDD_BLOCK_SIZE,
) -> torch.Tensor:
    """Descending video sigmas at which the eight PDD blocks start."""
    base = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return shifted_sigma(float(shift), base)[0:num_steps:block_size].float()


def pdd_sigma_boundaries(
    shift: float = PDD_VIDEO_SHIFT,
    num_steps: int = PDD_NUM_STEPS,
    block_size: int = PDD_BLOCK_SIZE,
) -> torch.Tensor:
    """Exact descending PDD block boundaries, including the terminal zero."""
    base = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return shifted_sigma(float(shift), base)[::block_size].float()


def swap_swiglu_up_halves(up: torch.Tensor) -> torch.Tensor:
    """Translate Diffusers [value; gate] rows to ComfyUI H3 [gate; value]."""
    if up.ndim != 2 or up.shape[0] % 2:
        raise ValueError(f"invalid SwiGLU LoRA up shape: {_shape_tuple(up)}")
    half = up.shape[0] // 2
    return torch.cat((up[half:], up[:half]), dim=0).contiguous()


class PDDParallelHead(nn.Module):
    """The 32 per-interval output heads stored in the original checkpoint."""

    comfy_cast_weights = True

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, plans: torch.Tensor):
        super().__init__()
        if weight.ndim != 3 or bias.ndim != 2:
            raise ValueError(
                f"PDD head expects weight [N,O,I] and bias [N,O], got {tuple(weight.shape)} / {tuple(bias.shape)}"
            )
        if weight.shape[:2] != bias.shape or weight.shape[0] != plans.shape[1]:
            raise ValueError("PDD head tensors and sampling plans do not agree")
        self.in_features = int(weight.shape[2])
        self.out_features = int(weight.shape[1])
        self.num_steps = int(weight.shape[0])
        self.num_directions = int(plans.shape[0])
        self.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
        self.bias = nn.Parameter(bias.contiguous(), requires_grad=False)
        self.register_buffer("plans", plans.contiguous(), persistent=False)
        self.active_index = 0
        # Attributes expected by ComfyUI's dynamic/low-VRAM weight caster.
        self.weight_function: list[Any] = []
        self.bias_function: list[Any] = []

    def set_index(self, index: int) -> None:
        self.active_index = max(0, min(int(index), self.num_directions - 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight, bias, offload = comfy.ops.cast_bias_weight(
            self,
            hidden_states,
            dtype=hidden_states.dtype,
            bias_dtype=hidden_states.dtype,
            offloadable=True,
        )
        try:
            plan = self.plans[self.active_index].to(device=weight.device, dtype=weight.dtype)
            nonzero = torch.nonzero(plan, as_tuple=False).flatten()
            selected_plan = plan.index_select(0, nonzero)
            selected_weight = weight.index_select(0, nonzero)
            selected_bias = bias.index_select(0, nonzero)
            fused_weight = torch.einsum("n,noi->oi", selected_plan, selected_weight)
            fused_bias = torch.einsum("n,no->o", selected_plan, selected_bias)
            return F.linear(hidden_states, fused_weight, fused_bias)
        finally:
            comfy.ops.uncast_bias_weight(self, weight, bias, offload)


@dataclass(frozen=True)
class AdapterTarget:
    model_key: str
    offset: tuple[int, int, int] | None = None


_BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)\.(.+)$")
_REFINER_RE = re.compile(r"^token_refiner\.refiner_blocks\.(\d+)\.(.+)$")


def adapter_target(diffusers_name: str, qkv_width: int) -> AdapterTarget:
    """Map one Diffusers PDD adapter base name to a ComfyUI model key."""
    match = _BLOCK_RE.match(diffusers_name)
    if match:
        prefix = f"diffusion_model.blocks.{int(match.group(1))}"
        suffix = match.group(2)
    else:
        match = _REFINER_RE.match(diffusers_name)
        if not match:
            raise KeyError(f"Unsupported PDD adapter target: {diffusers_name}")
        prefix = f"diffusion_model.token_refiner.blocks.{int(match.group(1))}"
        suffix = match.group(2)

    qkv = {
        "attn.to_q": 0,
        "attn.to_k": 1,
        "attn.to_v": 2,
    }
    if suffix in qkv:
        return AdapterTarget(
            f"{prefix}.attn.qkv_proj.weight",
            (0, qkv[suffix] * qkv_width, qkv_width),
        )
    direct = {
        "attn.to_out.0": "attn.out_proj.weight",
        "ff.net.0.proj": "mlp.fc1.weight",
        "ff.net.2": "mlp.fc2.weight",
        "adaln_proj.linear": "adaln_proj.linear.weight",
    }
    if suffix not in direct:
        raise KeyError(f"Unsupported PDD adapter target: {diffusers_name}")
    return AdapterTarget(f"{prefix}.{direct[suffix]}")


def _shape_tuple(value: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in value.shape)


def build_lora_patches(
    state_dict: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
) -> tuple[dict[Any, LoRAAdapter], set[str]]:
    """Build native ComfyUI LoRA adapters, including fused Q/K/V slices."""
    patches: dict[Any, LoRAAdapter] = {}
    consumed: set[str] = set()
    errors: list[str] = []

    down_names = sorted(name for name in state_dict if name.endswith(".lora_down"))
    for down_name in down_names:
        base_name = down_name[: -len(".lora_down")]
        up_name = f"{base_name}.lora_up"
        if up_name not in state_dict:
            errors.append(f"missing {up_name}")
            continue
        down = state_dict[down_name]
        up = state_dict[up_name]
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
            errors.append(f"invalid LoRA pair {base_name}: {_shape_tuple(down)} / {_shape_tuple(up)}")
            continue
        target = adapter_target(base_name, int(up.shape[0]))
        if base_name.endswith(".ff.net.0.proj"):
            up = swap_swiglu_up_halves(up)
        target_weight = model_state.get(target.model_key)
        if target_weight is None:
            errors.append(f"model key missing: {target.model_key}")
            continue
        expected = (int(up.shape[0]), int(down.shape[1]))
        actual = _shape_tuple(target_weight)
        if target.offset is not None:
            dim, start, length = target.offset
            if dim != 0 or start + length > actual[0]:
                errors.append(f"invalid fused slice for {target.model_key}: {target.offset} vs {actual}")
                continue
            actual = (length, *actual[1:])
        if actual != expected:
            errors.append(f"shape mismatch {base_name} -> {target.model_key}: LoRA {expected}, model {actual}")
            continue

        loaded_keys = {down_name, up_name}
        adapter = LoRAAdapter(
            loaded_keys,
            (up, down, float(down.shape[0]), None, None, None),
        )
        patch_key: Any = target.model_key if target.offset is None else (target.model_key, target.offset)
        patches[patch_key] = adapter
        consumed.update(loaded_keys)

    if errors:
        preview = "\n - ".join(errors[:8])
        curve_hint = ""
        if "diffusion_model.adaln_t_table" in model_state and any("adaln_proj" in item for item in errors):
            curve_hint = (
                "\nEste checkpoint usa AdaLN curve/pruned. O PDD oficial foi treinado para a arquitetura "
                "AdaLN completa e não pode ser aplicado fielmente a esse checkpoint."
            )
        raise RuntimeError(f"PDD LoRA incompatível com o modelo carregado:\n - {preview}{curve_hint}")
    return patches, consumed


class PDDModelWrapper:
    """Select the correct direction block from the current ComfyUI sigma."""

    def __init__(
        self,
        video_head: PDDParallelHead,
        audio_head: PDDParallelHead,
        sigma_starts: torch.Tensor,
        old_wrapper=None,
    ):
        self.video_head = video_head
        self.audio_head = audio_head
        self.sigma_starts = sigma_starts.float().cpu()
        self.old_wrapper = old_wrapper

    def to(self, *args, **kwargs):
        if self.old_wrapper is not None and hasattr(self.old_wrapper, "to"):
            updated = self.old_wrapper.to(*args, **kwargs)
            if updated is not None:
                self.old_wrapper = updated
        return self

    def models(self):
        if self.old_wrapper is not None and hasattr(self.old_wrapper, "models"):
            return self.old_wrapper.models()
        return []

    def cleanup(self, **kwargs):
        if self.old_wrapper is not None and hasattr(self.old_wrapper, "cleanup"):
            return self.old_wrapper.cleanup(**kwargs)
        return None

    def __call__(self, apply_model, args: dict):
        sigma = float(args["timestep"].detach().float().max().cpu())
        index = int(torch.argmin(torch.abs(self.sigma_starts - sigma)).item())
        self.video_head.set_index(index)
        self.audio_head.set_index(index)

        c = args["c"].copy()
        transformer_options = c.get("transformer_options", {}).copy()
        transformer_options["minimax_h3_sigma_shift_video"] = PDD_VIDEO_SHIFT
        transformer_options["minimax_h3_sigma_shift_audio"] = PDD_AUDIO_SHIFT
        transformer_options["bruxos_h3_pdd_index"] = index
        c["transformer_options"] = transformer_options
        wrapped_args = args | {"c": c}
        if self.old_wrapper is not None:
            return self.old_wrapper(apply_model, wrapped_args)
        return apply_model(wrapped_args["input"], wrapped_args["timestep"], **c)


def apply_pdd_to_model(model, state_dict: dict[str, torch.Tensor], source_name: str):
    """Clone and patch a ComfyUI MiniMax-H3 MODEL with the original PDD file."""
    if model.model_options.get("bruxos_h3_pdd"):
        raise RuntimeError("Este MODEL já possui um PDD LoRA aplicado.")

    diffusion_model = model.get_model_object("diffusion_model")
    if diffusion_model.__class__.__name__ != "MiniMaxH3Model":
        raise TypeError(
            "Bruxos H3 PDD Loader requer um Load Diffusion Model do MiniMax-H3 "
            f"(recebido: {diffusion_model.__class__.__name__})."
        )
    if len(state_dict) != EXPECTED_TENSORS:
        raise RuntimeError(
            f"Checkpoint PDD inesperado: {len(state_dict)} tensores; eram esperados {EXPECTED_TENSORS}."
        )

    head_names = {
        "proj_out.weight",
        "proj_out.bias",
        "audio_proj_out.weight",
        "audio_proj_out.bias",
    }
    missing_heads = sorted(head_names.difference(state_dict))
    if missing_heads:
        raise RuntimeError(f"Checkpoint PDD sem as cabeças obrigatórias: {', '.join(missing_heads)}")

    video_plans = pdd_plans(PDD_VIDEO_SHIFT)
    audio_plans = pdd_plans(PDD_AUDIO_SHIFT)
    video_head = PDDParallelHead(state_dict["proj_out.weight"], state_dict["proj_out.bias"], video_plans)
    audio_head = PDDParallelHead(
        state_dict["audio_proj_out.weight"],
        state_dict["audio_proj_out.bias"],
        audio_plans,
    )

    patched = model.clone()
    model_state = patched.model.state_dict()
    patches, consumed = build_lora_patches(state_dict, model_state)
    if len(patches) != EXPECTED_ADAPTERS:
        raise RuntimeError(
            f"Mapeamento PDD incompleto: {len(patches)} adaptadores; eram esperados {EXPECTED_ADAPTERS}."
        )
    unexpected = sorted(set(state_dict).difference(consumed).difference(head_names))
    if unexpected:
        raise RuntimeError(f"Tensores PDD não reconhecidos: {', '.join(unexpected[:8])}")

    applied = patched.add_patches(patches, strength_patch=1.0, strength_model=1.0)
    if len(applied) != len(patches):
        raise RuntimeError(
            f"O ComfyUI aceitou apenas {len(applied)} de {len(patches)} patches PDD. Atualize o ComfyUI."
        )

    patched.add_object_patch("diffusion_model.final_layer.video_out", video_head)
    patched.add_object_patch("diffusion_model.final_layer.audio_out", audio_head)

    class ModelSamplingPDD(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    original_sampling = patched.get_model_object("model_sampling")
    model_sampling = ModelSamplingPDD(patched.model.model_config)
    model_sampling.set_parameters(shift=PDD_VIDEO_SHIFT, audio_shift=PDD_AUDIO_SHIFT)
    if hasattr(original_sampling, "noise_scale"):
        model_sampling.set_noise_scale(original_sampling.noise_scale)
    patched.add_object_patch("model_sampling", model_sampling)

    transformer_options = patched.model_options.get("transformer_options", {}).copy()
    transformer_options["minimax_h3_sigma_shift_video"] = PDD_VIDEO_SHIFT
    transformer_options["minimax_h3_sigma_shift_audio"] = PDD_AUDIO_SHIFT
    patched.model_options["transformer_options"] = transformer_options

    old_wrapper = patched.model_options.get("model_function_wrapper")
    patched.set_model_unet_function_wrapper(
        PDDModelWrapper(video_head, audio_head, pdd_sigma_starts(), old_wrapper=old_wrapper)
    )
    patched.model_options["bruxos_h3_pdd"] = {
        "source": source_name,
        "grid": PDD_NUM_STEPS,
        "block_size": PDD_BLOCK_SIZE,
        "nfe": PDD_NFE,
        "video_shift": PDD_VIDEO_SHIFT,
        "audio_shift": PDD_AUDIO_SHIFT,
    }
    LOGGER.info(
        "Bruxos H3 PDD loaded %s: %d adapters, grid=%d, block=%d, NFE=%d",
        source_name,
        len(patches),
        PDD_NUM_STEPS,
        PDD_BLOCK_SIZE,
        PDD_NFE,
    )
    return patched
