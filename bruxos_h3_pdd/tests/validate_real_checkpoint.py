"""Structural validation for a downloaded Alibaba PAI PDD checkpoint."""

from __future__ import annotations

import sys

import torch
from safetensors import safe_open

from bruxos_h3_pdd.pdd import (
    EXPECTED_ADAPTERS,
    EXPECTED_TENSORS,
    PDD_AUDIO_SHIFT,
    PDD_VIDEO_SHIFT,
    PDDParallelHead,
    adapter_target,
    build_lora_patches,
    pdd_plans,
)


def validate(path: str) -> None:
    checkpoint = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if len(keys) != EXPECTED_TENSORS:
            raise AssertionError(f"expected {EXPECTED_TENSORS} tensors, found {len(keys)}")
        for name in keys:
            if name in {
                "proj_out.weight",
                "proj_out.bias",
                "audio_proj_out.weight",
                "audio_proj_out.bias",
            }:
                checkpoint[name] = handle.get_tensor(name)
            else:
                checkpoint[name] = torch.empty(handle.get_slice(name).get_shape(), device="meta")

    model_state = {}
    for down_name in (name for name in checkpoint if name.endswith(".lora_down")):
        base = down_name[: -len(".lora_down")]
        down = checkpoint[down_name]
        up = checkpoint[f"{base}.lora_up"]
        target = adapter_target(base, int(up.shape[0]))
        if target.offset is None:
            shape = (int(up.shape[0]), int(down.shape[1]))
        else:
            shape = (int(up.shape[0]) * 3, int(down.shape[1]))
        model_state[target.model_key] = torch.empty(shape, device="meta")

    patches, consumed = build_lora_patches(checkpoint, model_state)
    assert len(patches) == EXPECTED_ADAPTERS
    assert len(consumed) == EXPECTED_ADAPTERS * 2
    video = PDDParallelHead(checkpoint["proj_out.weight"], checkpoint["proj_out.bias"], pdd_plans(PDD_VIDEO_SHIFT))
    audio = PDDParallelHead(
        checkpoint["audio_proj_out.weight"],
        checkpoint["audio_proj_out.bias"],
        pdd_plans(PDD_AUDIO_SHIFT),
    )
    assert (video.num_steps, video.out_features, video.in_features) == (32, 96, 5376)
    assert (audio.num_steps, audio.out_features, audio.in_features) == (32, 32, 5376)
    print(f"checkpoint valid: {len(keys)} tensors, {len(patches)} adapters, 2 parallel heads")


if __name__ == "__main__":
    validate(sys.argv[1])
