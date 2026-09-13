import torch

from bruxos_h3_pdd.pdd import (
    EXPECTED_ADAPTERS,
    PDD_NFE,
    PDDParallelHead,
    adapter_target,
    build_lora_patches,
    pdd_plans,
    pdd_sigma_boundaries,
    pdd_sigma_starts,
    pdd_time_grid,
    swap_swiglu_up_halves,
)


def test_official_grid_and_plans():
    grid = pdd_time_grid(12.0)
    plans = pdd_plans(12.0)
    assert grid.shape == (33,)
    assert plans.shape == (PDD_NFE, 32)
    assert torch.all(grid[1:] >= grid[:-1])
    assert torch.allclose(plans.sum(dim=1), torch.ones(PDD_NFE))
    assert torch.equal((plans != 0).sum(dim=1), torch.full((PDD_NFE,), 4))
    assert torch.all(pdd_sigma_starts()[1:] < pdd_sigma_starts()[:-1])
    bounds = pdd_sigma_boundaries()
    assert bounds.shape == (PDD_NFE + 1,)
    assert bounds[0] == 1.0 and bounds[-1] == 0.0


def test_swiglu_lora_up_half_swap():
    up = torch.arange(24).reshape(8, 3)
    converted = swap_swiglu_up_halves(up)
    assert torch.equal(converted[:4], up[4:])
    assert torch.equal(converted[4:], up[:4])


def test_diffusers_to_comfy_mapping():
    q = adapter_target("transformer_blocks.9.attn.to_q", 7168)
    k = adapter_target("transformer_blocks.9.attn.to_k", 7168)
    v = adapter_target("transformer_blocks.9.attn.to_v", 7168)
    assert q.model_key == "diffusion_model.blocks.9.attn.qkv_proj.weight"
    assert q.offset == (0, 0, 7168)
    assert k.offset == (0, 7168, 7168)
    assert v.offset == (0, 14336, 7168)
    assert adapter_target("transformer_blocks.3.ff.net.2", 5376).model_key.endswith("mlp.fc2.weight")
    assert adapter_target("token_refiner.refiner_blocks.1.attn.to_out.0", 5376).model_key == (
        "diffusion_model.token_refiner.blocks.1.attn.out_proj.weight"
    )


def test_parallel_head_matches_weighted_heads():
    weights = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    biases = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
    plans = torch.tensor([[0.25, 0.75, 0.0, 0.0], [0.0, 0.0, 0.4, 0.6]])
    head = PDDParallelHead(weights, biases, plans)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    expected_weight = 0.25 * weights[0] + 0.75 * weights[1]
    expected_bias = 0.25 * biases[0] + 0.75 * biases[1]
    assert torch.allclose(head(x), torch.nn.functional.linear(x, expected_weight, expected_bias))
    head.set_index(1)
    expected_weight = 0.4 * weights[2] + 0.6 * weights[3]
    expected_bias = 0.4 * biases[2] + 0.6 * biases[3]
    assert torch.allclose(head(x), torch.nn.functional.linear(x, expected_weight, expected_bias))


def test_all_official_adapter_keys_map_without_allocating_weights():
    checkpoint = {}
    model_state = {}

    def add_pair(base, out_features, in_features):
        checkpoint[f"{base}.lora_down"] = torch.empty(64, in_features, device="meta")
        checkpoint[f"{base}.lora_up"] = torch.empty(out_features, 64, device="meta")
        target = adapter_target(base, out_features)
        if target.offset is None:
            model_state[target.model_key] = torch.empty(out_features, in_features, device="meta")
        else:
            model_state[target.model_key] = torch.empty(out_features * 3, in_features, device="meta")

    for block in range(50):
        prefix = f"transformer_blocks.{block}"
        for name in ("to_q", "to_k", "to_v"):
            add_pair(f"{prefix}.attn.{name}", 7168, 5376)
        add_pair(f"{prefix}.attn.to_out.0", 5376, 7168)
        add_pair(f"{prefix}.ff.net.0.proj", 28672, 5376)
        add_pair(f"{prefix}.ff.net.2", 5376, 14336)
        add_pair(f"{prefix}.adaln_proj.linear", 96768, 2688)

    for block in range(2):
        prefix = f"token_refiner.refiner_blocks.{block}"
        for name in ("to_q", "to_k", "to_v"):
            add_pair(f"{prefix}.attn.{name}", 7168, 5376)
        add_pair(f"{prefix}.attn.to_out.0", 5376, 7168)
        add_pair(f"{prefix}.ff.net.0.proj", 28672, 5376)
        add_pair(f"{prefix}.ff.net.2", 5376, 14336)

    patches, consumed = build_lora_patches(checkpoint, model_state)
    assert len(patches) == EXPECTED_ADAPTERS
    assert len(consumed) == EXPECTED_ADAPTERS * 2
