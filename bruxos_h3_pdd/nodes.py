"""ComfyUI node registration for the MiniMax-H3 PDD runtime loader."""

from __future__ import annotations

import os

import folder_paths
import comfy.utils

from .pdd import PDD_NFE, apply_pdd_to_model, pdd_sigma_boundaries


class BruxosMiniMaxH3PDDLoader:
    """Read an original Alibaba PAI PDD LoRA without an offline conversion."""

    def __init__(self):
        self._cache_path = None
        self._cache_state = None

    @classmethod
    def INPUT_TYPES(cls):
        names = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "pdd_lora": (names,),
            }
        }

    RETURN_TYPES = ("MODEL", "SIGMAS", "INT", "STRING")
    RETURN_NAMES = ("model", "sigmas", "steps", "status")
    FUNCTION = "load_pdd"
    CATEGORY = "Bruxos do VFX/MiniMax H3/PDD"
    DESCRIPTION = (
        "Lê diretamente o PDD LoRA oficial da Alibaba PAI e o aplica ao MiniMax-H3 em memória. "
        "Use Euler com a saída SIGMAS no SamplerCustomAdvanced, 8 avaliações e CFG 1.0."
    )

    def load_pdd(self, model, pdd_lora):
        path = folder_paths.get_full_path_or_raise("loras", pdd_lora)
        if not path.lower().endswith(".safetensors"):
            raise ValueError("O PDD Loader aceita o checkpoint oficial .safetensors.")
        if self._cache_path != path or self._cache_state is None:
            self._cache_state = comfy.utils.load_torch_file(path, safe_load=True)
            self._cache_path = path
        patched = apply_pdd_to_model(model, self._cache_state, os.path.basename(path))
        status = (
            f"PDD ativo: {os.path.basename(path)} | Euler + sigmas PDD exatos | "
            f"NFE={PDD_NFE} | CFG=1.0 | shifts vídeo/áudio=12/3"
        )
        return (patched, pdd_sigma_boundaries(), PDD_NFE, status)


NODE_CLASS_MAPPINGS = {
    "BruxosMiniMaxH3PDDLoader": BruxosMiniMaxH3PDDLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosMiniMaxH3PDDLoader": "MiniMax H3 PDD Loader · Original LoRA (Bruxos)",
}
