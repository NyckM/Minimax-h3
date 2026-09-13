"""bruxos_core — biblioteca comum dos pacotes "Bruxos do VFX".

NAO registra nenhum node. So expoe as funcoes de apoio que antes moravam no
nodes.py do pacote unico, pra que cada repositorio separado continue rodando
sem depender dos outros.

Uso dentro de um pacote:

    from .bruxos_core import _normalize_mask, _align_up_4n1, _mem_cleanup
"""

from .helpers import *          # noqa: F401,F403
from .helpers import (          # noqa: F401  (nomes com "_" nao vem no import *)
    _mem_gb,
    _mem_snapshot,
    _mem_report,
    _bx_patch_count,
    _mem_cleanup,
    _clone_conditioning_set_values,
    _resize_long_edge,
    _FIT_CROP,
    _resize_source_video,
    _video_frame_count,
    _split_video,
    _latent_shape,
    _make_empty_latent,
    _encode_video,
    _resize_latent_spatial,
    _prepare_init_latent,
    _lat_len,
    _align_up_4n1,
    _mirror_pad_frames,
    _normalize_mask,
    _grow_blur_mask,
    _resize_mask_spatial_temporal,
    _rect_feather_mask,
    _mask_to_latent,
    _mask_bbox,
    _collect_reference_latents,
    _merge_linear_overlap,
    _merge_latent_overlap,
    _decode_video,
    _decode_video_chunked,
    _ordered_offset,
    _context_windows,
    _window_blend_weights,
    _slice_temporal,
    _debug_dump_shapes,
    _make_context_wrapper,
)

# re-export dos samplers do ComfyUI que varios modulos puxavam via "from .nodes"
try:  # pragma: no cover
    from comfy_extras.nodes_custom_sampler import (
        BasicScheduler,
        KSamplerSelect,
        SamplerCustom,
        SplitSigmas,
    )
except Exception:  # pragma: no cover
    BasicScheduler = KSamplerSelect = SamplerCustom = SplitSigmas = None

# loader/cache do Qwen-VL (caption e prompt enhancer)
try:  # pragma: no cover
    from .qwen import (  # noqa: F401
        _BX_QWEN_DEFAULT_INSTRUCTION,
        _BX_QWEN_MODELS,
        _BX_QWEN_CACHE,
        _bx_qwen_load,
        _bx_tensor_to_pil,
        BruxosQwenVLCaption,
    )
except Exception:  # pragma: no cover
    _BX_QWEN_MODELS = ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]
    _BX_QWEN_CACHE = {"name": None, "model": None, "processor": None}
    _bx_qwen_load = _bx_tensor_to_pil = None
    BruxosQwenVLCaption = None

from .optional import get_node_class, import_from_pack  # noqa: F401

__version__ = "1.0.0"
