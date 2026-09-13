# -*- coding: utf-8 -*-
"""ComfyUI-Bruxos-MiniMaxH3

Tudo de MiniMax-H3: loader, frames 17n+5, Context-IR, prompt rapido, outpaint/mascara, upscale 2-pass e neural, rolling memory, latente no SSD, autocontext, sampler unlimited, PDD e latent I/O.

Separado do pacote unico ComfyUI-Bruxos-do-VFX.
"""

TAG = "Bruxos — MiniMax-H3"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _merge(modname):
    try:
        mod = __import__(f"{__name__}.{modname}", fromlist=["*"])
        NODE_CLASS_MAPPINGS.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except Exception as e:  # pragma: no cover
        import logging
        logging.warning(f"[{TAG}] modulo '{modname}' nao carregou: {e}")


_merge("minimax_h3_bruxos")
_merge("minimax_h3_mask_bruxos")
_merge("minimax_encode_bruxos")
_merge("bruxos_h3_loader")
_merge("bruxos_h3_frames")
_merge("bruxos_h3_cond_forca")
_merge("bruxos_qwen3_enhancer")
_merge("bruxos_h3_context_ir")
_merge("bruxos_h3_prompt_rapido")
_merge("bruxos_h3_latent_upscale")
_merge("bruxos_h3_neural_upscale")
_merge("bruxos_h3_ultimate_upscale")
_merge("bruxos_h3_rolling_memory")
_merge("bruxos_h3_contex_loop")
_merge("bruxos_h3_latent_ssd")
_merge("bruxos_h3_clay_reference")
_merge("bruxos_h3_row_chunk")
_merge("bruxos_h3_reference_isolate")
_merge("bruxos_h3_face_refine")
_merge("bruxos_autocontext")
_merge("bruxos_h3_sampler_unlimited")
_merge("bruxos_h3_latent_io")
_merge("bruxos_h3_pdd.nodes")


# ---- catalogo das macros # e @ do Prompt Rapido do H3 ----
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/bruxos/h3_macros")
    async def _bruxos_h3_macros(request):  # pragma: no cover
        from .bruxos_h3_prompt_rapido import macros_payload
        return web.json_response(macros_payload())
except Exception as e:  # pragma: no cover
    import logging
    logging.info(f"[{TAG}] rota de macros nao registrada (ok fora do server): {e}")


# Banner de inicializacao
try:
    from .banner import print_banner
    print_banner(area="MiniMax-H3", node_count=len(NODE_CLASS_MAPPINGS), version="v2")
except Exception:
    pass

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
