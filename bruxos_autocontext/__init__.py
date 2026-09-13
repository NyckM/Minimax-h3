"""MiniMax H3 Auto Chunks integrado à Bruxos do VFX.

Baseado em supElement/ComfyUI_MinimaxH3_AutoContext, Apache-2.0.
A licença upstream completa está em LICENSE.upstream.
Native Masked adaptado de ComfyUI-H3-Continuum, MIT; veja
LICENSE.h3_continuum.
"""

from .nodes import H3AutoContextSampler, H3ParameterNode
from .seam_correction import H3SeamCorrection
from .seam_correction_v2 import H3SeamCorrectionV2

NODE_CLASS_MAPPINGS = {
    "BruxosH3AutoContextParameter": H3ParameterNode,
    "BruxosH3AutoContextSampler": H3AutoContextSampler,
    "BruxosH3AutoContextSeamCorrection": H3SeamCorrection,
    "BruxosH3AutoContextSeamCorrectionV2": H3SeamCorrectionV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosH3AutoContextParameter": "H3 Auto Chunks · Parâmetros (Bruxos)",
    "BruxosH3AutoContextSampler": "H3 Auto Chunks · I2V / Ref2V (Bruxos)",
    "BruxosH3AutoContextSeamCorrection": "H3 Auto Chunks · Corrigir Emenda (Bruxos)",
    "BruxosH3AutoContextSeamCorrectionV2": "H3 Auto Chunks · Corrigir Emenda 2.0 (Bruxos)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
