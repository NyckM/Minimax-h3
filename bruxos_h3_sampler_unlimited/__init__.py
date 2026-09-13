from .nodes import MiniMaxH3SamplerCustomAdvancedUnlimited
from .preview import MiniMaxH3UnlimitedPreview


NODE_CLASS_MAPPINGS = {
    "BruxosH3SamplerCustomAdvancedUnlimited": MiniMaxH3SamplerCustomAdvancedUnlimited,
    "BruxosH3UnlimitedPreview": MiniMaxH3UnlimitedPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosH3SamplerCustomAdvancedUnlimited": "H3 Sampler Unlimited (Bruxos TESTE)",
    "BruxosH3UnlimitedPreview": "H3 Preview Acumulado Unlimited (Bruxos TESTE)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
