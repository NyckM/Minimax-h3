# -*- coding: utf-8 -*-
"""Integração Bruxos para o neural latent upscaler do MiniMax H3.

O primeiro node mantém o modelo neural treinado do pacote original. O segundo
sincroniza os blocos reais ``minimax_refs`` (latentes 4D/5D + latent_h/w) com
a escala aplicada, sem tocar em embeddings ou no áudio.
"""

from __future__ import annotations

import importlib
import inspect
import math
import os
import sys

import torch
import torch.nn.functional as F

try:
    from .minimax_h3_bruxos import _componentes
except Exception:  # pragma: no cover
    from minimax_h3_bruxos import _componentes


CAT = "Bruxos do VFX/MiniMax H3"
META_KEY = "bruxos_h3_neural_upscale"
MODEL_PACKAGE_CURRENT = "Comfyui_Minimax_h3_latent_Upscaler.nodes.minimax_h3_latent_upscaler_3d"
MODEL_PACKAGE_LEGACY = "Comfyui_Minimax_h3_latent_Upscaler.nodes.h3_upscaler_common"


def _backend_supported(module):
    return (
        hasattr(module, "run_upscale")
        or (
            hasattr(module, "MinimaxH3LatentUpscaler3D")
            and hasattr(module, "UpscaleMode")
        )
    )


def _backend():
    first_error = None
    for package in (MODEL_PACKAGE_CURRENT, MODEL_PACKAGE_LEGACY):
        try:
            module = importlib.import_module(package)
            if _backend_supported(module):
                return module
            raise RuntimeError(f"módulo sem API compatível: {package}")
        except Exception as error:
            first_error = first_error or error
    if first_error is not None:
        # O loader do ComfyUI importa cada custom node por caminho e nem sempre
        # deixa custom_nodes no sys.path. Descobre a raiz oficial sem fixar uma
        # instalação específica e tenta novamente.
        candidates = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
        try:
            import folder_paths
            candidates.insert(0, os.path.join(folder_paths.base_path, "custom_nodes"))
        except Exception:
            pass
        for candidate in candidates:
            if os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)
        last_error = first_error
        for package in (MODEL_PACKAGE_CURRENT, MODEL_PACKAGE_LEGACY):
            try:
                module = importlib.import_module(package)
                if _backend_supported(module):
                    return module
                raise RuntimeError(f"módulo sem API compatível: {package}")
            except Exception as error:
                last_error = error
        raise RuntimeError(
            "[Bruxos H3 Neural Upscale] O backend neural não está disponível. "
            "Mantenha instalado o pacote 'Comfyui_Minimax_h3_latent_Upscaler'; "
            "a versão Bruxos reutiliza a arquitetura e os pesos treinados dele. "
            f"Erro original: {last_error}"
        ) from last_error


def _models():
    try:
        names = _backend().scan_models()
        h3 = [name for name in names if "minimax" in name.lower() and "h3" in name.lower()]
        return h3 or names
    except Exception:
        return ["(instale o backend/modelo H3 neural)"]


def _video_component(latent):
    samples = latent.get("samples", latent) if isinstance(latent, dict) else latent
    components, _kind = _componentes(samples)
    if not components or not torch.is_tensor(components[0]):
        raise ValueError(
            "[Bruxos H3 Neural Upscale] LATENT não reconhecido. Ligue a saída LATENT "
            "do MiniMax H3; o primeiro componente precisa ser o vídeo."
        )
    video = components[0]
    if video.ndim not in (4, 5):
        raise ValueError(
            f"[Bruxos H3 Neural Upscale] vídeo precisa ser [B,C,H,W] ou "
            f"[B,C,T,H,W], mas chegou {tuple(video.shape)}."
        )
    if int(video.shape[1]) != 24:
        raise ValueError(
            f"[Bruxos H3 Neural Upscale] esperado latente H3 de 24 canais; "
            f"chegaram {int(video.shape[1])} canais."
        )
    return video


def _aligned_target(source_w, source_h, megapixels, align):
    target_pixels = float(megapixels) * 1_000_000.0
    aspect = source_w / source_h
    pixel_h = math.sqrt(target_pixels / aspect)
    pixel_w = pixel_h * aspect
    # ``align`` é alinhamento em PIXELS. A versão anterior o aplicava como
    # células latentes, então align=32 virava blocos de 512 px e podia mudar
    # 16:9 para 3:2 (por exemplo 84x48 -> 96x64), esticando pessoas.
    pixel_grid = max(32, int(math.ceil(max(1, int(align)) / 32.0)) * 32)
    latent_grid = max(2, pixel_grid // 16)
    ideal_w = pixel_w / 16.0
    ideal_h = pixel_h / 16.0
    target_area = max(1.0, ideal_w * ideal_h)

    def nearby(value):
        center = int(round(value / latent_grid))
        return {
            max(latent_grid, (center + delta) * latent_grid)
            for delta in range(-2, 3)
            if center + delta > 0
        }

    candidates = []
    for latent_w in nearby(ideal_w):
        for latent_h in nearby(ideal_h):
            if latent_w < source_w or latent_h < source_h:
                continue
            ratio_error = abs(math.log((latent_w / latent_h) / aspect))
            area_error = abs(math.log((latent_w * latent_h) / target_area))
            # Anatomia/proporção vale mais que acertar o MP na casa decimal.
            candidates.append((ratio_error * 8.0 + area_error, area_error, latent_w, latent_h))
    if not candidates:
        return int(source_w), int(source_h)
    _, _, latent_w, latent_h = min(candidates)
    return int(latent_w), int(latent_h)


def _resolve_compute(device, precision):
    requested_device = str(device)
    if requested_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "rocm":
        if getattr(torch.version, "hip", None) is None or not torch.cuda.is_available():
            raise RuntimeError(
                "[Bruxos H3 Neural Upscale] ROCm foi selecionado, mas este PyTorch não possui suporte HIP/AMD."
            )
        # No PyTorch ROCm, o dispositivo continua sendo exposto como "cuda".
        resolved_device = "cuda"
    elif requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("[Bruxos H3 Neural Upscale] CUDA foi selecionado, mas não está disponível.")
    else:
        resolved_device = requested_device

    requested_precision = str(precision)
    if requested_precision == "auto":
        if resolved_device == "cpu":
            resolved_precision = "fp32"
        elif torch.cuda.is_bf16_supported():
            resolved_precision = "bf16"
        else:
            resolved_precision = "fp16"
    else:
        resolved_precision = requested_precision
    if resolved_device == "cpu" and resolved_precision != "fp32":
        raise ValueError(
            "[Bruxos H3 Neural Upscale] use precision=fp32 no CPU; Conv3D em fp16/bf16 "
            "não é suportado de forma confiável nesse caminho."
        )
    return resolved_device, resolved_precision


def _release_cached_model(common, model_name, device, precision):
    cache = getattr(common, "MODEL_CACHE", {})
    key = f"{model_name}::{device}::{precision}"
    model = cache.pop(key, None)
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
        del model
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _node_output_values(value):
    """Normaliza retornos legacy tuple e o NodeOutput do ComfyUI 0.33+."""
    args = getattr(value, "args", None)
    if args is not None:
        return tuple(args)
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _first_node_output(value):
    values = _node_output_values(value)
    if not values:
        raise RuntimeError("[Bruxos H3 Neural Upscale] backend retornou NodeOutput vazio.")
    return values[0]


def _run_backend_upscale(common, video_latent, audio_latent, model_name,
                         device, precision, target_w, target_h, scale,
                         enable_native_chunking=True, force_unload=True):
    """Executa uma vez e deixa o próprio modelo 3D controlar os chunks."""
    if hasattr(common, "run_upscale"):
        return common.run_upscale(
            video_latent, audio_latent, model_name, device, precision,
            target_w, target_h, scale,
        )

    node = getattr(common, "MinimaxH3LatentUpscaler3D", None)
    mode_enum = getattr(common, "UpscaleMode", None)
    if node is None or mode_enum is None:
        raise RuntimeError(
            "[Bruxos H3 Neural Upscale] O backend instalado não oferece a API 3D esperada."
        )
    mode = {
        "mode": mode_enum.TARGET_DIMENSIONS,
        "width": int(target_w) * 16,
        "height": int(target_h) * 16,
    }
    parameters = inspect.signature(node.execute).parameters
    kwargs = {
        "latent": video_latent,
        "model_name": model_name,
        "mode": mode,
        "align": 32,
        "device": device,
        "precision": precision,
    }
    if "enable_temporal_chunking" in parameters:
        kwargs["enable_temporal_chunking"] = bool(enable_native_chunking)
    elif "enable_chunking" in parameters:
        kwargs["enable_chunking"] = bool(enable_native_chunking)
    if "force_unload" in parameters:
        kwargs["force_unload"] = bool(force_unload)
    result = node.execute(**kwargs)
    video_upscaled = _first_node_output(result)
    from comfy_extras.nodes_lt import LTXVConcatAVLatent
    return LTXVConcatAVLatent.execute(video_upscaled, audio_latent)


def _resize_spatial(tensor, height, width, method):
    original_dtype = tensor.dtype
    kwargs = {"align_corners": False} if method in ("bilinear", "bicubic") else {}
    if tensor.ndim == 5:
        b, c, time, old_h, old_w = tensor.shape
        frames = tensor.permute(0, 2, 1, 3, 4).reshape(b * time, c, old_h, old_w)
        frames = F.interpolate(frames.float(), size=(height, width), mode=method, **kwargs)
        return frames.reshape(b, time, c, height, width).permute(0, 2, 1, 3, 4).to(original_dtype).contiguous()
    if tensor.ndim == 4:
        return F.interpolate(tensor.float(), size=(height, width), mode=method, **kwargs).to(original_dtype).contiguous()
    return tensor


def _even(value):
    return max(2, int(round(float(value) / 2.0)) * 2)


def _sync_refs(obj, scale_y, scale_x, method, stats, depth=0):
    if depth > 10:
        return obj
    if isinstance(obj, dict):
        result = dict(obj)
        latent_key = next(
            (key for key, val in obj.items()
             if "latent" in str(key).lower() and "audio" not in str(key).lower()
             and torch.is_tensor(val) and val.ndim in (4, 5)),
            None,
        )
        has_h = "latent_h" in obj and isinstance(obj.get("latent_h"), (int, float))
        has_w = "latent_w" in obj and isinstance(obj.get("latent_w"), (int, float))
        if latent_key is not None and has_h and has_w:
            tensor = obj[latent_key]
            old_h, old_w = int(tensor.shape[-2]), int(tensor.shape[-1])
            new_h, new_w = _even(old_h * scale_y), _even(old_w * scale_x)
            result[latent_key] = _resize_spatial(tensor, new_h, new_w, method)
            result["latent_h"] = new_h
            result["latent_w"] = new_w
            stats.append(f"{latent_key} {old_w}x{old_h} -> {new_w}x{new_h}")
            # O mesmo bloco pode conter audio_latent; ele permanece intocado.
            return result
        return {key: _sync_refs(val, scale_y, scale_x, method, stats, depth + 1)
                for key, val in result.items()}
    if isinstance(obj, list):
        return [_sync_refs(val, scale_y, scale_x, method, stats, depth + 1) for val in obj]
    if isinstance(obj, tuple):
        return tuple(_sync_refs(val, scale_y, scale_x, method, stats, depth + 1) for val in obj)
    return obj


def _sync_conditioning(conditioning, scale_y, scale_x, method, stats):
    if conditioning is None:
        return None
    output = []
    for item in conditioning:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            copied = list(item)
            copied[1] = _sync_refs(item[1], scale_y, scale_x, method, stats)
            output.append(copied)
        else:
            output.append(item)
    return output


class BruxosH3NeuralUpscaleMegapixels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Latente AV do MiniMax H3. O vídeo é ampliado; o áudio permanece intacto."}),
                "model_name": (_models(),),
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "align": ("INT", {"default": 32, "min": 32, "max": 512, "step": 32,
                    "tooltip": "Alinhamento em pixels. 32 preserva o aspect ratio e mantém o grid seguro do H3."}),
                "device": (["auto", "cuda", "rocm", "cpu"], {"default": "cuda",
                    "tooltip": "CUDA para NVIDIA; ROCm somente com PyTorch HIP em GPU AMD; CPU é muito mais lento."}),
                "precision": (["auto", "bf16", "fp32", "fp16"], {"default": "bf16",
                    "tooltip": "BF16 é o padrão recomendado para os pesos H3. FP32 prioriza fidelidade; FP16 é legado e pode perder faixa numérica."}),
            },
            "optional": {
                "manter_modelo_na_vram": ("BOOLEAN", {"default": False, "advanced": True,
                    "tooltip": "Ligado acelera execuções seguintes. Desligado libera o upscaler neural após cada execução."}),
                "usar_chunks_temporais": ("BOOLEAN", {"default": True, "advanced": True,
                    "tooltip": "Usa somente o chunking nativo do modelo 3D, com padding replicado e blend treinado. Desligue apenas para clips curtos."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Upscaler neural MiniMax H3 por megapixels. Amplia somente o vídeo, preserva o áudio e usa o "
        "chunking temporal nativo da rede 3D para evitar borrão e padrões repetidos. A saída decodificada "
        "diretamente é um preview; para recuperar detalhe fino, use Cond Sync e um segundo passe de refino leve."
    )

    def run(self, latent, model_name, target_megapixels, align, device, precision,
            manter_modelo_na_vram=False, usar_chunks_temporais=True):
        if str(model_name).startswith("("):
            raise ValueError("[Bruxos H3 Neural Upscale] coloque o modelo H3 em models/latent_upscale_models.")
        if "minimax" not in str(model_name).lower() or "h3" not in str(model_name).lower():
            raise ValueError("[Bruxos H3 Neural Upscale] selecione um modelo minimax_h3_latent_upscaler.")

        video = _video_component(latent)
        source_h, source_w = int(video.shape[-2]), int(video.shape[-1])
        target_w, target_h = _aligned_target(source_w, source_h, target_megapixels, align)
        if target_w < source_w or target_h < source_h:
            current_mp = (source_w * 16) * (source_h * 16) / 1_000_000.0
            raise ValueError(
                f"[Bruxos H3 Neural Upscale] alvo {float(target_megapixels):.2f} MP é menor "
                f"que o canvas atual (~{current_mp:.2f} MP). Este modelo faz somente upscale."
            )

        common = _backend()
        resolved_device, resolved_precision = _resolve_compute(device, precision)
        scale = math.sqrt((target_w * target_h) / (source_w * source_h))
        if scale > 4.0001:
            raise ValueError(
                f"[Bruxos H3 Neural Upscale] escala calculada {scale:.3f}x excede 4x, "
                "limite do treinamento do modelo. Reduza target_megapixels ou faça dois passes menores."
            )

        from comfy_extras.nodes_lt import LTXVSeparateAVLatent
        separated = LTXVSeparateAVLatent.execute(latent)
        separated_values = _node_output_values(separated)
        if len(separated_values) < 2:
            raise RuntimeError(
                "[Bruxos H3 Neural Upscale] SeparateAV retornou menos de dois streams."
            )
        video_latent, audio_latent = separated_values[:2]
        result = _run_backend_upscale(
            common,
            video_latent, audio_latent, model_name,
            resolved_device, resolved_precision,
            target_w, target_h, scale,
            enable_native_chunking=bool(usar_chunks_temporais),
            force_unload=not bool(manter_modelo_na_vram),
        )
        out = _first_node_output(result)
        if not isinstance(out, dict):
            out = {"samples": out}
        else:
            out = dict(out)
        out[META_KEY] = {
            "source_h": source_h,
            "source_w": source_w,
            "target_h": target_h,
            "target_w": target_w,
            "scale_y": target_h / source_h,
            "scale_x": target_w / source_w,
            "target_megapixels_requested": float(target_megapixels),
            "target_megapixels_real": (target_w * 16) * (target_h * 16) / 1_000_000.0,
            "align": int(align),
            "chunking": "nativo_3d" if bool(usar_chunks_temporais) else "contexto_completo",
            "precision": str(resolved_precision),
        }

        backend_node = getattr(common, "MinimaxH3LatentUpscaler3D", None)
        backend_has_force_unload = bool(
            backend_node is not None
            and "force_unload" in inspect.signature(backend_node.execute).parameters
        )
        if not bool(manter_modelo_na_vram) and not backend_has_force_unload:
            _release_cached_model(common, model_name, resolved_device, resolved_precision)

        print(
            f"[Bruxos H3 Neural Upscale] latent {source_w}x{source_h} -> {target_w}x{target_h} | "
            f"pixels {source_w*16}x{source_h*16} -> {target_w*16}x{target_h*16} | "
            f"{resolved_device}/{resolved_precision} | escala {scale:.4f} | "
            f"chunks={'nativo 3D' if bool(usar_chunks_temporais) else 'contexto completo'}",
            flush=True,
        )
        return (out,)


class BruxosH3LatentCondSync:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"latent": ("LATENT",)},
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "metodo": (["bilinear", "bicubic", "nearest-exact"], {"default": "bilinear", "advanced": True}),
            },
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("latent", "positive", "negative")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Passthrough do LATENT. Sincroniza somente os latentes visuais dos blocos minimax_refs/keyframes, "
        "incluindo tensors 5D e latent_h/latent_w; áudio e embeddings não são alterados."
    )

    def run(self, latent, positive=None, negative=None, metodo="bilinear"):
        meta = latent.get(META_KEY) if isinstance(latent, dict) else None
        if not isinstance(meta, dict):
            print(
                "[Bruxos H3 Cond Sync] LATENT não contém a escala do Neural Upscale Bruxos; "
                "vou manter o conditioning sem alteração para não redimensionar refs no chute.",
                flush=True,
            )
            return latent, positive, negative

        scale_y = float(meta.get("scale_y", 1.0))
        scale_x = float(meta.get("scale_x", 1.0))
        pos_stats, neg_stats = [], []
        out_positive = _sync_conditioning(positive, scale_y, scale_x, metodo, pos_stats)
        out_negative = _sync_conditioning(negative, scale_y, scale_x, metodo, neg_stats)
        unique = list(dict.fromkeys(pos_stats + neg_stats))
        print(
            f"[Bruxos H3 Cond Sync] escala refs x={scale_x:.4f} y={scale_y:.4f} | "
            f"{len(unique)} referência(s): " + ("; ".join(unique) if unique else "nenhuma encontrada"),
            flush=True,
        )
        return latent, out_positive, out_negative


NODE_CLASS_MAPPINGS = {
    "BruxosH3NeuralUpscaleMegapixels": BruxosH3NeuralUpscaleMegapixels,
    "BruxosH3LatentCondSync": BruxosH3LatentCondSync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosH3NeuralUpscaleMegapixels": "MiniMax H3 · Neural Latent Upscale (MP) (Bruxos)",
    "BruxosH3LatentCondSync": "MiniMax H3 · Latent Cond Sync (Bruxos)",
}
