"""Loader/cache compartilhado do Qwen-VL (caption e prompt enhancer).

Extraido do nodes.py do pacote unico. NAO registra node: quem registra e o
pacote que precisar (ex.: ComfyUI-Bruxos-Bernini registra o BruxosQwenVLCaption).
"""

import torch


_BX_QWEN_DEFAULT_INSTRUCTION = (
    "Describe this video in one rich paragraph for a text-to-video upscale "
    "prompt: subjects, clothing, materials, environment, lighting, camera, "
    "color palette, and motion. Be concrete and visual. No meta commentary."
)

_BX_QWEN_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
]

_BX_QWEN_CACHE = {"name": None, "model": None, "processor": None}


def _bx_qwen_load(model_name: str, dtype_str: str, device: str):
    """Carrega Qwen-VL via transformers, com cache em memoria."""
    if _BX_QWEN_CACHE["name"] == (model_name, dtype_str, device) and _BX_QWEN_CACHE["model"] is not None:
        return _BX_QWEN_CACHE["model"], _BX_QWEN_CACHE["processor"]
    try:
        from transformers import AutoProcessor
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "transformers nao esta instalado. Rode: pip install -U transformers accelerate"
        ) from e

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(
        dtype_str, torch.float16
    )

    # Qwen2.5-VL e Qwen2-VL usam classes diferentes; tentamos os dois.
    Model = None
    last_err = None
    for cls_name in ("Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration"):
        try:
            import transformers
            Model = getattr(transformers, cls_name, None)
            if Model is None:
                continue
            model = Model.from_pretrained(model_name, torch_dtype=dtype, device_map=device)
            processor = AutoProcessor.from_pretrained(model_name)
            _BX_QWEN_CACHE.update(
                {"name": (model_name, dtype_str, device), "model": model, "processor": processor}
            )
            return model, processor
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Nao consegui carregar {model_name}. Verifique o nome do modelo e a versao "
        f"do transformers (precisa de >=4.45 para Qwen2.5-VL). Ultimo erro: {last_err}"
    )


def _bx_tensor_to_pil(frame):
    """(H,W,C) torch 0..1 -> PIL.Image RGB."""
    from PIL import Image
    arr = (frame.detach().cpu().clamp(0, 1).numpy() * 255.0).astype("uint8")
    if arr.shape[-1] == 1:
        arr = arr.repeat(3, axis=-1)
    return Image.fromarray(arr[..., :3])


class BruxosQwenVLCaption:
    """Caption de imagem/video com Qwen2.5-VL.
    Saida 'caption' (STRING) e drop-in pro lugar do Florence2Run.caption."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Frames de entrada que serao descritos."}),
                "model_name": (_BX_QWEN_MODELS, {"default": _BX_QWEN_MODELS[0],
                    "tooltip": "Qual modelo Qwen-VL usar (3B leve; 7B mais forte). Baixa do HuggingFace na 1a vez."}),
                "mode": (["single_frame", "keyframes_merge"], {"default": "keyframes_merge",
                    "tooltip": "single_frame: 1 frame (como o Florence). keyframes_merge: varios frames -> UM prompt unico."}),
            },
            "optional": {
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": _BX_QWEN_DEFAULT_INSTRUCTION,
                    "tooltip": "O que o modelo deve descrever. Ja vem ajustada p/ prompt de upscale.",
                }),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 10_000_000,
                    "tooltip": "Indice do frame em single_frame."}),
                "num_keyframes": ("INT", {"default": 6, "min": 2, "max": 32,
                    "tooltip": "Quantos keyframes amostrar em keyframes_merge."}),
                "max_new_tokens": ("INT", {"default": 220, "min": 16, "max": 2048,
                    "tooltip": "Tamanho maximo do texto gerado (mais tokens = descricao mais longa)."}),
                "dtype": (["fp16", "bf16", "fp32"], {"default": "fp16",
                    "tooltip": "Precisao do modelo. fp16 economiza VRAM; bf16 se a GPU suportar."}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto",
                    "tooltip": "Onde rodar. auto escolhe GPU se houver."}),
                "keep_loaded": ("BOOLEAN", {"default": True,
                    "tooltip": "Mantem o modelo na memoria entre execucoes (mais rapido, usa VRAM)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Semente da geracao de texto (0 = livre)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "run"
    CATEGORY = "Bruxos do VFX/Caption"

    def _sample_frames(self, images, mode, frame_index, num_keyframes):
        n = int(images.shape[0])
        if mode == "single_frame" or n == 1:
            idx = max(0, min(n - 1, frame_index))
            return [images[idx]]
        k = max(2, min(num_keyframes, n))
        # amostragem uniforme cobrindo o video
        step = (n - 1) / float(k - 1) if k > 1 else 0
        idxs = sorted({int(round(i * step)) for i in range(k)})
        return [images[i] for i in idxs]

    def run(self, images, model_name, mode,
            instruction=_BX_QWEN_DEFAULT_INSTRUCTION,
            frame_index=0, num_keyframes=6, max_new_tokens=220,
            dtype="fp16", device="auto", keep_loaded=True, seed=0):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model, processor = _bx_qwen_load(model_name, dtype, device)

        # Monta a mensagem multimodal (formato Qwen2-VL/2.5-VL)
        pil_frames = [_bx_tensor_to_pil(f) for f in self._sample_frames(
            images, mode, frame_index, num_keyframes
        )]
        content = [{"type": "image", "image": img} for img in pil_frames]
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "user", "content": content}]

        # Os processors do Qwen-VL aceitam tanto apply_chat_template quanto
        # a API "imagens + texto" direta. Tentamos chat_template (oficial).
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=pil_frames, return_tensors="pt", padding=True,
            ).to(device)
        except Exception:
            # fallback simples
            inputs = processor(
                text=[instruction], images=pil_frames, return_tensors="pt", padding=True,
            ).to(device)

        if seed:
            try:
                torch.manual_seed(int(seed))
            except Exception:
                pass

        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=int(max_new_tokens))
        # remove o prompt do output
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        out_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        if not keep_loaded:
            _BX_QWEN_CACHE.update({"name": None, "model": None, "processor": None})
            try:
                del model, processor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception:
                pass

        return (out_text,)
