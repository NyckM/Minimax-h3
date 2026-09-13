"""
bruxos_core — biblioteca COMUM dos pacotes "Bruxos do VFX".

Este modulo NAO registra nenhum node. Ele so guarda as funcoes de apoio que
antes moravam dentro do nodes.py do pacote unico (mascara, 4n+1, encode/decode
de video, latentes, limpeza de memoria, janelas de contexto e o loader do
Qwen-VL).

Ele e VENDORIZADO (copiado) dentro de cada repositorio Bruxos, de proposito:
assim cada pacote instala sozinho, sem depender da ordem de instalacao dos
outros nem de importar pastas com "-" no nome (que o Python nao aceita).

Fonte canonica: https://github.com/<seu-usuario>/ComfyUI-Bruxos-Core
"""

import copy
import gc
import math

import torch
import torch.nn.functional as F

# numpy e cv2 sao usados por varios nodes deste arquivo pelos nomes curtos
# (np / cv2). Sem estes imports o modulo levanta NameError em tempo de
# execucao (ex.: BruxosFrameInterpolator, BruxosColorMatch).
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

import comfy.model_management
import comfy.samplers
import comfy.utils
from comfy_extras.nodes_custom_sampler import BasicScheduler, KSamplerSelect, SamplerCustom, SplitSigmas

# ---------------------------------------------------------------------------
# Gerenciamento de memoria (portavel): limpeza de VRAM/RAM entre passos e
# entre blocos, prevencao de vazamento (gc) e monitoramento em tempo real.
# Tudo em try/except -> funciona em CPU, sem CUDA, sem psutil e em versoes
# antigas do ComfyUI. Nao depende de nenhuma flag de launch.
# ---------------------------------------------------------------------------
try:
    import psutil as _psutil
except Exception:
    _psutil = None



# === memoria / limpeza ===
def _mem_gb(n):
    try:
        return float(n) / (1024.0 ** 3)
    except Exception:
        return 0.0


def _mem_snapshot():
    """Foto atual de RAM (sistema) e VRAM (GPU corrente). Campos ausentes se
    a fonte nao estiver disponivel."""
    snap = {}
    if _psutil is not None:
        try:
            vm = _psutil.virtual_memory()
            snap["ram_used"] = _mem_gb(vm.total - vm.available)
            snap["ram_total"] = _mem_gb(vm.total)
        except Exception:
            pass
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            snap["vram_free"] = _mem_gb(free)
            snap["vram_total"] = _mem_gb(total)
            snap["vram_alloc"] = _mem_gb(torch.cuda.memory_allocated())
            snap["vram_reserved"] = _mem_gb(torch.cuda.memory_reserved())
    except Exception:
        pass
    return snap


def _mem_report(tag):
    """Imprime uma linha compacta de uso de memoria (VRAM + RAM)."""
    s = _mem_snapshot()
    parts = []
    if "vram_total" in s:
        used = s["vram_total"] - s.get("vram_free", s["vram_total"])
        parts.append(
            f"VRAM {used:.2f}/{s['vram_total']:.2f}GB "
            f"(alloc {s.get('vram_alloc', 0.0):.2f} reserv {s.get('vram_reserved', 0.0):.2f})"
        )
    if "ram_total" in s:
        parts.append(f"RAM {s['ram_used']:.1f}/{s['ram_total']:.1f}GB")
    if not parts:
        parts.append("sem dados (cuda/psutil indisponivel)")
    print(f"[Bernini Infinity][mem] {tag}: " + " | ".join(parts), flush=True)


def _bx_patch_count(model):
    """Quantos pesos estao 'patchados' (LoRA etc.) no ModelPatcher. Sob
    DynamicVRAM/async offload, descarregar um modelo assim obriga a REFAZER o
    staging (GBs) e RE-APLICAR todos esses patches na passada seguinte -- caro."""
    if model is None:
        return 0
    for attr in ("patches", "object_patches"):
        try:
            p = getattr(model, attr, None)
            if p:
                return len(p)
        except Exception:
            pass
    return 0


# acima disto, um unload entre passos custa re-stage + re-patch (nao compensa)
_BX_PATCH_HEAVY = 32
_BX_WARNED = {"unload": False}


def _mem_cleanup(level="leve", model=None, between_passes=False, force_unload=False):
    """Limpeza de memoria. Niveis:
      off        -> nao faz nada (comportamento legado).
      leve       -> gc.collect() + esvazia cache de VRAM (soft_empty_cache +
                    torch empty_cache/ipc_collect). Barato e seguro.
      agressivo  -> alem do acima, DESCARREGA os modelos da VRAM
                    (unload_all_models). Menor pico de VRAM, ao custo de
                    recarregar o modelo.

    IMPORTANTE (DynamicVRAM + LoRA): se o modelo tem MUITOS patches (LoRA
    distill = centenas), o unload ENTRE os passos high/low e contraproducente:
    forca re-stage de GBs + re-aplicacao de todos os patches, e isso pode custar
    minutos por passo. Nesse caso o 'agressivo' vira 'leve' automaticamente
    entre passos (o unload do FIM da run continua valendo).

    force_unload=True: DESCARREGA os modelos IGNORANDO o guard acima. E o que o
    'force_unload_between_passes' liga: o usuario esta estourando VRAM na
    transicao high->low (os dois modelos nao cabem juntos na placa) e aceita o
    custo do re-stage pra que o high saia da VRAM ANTES do low entrar. Vale
    mesmo com level='off' (a intencao e clara: libere a VRAM aqui)."""
    if level == "off" and not force_unload:
        return
    do_unload = (level == "agressivo") or bool(force_unload)

    # O guard de re-stage so vale pro caminho AUTOMATICO (agressivo). Se o
    # usuario FORCOU (force_unload), respeitamos a escolha dele -- e exatamente
    # o caso de OOM em placa unica onde high+low nao cabem juntos.
    if do_unload and between_passes and not force_unload:
        n = _bx_patch_count(model)
        if n >= _BX_PATCH_HEAVY:
            do_unload = False
            if not _BX_WARNED["unload"]:
                _BX_WARNED["unload"] = True
                print(
                    f"[Bernini Infinity][mem] limpar_vram=agressivo IGNORADO entre passos: "
                    f"o modelo tem {n} patches (LoRA). Sob DynamicVRAM/async offload, "
                    f"descarregar aqui forca re-stage + re-aplicar {n} patches na passada "
                    f"seguinte (custa MUITO mais do que economiza). Usando limpeza leve. "
                    f"Se voce esta estourando VRAM na transicao high->low, ligue "
                    f"'force_unload_between_passes' pra forcar o unload mesmo assim.",
                    flush=True,
                )
    if force_unload and between_passes and do_unload and not _BX_WARNED.get("forced", False):
        _BX_WARNED["forced"] = True
        print("[Bernini Infinity][mem] force_unload_between_passes LIGADO: descarregando o "
              "modelo high antes do low (evita o pico de 2 modelos na VRAM). Custa um "
              "re-stage por passo -- ligue so se estava dando OOM na transicao.", flush=True)
    try:
        gc.collect()
    except Exception:
        pass
    if do_unload:
        try:
            comfy.model_management.unload_all_models()
        except Exception:
            pass
    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# === video / latente / 4n+1 ===
def _clone_conditioning_set_values(conditioning, values):
    updated = []
    for item in conditioning:
        if len(item) != 2:
            updated.append(copy.deepcopy(item))
            continue
        cond, metadata = item
        metadata = copy.deepcopy(metadata)
        metadata.update(values)
        updated.append([cond, metadata])
    return updated


def _resize_long_edge(image, max_size, stride=16):
    h, w = image.shape[1], image.shape[2]
    scale = min(max_size / max(h, w), 1.0)
    nh = max(stride, round(h * scale / stride) * stride)
    nw = max(stride, round(w * scale / stride) * stride)
    return comfy.utils.common_upscale(
        image[:, :, :, :3].movedim(-1, 1), nw, nh, "area", "disabled"
    ).movedim(1, -1)


# Modo de encaixe quando o aspect ratio do video difere do width/height pedido:
#   stretch -> "disabled": estica (sem cortar; pode distorcer um pouco)
#   crop    -> "center":   corta as bordas pra preservar o aspect ratio
_FIT_CROP = {"stretch": "disabled", "crop": "center"}


def _resize_source_video(video, width, height, mode="stretch"):
    crop = _FIT_CROP.get(mode, "disabled")
    return comfy.utils.common_upscale(
        video[:, :, :, :3].movedim(-1, 1), width, height, "area", crop
    ).movedim(1, -1)


def _video_frame_count(video):
    if video is None or not hasattr(video, "shape") or len(video.shape) < 1:
        raise ValueError("source_video must be a ComfyUI IMAGE/video tensor with frames on dimension 0.")
    return int(video.shape[0])


def _split_video(video, chunk_size, overlap):
    frame_count = _video_frame_count(video)
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    step = chunk_size - overlap
    chunks = []
    ranges = []
    for start in range(0, frame_count, step):
        end = min(start + chunk_size, frame_count)
        if end <= start:
            continue
        chunks.append(video[start:end])
        ranges.append((start, end))
        if end == frame_count:
            break
    return chunks, ranges


def _latent_shape(frame_count, width, height, batch_size):
    return [batch_size, 16, ((frame_count - 1) // 4) + 1, height // 8, width // 8]


def _make_empty_latent(frame_count, width, height, batch_size):
    return torch.zeros(
        _latent_shape(frame_count, width, height, batch_size),
        device=comfy.model_management.intermediate_device(),
    )


def _encode_video(vae, video):
    encoded = vae.encode(video)
    if isinstance(encoded, dict) and "samples" in encoded:
        return encoded["samples"]
    return encoded


def _resize_latent_spatial(samples, new_h, new_w, mode="bicubic"):
    """[B,C,T,H,W] -> mesma coisa com H,W novos (so espacial; T intocado)."""
    B, C, T, H, W = (int(v) for v in samples.shape)
    new_h, new_w = int(new_h), int(new_w)
    if (H, W) == (new_h, new_w):
        return samples
    x = samples.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    x = torch.nn.functional.interpolate(x.float(), size=(new_h, new_w), mode=mode, **kwargs)
    return x.reshape(B, T, C, new_h, new_w).permute(0, 2, 1, 3, 4).to(samples.dtype).contiguous()


def _prepare_init_latent(init_latent, target_t, target_h, target_w, mode="bicubic"):
    """Adapta um LATENT externo (saida de um passe anterior, tipicamente ja
    upscalado) pro grid exato deste passe: [1,16,target_t,target_h,target_w].
    Redimensiona espaco por interpolacao; tempo por corte/repeticao do ultimo
    latente (o uso esperado e MESMO source_video nos dois passes, entao T
    normalmente ja bate)."""
    samples = init_latent.get("samples") if isinstance(init_latent, dict) else init_latent
    if not torch.is_tensor(samples):
        raise ValueError("[Bernini Infinity] init_latent invalido: nao encontrei 'samples' (tensor).")
    if samples.ndim != 5:
        raise ValueError(f"[Bernini Infinity] init_latent esperava [B,C,T,H,W]; veio {tuple(samples.shape)}.")
    B, C, T, H, W = (int(v) for v in samples.shape)
    if C != 16:
        raise ValueError(
            f"[Bernini Infinity] init_latent tem {C} canais; o latente do Wan/Bernini tem 16. "
            "Confirme que ele veio de um passe do BerniniInfinity (ou de um upscaler compativel), "
            "nao de outro modelo (ex.: MiniMax H3 tem 24 canais)."
        )
    out = _resize_latent_spatial(samples, target_h, target_w, mode)
    if T != target_t:
        print(f"[Bernini Infinity] init_latent: T={T} != alvo T={target_t}; ajustando "
              f"(corte/repeticao do ultimo latente). Espera-se que os dois passes usem o "
              f"MESMO source_video -- se T bate normalmente, confira max_frames/chunk_size.", flush=True)
        if T > target_t:
            out = out[:, :, :target_t]
        else:
            pad = out[:, :, -1:].repeat(1, 1, target_t - T, 1, 1)
            out = torch.cat([out, pad], dim=2)
    device = comfy.model_management.intermediate_device()
    return out.to(device=device)


# ----------------------------------------------------------------------
# ALINHAMENTO TEMPORAL (4n+1) -- por que o video "perde" frames:
#   o Wan VAE comprime o tempo em ~4x. N frames viram T_lat = ((N-1)//4)+1
#   latentes, e o decode devolve (T_lat-1)*4 + 1 frames. So sobrevivem
#   comprimentos da forma 4n+1 (1,5,9,...,109,113,...). 111 -> 28 latentes
#   -> 109 frames. A correcao: por dentro trabalhamos no proximo 4n+1
#   (padding espelhado) e no final cortamos de volta ao alvo do usuario.
# ----------------------------------------------------------------------
def _lat_len(frames):
    """Numero de frames latentes para `frames` frames de pixel (compressao 4x)."""
    return ((int(frames) - 1) // 4) + 1


def _align_up_4n1(n):
    """Proximo comprimento valido para o grid temporal do Wan VAE (4n+1)."""
    n = int(n)
    if n < 1:
        return 1
    r = (n - 1) % 4
    return n if r == 0 else n + (4 - r)


def _mirror_pad_frames(video, target_len):
    """Estende o video no eixo temporal (dim 0) ate target_len por reflexao
    (espelho ping-pong, igual ao truque do Kijai no Wan Animate), evitando o
    frame congelado que a simples duplicacao do ultimo frame causaria."""
    cur = int(video.shape[0])
    target_len = int(target_len)
    if cur >= target_len:
        return video
    need = target_len - cur
    if cur == 1:
        pad = video[-1:].repeat(need, *([1] * (video.dim() - 1)))
        return torch.cat([video, pad], dim=0)
    idx = []
    i = cur - 2          # comeca refletindo a partir do penultimo
    direction = -1
    while len(idx) < need:
        idx.append(i)
        i += direction
        if i < 0:                 # bate na borda inicial e volta
            i = 1
            direction = 1
        elif i > cur - 1:         # bate na borda final e volta
            i = cur - 2
            direction = -1
    pad = video[idx]
    return torch.cat([video, pad], dim=0)


# === mascaras ===
def _normalize_mask(mask):
    if mask is None:
        return None
    m = mask
    if m.dim() == 4:                       # IMAGE [T,H,W,C] (mascara colorida)
        m = m[..., :3].amax(dim=-1)        # qualquer canal aceso = dentro
    elif m.dim() == 2:                     # [H,W]
        m = m.unsqueeze(0)
    return m.float().clamp(0.0, 1.0)


def _grow_blur_mask(m, grow=0, blur=0):
    """m: [T,H,W] -> [T,H,W]. grow>0 dilata, grow<0 contrai, blur suaviza a borda."""
    x = m.unsqueeze(1)                      # [T,1,H,W]
    grow = int(grow)
    if grow > 0:
        k = grow * 2 + 1
        x = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=1, padding=grow)
    elif grow < 0:
        g = -grow
        k = g * 2 + 1
        x = -torch.nn.functional.max_pool2d(-x, kernel_size=k, stride=1, padding=g)
    blur = int(blur)
    if blur > 0:
        k = blur * 2 + 1
        coords = torch.arange(k, dtype=torch.float32, device=m.device) - blur
        sigma = blur * 0.5 + 1e-6
        g1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        g1d = (g1d / g1d.sum())
        kh = g1d.view(1, 1, k, 1)
        kw = g1d.view(1, 1, 1, k)
        x = torch.nn.functional.conv2d(x, kh, padding=(blur, 0))
        x = torch.nn.functional.conv2d(x, kw, padding=(0, blur))
    return x.squeeze(1).clamp(0.0, 1.0)

def _resize_mask_spatial_temporal(m, T, H, W, mode="stretch"):
    """m: [Tm,Hm,Wm] -> [T,H,W]. Usa o MESMO encaixe (stretch/crop) do vídeo,
    para máscara e fonte ficarem alinhadas. Tempo por amostragem nearest."""
    Tm = int(m.shape[0])
    crop = _FIT_CROP.get(mode, "disabled")

    m = comfy.utils.common_upscale(
        m.unsqueeze(1),
        int(W),
        int(H),
        "bilinear",
        crop,
    ).squeeze(1)

    if Tm != int(T):
        idx = (
            torch.linspace(0, Tm - 1, steps=int(T))
            .round()
            .long()
            .clamp(0, Tm - 1)
        )
        m = m[idx]

    return m

def _rect_feather_mask(n, ch, cw, feather, device=None):
    """Alpha retangular [n,ch,cw]: 1.0 no interior, com borda suave (feather px)
    caindo pra 0 nas bordas do retangulo. Usado pra compor pelo BBOX (sem a
    linha do contorno da silhueta)."""
    device = device or torch.device("cpu")
    ys = torch.arange(ch, dtype=torch.float32, device=device).view(ch, 1)
    xs = torch.arange(cw, dtype=torch.float32, device=device).view(1, cw)
    f = max(1, int(feather))
    dist_y = torch.minimum(ys, (ch - 1) - ys)
    dist_x = torch.minimum(xs, (cw - 1) - xs)
    edge = torch.minimum(dist_y, dist_x)              # dist ate a borda mais proxima
    alpha = (edge / float(f)).clamp(0.0, 1.0)         # 0 na borda -> 1 apos feather px
    return alpha.unsqueeze(0).expand(n, ch, cw).contiguous()



    """m: [Tm,Hm,Wm] -> [T,H,W]. Usa o MESMO encaixe (stretch/crop) do video,
    pra mascara e fonte ficarem alinhadas. Tempo por amostragem nearest."""
    Tm = int(m.shape[0])
    crop = _FIT_CROP.get(mode, "disabled")
    m = comfy.utils.common_upscale(
        m.unsqueeze(1), int(W), int(H), "bilinear", crop
    ).squeeze(1)                              # [Tm,H,W]
    if Tm != int(T):
        idx = torch.linspace(0, Tm - 1, steps=int(T)).round().long().clamp(0, Tm - 1)
        m = m[idx]
    return m


def _mask_to_latent(m, T_lat, lat_h, lat_w, device, dtype):
    """Reduz a mascara de pixel [Tpix,H,W] para o grid latente
    [1,1,T_lat,lat_h,lat_w]. No tempo agrupa em blocos de 4 (1 + 4k) usando o
    maximo (qualquer frame do bloco dentro => latente dentro)."""
    m = m.to(device=device, dtype=torch.float32)
    m = torch.nn.functional.interpolate(
        m.unsqueeze(1), size=(int(lat_h), int(lat_w)), mode="bilinear", align_corners=False
    ).squeeze(1)                             # [Tpix,lat_h,lat_w]
    Tpix = int(m.shape[0])
    out = torch.zeros(int(T_lat), int(lat_h), int(lat_w), device=device, dtype=torch.float32)
    out[0] = m[0]
    for li in range(1, int(T_lat)):
        a = 1 + (li - 1) * 4
        b = min(Tpix, a + 4)
        out[li] = m[a:b].amax(dim=0) if a < Tpix else m[-1]
    return out.view(1, 1, int(T_lat), int(lat_h), int(lat_w)).to(dtype=dtype)


def _mask_bbox(m, pad, stride, W, H, thr=0.02):
    """bbox (x0,y0,x1,y1) cobrindo a regiao em TODOS os frames, com folga `pad`
    e alinhada a `stride` (multiplo exigido por largura/altura)."""
    any2d = (m.amax(dim=0) > thr)
    rows = torch.where(any2d.any(dim=1))[0]
    cols = torch.where(any2d.any(dim=0))[0]
    if rows.numel() == 0 or cols.numel() == 0:
        return 0, 0, int(W), int(H)
    y0 = int(rows.min()); y1 = int(rows.max()) + 1
    x0 = int(cols.min()); x1 = int(cols.max()) + 1
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(int(W), x1 + pad); y1 = min(int(H), y1 + pad)
    x0 -= x0 % stride
    y0 -= y0 % stride
    if x1 % stride:
        x1 = min(int(W), x1 + (stride - x1 % stride))
    if y1 % stride:
        y1 = min(int(H), y1 + (stride - y1 % stride))
    if x1 - x0 < stride:
        x1 = min(int(W), x0 + stride)
    if y1 - y0 < stride:
        y1 = min(int(H), y0 + stride)
    return x0, y0, x1, y1


def _collect_reference_latents(vae, length, ref_max_size, reference_video=None, reference_images=None,
                                scale_vid=1.0, scale_img=1.0):
    latents = []
    if reference_video is not None:
        ref_vid = _resize_long_edge(reference_video[:length], ref_max_size)
        vid_latent = _encode_video(vae, ref_vid[:, :, :, :3])
        # scale_vid ajusta a influencia do VIDEO de referencia, fora do modo multi
        # (ver ref_influence_vid_off no BerniniInfinity.render). 1.0 = sem mudanca.
        if scale_vid != 1.0:
            vid_latent = vid_latent * scale_vid
        latents.append(vid_latent)

    if reference_images:
        for name in sorted(reference_images):
            imgs = reference_images[name]
            if imgs is None:
                continue
            for i in range(imgs.shape[0]):
                img = _resize_long_edge(imgs[i:i + 1], ref_max_size)
                img_latent = _encode_video(vae, img[:, :, :, :3])
                # scale_img ajusta a influencia das IMAGENS de referencia, fora do
                # modo multi (ver ref_influence_img_off no BerniniInfinity.render).
                if scale_img != 1.0:
                    img_latent = img_latent * scale_img
                latents.append(img_latent)

    return latents


def _merge_linear_overlap(first, second, overlap):
    if overlap <= 0:
        return torch.cat([first, second], dim=0)
    if first.shape[0] < overlap or second.shape[0] < overlap:
        overlap = min(int(first.shape[0]), int(second.shape[0]), overlap)
    if overlap <= 0:
        return torch.cat([first, second], dim=0)

    left = first[:-overlap]
    right = second[overlap:]
    first_tail = first[-overlap:]
    second_head = second[:overlap]
    weights = torch.linspace(0.0, 1.0, overlap, dtype=first.dtype, device=first.device)
    while weights.ndim < first_tail.ndim:
        weights = weights.unsqueeze(-1)
    blended = first_tail * (1.0 - weights) + second_head * weights
    return torch.cat([left, blended, right], dim=0)


def _merge_latent_overlap(first, second, overlap):
    # Latente de vídeo no formato [B, C, T, H, W]; o eixo temporal é a dim 2.
    if overlap <= 0:
        return torch.cat([first, second], dim=2)
    overlap = min(int(first.shape[2]), int(second.shape[2]), int(overlap))
    if overlap <= 0:
        return torch.cat([first, second], dim=2)

    left = first[:, :, :-overlap]
    right = second[:, :, overlap:]
    first_tail = first[:, :, -overlap:]
    second_head = second[:, :, :overlap]
    weights = torch.linspace(0.0, 1.0, overlap, dtype=first.dtype, device=first.device)
    weights = weights.view(1, 1, overlap, 1, 1)
    blended = first_tail * (1.0 - weights) + second_head * weights
    return torch.cat([left, blended, right], dim=2)


# === decode / janelas de contexto ===
def _decode_video(vae, latent_samples, tiled=False):
    if tiled:
        # Passa tile/overlap explicitos: em algumas versoes do ComfyUI o
        # decode_tiled 3D deixa overlap=None num eixo e quebra em "tile - overlap".
        try:
            images = vae.decode_tiled(
                latent_samples,
                tile_x=256, tile_y=256, overlap=64,
                tile_t=32, overlap_t=8,
            )
        except TypeError:
            # assinaturas mais antigas nao aceitam tile_t/overlap_t
            try:
                images = vae.decode_tiled(latent_samples, tile_x=256, tile_y=256, overlap=64)
            except Exception:
                images = vae.decode(latent_samples)
        except Exception:
            images = vae.decode(latent_samples)
    else:
        images = vae.decode(latent_samples)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def _decode_video_chunked(vae, latent_samples, tiled=False, lat_chunk=0, lat_overlap=1):
    """Decodifica o latente em blocos temporais (eixo dim 2) para evitar pico de VRAM
    no VAE em videos longos. lat_chunk<=0 (ou >= T) decodifica tudo de uma vez."""
    total = int(latent_samples.shape[2])
    if lat_chunk is None or int(lat_chunk) <= 0 or total <= int(lat_chunk):
        return _decode_video(vae, latent_samples, tiled)
    lat_chunk = int(lat_chunk)
    stride = max(1, lat_chunk - int(lat_overlap))
    pix_overlap = max(0, int(lat_overlap)) * 4  # fator de compressao temporal ~4 do VAE Wan
    result = None
    start = 0
    while start < total:
        end = min(start + lat_chunk, total)
        sub = latent_samples[:, :, start:end]
        imgs = _decode_video(vae, sub, tiled).cpu()
        result = imgs if result is None else _merge_linear_overlap(result, imgs, pix_overlap)
        if end == total:
            break
        start += stride
    return result


def _ordered_offset(idx, stride):
    """Espalha offsets quase uniformemente em [0, stride) conforme os passos avancam."""
    if stride <= 1:
        return 0
    bits = max(1, int(math.ceil(math.log2(stride))))
    r = 0
    for b in range(bits):
        r = (r << 1) | ((idx >> b) & 1)
    return int(r) % stride


def _context_windows(total, win, overlap, offset=0):
    """Janelas (start, end) em frames LATENTES. As pontas reais (0 e total-win) ficam
    SEMPRE ancoradas (video ABERTO, sem wrap end->start); so as fronteiras internas
    deslizam com `offset` para nao travar nos mesmos frames a cada passo de denoise."""
    if win <= 0 or total <= win:
        return [(0, total)]
    stride = max(1, win - overlap)
    offset = int(offset) % stride
    starts = {0, total - win}
    s = offset
    while s < total - win:
        if s > 0:
            starts.add(s)
        s += stride
    return [(p, p + win) for p in sorted(starts)]


def _window_blend_weights(length, ramp_left, ramp_right, device, dtype):
    """Pesos de blend (rampa Hann) nas bordas esquerda/direita da janela.

    ramp_left/ramp_right sao a sobreposicao REAL com a janela vizinha anterior/
    seguinte (nao um valor fixo de overlap configurado): como as pontas 0 e
    total-win ficam sempre ancoradas (ver _context_windows), a sobreposicao com
    a janela vizinha pode ficar maior OU menor que `overlap` perto das bordas do
    video e conforme o jitter desliza as fronteiras internas. Usar sempre o
    `overlap` configurado como rampa (comportamento antigo) descasava do
    tamanho real da sobreposicao: sobra uma faixa "achatada" (peso 1.0 dos dois
    lados, sem rampa) sempre que a sobreposicao real > overlap configurado --
    essa faixa acaba sendo uma media 50/50 sem suavizacao entre duas predicoes
    independentes, o que aparece como emenda/instabilidade nas transicoes.
    """
    w = torch.ones(length, device=device, dtype=dtype)
    ramp_left = int(min(max(0, ramp_left), length // 2))
    ramp_right = int(min(max(0, ramp_right), length // 2))
    if ramp_left > 0:
        t = torch.linspace(0.0, math.pi, steps=ramp_left + 2, device=device, dtype=dtype)[1:-1]
        w[:ramp_left] = (1.0 - torch.cos(t)) * 0.5
    if ramp_right > 0:
        t = torch.linspace(0.0, math.pi, steps=ramp_right + 2, device=device, dtype=dtype)[1:-1]
        edge = (1.0 - torch.cos(t)) * 0.5
        w[-ramp_right:] = torch.flip(edge, dims=[0])
    return w


def _slice_temporal(obj, s, e, total):
    """Fatia recursivamente qualquer tensor cujo eixo temporal (dim 2 em [B,C,T,H,W])
    tenha comprimento == total. O que nao for temporal passa intacto (refs, texto)."""
    if torch.is_tensor(obj):
        if obj.dim() >= 5 and obj.shape[2] == total:
            return obj[:, :, s:e]
        return obj
    if isinstance(obj, dict):
        return {k: _slice_temporal(v, s, e, total) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_slice_temporal(v, s, e, total) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_slice_temporal(v, s, e, total) for v in obj)
    return obj


def _debug_dump_shapes(obj, total, prefix="c", depth=0, acc=None):
    if acc is None:
        acc = []
    if depth > 3:
        return acc
    if torch.is_tensor(obj):
        tag = " <== T_lat" if (obj.dim() >= 5 and obj.shape[2] == total) else ""
        acc.append(f"{prefix}={tuple(obj.shape)}{tag}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _debug_dump_shapes(v, total, f"{prefix}.{k}", depth + 1, acc)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _debug_dump_shapes(v, total, f"{prefix}[{i}]", depth + 1, acc)
    return acc


def _make_context_wrapper(win_len, win_overlap, ramp=None, jitter=True, debug_holder=None):
    """model_function_wrapper estilo WanVideoWrapper: divide o latente completo em
    janelas sobrepostas a cada passo, roda o modelo em cada uma e compoe as predicoes
    com blend. Com jitter, as fronteiras internas deslizam por passo.

    Duas correcoes em relacao a versao anterior (ambas sobre a MESMA causa: as
    janelas nao ficavam consistentes entre si, entao a transicao entre elas
    "nao seguia o frame certo"):

    1) O offset do jitter avancava por CHAMADA do wrapper, nao por PASSO de
       sampling. Um unico passo de denoise pode chamar o model_function mais
       de uma vez (positivo/negativo com cfg>1, os varios forwards de
       guidance_mode=stream, samplers tipo Heun/DPM++ que corrigem em 2
       sub-chamadas). Isso fazia, por exemplo, positivo e negativo do MESMO
       passo caírem em janelas DIFERENTES -- o delta do CFG (pos-neg) virava
       a diferenca entre duas predicoes recortadas em pontos diferentes do
       video, o que e sem sentido e gera instabilidade/tremedeira exatamente
       nas bordas das janelas. Agora o offset so avanca quando o timestep
       muda: todas as chamadas do mesmo passo usam a MESMA janela.

    2) A rampa de blend usava sempre o `overlap` configurado, mas a
       sobreposicao REAL entre duas janelas vizinhas varia (as pontas 0 e
       total-win ficam sempre ancoradas -- ver _context_windows -- entao a
       sobreposicao com a janela vizinha pode ficar bem maior ou menor que o
       overlap configurado, sobretudo perto das pontas do video e conforme o
       jitter desliza). Rampa fixa != sobreposicao real deixava uma faixa sem
       suavizacao (peso 1.0 dos dois lados) sendo apenas MEDIADA 50/50 entre
       duas predicoes independentes -- variavel a cada passo -- em vez de uma
       transicao suave. Agora a rampa de cada lado usa a sobreposicao REAL
       com a janela anterior/seguinte.
    """
    del ramp  # obsoleto -- so por compatibilidade de assinatura, ver acima
    state = {"step": -1, "last_t": None}
    stride = max(1, win_len - win_overlap)

    def wrapper(model_function, params):
        x = params["input"]
        t = params["timestep"]
        c = params["c"]
        total = int(x.shape[2])

        # chave do passo = valor do timestep (sigma). So avanca o jitter
        # quando o timestep muda, entao todas as chamadas do mesmo passo de
        # sampling (pos/neg, streams, sub-chamadas do sampler) usam a mesma
        # janela -- ver ponto (1) da docstring acima.
        try:
            t_key = round(float(t.flatten()[0].item()), 6) if torch.is_tensor(t) else float(t)
        except Exception:
            t_key = None
        if t_key is None or t_key != state["last_t"]:
            state["last_t"] = t_key
            state["step"] += 1

        offset = _ordered_offset(max(0, state["step"]), stride) if jitter else 0
        windows = _context_windows(total, win_len, win_overlap, offset)

        if debug_holder is not None and not debug_holder.get("printed"):
            debug_holder["printed"] = True
            try:
                print(f"[Bernini Infinity][ctx] x={tuple(x.shape)} T_lat={total} offset={offset} janelas={windows}", flush=True)
                for line in _debug_dump_shapes(c, total):
                    print(f"[Bernini Infinity][ctx]   {line}", flush=True)
            except Exception:
                pass

        if len(windows) <= 1:
            return model_function(x, t, **c)

        out = torch.zeros_like(x)
        counter = torch.zeros((1, 1, total, 1, 1), device=x.device, dtype=x.dtype)
        n = len(windows)
        for i, (s, e) in enumerate(windows):
            xw = x[:, :, s:e]
            cw = _slice_temporal(c, s, e, total)
            ow = model_function(xw, t, **cw)
            # sobreposicao REAL com a vizinha anterior/seguinte (ver ponto 2)
            ramp_left = max(0, windows[i - 1][1] - s) if i > 0 else 0
            ramp_right = max(0, e - windows[i + 1][0]) if i < n - 1 else 0
            wts = _window_blend_weights(e - s, ramp_left, ramp_right, x.device, x.dtype).view(1, 1, e - s, 1, 1)
            out[:, :, s:e] += ow * wts
            counter[:, :, s:e] += wts
        return out / counter.clamp(min=1e-6)

    return wrapper
