"""Bruxos H3 Ultimate Latent Upscale — chunk temporal + tiles espaciais.

Integração Bruxos baseada no projeto MIT Comfyui-MMH3-UltimateUpscale,
Copyright (c) 2026 bbaudio-2025. A versão Bruxos acrescenta alvo por
megapixels, alinhamento final realmente múltiplo de 32 e sincronização dos
minimax_refs/latent_h/latent_w usados pelo ref2va.

Pipeline (auto, no graph wiring):
    input AV latent
      -> temporal split (outer loop)
      ->   latent upscale  (H3 3D upscaler, per chunk, video only)
      ->   spatial split   (inner loop)
      ->   per-tile sampling with preview
      ->   spatial stitch
      -> temporal stitch
      -> output AV latent

Helpers (frame/token mapping, re-anchoring, spatial tiling, seam blending,
stitching) are self-contained copies of the logic used by the
Comfyui-MiniMax-H3-LatentSplit project so this plugin has no dependency on it.

Frame/token mapping mirrors comfy.ldm.minimax.model:
  * video latent token k covers FRAME_PER_TOKEN[k % 5] = (1, 4, 4, 4, 4) pixel
    frames (periodic grid, 17 frames per 5 tokens)
  * audio latent frames run at FRAME_RESCALE = 5/3 per pixel frame (40 vs 24 Hz)

The H3 3D upscaler inference code (model classes, loading, normalization stats)
is copied from the Comfyui_Minimax_h3_latent_Upscaler plugin so it works with
the minimax_h3_latent_upscaler_3d checkpoints directly.
"""

import glob
import math
import os
import re
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview
from comfy_api.latest import io

try:
    from .bruxos_h3_neural_upscale import _sync_conditioning as _bruxos_sync_conditioning
except Exception:  # pragma: no cover
    from bruxos_h3_neural_upscale import _sync_conditioning as _bruxos_sync_conditioning

try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
except Exception:
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    FRAME_RESCALE = 5.0 / 3.0

CAT = "Bruxos do VFX/MiniMax H3"
H3_UPSCALE_PARAM = io.Custom("BRUXOS_H3_UPSCALE_PARAM")
H3_TEMPORAL_PARAM = io.Custom("BRUXOS_H3_TEMPORAL_PARAM")
H3_SPATIAL_PARAM = io.Custom("BRUXOS_H3_SPATIAL_PARAM")

# Spatial compression factor of the Minimax H3 3D VAE (16x).
VAE_DOWNSAMPLE = 16

# ---------------------------------------------------------------------------
# frame <-> token helpers (copied from Comfyui-MiniMax-H3-LatentSplit)
# ---------------------------------------------------------------------------

def frames_for_tokens(n):
    """Pixel frames covered by the first `n` video latent tokens."""
    return sum(FRAME_PER_TOKEN[i % 5] for i in range(n))


def tokens_for_frames(f):
    """Smallest token count whose cumulative frames reach at least `f`."""
    n, acc = 0, 0
    while acc < f:
        acc += FRAME_PER_TOKEN[n % 5]
        n += 1
    return n


def audio_range(f0, f1):
    """Audio latent token range [a0, a1) for the pixel-frame span [f0, f1)."""
    return round(f0 * FRAME_RESCALE), round(f1 * FRAME_RESCALE)


def compute_segments(tv, chunk_length, overlap):
    """Per-chunk (video_token_start, frame_start, video_token_end, frame_end).

    Same rules as the Split node: every boundary is snapped to a keyframe token
    (index % 5 == 0), the realized overlap is a whole number of 17-frame grid
    steps, and the last chunk always ends on the exact total frame count.
    """
    frame_count = frames_for_tokens(tv)
    if chunk_length <= 0:
        raise ValueError("O tamanho do chunk (chunk_length) precisa ser maior que zero.")
    if overlap < 0:
        raise ValueError("A sobreposição (overlap) não pode ser negativa.")
    if chunk_length <= overlap:
        raise ValueError("A sobreposição precisa ser menor que o tamanho do chunk.")

    hop = chunk_length - overlap
    bounds = []
    prev_end_k = 0
    i = 0
    while True:
        s = i * hop
        e = min(s + chunk_length, frame_count)
        if i == 0:
            k0, f0 = 0, 0
        else:
            k0, f0 = snap_frame_boundary(s, tv, phase=5)
            if k0 > prev_end_k:
                k0, f0 = prev_end_k, frames_for_tokens(prev_end_k)
        if e >= frame_count:
            k1, f1 = tv, frame_count
        else:
            k1, f1 = snap_frame_boundary(e, tv, phase=5)
            if k1 <= k0:
                k1 = k0 + 5
                f1 = frames_for_tokens(k1)
            if k1 >= tv:
                k1, f1 = tv, frame_count
        bounds.append((k0, f0, k1, f1))
        if k1 >= tv:
            break
        prev_end_k = k1
        i += 1
    return bounds, frame_count


def snap_frame_boundary(f, max_tokens, phase=None):
    """Nearest video-token boundary to pixel frame f (optionally on a phase grid)."""
    step = phase if phase is not None else 1
    best_k, best_f, best_d = 0, 0, f
    for k in range(0, max_tokens + 1, step):
        acc = frames_for_tokens(k)
        d = abs(acc - f)
        if d < best_d:
            best_k, best_f, best_d = k, acc, d
    return best_k, best_f


def is_h3_av_latent(samples):
    return (samples is not None and samples.is_nested and len(samples.tensors) == 2
            and samples.tensors[0].ndim == 5 and samples.tensors[0].shape[1] == 24
            and samples.tensors[1].ndim == 4 and samples.tensors[1].shape[1] == 32)


# ---------------------------------------------------------------------------
# spatial tiling helpers (copied from Comfyui-MiniMax-H3-LatentSplit)
# ---------------------------------------------------------------------------

def _grid_1d(size, tile, ol, min_tile):
    """Tile origins/dims for one axis plus per-seam overlaps.

    If the leftover edge tile would be smaller than min_tile, the last origin is
    pulled left until the edge reaches min_tile; the extra overlap that creates
    is reported per-seam so stitching blends over its full width."""
    if size <= tile:
        return [0], [size], [0]
    sh = tile - ol
    n = math.ceil((size - ol) / sh)
    if (n - 1) * sh + tile < size:
        n += 1
    rows = [i * sh for i in range(n)]
    trows = [min(tile, size - r) for r in rows]
    if min_tile > 0 and n >= 2:
        edge = size - rows[-1]
        if edge < min_tile:
            new_last = size - min_tile
            if rows[-2] < new_last < rows[-2] + trows[-2]:
                rows[-1] = new_last
                trows[-1] = size - new_last
    ovl = [0] * n
    for i in range(1, n):
        ovl[i] = max(0, rows[i - 1] + trows[i - 1] - rows[i])
    return rows, trows, ovl


def compute_spatial_grid(h, w, th, tw, ol_h, ol_w, min_th=0, min_tw=0):
    """Tile a latent of size (h, w) with tiles (th, tw) and overlap (ol_h, ol_w).

    Returns (row_offsets, col_offsets, true_row_dims, true_col_dims,
    row_overlaps, col_overlaps) in latent units. Horizontal and vertical
    overlaps are independent. min_th/min_tw (0 = disabled) force the leftover
    edge tile to at least that size when possible, growing the seam overlap."""
    if th <= 0 or tw <= 0:
        raise ValueError("A largura e a altura dos tiles precisam ser maiores que zero.")
    if ol_h >= th or ol_w >= tw:
        raise ValueError("A sobreposição precisa ser menor que o tamanho do tile.")
    if min_th < 0 or min_tw < 0:
        raise ValueError("O tamanho mínimo do tile não pode ser negativo.")
    if min_th > th or min_tw > tw:
        raise ValueError("O tamanho mínimo do tile não pode ser maior que o próprio tile.")
    rows, trows, row_ovl = _grid_1d(h, th, ol_h, min_th)
    cols, tcols, col_ovl = _grid_1d(w, tw, ol_w, min_tw)
    return rows, cols, trows, tcols, row_ovl, col_ovl


def spatial_fade_mask(tile_h, tile_w, ol_h, ol_w, done_top, done_left, fade_h=0, fade_w=0):
    """Per-tile video noise mask [tile_h, tile_w]: 1 = re-sample freely, 0 = frozen.

    Every tile is sampled at its true extent (no padding), so the mask only
    freezes the overlap strips shared with an already-processed neighbor
    (done_top / done_left). Those strips are frozen by fade_width/fade_height
    ALONE: 0 (default) = whole strip frozen (keeps the neighbor's content),
    otherwise only the first fade latent units fade 0->1 toward the tile
    interior. The two axes use independent fade widths."""
    mask = torch.ones(tile_h, tile_w, dtype=torch.float32)
    if done_left and ol_w > 0:
        if fade_w == 0:
            mask[:, :ol_w] = 0.0
        else:
            f = min(fade_w, ol_w)
            frozen_w = ol_w - f
            w = torch.linspace(0.0, 1.0, f)
            mask[:, :frozen_w] = 0.0
            mask[:, frozen_w:ol_w] = torch.minimum(
                mask[:, frozen_w:ol_w], w[None, :]
            )
    if done_top and ol_h > 0:
        if fade_h == 0:
            mask[:ol_h, :] = 0.0
        else:
            f = min(fade_h, ol_h)
            frozen_h = ol_h - f
            w = torch.linspace(0.0, 1.0, f)
            mask[:frozen_h, :] = 0.0
            mask[frozen_h:ol_h, :] = torch.minimum(
                mask[frozen_h:ol_h, :], w[:, None]
            )
    return mask


def blend_weights(t, overlap_blend, overlap_mode):
    """Weight given to the NEW tile's content across an overlap band.

    t runs 0..1 from the done-seam toward the tile interior. overlap_mode 'later'
    hands the band to the new tile; 'earlier' to the accumulated content.
    overlap_blend selects the transition shape."""
    if overlap_blend == "overwrite":
        return torch.ones_like(t) if overlap_mode == "later" else torch.zeros_like(t)
    if overlap_blend == "midpoint":
        step = (t >= 0.5).to(t.dtype)
    elif overlap_blend == "smoothstep":
        step = t * t * (3.0 - 2.0 * t)
    else:
        step = t
    # Compatível com a correção upstream de agosto/2026. context_only não usa
    # blend; estes pesos ficam apenas para os dois modos legados.
    if overlap_mode == "earlier":
        return step
    return 1.0 - step


def add_global_context_refs(cond, video, max_side_pixels=512,
                            frame_mode="first_middle_last"):
    """Acrescenta ao conditioning uma visão reduzida do chunk inteiro.

    Os keyframes continuam recortados para a área local do tile, enquanto
    estas refs mantêm composição, número de personagens, roupa e iluminação
    globais. Refs H3 têm grade espacial própria, portanto podem ser menores que
    o canvas amostrado sem distorcer as coordenadas do tile.
    """
    if cond is None or video is None or video.ndim != 5:
        return cond

    _, _, total_t, h, w = video.shape
    if total_t <= 0:
        return cond
    max_side_latent = max(2, int(max_side_pixels) // VAE_DOWNSAMPLE)
    scale = min(1.0, float(max_side_latent) / max(h, w))

    def even_size(value):
        return max(2, int(round(float(value) * scale / 2.0)) * 2)

    gh, gw = even_size(h), even_size(w)
    if frame_mode == "middle":
        indices = [total_t // 2]
    else:
        indices = [0, total_t // 2, total_t - 1]
    indices = list(dict.fromkeys(max(0, min(total_t - 1, int(i))) for i in indices))

    global_blocks = []
    for index in indices:
        frame = video[:, :, index:index + 1].float()
        if (gh, gw) != (h, w):
            frame_2d = frame[:, :, 0]
            frame_2d = F.interpolate(
                frame_2d, size=(gh, gw), mode="bilinear", align_corners=False
            )
            frame = frame_2d[:, :, None]
        frame = frame.to(device=video.device, dtype=video.dtype).contiguous()
        global_blocks.append({
            "kind": "image",
            "latent_h": gh,
            "latent_w": gw,
            "latent": frame,
            "bruxos_global_context": True,
        })

    out = []
    for tensor, data in cond:
        copied = dict(data)
        refs = list(copied.get("minimax_refs") or [])
        copied["minimax_refs"] = refs + global_blocks
        out.append([tensor, copied])
    return out


def count_minimax_refs(cond):
    if not cond:
        return 0
    return sum(len(data.get("minimax_refs") or []) for _tensor, data in cond)


def prepare_tile_conditioning(cond, src_h, src_w, r0, c0, tr, tc,
                              reference_policy="safe_strip_latent_refs",
                              first_tile=False):
    """Recorta keyframes locais e controla refs latentes globais por tile.

    A embedding Qwen/CLIP do conditioning é preservada. No modo seguro apenas
    os blocos `minimax_refs` latentes são retirados, pois reapresentar a mesma
    imagem/vídeo latente a cada tile faz o Ref2VA tentar recriar o sujeito
    inteiro em cada região. O latent já denoisado e o overlap congelado mantêm
    identidade e composição no segundo passe.
    """
    if cond is None:
        return None, 0
    local = crop_keyframes_to_tile(cond, src_h, src_w, r0, c0, tr, tc)
    keep_refs = reference_policy == "keep_all_refs"
    if reference_policy == "first_tile_only" and first_tile:
        keep_refs = True
    if keep_refs:
        return local, 0

    out = []
    removed = 0
    for tensor, data in local:
        copied = dict(data)
        refs = copied.pop("minimax_refs", None)
        if refs:
            removed += len(refs)
        out.append([tensor, copied])
    return out, removed


def crop_keyframes_to_tile(cond, src_h, src_w, r0, c0, tr, tc):
    """Spatially crop every keyframe's video latent to a tile of the source frame.

    Keyframes whose latent already matches the source spatial size are cropped to
    the tile's latent region; others are passed through unchanged. Audio
    keyframes are untouched (audio is not spatial)."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            cropped = []
            for kf in kfs:
                nkf = dict(kf)
                lt = kf.get("latent")
                if (lt is not None and lt.shape[3] == src_h and lt.shape[4] == src_w):
                    nkf["latent"] = lt[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                cropped.append(nkf)
            nd["minimax_keyframes"] = cropped
        out.append([tensor, nd])
    return out


def trim_keyframe(kf, f0, f1):
    """Copy a keyframe cut to the portion fully inside pixel frames [f0, f1)."""
    idx = kf["resolved_frame_index"]
    latent = kf.get("latent")
    audio_latent = kf.get("audio_latent")
    has_v = latent is not None
    has_a = audio_latent is not None

    if not has_v and not has_a:
        if idx < f0 or idx >= f1:
            return None
        return {"resolved_frame_index": idx - f0}

    out = {}
    if has_v:
        t_start = t_end = None
        pos = idx
        for k in range(latent.shape[2]):
            span = FRAME_PER_TOKEN[k % 5]
            if f0 <= pos and pos + span <= f1:
                if t_start is None:
                    t_start = k
                t_end = k + 1
            pos += span
        if t_start is None:
            return None
        out["latent"] = latent[:, :, t_start:t_end].contiguous()
        out["resolved_frame_index"] = idx + frames_for_tokens(t_start) - f0
    if has_a:
        rt = audio_latent.shape[-1]
        a_start = max(0, math.ceil((f0 - idx) * FRAME_RESCALE))
        a_end = min(rt, math.floor((f1 - idx) / FRAME_RESCALE))
        if a_end > a_start:
            out["audio_latent"] = audio_latent[..., a_start:a_end].contiguous()
            if "resolved_frame_index" not in out:
                out["resolved_frame_index"] = max(0, idx - f0)
    if "latent" not in out and "audio_latent" not in out:
        return None
    return out


def reanchor_conditioning(cond, f0, f1, spatial=None):
    """Cut/re-anchor minimax_keyframes to the pixel-frame segment [f0, f1).

    When `spatial` (latent_h, latent_w) is given, keyframe video latents whose
    spatial size differs are resized to it (bilinear)."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            trimmed = [trim_keyframe(kf, f0, f1) for kf in kfs]
            trimmed = [kf for kf in trimmed if kf is not None]
            if trimmed:
                if spatial is not None:
                    for kf in trimmed:
                        lt = kf.get("latent")
                        if lt is not None and (lt.shape[3] != spatial[0] or lt.shape[4] != spatial[1]):
                            B, C, T, H, W = lt.shape
                            kf["latent"] = F.interpolate(
                                lt.view(B * T, C, H, W), size=spatial, mode="bilinear", align_corners=False
                            ).view(B, C, T, spatial[0], spatial[1])
                nd["minimax_keyframes"] = trimmed
            else:
                nd.pop("minimax_keyframes", None)
        out.append([tensor, nd])
    return out


def validate_keyframe_spatial_size(cond, target_h, target_w, label="conditioning"):
    """Impede que keyframes incompatíveis sejam interpolados silenciosamente.

    O H3 empacota os keyframes em patches 2x2. Interpolar esses latents para uma
    nova resolução pode gravar grades e padrões repetidos no vídeo. O caminho
    seguro é recriar o conditioning diretamente na resolução final.
    """
    if cond is None:
        return
    mismatches = []
    for _tensor, data in cond:
        for keyframe in data.get("minimax_keyframes") or []:
            latent = keyframe.get("latent")
            if latent is None or latent.ndim != 5:
                continue
            height, width = int(latent.shape[-2]), int(latent.shape[-1])
            if height != target_h or width != target_w:
                mismatches.append((width, height))
    if mismatches:
        sizes = ", ".join(f"{w * VAE_DOWNSAMPLE}x{h * VAE_DOWNSAMPLE}px" for w, h in sorted(set(mismatches)))
        target = f"{target_w * VAE_DOWNSAMPLE}x{target_h * VAE_DOWNSAMPLE}px"
        raise ValueError(
            f"[Bruxos H3 Ultimate] O {label} contém keyframes em {sizes}, mas o "
            f"latent será amostrado em {target}. Recrie o node MiniMax H3 Image/Video "
            "to Video usando a resolução final e conecte o novo conditioning. "
            "Se não for possível, selecione 'redimensionar_fallback' no node de "
            "upscale, sabendo que essa interpolação pode criar grades e padrões repetidos."
        )


def anchor_conditioning(cond, prev_video, f0, strength):
    """Replace the frame-0 keyframe with the previous chunk's re-sampled frame.

    Mirrors the 'Anchor MiniMax H3 Latent' node: keyframes are frozen rows in
    the H3 packed sequence, so pinning frame 0 to the content the previous chunk
    ended with removes the detail mismatch at the seam. `strength` becomes
    minimax_visual_cond_noise_aug (0.999 = model default)."""
    t = tokens_for_frames(f0)
    if t >= prev_video.shape[2]:
        raise ValueError("O resultado anterior não alcança o quadro inicial do segmento atual.")
    anchor_kf = {"resolved_frame_index": 0, "latent": prev_video[:, :, t:t + 1].contiguous()}
    aug = max(0.0, min(1.0, float(strength)))
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            kept = [kf for kf in kfs if kf.get("resolved_frame_index") != 0 or "latent" not in kf]
            nd["minimax_keyframes"] = [anchor_kf] + kept
        else:
            nd["minimax_keyframes"] = [anchor_kf]
        nd["minimax_visual_cond_noise_aug"] = aug
        out.append([tensor, nd])
    return out


def _crossfade(a, b, dim):
    n = a.shape[dim]
    w = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype)
    shape = [1] * a.ndim
    shape[dim] = n
    w = w.view(shape)
    return a + (b - a) * w


# ---------------------------------------------------------------------------
# H3 3D latent upscaler (copied from Comfyui_Minimax_h3_latent_Upscaler)
# ---------------------------------------------------------------------------

_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER)
    )

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180265264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]


def _make_norm_tensors(device, dtype):
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    return mean, std


def _normalization(channels):
    return nn.GroupNorm(32, channels)


def _zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class _AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = _normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        b, c, t, hh, w = h.shape
        q = self.q(h).flatten(2).transpose(1, 2)
        k = self.k(h).flatten(2).transpose(1, 2)
        v = self.v(h).flatten(2).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).view(b, c, t, hh, w)
        return x + self.proj_out(h)


class _ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            _normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = _normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            _zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class _TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h


class _LatentResizer3D(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512, dropout=0.1, attn=False,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(_AttnBlock3D(channels))
            self.in_blocks.append(_ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(_TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(_AttnBlock3D(channels))
            self.out_blocks.append(_ResBlockEmb3D(channels, embed_dim, dropout))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(_TemporalConv(channels, temporal_kernel))

        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale=None, target_size=None, enable_chunking=True):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        # Atualização 23/08/2026 do backend original: chunks internos recebem
        # contexto temporal com padding replicado. A saída compartilhada é
        # acumulada com pesos, evitando a borda artificial de Conv3D e o flicker
        # nos últimos frames de cada janela.
        temporal_kernel = 0
        for block in self.in_blocks:
            if isinstance(block, _TemporalConv):
                temporal_kernel = int(block.dwconv.weight.shape[2])
                break
        overlap = temporal_kernel
        chunk = 24
        total_t = int(x.shape[2])
        if enable_chunking and total_t > chunk and overlap > 0:
            batch, channels = int(x.shape[0]), int(x.shape[1])
            padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate")
            accumulated = torch.zeros(
                batch, channels, total_t, size[-2], size[-1],
                device=x.device, dtype=x.dtype,
            )
            accumulated_weight = torch.zeros(
                1, 1, total_t, 1, 1, device=x.device, dtype=x.dtype,
            )
            print(
                f"[Bruxos H3 Ultimate] chunking interno do upscaler: T={total_t}, "
                f"janela={chunk}, contexto={overlap}.",
                flush=True,
            )
            start = 0
            while start < total_t:
                core_start = start
                core_end = min(total_t, start + chunk)
                out_start = max(0, core_start - overlap)
                out_end = min(total_t, core_end + overlap)
                read_start = max(0, out_start - overlap)
                read_end = min(total_t + 2 * overlap, out_end + overlap)
                segment = padded[:, :, read_start:read_end]
                segment_size = (read_end - read_start, size[-2], size[-1])
                segment_out = self._forward_segment(segment, scale, segment_size)
                valid_start = (out_start + overlap) - read_start
                valid_end = valid_start + (out_end - out_start)
                valid = segment_out[:, :, valid_start:valid_end]
                valid_length = out_end - out_start
                weight = torch.ones(valid_length, device=x.device, dtype=x.dtype)
                if core_start > out_start:
                    blend = core_start - out_start
                    weight[:blend] = torch.arange(
                        1, blend + 1, device=x.device, dtype=x.dtype
                    ) / (blend + 1)
                if out_end > core_end:
                    blend = out_end - core_end
                    weight[-blend:] = torch.arange(
                        blend, 0, -1, device=x.device, dtype=x.dtype
                    ) / (blend + 1)
                shaped_weight = weight.view(1, 1, valid_length, 1, 1)
                accumulated[:, :, out_start:out_end] += valid * shaped_weight
                accumulated_weight[:, :, out_start:out_end] += shaped_weight
                start += chunk
            return accumulated / accumulated_weight.clamp_min(1e-8)

        return self._forward_segment(x, scale, size)

    def _forward_segment(self, x, scale, size):

        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, _ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, _ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


_MODEL_CACHE = {}


def _get_models_dir():
    return folder_paths.get_folder_paths(_LATENT_UPSCALE_FOLDER)[0]


def _scan_models():
    files = []
    model_dir = _get_models_dir()
    for ext in ("*.pth", "*.safetensors"):
        files.extend(glob.glob(os.path.join(model_dir, ext)))
    names = sorted(os.path.basename(f) for f in files)
    if not names:
        return [f"(no upscale models found in: {model_dir})"]
    return names


def _load_raw_sd(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path, device='cpu')
    else:
        sd = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    sd = {k: v.to(torch.float16) if v.dtype == torch.float8_e4m3fn else v
          for k, v in sd.items()}
    return sd


def _extract_upscaler_sd(sd):
    if any(k.startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    return sd


def _detect_arch(sd):
    cfg = {
        "in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
        "dropout": 0.1, "attn": False, "temporal_every": 2, "temporal_kernel": 5,
    }
    conv_key = 'conv_in.weight'
    if conv_key in sd:
        cfg["in_channels"] = sd[conv_key].shape[1]
        cfg["channels"] = sd[conv_key].shape[0]

    in_ids, out_ids = set(), set()
    temporal_in_indices, temporal_out_indices = set(), set()
    for k in sd.keys():
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m:
            in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m:
            out_ids.add(int(m.group(1)))
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m:
            temporal_in_indices.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.dwconv\.weight', k)
        if m:
            temporal_out_indices.add(int(m.group(1)))

    if in_ids:
        cfg["in_blocks"] = len(in_ids)
    if out_ids:
        cfg["out_blocks"] = len(out_ids)

    if temporal_in_indices or temporal_out_indices:
        cfg["temporal_every"] = 2
        for k in sd.keys():
            if 'dwconv.weight' in k and k.endswith('dwconv.weight'):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0

    cfg["attn"] = False
    return cfg


def load_upscale_model(name, device, precision):
    cache_key = f"{name}::{device}::{precision}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key].to(device)

    path = os.path.join(_get_models_dir(), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo do modelo de upscale não encontrado: {path}")

    raw_sd = _load_raw_sd(path)
    up_sd = _extract_upscaler_sd(raw_sd)
    cfg = _detect_arch(up_sd)
    if cfg["in_channels"] != 24:
        raise ValueError(
            f"O checkpoint '{name}' não é um upscaler latent do H3 "
            f"(eram esperados 24 canais de entrada, mas foram encontrados {cfg['in_channels']})."
        )

    model = _LatentResizer3D(
        in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
        channels=cfg["channels"], dropout=cfg["dropout"], attn=cfg["attn"],
        temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"],
    )
    model.load_state_dict(up_sd, strict=True)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)
    model = model.to(device).eval().requires_grad_(False)
    if dtype != torch.float32:
        model = model.to(dtype)

    _MODEL_CACHE[cache_key] = model
    print(f"[Bruxos H3 Ultimate] Modelo de upscale carregado: {name}")
    return model


def unload_upscale_model(name, device, precision):
    """Free VRAM after upscaling: move the cached upscale model back to CPU. It stays
    in _MODEL_CACHE so the next chunk only re-copies weights to GPU, not re-reads disk."""
    cache_key = f"{name}::{device}::{precision}"
    model = _MODEL_CACHE.get(cache_key)
    if model is not None and str(next(model.parameters()).device) != "cpu":
        model.to("cpu")
        print(f"[Bruxos H3 Ultimate] Modelo de upscale retirado da memória: {name}")
    if str(device) == "cuda":
        torch.cuda.empty_cache()


def _compute_upscale_target(width, height, h_in, w_in):
    """Pixel target W/H + effective scale from EXPLICIT target dimensions.

    The upscale target is always an exact pixel size (it must match the
    conditioning's generation size)."""
    ds = VAE_DOWNSAMPLE
    w_px = float(width)
    h_px = float(height)
    eff = (w_px / (w_in * ds) + h_px / (h_in * ds)) / 2.0

    w_px_f = round(w_px / ds) * ds
    h_px_f = round(h_px / ds) * ds
    w_out = max(1, int(w_px_f // ds))
    h_out = max(1, int(h_px_f // ds))
    return h_out, w_out, eff


def _compute_megapixel_target(megapixels, h_in, w_in, align=32):
    """Preserva o aspecto e produz pixels realmente múltiplos de 32.

    O projeto de origem arredondava internamente por 16 apesar de anunciar 32.
    Aqui o alinhamento ocorre em pixels antes da conversão para o latente /16,
    garantindo H/W latentes pares para o patch 2x2 do DiT H3.
    """
    target_pixels = max(0.01, float(megapixels)) * 1_000_000.0
    aspect = float(w_in) / max(1.0, float(h_in))
    height = math.sqrt(target_pixels / aspect)
    width = height * aspect
    grid = max(32, int(align))
    grid = max(32, int(math.ceil(grid / 32.0)) * 32)
    width = max(grid, int(math.ceil(width / grid)) * grid)
    height = max(grid, int(math.ceil(height / grid)) * grid)
    return _compute_upscale_target(width, height, h_in, w_in)


def upscale_video(video, param):
    """Upscale one chunk's video latent with the H3 3D upscaler. Audio untouched.

    Returns (upscaled_video, new_h, new_w). The target is computed in pixel
    space (explicit width/height, snapped to the VAE 16x grid), then the
    H3 network resizes to it. scale 1.0 (or an equivalent target) is a no-op."""
    model_name = param["model_name"]
    width = int(param.get("width", 0))
    height = int(param.get("height", 0))
    device = param["device"]
    precision = param["precision"]

    orig_dtype = video.dtype
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    compute_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]

    _, c, t, h_in, w_in = video.shape
    if "target_megapixels" in param:
        h_out, w_out, eff = _compute_megapixel_target(
            param["target_megapixels"], h_in, w_in, param.get("align", 32)
        )
    else:
        h_out, w_out, eff = _compute_upscale_target(width, height, h_in, w_in)

    if eff < 1.0 and (w_out < w_in or h_out < h_in):
        raise ValueError("Este modelo aceita somente aumento de resolução (escala efetiva maior ou igual a 1,0).")
    if w_out == w_in and h_out == h_in:
        return video, h_in, w_in

    if str(model_name).startswith('('):
        raise ValueError("Coloque os modelos de upscale H3 na pasta 'latent_upscale_models'.")

    s = video.to(device=dev, dtype=compute_dtype, copy=True)
    model = load_upscale_model(model_name, dev, precision)
    norm_mean, norm_std = _make_norm_tensors(dev, compute_dtype)

    with torch.inference_mode():
        s = s.sub(norm_mean).div(norm_std)
        out = model(
            s, scale=eff, target_size=(t, h_out, w_out),
            enable_chunking=bool(param.get("enable_model_chunking", True)),
        )
        del s
        out = out.mul(norm_std).add(norm_mean)

    out = out.to(device="cpu", dtype=orig_dtype)
    unload_upscale_model(model_name, dev, precision)
    return out, h_out, w_out


def upscale_video_interp(video, param):
    """Model-free upscale of one chunk's video latent via interpolation (audio
    untouched) - mirrors ComfyUI's 'Upscale Latent' node. Returns (upscaled_video,
    new_h, new_w); the video latent [B,24,T,H,W] is resized in HxW only."""
    method = param["method"]
    width = int(param.get("width", 0))
    height = int(param.get("height", 0))

    _, c, t, h_in, w_in = video.shape
    if "target_megapixels" in param:
        h_out, w_out, _ = _compute_megapixel_target(
            param["target_megapixels"], h_in, w_in, param.get("align", 32)
        )
    else:
        h_out, w_out, _ = _compute_upscale_target(width, height, h_in, w_in)
    if h_out == h_in and w_out == w_in:
        return video, h_in, w_in

    video_bt = video.permute(0, 2, 1, 3, 4).reshape(-1, c, h_in, w_in)
    up = torch.nn.functional.interpolate(video_bt, size=(h_out, w_out), mode=method)
    up = up.reshape(video.shape[0], t, c, h_out, w_out).permute(0, 2, 1, 3, 4).contiguous()
    return up, h_out, w_out


def upscale_latent(video, param):
    """Dispatch a chunk's video upscale: H3 3D model (param has 'model_name') or
    model-free interpolation (param has 'method'). Audio is never touched."""
    if "model_name" in param:
        return upscale_video(video, param)
    return upscale_video_interp(video, param)


# ---------------------------------------------------------------------------
# sampling helpers
# ---------------------------------------------------------------------------

def build_guider(model, cond, negative, cfg):
    guider = comfy.samplers.CFGGuider(model)
    if negative is not None:
        guider.set_conds(cond, negative)
        guider.set_cfg(cfg)
    else:
        guider.inner_set_conds({"positive": cond})
    return guider


def sample_piece(piece, cond, model, noise, sampler, sigmas, negative, cfg):
    """Sample one piece (full chunk or tile). Mirrors SamplerCustomAdvanced,
    including the x0 preview callback. Returns nested samples (video+audio)."""
    latent = dict(piece)
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    latent["samples"] = latent_image
    noise_mask = latent.get("noise_mask")

    guider = build_guider(model, cond, negative, cfg)
    x0_output = {}
    callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = guider.sample(
        noise.generate_noise(latent), latent_image, sampler, sigmas,
        denoise_mask=noise_mask, callback=callback,
        disable_pbar=disable_pbar, seed=noise.seed,
    )
    samples = samples.to(comfy.model_management.intermediate_device())
    return samples


# ---------------------------------------------------------------------------
# stitching helpers
# ---------------------------------------------------------------------------

def temporal_append(acc_v, acc_a, chunk_v, chunk_a, index, k0, f0):
    """Stitch one re-sampled chunk into the accumulated latent (cross-fade).
    Mirrors 'Append MiniMax H3 Latents'. Returns (result_v, result_a)."""
    if acc_v is None:
        return chunk_v, chunk_a

    gi = k0
    agi = round(f0 * FRAME_RESCALE)
    total_v = max(acc_v.shape[2], gi + chunk_v.shape[2])
    total_a = max(acc_a.shape[-1], agi + chunk_a.shape[-1])
    result_v = torch.zeros((1, acc_v.shape[1], total_v, acc_v.shape[3], acc_v.shape[4]),
                           device=acc_v.device, dtype=acc_v.dtype)
    result_a = torch.zeros((1, 32, 2, total_a), device=acc_a.device, dtype=acc_a.dtype)
    result_v[:, :, :acc_v.shape[2]] = acc_v
    result_a[:, :, :, :acc_a.shape[-1]] = acc_a

    v = chunk_v
    a = chunk_a
    ov = (acc_v.shape[2] - gi) if index > 0 else 0
    if ov > 0:
        ov = min(ov, v.shape[2])
        tail = result_v[:, :, gi:gi + ov].clone()
        result_v[:, :, gi:gi + ov] = _crossfade(tail, v[:, :, :ov], dim=2)
        v = v[:, :, ov:]
    write_v = gi + max(ov, 0)
    if v.shape[2] > 0:
        result_v[:, :, write_v:write_v + v.shape[2]] = v

    ova = (acc_a.shape[-1] - agi) if index > 0 else 0
    if ova > 0:
        ova = min(ova, a.shape[-1])
        tail = result_a[:, :, :, agi:agi + ova].clone()
        result_a[:, :, :, agi:agi + ova] = _crossfade(tail, a[:, :, :, :ova], dim=3)
        a = a[:, :, :, ova:]
    write_a = agi + max(ova, 0)
    if a.shape[-1] > 0:
        result_a[:, :, :, write_a:write_a + a.shape[-1]] = a

    return result_v, result_a


def spatial_process(chunk_v, chunk_a, cond, sp, model, noise, sampler, sigmas, negative, cfg):
    """Inner loop: spatial split -> per-tile sampling -> spatial stitch.
    Mirrors the spatial split/extract/append trio. Audio is carried unchanged
    (frozen in every tile, never re-sampled). Returns (reassembled_video, info)."""
    tw = int(sp["tile_width"]) // 16
    th = int(sp["tile_height"]) // 16
    ol_w = int(sp["spatial_w_overlap"]) // 16
    ol_h = int(sp["spatial_h_overlap"]) // 16
    fw = int(sp["fade_width"]) // 16
    fh = int(sp["fade_height"]) // 16
    min_tile = int(sp["min_tile_size"]) // 16
    overlap_mode = sp["overlap_mode"]
    overlap_blend = sp["overlap_blend"]
    context_margin = max(0, int(sp.get("context_margin", 128)) // 16)
    global_reference = bool(sp.get("global_reference", False))
    global_reference_max_side = max(64, int(sp.get("global_reference_max_side", 512)))
    global_reference_frames = str(sp.get("global_reference_frames", "middle"))
    reference_policy = str(sp.get("reference_policy", "safe_strip_latent_refs"))

    if tw <= 0 or th <= 0:
        raise ValueError("A largura e a altura dos tiles precisam ser múltiplas de 32 pixels.")
    if ol_w >= tw or ol_h >= th:
        raise ValueError("A sobreposição horizontal/vertical precisa ser menor que o tamanho do tile.")
    if min_tile > th or min_tile > tw:
        raise ValueError("O tamanho mínimo do tile não pode ser maior que o tamanho definido para ele.")

    _, c, t, h, w = chunk_v.shape
    rows, cols, trows, tcols, row_ovl, col_ovl = compute_spatial_grid(h, w, th, tw, ol_h, ol_w, min_tile, min_tile)
    nrows, ncols = len(rows), len(cols)
    ta = chunk_a.shape[-1]

    acc_v = chunk_v.clone()
    tile_info = {
        "rows": rows, "cols": cols, "tile_h": th, "tile_w": tw,
        "overlap_h": ol_h, "overlap_w": ol_w,
        "row_overlaps": row_ovl, "col_overlaps": col_ovl, "min_tile": min_tile,
        "tile_rows": trows, "tile_cols": tcols, "n_cols": ncols,
        "orig_h": h, "orig_w": w, "overlap_mode": overlap_mode, "overlap_blend": overlap_blend,
        "context_margin": context_margin,
        "global_reference": global_reference,
        "global_reference_max_side": global_reference_max_side,
        "global_reference_frames": global_reference_frames,
        "reference_policy": reference_policy,
        "positive_refs_in": count_minimax_refs(cond),
        "negative_refs_in": count_minimax_refs(negative),
    }

    cond_with_global = (
        add_global_context_refs(
            cond, chunk_v, global_reference_max_side, global_reference_frames
        ) if global_reference else cond
    )

    total_refs = count_minimax_refs(cond_with_global) + count_minimax_refs(negative)
    if total_refs and reference_policy == "keep_all_refs" and (nrows > 1 or ncols > 1):
        print(
            "[MMH3-UltimateUpscale/Bruxos] AVISO: keep_all_refs em múltiplos tiles "
            f"reapresentará {total_refs} minimax_ref(s) a cada tile e pode duplicar "
            "personagens/padrões. Prefira safe_strip_latent_refs.",
            flush=True,
        )

    refs_removed_total = 0

    if overlap_mode == "context_only":
        # Um único tile é dono de cada posição. Tiles seguintes recebem a área
        # já refinada e uma margem ao redor como contexto congelado, mas só
        # amostram pixels ainda não processados dentro do seu core. Não há duas
        # versões conflitantes para misturar depois, logo não há linha de blend.
        processed = torch.zeros((h, w), device=chunk_v.device, dtype=torch.bool)
        for i in range(nrows):
            for j in range(ncols):
                r0, c0 = rows[i], cols[j]
                tr, tc = trows[i], tcols[j]
                r1, c1 = r0 + tr, c0 + tc

                pr0 = max(0, r0 - context_margin)
                pc0 = max(0, c0 - context_margin)
                pr1 = min(h, r1 + context_margin)
                pc1 = min(w, c1 + context_margin)
                ptr, ptc = pr1 - pr0, pc1 - pc0

                tile = chunk_v[:, :, :, pr0:pr1, pc0:pc1].clone()
                processed_patch = processed[pr0:pr1, pc0:pc1]
                accumulated_patch = acc_v[:, :, :, pr0:pr1, pc0:pc1]
                tile = torch.where(
                    processed_patch[None, None, None], accumulated_patch, tile
                )

                local_r0, local_c0 = r0 - pr0, c0 - pc0
                core = torch.zeros((ptr, ptc), device=chunk_v.device, dtype=torch.bool)
                core[local_r0:local_r0 + tr, local_c0:local_c0 + tc] = True
                sample_area = core & ~processed_patch

                mv = sample_area.to(torch.float32)[None, None, None]
                ma = torch.zeros((1, 32, 2, ta), device=chunk_a.device, dtype=chunk_a.dtype)
                piece = {
                    "samples": comfy.nested_tensor.NestedTensor((tile, chunk_a)),
                    "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma)),
                }

                cond_tile, removed_pos = prepare_tile_conditioning(
                    cond_with_global, h, w, pr0, pc0, ptr, ptc,
                    reference_policy, first_tile=(i == 0 and j == 0),
                )
                negative_tile, removed_neg = prepare_tile_conditioning(
                    negative, h, w, pr0, pc0, ptr, ptc,
                    reference_policy, first_tile=(i == 0 and j == 0),
                )
                refs_removed_total += removed_pos + removed_neg
                out = sample_piece(
                    piece, cond_tile, model, noise, sampler, sigmas,
                    negative_tile, cfg,
                )
                tile_v = out.tensors[0]
                core_v = tile_v[:, :, :,
                                local_r0:local_r0 + tr,
                                local_c0:local_c0 + tc]

                already = processed[r0:r1, c0:c1]
                region = acc_v[:, :, :, r0:r1, c0:c1]
                acc_v[:, :, :, r0:r1, c0:c1] = torch.where(
                    already[None, None, None], region, core_v
                )
                processed[r0:r1, c0:c1] = True

        tile_info["processed_coverage"] = float(processed.float().mean().item())
        tile_info["refs_removed_across_tiles"] = refs_removed_total
        print(
            "[MMH3-UltimateUpscale/Bruxos] spatial "
            f"{nrows}x{ncols} context_only | margin={context_margin * 16}px | "
            f"reference_policy={reference_policy} | refs removidas={refs_removed_total}",
            flush=True,
        )
        return acc_v, tile_info

    for i in range(nrows):
        for j in range(ncols):
            r0, c0 = rows[i], cols[j]
            tr, tc = trows[i], tcols[j]
            ovh = row_ovl[i]
            ovw = col_ovl[j]

            tile = torch.zeros((1, c, t, tr, tc), device=chunk_v.device, dtype=chunk_v.dtype)
            tile[:, :, :, :, :] = chunk_v[:, :, :, r0:r0 + tr, c0:c0 + tc]
            # pre-fill done-overlap strips from the accumulated re-sampled result
            if j > 0 and ovw > 0:
                tile[:, :, :, :, :ovw] = acc_v[:, :, :, r0:r0 + tr, c0:c0 + ovw]
            if i > 0 and ovh > 0:
                tile[:, :, :, :ovh, :] = acc_v[:, :, :, r0:r0 + ovh, c0:c0 + tc]

            m = spatial_fade_mask(tr, tc, ovh, ovw,
                                  done_top=(i > 0), done_left=(j > 0),
                                  fade_h=fh, fade_w=fw)
            mv = m[None, None, None]
            ma = torch.zeros((1, 32, 2, ta), device=chunk_a.device, dtype=chunk_a.dtype)
            piece = {
                "samples": comfy.nested_tensor.NestedTensor((tile, chunk_a)),
                "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma)),
            }

            cond_tile, removed_pos = prepare_tile_conditioning(
                cond_with_global, h, w, r0, c0, tr, tc,
                reference_policy, first_tile=(i == 0 and j == 0),
            )
            negative_tile, removed_neg = prepare_tile_conditioning(
                negative, h, w, r0, c0, tr, tc,
                reference_policy, first_tile=(i == 0 and j == 0),
            )
            refs_removed_total += removed_pos + removed_neg
            out = sample_piece(
                piece, cond_tile, model, noise, sampler, sigmas,
                negative_tile, cfg,
            )
            tile_v = out.tensors[0]

            region = acc_v[:, :, :, r0:r0 + tr, c0:c0 + tc].clone()
            if j > 0 and ovw > 0:
                tt = torch.linspace(0.0, 1.0, ovw, device=region.device, dtype=region.dtype)
                wts = blend_weights(tt, overlap_blend, overlap_mode)
                region[:, :, :, :, :ovw] = (region[:, :, :, :, :ovw] * (1.0 - wts[None, None, None, None, :])
                                            + tile_v[:, :, :, :, :ovw] * wts[None, None, None, None, :])
            if i > 0 and ovh > 0:
                tt = torch.linspace(0.0, 1.0, ovh, device=region.device, dtype=region.dtype)
                wts = blend_weights(tt, overlap_blend, overlap_mode)
                region[:, :, :, :ovh, :] = (region[:, :, :, :ovh, :] * (1.0 - wts[None, None, None, :, None])
                                            + tile_v[:, :, :, :ovh, :] * wts[None, None, None, :, None])
            band = torch.zeros((1, 1, 1, tr, tc), device=region.device, dtype=torch.bool)
            if j > 0 and ovw > 0:
                band[:, :, :, :, :ovw] = True
            if i > 0 and ovh > 0:
                band[:, :, :, :ovh, :] = True
            region = torch.where(band, region, tile_v)
            acc_v[:, :, :, r0:r0 + tr, c0:c0 + tc] = region

    tile_info["refs_removed_across_tiles"] = refs_removed_total
    print(
        "[MMH3-UltimateUpscale/Bruxos] spatial "
        f"{nrows}x{ncols} {overlap_mode}/{overlap_blend} | "
        f"reference_policy={reference_policy} | refs removidas={refs_removed_total}",
        flush=True,
    )
    return acc_v, tile_info


# ---------------------------------------------------------------------------
# parameter nodes
# ---------------------------------------------------------------------------

class MMH3LatentUpscaleWithModelParams(io.ComfyNode):
    """Agrupa as configurações do upscale neural 3D usadas pelo H3 Ultimate."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BruxosH3UltimateModelParams",
            display_name="H3 Ultimate · Upscale Neural por MP (Bruxos)",
            category=CAT,
            description=(
                "Configura o upscale neural 3D do MiniMax H3 usado pelo node H3 Ultimate. "
                "Escolha o modelo e informe a área final em megapixels; a proporção do vídeo "
                "é preservada. Os checkpoints minimax_h3_latent_upscaler_3d devem ficar na "
                "pasta latent_upscale_models e não funcionam no carregador LatentUpscale comum."
            ),
            search_aliases=["h3 upscale params", "upscale param", "h3 upscale"],
            inputs=[
                io.Combo.Input("model_name", options=_scan_models(),
                               tooltip="Modelo neural H3 localizado em latent_upscale_models. Use um arquivo minimax_h3_latent_upscaler_3d compatível; outro tipo de modelo causará erro."),
                io.Float.Input("target_megapixels", default=1.0, min=0.1, max=16.0, step=0.1,
                               tooltip="Área alvo em megapixels. O aspecto original é preservado e W/H finais são alinhados a múltiplos de 32."),
                io.Int.Input("align", default=32, min=32, max=512, step=32,
                             tooltip="Grade final em pixels. 32 é o valor correto para o patch 2x2 do H3."),
                io.Combo.Input("device", options=["cuda", "cpu"], default="cuda",
                               tooltip="Dispositivo que executará o upscale. CUDA é mais rápido; CPU é muito mais lenta e usa memória RAM."),
                io.Combo.Input("precision", options=["fp16", "fp32", "bf16"], default="fp16",
                               tooltip="Precisão do modelo. FP16 é o padrão recomendado para economizar VRAM; FP32 usa mais memória."),
                io.Combo.Input(
                    "conditioning_size_mode",
                    options=["exigir_tamanho_final", "redimensionar_fallback"],
                    default="exigir_tamanho_final",
                    tooltip="exigir_tamanho_final (recomendado) interrompe antes da amostragem se os keyframes estiverem na resolução antiga. redimensionar_fallback interpola esses latents, mas pode criar grades e padrões repetidos.",
                ),
                io.Boolean.Input(
                    "enable_model_chunking",
                    default=True,
                    tooltip="Ligado divide internamente latents longos em janelas com contexto replicado e mistura ponderada, economizando VRAM e evitando flicker nas bordas. Desligue apenas em clipes curtos para usar contexto temporal completo.",
                ),
            ],
            outputs=[
                H3_UPSCALE_PARAM.Output("latent_upscale_param",
                                        tooltip="Configuração pronta para ligar na entrada latent_upscale_param do H3 Ultimate."),
            ],
        )

    @classmethod
    def execute(cls, model_name, target_megapixels, align, device, precision,
                conditioning_size_mode="exigir_tamanho_final",
                enable_model_chunking=True) -> io.NodeOutput:
        param = {
            "model_name": model_name,
            "target_megapixels": float(target_megapixels),
            "align": int(align),
            "device": device,
            "precision": precision,
            "conditioning_size_mode": conditioning_size_mode,
            "enable_model_chunking": bool(enable_model_chunking),
        }
        return io.NodeOutput(param)


class MMH3LatentUpscaleParams(io.ComfyNode):
    """Agrupa o upscale por interpolação usado pelo H3 Ultimate. Redimensiona
    somente o latent de vídeo e mantém intacta a estrutura aninhada de áudio."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BruxosH3UltimateInterpParams",
            display_name="H3 Ultimate · Interpolação por MP (Bruxos)",
            category=CAT,
            description=(
                "Configura um redimensionamento simples do latent por interpolação, sem "
                "carregar modelo neural. Apenas a parte de vídeo muda; o áudio é preservado. "
                "É mais rápido, mas não recupera detalhes como o upscaler neural. O tamanho "
                "final precisa corresponder ao tamanho usado no conditioning."
            ),
            search_aliases=["h3 upscale params", "upscale param", "h3 latent upscale", "model-free upscale"],
            inputs=[
                io.Combo.Input("method", options=["nearest-exact", "bilinear", "area", "bicubic"],
                               default="bilinear",
                               tooltip="Método usado para redimensionar largura e altura do latent. Bilinear é o padrão equilibrado; bicubic tende a ser mais suave."),
                io.Float.Input("target_megapixels", default=1.0, min=0.1, max=16.0, step=0.1,
                               tooltip="Área final aproximada em megapixels, preservando a proporção original."),
                io.Int.Input("align", default=32, min=32, max=512, step=32,
                             tooltip="Alinha largura e altura finais a esta grade. Para MiniMax H3, use 32."),
                io.Combo.Input(
                    "conditioning_size_mode",
                    options=["exigir_tamanho_final", "redimensionar_fallback"],
                    default="exigir_tamanho_final",
                    tooltip="exigir_tamanho_final (recomendado) exige keyframes já codificados na resolução final. redimensionar_fallback evita o erro por interpolação, porém pode introduzir grades e padrões.",
                ),
            ],
            outputs=[
                H3_UPSCALE_PARAM.Output("latent_upscale_param",
                                        tooltip="Configuração de interpolação pronta para ligar no H3 Ultimate."),
            ],
        )

    @classmethod
    def execute(cls, method, target_megapixels, align,
                conditioning_size_mode="exigir_tamanho_final") -> io.NodeOutput:
        param = {
            "method": method,
            "target_megapixels": float(target_megapixels),
            "align": int(align),
            "conditioning_size_mode": conditioning_size_mode,
        }
        return io.NodeOutput(param)


class MMH3TemporalSplitParams(io.ComfyNode):
    """Agrupa as configurações dos chunks temporais usadas pelo H3 Ultimate."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BruxosH3UltimateTemporalParams",
            display_name="H3 Ultimate · Chunks Temporais (Bruxos)",
            category=CAT,
            description=(
                "Divide vídeos longos em chunks temporais sobrepostos para o H3 Ultimate. "
                "Isso reduz o pico de VRAM e permite processar centenas de frames. A "
                "sobreposição e a âncora ajudam a esconder cortes entre os chunks."
            ),
            search_aliases=["h3 temporal params", "temporal split param", "time split"],
            inputs=[
                io.Int.Input("chunk_length", default=136, min=17, max=100000, step=17,
                             tooltip="Quantidade de frames por chunk. Precisa ser múltipla de 17 por causa da grade temporal do H3. Em 24 fps: 136 ≈ 5,7 s e 153 ≈ 6,4 s."),
                io.Int.Input("temporal_overlap", default=17, min=0, max=100000, step=17,
                             tooltip="Frames compartilhados entre chunks consecutivos. Precisa ser múltiplo de 17; use 17 como ponto inicial."),
                io.Float.Input("anchor_strength", default=0.999, min=0.0, max=1.0, step=0.01,
                               tooltip="Força usada para manter a continuidade com o chunk anterior. 1,0 preserva exatamente a borda anterior; 0,999 é recomendado; 0 desativa a âncora."),
            ],
            outputs=[
                H3_TEMPORAL_PARAM.Output("temporal_split_param",
                                         tooltip="Configuração de chunks pronta para ligar na entrada temporal_split_param do H3 Ultimate."),
            ],
        )

    @classmethod
    def execute(cls, chunk_length, temporal_overlap, anchor_strength) -> io.NodeOutput:
        if chunk_length % 17 != 0:
            raise ValueError(f"O tamanho do chunk precisa ser múltiplo de 17, conforme a grade temporal do H3. Valor recebido: {chunk_length}.")
        if temporal_overlap % 17 != 0:
            raise ValueError(f"A sobreposição temporal precisa ser múltipla de 17. Valor recebido: {temporal_overlap}.")
        param = {
            "chunk_length": chunk_length,
            "temporal_overlap": temporal_overlap,
            "anchor_strength": anchor_strength,
        }
        return io.NodeOutput(param)


class MMH3SpatialSplitParams(io.ComfyNode):
    """Agrupa as configurações dos tiles espaciais usadas pelo H3 Ultimate."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BruxosH3UltimateSpatialParams",
            display_name="H3 Ultimate · Tiles Espaciais (Bruxos)",
            category=CAT,
            description=(
                "Configura tiles espaciais para processar resoluções maiores com menos VRAM. "
                "Define tamanho, sobreposição, transição e contexto de cada região. Tiles podem "
                "criar emendas ou repetir sujeitos; use context_only e safe_strip_latent_refs "
                "como configuração segura inicial."
            ),
            search_aliases=["h3 spatial params", "spatial split param", "tile param"],
            inputs=[
                io.Int.Input("tile_width", default=512, min=32, max=100000, step=32,
                             tooltip="Largura de cada tile em pixels na resolução final. Precisa ser múltipla de 32."),
                io.Int.Input("tile_height", default=512, min=32, max=100000, step=32,
                             tooltip="Altura de cada tile em pixels na resolução final. Precisa ser múltipla de 32."),
                io.Int.Input("spatial_w_overlap", default=128, min=0, max=100000, step=32,
                             tooltip="Sobreposição horizontal entre tiles vizinhos, em pixels. Precisa ser múltipla de 32 e menor que a largura do tile."),
                io.Int.Input("spatial_h_overlap", default=128, min=0, max=100000, step=32,
                             tooltip="Sobreposição vertical entre tiles vizinhos, em pixels. Precisa ser múltipla de 32 e menor que a altura do tile."),
                io.Int.Input("fade_width", default=32, min=0, max=100000, step=32,
                             tooltip="Faixa horizontal de transição da área congelada para a área livre. 32 é o padrão seguro para reduzir emendas."),
                io.Int.Input("fade_height", default=32, min=0, max=100000, step=32,
                             tooltip="Faixa vertical de transição da área congelada para a área livre. 32 é o padrão seguro para reduzir emendas."),
                io.Int.Input("min_tile_size", default=256, min=0, max=100000, step=32,
                             tooltip="Menor tamanho permitido para tiles nas bordas. Evita uma sobra muito estreita; o último tile é reposicionado e sua sobreposição aumenta. Não pode exceder o tamanho do tile."),
                io.Combo.Input("overlap_mode", options=["context_only", "earlier", "later"], default="context_only",
                               tooltip="context_only (recomendado): overlap e margem já processados entram como contexto congelado e nunca são redesenhados. earlier/later mantêm o blend legado."),
                io.Combo.Input("overlap_blend", options=["linear", "smoothstep", "overwrite", "midpoint"], default="linear",
                               tooltip="Transição usada para unir a sobreposição: linear mistura gradualmente; smoothstep suaviza a curva; overwrite substitui toda a faixa; midpoint faz um corte no meio."),
                io.Int.Input("context_margin", default=128, min=0, max=2048, step=32,
                             tooltip="Margem PIXEL extra ao redor do tile. Ela participa da atenção, mas fica congelada; 128 é o ponto inicial recomendado."),
                io.Boolean.Input("global_reference", default=False,
                                 tooltip="EXPERIMENTAL. Injeta uma visão reduzida como minimax_ref. Pode duplicar personagens/padrões; mantenha desligado no uso normal."),
                io.Int.Input("global_reference_max_side", default=512, min=64, max=2048, step=64,
                             tooltip="Maior lado em PIXELS da referência global reduzida. 512 mantém contexto sem inflar demais a atenção."),
                io.Combo.Input("global_reference_frames", options=["middle", "first_middle_last"], default="middle",
                               tooltip="Quando a referência experimental estiver ligada: um quadro central é menos propenso a duplicações; três quadros são apenas para diagnóstico."),
                io.Combo.Input(
                    "reference_policy",
                    options=["safe_strip_latent_refs", "first_tile_only", "keep_all_refs"],
                    default="safe_strip_latent_refs",
                    tooltip="Ref2VA tiled: safe_strip_latent_refs remove somente os blocos latentes minimax_refs de cada tile e preserva a embedding Qwen, evitando repetir o sujeito. first_tile_only é experimental; keep_all_refs mantém o comportamento upstream e pode duplicar padrões.",
                ),
            ],
            outputs=[
                H3_SPATIAL_PARAM.Output("spatial_split_param",
                                        tooltip="Configuração de tiles pronta para ligar na entrada spatial_split_param do H3 Ultimate."),
            ],
        )

    @classmethod
    def execute(cls, tile_width, tile_height, spatial_w_overlap, spatial_h_overlap,
                fade_width, fade_height, min_tile_size, overlap_mode, overlap_blend,
                context_margin=128, global_reference=False,
                global_reference_max_side=512,
                global_reference_frames="middle",
                reference_policy="safe_strip_latent_refs") -> io.NodeOutput:
        for name, v in (("tile_width", tile_width), ("tile_height", tile_height),
                        ("spatial_w_overlap", spatial_w_overlap), ("spatial_h_overlap", spatial_h_overlap),
                        ("fade_width", fade_width), ("fade_height", fade_height),
                        ("min_tile_size", min_tile_size),
                        ("context_margin", context_margin)):
            if v % 32 != 0:
                raise ValueError(f"O campo '{name}' precisa ser múltiplo de 32 pixels por causa da grade latent do H3. Valor recebido: {v}.")
        if spatial_w_overlap >= tile_width:
            raise ValueError("A sobreposição horizontal precisa ser menor que a largura do tile.")
        if spatial_h_overlap >= tile_height:
            raise ValueError("A sobreposição vertical precisa ser menor que a altura do tile.")
        if fade_width > spatial_w_overlap:
            raise ValueError("A transição horizontal não pode exceder a sobreposição horizontal.")
        if fade_height > spatial_h_overlap:
            raise ValueError("A transição vertical não pode exceder a sobreposição vertical.")
        if min_tile_size > tile_width or min_tile_size > tile_height:
            raise ValueError("O tamanho mínimo não pode ser maior que a largura ou a altura do tile.")
        param = {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "spatial_w_overlap": spatial_w_overlap,
            "spatial_h_overlap": spatial_h_overlap,
            "fade_width": fade_width,
            "fade_height": fade_height,
            "min_tile_size": min_tile_size,
            "overlap_mode": overlap_mode,
            "overlap_blend": overlap_blend,
            "context_margin": context_margin,
            "global_reference": global_reference,
            "global_reference_max_side": global_reference_max_side,
            "global_reference_frames": global_reference_frames,
            "reference_policy": reference_policy,
        }
        return io.NodeOutput(param)


# ---------------------------------------------------------------------------
# main node
# ---------------------------------------------------------------------------

class MMH3UltimateUpscale(io.ComfyNode):
    """Executa em um único node o refinamento completo do latent MiniMax H3."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BruxosH3UltimateUpscale",
            display_name="MiniMax H3 · Upscale Latent Completo (Bruxos)",
            category=CAT,
            description=(
                "Refina e aumenta um latent MiniMax H3 de vídeo e áudio já gerado. Pode "
                "dividir o vídeo em chunks temporais e, opcionalmente, em tiles espaciais "
                "para reduzir o uso máximo de VRAM. O node redimensiona, reamostra, une os "
                "tiles e depois une os chunks. As três entradas de configuração são opcionais: "
                "sem upscale o tamanho é mantido; sem chunks o vídeo inteiro é processado junto; "
                "sem tiles cada chunk é processado como uma imagem completa."
            ),
            search_aliases=["h3 ultimate upscale", "ultimate upscale", "h3 auto upscale", "h3 enhance"],
            inputs=[
                io.Model.Input("model", tooltip="Modelo MiniMax H3 usado para reamostrar cada chunk ou tile. O guider é criado internamente."),
                io.Conditioning.Input("conditioning",
                                      tooltip="Conditioning original da geração. Em cada chunk, o tempo é realinhado; em cada tile, os keyframes são recortados para a região correta."),
                io.Latent.Input("latent", tooltip="Latent MiniMax H3 AV já amostrado que será refinado e, se configurado, ampliado."),
                io.Noise.Input("noise", tooltip="Fonte de ruído. O node gera um tensor de ruído separado para cada chunk ou tile."),
                io.Sampler.Input("sampler", tooltip="Sampler utilizado para reamostrar todos os chunks e tiles."),
                io.Sigmas.Input("sigmas", tooltip="Agenda de sigmas utilizada em todos os chunks e tiles. Ela controla as etapas e a intensidade da reamostragem."),
                io.Conditioning.Input("negative", optional=True,
                                      tooltip="Conditioning negativo opcional. Quando ligado, ativa CFG com o valor abaixo; sem ele, o node usa apenas o conditioning positivo."),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,
                               tooltip="Escala CFG usada somente quando a entrada negative está conectada."),
                H3_UPSCALE_PARAM.Input("latent_upscale_param", optional=True,
                                       tooltip="Ligue o node de Upscale Neural por MP ou Interpolação por MP. Por segurança, gere o conditioning diretamente na resolução final; deixe vazio para manter a resolução do latent."),
                H3_TEMPORAL_PARAM.Input("temporal_split_param", optional=True,
                                        tooltip="Ligue o node Chunks Temporais para vídeos longos. Deixe vazio para processar todos os frames de uma vez."),
                H3_SPATIAL_PARAM.Input("spatial_split_param", optional=True,
                                       tooltip="Ligue o node Tiles Espaciais para economizar VRAM em alta resolução. Deixe vazio para processar cada chunk inteiro, sem tiles."),
            ],
            outputs=[
                io.Latent.Output("latent", tooltip="Latent MiniMax H3 AV final, já ampliado, reamostrado e unido."),
                io.Dict.Output("segments_info",
                               tooltip="Diagnóstico dos chunks: início, quantidade de frames, faixas de tokens de vídeo/áudio e informação de upscale."),
                io.Dict.Output("tiles_info",
                               tooltip="Diagnóstico dos tiles: posições, dimensões, sobreposições e modo usado para unir as regiões."),
            ],
        )

    @classmethod
    def execute(cls, latent, conditioning, model, noise, sampler, sigmas,
                negative=None, cfg=1.0,
                temporal_split_param=None, spatial_split_param=None,
                latent_upscale_param=None) -> io.NodeOutput:
        samples = latent["samples"]
        if not is_h3_av_latent(samples):
            raise ValueError("O H3 Ultimate precisa receber um latent AV do MiniMax H3 contendo vídeo [B,24,T,H,W] e áudio [B,32,2,T].")
        video = samples.tensors[0]
        audio = samples.tensors[1]
        if video.shape[0] != 1:
            raise ValueError("O H3 Ultimate processa um vídeo por execução. O latent recebido possui batch maior que 1.")

        tv = video.shape[2]
        ta = audio.shape[-1]

        if temporal_split_param is not None:
            chunk_length = int(temporal_split_param["chunk_length"])
            overlap = int(temporal_split_param["temporal_overlap"])
            bounds, frame_count = compute_segments(tv, chunk_length, overlap)
            anchor_strength = temporal_split_param["anchor_strength"]
        else:
            frame_count = frames_for_tokens(tv)
            bounds = [(0, 0, tv, frame_count)]
            anchor_strength = 0.999

        acc_v = None
        acc_a = None
        segments_debug = []
        tiles_debug = []

        for i, (k0, f0, k1, f1) in enumerate(bounds):
            chunk_v = video[:, :, k0:k1].contiguous()
            source_h, source_w = int(chunk_v.shape[3]), int(chunk_v.shape[4])
            a0, a1 = audio_range(f0, f1)
            a1 = min(a1, ta)
            chunk_a = audio[:, :, :, a0:a1].contiguous()

            # 1. upscale this chunk's video (audio untouched). While the 3D upscaler
            #    is on the GPU the diffusion model isn't needed, so offload it first
            #    to avoid H3 + upscaler resident simultaneously; the next sample
            #    reloads H3 automatically.
            upscaled = False
            if latent_upscale_param is not None:
                use_model = "model_name" in latent_upscale_param
                if use_model and str(latent_upscale_param["device"]) == "cuda" and hasattr(model, "clone_base_uuid"):
                    # the 3D upscaler is on the GPU during upscale; offload the
                    # diffusion model so they don't reside simultaneously
                    comfy.model_management.unload_model_and_clones(model, unload_additional_models=False)
                    comfy.model_management.soft_empty_cache()
                chunk_v, _, _ = upscale_latent(chunk_v, latent_upscale_param)
                upscaled = True

            # 2. Sincroniza referências e realinha o tempo. Keyframes do H3 usam
            #    patches 2x2: redimensioná-los em latent space pode gravar uma grade
            #    no resultado. O padrão seguro exige conditioning já codificado no
            #    tamanho final; a interpolação fica disponível apenas como fallback.
            scale_y = float(chunk_v.shape[3]) / max(1, source_h)
            scale_x = float(chunk_v.shape[4]) / max(1, source_w)
            ref_stats = []
            cond_scaled = _bruxos_sync_conditioning(
                conditioning, scale_y, scale_x, "bilinear", ref_stats
            )
            negative_scaled = _bruxos_sync_conditioning(
                negative, scale_y, scale_x, "bilinear", []
            )
            conditioning_size_mode = (
                latent_upscale_param.get("conditioning_size_mode", "exigir_tamanho_final")
                if latent_upscale_param is not None else "exigir_tamanho_final"
            )
            resize_spatial = (
                (chunk_v.shape[3], chunk_v.shape[4])
                if conditioning_size_mode == "redimensionar_fallback" else None
            )
            cond_i = reanchor_conditioning(
                cond_scaled, f0, f1, resize_spatial
            )
            negative_i = reanchor_conditioning(
                negative_scaled, f0, f1, resize_spatial
            )
            if conditioning_size_mode != "redimensionar_fallback":
                validate_keyframe_spatial_size(
                    cond_i, int(chunk_v.shape[3]), int(chunk_v.shape[4]), "conditioning positivo"
                )
                validate_keyframe_spatial_size(
                    negative_i, int(chunk_v.shape[3]), int(chunk_v.shape[4]), "conditioning negativo"
                )

            # 3. pin frame-0 keyframe to the previous chunk's re-sampled frame
            if i > 0 and acc_v is not None:
                cond_i = anchor_conditioning(cond_i, acc_v, f0, anchor_strength)

            # 4. inner loop: spatial split -> sample -> stitch
            if spatial_split_param is not None:
                chunk_out_v, tile_info = spatial_process(
                    chunk_v, chunk_a, cond_i, spatial_split_param,
                    model, noise, sampler, sigmas, negative_i, cfg,
                )
                tile_info = dict(tile_info)
                tile_info["chunk"] = i
                tiles_debug.append(tile_info)
            else:
                piece = {"samples": comfy.nested_tensor.NestedTensor((chunk_v, chunk_a))}
                out = sample_piece(piece, cond_i, model, noise, sampler, sigmas, negative_i, cfg)
                chunk_out_v = out.tensors[0]

            # 5. temporal stitch
            acc_v, acc_a = temporal_append(acc_v, acc_a, chunk_out_v, chunk_a, i, k0, f0)

            segments_debug.append({
                "chunk": i,
                "frame_start": f0,
                "frame_count": f1 - f0,
                "video_tokens": [k0, k1],
                "audio_tokens": list(audio_range(f0, f1)),
                "upscaled": upscaled,
                "spatial_h": chunk_v.shape[3],
                "spatial_w": chunk_v.shape[4],
                "reference_latents_synced": len(list(dict.fromkeys(ref_stats))),
                "conditioning_size_mode": conditioning_size_mode,
            })

        # all chunks sampled & stitched: the diffusion model is no longer needed,
        # unload it so the caller (e.g. VAE decode of the large latent) gets the VRAM
        if hasattr(model, "clone_base_uuid"):
            comfy.model_management.unload_model_and_clones(model, unload_additional_models=False)
            comfy.model_management.soft_empty_cache()

        out = {"samples": comfy.nested_tensor.NestedTensor((acc_v, acc_a))}
        return io.NodeOutput(out, segments_debug, tiles_debug)


NODE_CLASS_MAPPINGS = {
    "BruxosH3UltimateUpscale": MMH3UltimateUpscale,
    "BruxosH3UltimateModelParams": MMH3LatentUpscaleWithModelParams,
    "BruxosH3UltimateInterpParams": MMH3LatentUpscaleParams,
    "BruxosH3UltimateTemporalParams": MMH3TemporalSplitParams,
    "BruxosH3UltimateSpatialParams": MMH3SpatialSplitParams,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosH3UltimateUpscale": "MiniMax H3 · Upscale Latent Completo (Bruxos)",
    "BruxosH3UltimateModelParams": "H3 Ultimate · Upscale Neural por MP (Bruxos)",
    "BruxosH3UltimateInterpParams": "H3 Ultimate · Interpolação por MP (Bruxos)",
    "BruxosH3UltimateTemporalParams": "H3 Ultimate · Chunks Temporais (Bruxos)",
    "BruxosH3UltimateSpatialParams": "H3 Ultimate · Tiles Espaciais (Bruxos)",
}
