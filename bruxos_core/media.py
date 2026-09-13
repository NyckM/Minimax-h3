# Helpers de midia (decode/encode/resize de video) — copia das funcoes
# auxiliares do video_nodes.py do pacote unico. NAO registra nodes.
# Quem registra Load/Save Video e o ComfyUI-Bruxos-MediaIO.
"""
ComfyUI-Bruxos-do-VFX — Video I/O
=================================
Dois nodes no estilo do VideoHelperSuite, porem compativeis com o tipo VIDEO
nativo dos nodes "2.0" do ComfyUI:

  * Bruxos Load Video   -> images, video (VIDEO nativo), audio, fps, frame_count, video_info
  * Bruxos Save Video   -> exporta MP4 e/ou sequencia de PNG, criando pastas.

Backends de decode/encode em camadas: PyAV (vem com o ComfyUI) -> imageio-ffmpeg
-> OpenCV. O codigo degrada com elegancia se algum nao estiver disponivel.
"""

import os
import json
import hashlib
import logging
import datetime
from fractions import Fraction

import numpy as np
import torch

# ---- fit/crop compartilhado com o Load Image ------------------------------
try:
    from .load_media_fit import _bx_apply_fit, ASPECTS as _BX_ASPECTS, FIT_MODES as _BX_FIT_MODES
except Exception:  # pragma: no cover
    _bx_apply_fit = None
    _BX_ASPECTS = ["livre", "1:1", "3:4", "4:3", "16:9", "9:16"]
    _BX_FIT_MODES = ["off (original)", "crop", "stretch", "pad (letterbox)"]

# ---- ComfyUI helpers (guardados p/ rodar fora do Comfy em teste) ----------
try:
    import folder_paths
    _HAS_FP = True
except Exception:
    folder_paths = None
    _HAS_FP = False

# tipo VIDEO nativo (nodes 2.0). Import guardado: varia por versao do ComfyUI.
_VIDEO_API = None
try:
    from comfy_api.latest import InputImpl as _InputImpl, Types as _Types
    _VIDEO_API = "latest"
except Exception:
    try:
        from comfy_api.input_impl import VideoFromComponents as _VFC  # type: ignore
        from comfy_api.util import VideoComponents as _VComp           # type: ignore
        _VIDEO_API = "legacy"
    except Exception:
        _VIDEO_API = None

# backends de midia
try:
    import av  # PyAV (vem com o ComfyUI)
    _HAS_AV = True
except Exception:
    _HAS_AV = False
try:
    import imageio
    import imageio_ffmpeg  # noqa: F401
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".gif", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv")


# ===========================================================================
# Helpers de decode
# ===========================================================================
def _list_input_videos():
    if not _HAS_FP:
        return []
    d = folder_paths.get_input_directory()
    out = []
    try:
        for f in os.listdir(d):
            if f.lower().endswith(VIDEO_EXTS) and os.path.isfile(os.path.join(d, f)):
                out.append(f)
    except Exception:
        pass
    return sorted(out)


def _resolve_path(video, video_path):
    if video_path and str(video_path).strip():
        p = str(video_path).strip().strip('"')
        if os.path.isfile(p):
            return p
    if video and _HAS_FP:
        cand = os.path.join(folder_paths.get_input_directory(), video)
        if os.path.isfile(cand):
            return cand
    if video and os.path.isfile(video):
        return video
    raise FileNotFoundError(f"[Bruxos Load Video] video nao encontrado: video={video!r} video_path={video_path!r}")


def _input_preview_ref(video, video_path):
    """Devolve {filename, subfolder, type, format} se o video estiver no diretorio
    de input do ComfyUI (pro preview via /view). Retorna None caso contrario
    (ex.: caminho absoluto fora do input, que o /view nao serve)."""
    if video_path and str(video_path).strip():
        return None
    if not (video and _HAS_FP):
        return None
    try:
        in_dir = folder_paths.get_input_directory()
        cand = os.path.join(in_dir, video)
        if not os.path.isfile(cand):
            return None
        rel = os.path.relpath(cand, in_dir).replace("\\", "/")
        subfolder = os.path.dirname(rel)
        filename = os.path.basename(rel)
        ext = os.path.splitext(filename)[1].lower().lstrip(".") or "mp4"
        return {"filename": filename, "subfolder": subfolder,
                "type": "input", "format": "video/" + ext}
    except Exception:
        return None


def _iter_frames_imageio(path):
    rdr = imageio.get_reader(path, "ffmpeg")
    meta = rdr.get_meta_data()
    fps = float(meta.get("fps", 0) or 0)
    for frame in rdr:
        yield np.asarray(frame)[..., :3], fps
    rdr.close()


def _iter_frames_cv2(path):
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), fps
    cap.release()


def _iter_frames_av(path):
    container = av.open(path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    for frame in container.decode(stream):
        yield frame.to_ndarray(format="rgb24"), fps
    container.close()


def _iter_frames_av_hi(path):
    """Como _iter_frames_av, mas decodifica em 16-bit (rgb48le) em vez de
    8-bit (rgb24) -- preserva a precisao real de fontes 10/12-bit (ProRes,
    H.265 Main10, video vindo de outra ferramenta em alta profundidade etc.)
    em vez de truncar pra 8-bit ja no import. Fonte que so tem 8-bit mesmo:
    o resultado final e numericamente identico ao caminho normal (upsample
    exato, sem perda nem invencao de informacao) -- so custa mais CPU.

    Funcao ISOLADA de proposito: _iter_frames_av/_frame_iterator continuam
    8-bit sem mudanca nenhuma, porque bruxos_disk_stream.py consome os
    frames de _frame_iterator direto e assume uint8 (cache SSD "lossless-u8");
    misturar 16-bit ali corromperia esse cache."""
    container = av.open(path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    for frame in container.decode(stream):
        yield frame.to_ndarray(format="rgb48le"), fps
    container.close()


def _frame_iterator(path):
    """Itera (frame_rgb_uint8, source_fps). Tenta av -> imageio -> cv2."""
    if _HAS_AV:
        try:
            yield from _iter_frames_av(path); return
        except Exception as e:
            logging.warning(f"[Bruxos] av falhou ({e}); tentando imageio")
    if _HAS_IMAGEIO:
        try:
            yield from _iter_frames_imageio(path); return
        except Exception as e:
            logging.warning(f"[Bruxos] imageio falhou ({e}); tentando cv2")
    if _HAS_CV2:
        yield from _iter_frames_cv2(path); return
    raise RuntimeError("[Bruxos] Nenhum backend de video disponivel (av / imageio-ffmpeg / opencv).")


def _resize_frame(f, cw, ch):
    H, W = f.shape[:2]
    if cw <= 0 and ch <= 0:
        return f
    if cw > 0 and ch > 0:
        tw, th = cw, ch
    elif cw > 0:
        tw = cw; th = max(1, round(H * cw / W))
    else:
        th = ch; tw = max(1, round(W * ch / H))
    if (tw, th) == (W, H):
        return f
    if _HAS_CV2:
        interp = cv2.INTER_AREA if (tw < W or th < H) else cv2.INTER_CUBIC
        return cv2.resize(f, (tw, th), interpolation=interp)
    # fallback torch
    t = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, size=(th, tw), mode="bicubic", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()


def decode_video(path, skip_first_frames=0, frame_load_cap=0, select_every_nth=1,
                 force_rate=0.0, custom_width=0, custom_height=0):
    """Retorna (images_tensor[B,H,W,3] float 0..1, source_fps, out_fps)."""
    select_every_nth = max(1, int(select_every_nth))
    frames = []
    src_fps = 0.0
    kept = 0
    next_tick = 0.0
    step = None
    idx = -1
    for raw, sfps in _frame_iterator(path):
        if sfps:
            src_fps = sfps
        idx += 1
        if idx < skip_first_frames:
            continue
        j = idx - skip_first_frames
        if force_rate and src_fps:
            if step is None:
                step = src_fps / float(force_rate)
            if j < next_tick - 1e-9:
                continue
            next_tick += step
        else:
            if j % select_every_nth != 0:
                continue
        frames.append(_resize_frame(raw, custom_width, custom_height))
        kept += 1
        if frame_load_cap and kept >= frame_load_cap:
            break

    if not frames:
        raise RuntimeError("[Bruxos Load Video] nenhum frame decodificado (cheque skip/cap/path).")

    # garante shape uniforme (caso custom resize off e video tenha mudanca rara)
    h0, w0 = frames[0].shape[:2]
    arr = np.stack([f if f.shape[:2] == (h0, w0) else _resize_frame(f, w0, h0) for f in frames], 0)
    frames = None  # libera a lista (mesmos dados do arr) antes de ir pra float32 -- manter
    # os dois vivos ao mesmo tempo dobra a memoria a toa em lotes grandes.
    # torch.from_numpy(arr) e uma VIEW (sem copia); .float() faz UMA copia pra float32;
    # div_ e IN-PLACE (sem copia extra). "arr.astype(np.float32) / 255.0" fazia DUAS
    # copias extras do lote inteiro -- foi essa linha que estourou a RAM do usuario
    # (so 4GB, mas a maquina ja estava com bastante RAM ocupada por modelos grandes
    # rodando na mesma sessao do ComfyUI).
    images = torch.from_numpy(arr).float()
    images.div_(255.0)
    del arr

    if force_rate:
        out_fps = float(force_rate)
    elif src_fps:
        out_fps = src_fps / select_every_nth
    else:
        out_fps = 0.0
    return images, src_fps, out_fps


def _resize_frame_float(f, cw, ch):
    """Como _resize_frame, mas mantem float32 0..1 (nunca quantiza pra uint8) --
    usado so no decode de alta precisao (decode_video_hi)."""
    H, W = f.shape[:2]
    if cw <= 0 and ch <= 0:
        return f
    if cw > 0 and ch > 0:
        tw, th = cw, ch
    elif cw > 0:
        tw = cw; th = max(1, round(H * cw / W))
    else:
        th = ch; tw = max(1, round(W * ch / H))
    if (tw, th) == (W, H):
        return f
    if _HAS_CV2:
        interp = cv2.INTER_AREA if (tw < W or th < H) else cv2.INTER_CUBIC
        return cv2.resize(f, (tw, th), interpolation=interp)
    t = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, size=(th, tw), mode="bicubic", align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()


def decode_video_hi(path, skip_first_frames=0, frame_load_cap=0, select_every_nth=1,
                    force_rate=0.0, custom_width=0, custom_height=0):
    """Igual a decode_video, mas decodifica via PyAV em 16-bit (rgb48le)
    quando disponivel -- ve 'Importar em mais bits' no Load Video. Sem PyAV,
    cai pro decode_video normal (8-bit)."""
    if not _HAS_AV:
        return decode_video(path, skip_first_frames, frame_load_cap, select_every_nth,
                            force_rate, custom_width, custom_height)
    select_every_nth = max(1, int(select_every_nth))
    frames = []
    src_fps = 0.0
    kept = 0
    next_tick = 0.0
    step = None
    idx = -1
    try:
        for raw16, sfps in _iter_frames_av_hi(path):
            if sfps:
                src_fps = sfps
            idx += 1
            if idx < skip_first_frames:
                continue
            j = idx - skip_first_frames
            if force_rate and src_fps:
                if step is None:
                    step = src_fps / float(force_rate)
                if j < next_tick - 1e-9:
                    continue
                next_tick += step
            else:
                if j % select_every_nth != 0:
                    continue
            f32 = raw16.astype(np.float32) / 65535.0
            frames.append(_resize_frame_float(f32, custom_width, custom_height))
            kept += 1
            if frame_load_cap and kept >= frame_load_cap:
                break
    except Exception as e:
        logging.warning(f"[Bruxos Load Video] decode em 16-bit falhou ({e}); caindo pro decode 8-bit normal.")
        return decode_video(path, skip_first_frames, frame_load_cap, select_every_nth,
                            force_rate, custom_width, custom_height)

    if not frames:
        raise RuntimeError("[Bruxos Load Video] nenhum frame decodificado (cheque skip/cap/path).")

    h0, w0 = frames[0].shape[:2]
    arr = np.stack([f if f.shape[:2] == (h0, w0) else _resize_frame_float(f, w0, h0) for f in frames], 0)
    frames = None  # libera a lista antes de seguir -- ver comentario equivalente em decode_video
    # frames aqui ja sao float32 (convertidos por frame antes do stack) -- .astype(np.float32)
    # faria mais uma copia inteira do lote a toa. torch.from_numpy sem astype e zero-copy.
    images = torch.from_numpy(arr)

    if force_rate:
        out_fps = float(force_rate)
    elif src_fps:
        out_fps = src_fps / select_every_nth
    else:
        out_fps = 0.0
    return images, src_fps, out_fps


def _extract_audio_av(path):
    """Best-effort: AUDIO dict {'waveform':[1,C,N], 'sample_rate'} via PyAV."""
    if not _HAS_AV:
        return None
    try:
        container = av.open(path)
        if not container.streams.audio:
            container.close(); return None
        astream = container.streams.audio[0]
        sr = astream.rate
        chunks = []
        for frame in container.decode(astream):
            chunks.append(frame.to_ndarray())
        container.close()
        if not chunks:
            return None
        data = np.concatenate(chunks, axis=1) if chunks[0].ndim == 2 else np.concatenate(chunks)[None, :]
        wav = torch.from_numpy(np.ascontiguousarray(data)).float()
        if wav.abs().max() > 1.5:  # int PCM
            wav = wav / 32768.0
        return {"waveform": wav.unsqueeze(0), "sample_rate": int(sr)}
    except Exception as e:
        logging.warning(f"[Bruxos] extracao de audio falhou: {e}")
        return None


def _make_video_obj(images, audio, fps):
    """Constroi o objeto VIDEO nativo, se a API existir; senao None."""
    if _VIDEO_API == "latest":
        try:
            comps = _Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps).limit_denominator(100000))
            return _InputImpl.VideoFromComponents(comps)
        except Exception as e:
            logging.warning(f"[Bruxos] VideoFromComponents (latest) falhou: {e}")
    elif _VIDEO_API == "legacy":
        try:
            comps = _VComp(images=images, audio=audio, frame_rate=Fraction(fps).limit_denominator(100000))
            return _VFC(comps)
        except Exception as e:
            logging.warning(f"[Bruxos] VideoFromComponents (legacy) falhou: {e}")
    return None


# ===========================================================================
# NODE: Load Video
# ===========================================================================
_BX_GIRO = ["off", "90 (horario)", "-90 (anti-horario)", "180"]


def bx_rotacionar(t, giro):
    """Gira um lote [B,H,W,C] ou uma mascara [B,H,W]. Devolve o mesmo tipo.

    torch.rot90 com k=1 gira ANTI-HORARIO no plano (dims na ordem dada), entao
    o horario e k=-1. Escrevi os rotulos com a direcao por extenso porque "90"
    sozinho e ambiguo -- editor de video, camera e biblioteca de imagem nao
    concordam sobre o sinal, e trocar isso depois quebraria grafos salvos.

    -180 nao existe como opcao: e o mesmo que 180.
    """
    g = str(giro or "off")
    if g.startswith("off"):
        return t
    dims = (1, 2)                      # (H, W) tanto em [B,H,W,C] quanto em [B,H,W]
    if g.startswith("90"):
        k = -1
    elif g.startswith("-90"):
        k = 1
    elif g.startswith("180"):
        k = 2
    else:
        return t
    return torch.rot90(t, k, dims).contiguous()