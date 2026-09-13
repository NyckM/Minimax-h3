# Layout/config de ladrilho (MARCA, ler_config, montar_layout, fundir) —
# copia dos helpers do bruxos_video_tiler.py. NAO registra nodes.
# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Video Tiler: ladrilho em PIXEL pra upscale (LTX / MiniMax H3)
=============================================================================
POR QUE ESTE PACOTE EXISTE (e por que o `ltx_tiled_bruxos.py` nao servia)
    O `BruxosLTXTiledGuider` corta o LATENTE e funde as predicoes de ruido a
    cada passo. Isso e otimo pra GERAR coerente -- os ladrilhos se enxergam
    durante o processo -- mas ele nao tem onde enfiar um condicionamento
    DIFERENTE por ladrilho.

    Upscale E condicionamento por ladrilho: cada pedaco precisa ser guiado
    pelo seu proprio recorte da fonte. Sao arquiteturas diferentes:

        step-fused (latente)      | ladrilho em pixel (aqui)
        --------------------------|---------------------------------
        corta o latente           | corta a IMAGEM
        funde a cada passo        | funde no fim
        condicionamento global    | UM RAMO COMPLETO POR LADRILHO
        serve pra gerar           | serve pra UPSCALAR

CREDITO
    A geometria de posicionamento par (equiparticao inteira das passadas) e a
    rampa de costura por "distancia a borda interna mais proxima" vieram do
    comfyui-video-tiler de maDcaDDie2000 (Apache-2.0). Reescrevi em vez de
    copiar, mas a ideia central e dele e merece o credito.

O QUE MUDA AQUI
    * Um TILE_CONFIG proprio ("BXT1"), sem versoes legadas pra carregar.
    * Helper de GRADE POR MODELO: LTX comprime 32x no espaco, o H3 comprime
      16x + patch 2x2. Ladrilho fora da grade da costura ou erro de shape, e
      nenhum slicer generico sabe qual modelo voce esta usando.
    * Avisos altos onde da pra errar em silencio: ladrilho maior que o quadro,
      sobreposicao zero, contagem de ladrilhos que nao bate na fusao.

FLUXO EM MEMORIA
    Load Video ─> Tile Slice ─┬─ tiles (LISTA) ─> [seu ramo de upscale] ─┐
                              └─ tile_config ───────────────────────────┼─> Tile Merge
    (referencia) ─> Tile Ref Slice (mesma geometria) ─> ramo             ┘

FLUXO EM DISCO (quando nem a lista de ladrilhos cabe)
    passe 1: Disk Job ─> Disk Get Tile ─> [upscale] ─> Disk Save Tile
             (uma execucao por indice)
    passe 2: Disk Merge (le os .pt um por um e funde)
"""

import json
import logging
import math
import os

try:
    import torch
    import torch.nn.functional as F
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/Tiler"
MARCA = "BXT1"

# Compressao espacial efetiva de cada modelo. Ladrilho tem que ser multiplo
# disso, senao o encoder arredonda e a costura desalinha.
GRADE = {
    "LTX 2.x  (32)": 32,
    "MiniMax H3  (32)": 32,
    "Bernini / Wan 2.2-14B  (16)": 16,
    "Wan 2.1 / 2.2-14B  (16)": 16,
    "Wan 2.2-5B  (32)": 32,
    "SDXL / generico  (8)": 8,
}


# ---------------------------------------------------------------------------
# geometria
# ---------------------------------------------------------------------------
def _snap(v, m):
    return max(m, (int(v) // m) * m)


def _posicoes(tamanho, ladrilho, sobrep_min, m):
    """Inicios de ladrilho no eixo, com o ULTIMO encostando na borda.

    Usa equiparticao INTEIRA do resto entre as passadas:
        passo_i = (resto*(i+1))//n - (resto*i)//n
    Assim a soma fecha exata e passadas vizinhas diferem no maximo 1 pixel --
    sem aquele ladrilho final espremido que aparece quando voce vai somando
    passo fixo e joga o resto todo na ultima junta.
    """
    ladrilho = max(1, int(ladrilho))
    tamanho = max(1, int(tamanho))
    m = max(1, min(64, int(m)))
    if ladrilho >= tamanho:
        return [0], 0
    resto = tamanho - ladrilho
    sobrep_min = max(0, min(int(sobrep_min), ladrilho // 2))
    passo_max = max(m, ladrilho - sobrep_min)

    n_min = max(1, math.ceil(resto / passo_max))
    for n in range(n_min, n_min + max(8, resto // max(m, 1) + 4) + 1):
        passos = [(resto * (i + 1)) // n - (resto * i) // n for i in range(n)]
        if not passos or max(passos) > passo_max or min(passos) < 1:
            continue
        xs = [0]
        for p in passos:
            xs.append(xs[-1] + p)
        if xs[-1] != resto:
            continue
        return xs, ladrilho - max(passos)

    # fallback: passo fixo (pode deixar a ultima junta desigual)
    passo = max(1, ladrilho - sobrep_min)
    xs, i = [0], 0
    while xs[-1] + ladrilho < tamanho:
        i += passo
        xs.append(min(i, tamanho - ladrilho))
        if xs[-1] == xs[-2]:
            break
    return sorted(set(xs)), max(0, ladrilho - passo)


def _ordem(padrao, nr, nc):
    if padrao == "coluna":
        return [(r, c) for c in range(nc) for r in range(nr)]
    if padrao == "espiral":
        saida, cima, baixo, esq, dir_ = [], 0, nr - 1, 0, nc - 1
        while cima <= baixo and esq <= dir_:
            for c in range(esq, dir_ + 1):
                saida.append((cima, c))
            cima += 1
            for r in range(cima, baixo + 1):
                saida.append((r, dir_))
            dir_ -= 1
            if cima <= baixo:
                for c in range(dir_, esq - 1, -1):
                    saida.append((baixo, c))
                baixo -= 1
            if esq <= dir_:
                for r in range(baixo, cima - 1, -1):
                    saida.append((r, esq))
                esq += 1
        return saida
    return [(r, c) for r in range(nr) for c in range(nc)]


def montar_layout(W, H, lw, lh, mult, sobrep_frac, padrao):
    """-> (lista de dicts, config, sobrep_x, sobrep_y)"""
    m = max(1, min(64, int(mult)))
    lw = _snap(max(m, lw), m)
    lh = _snap(max(m, lh), m)
    sx = min(lw // 2, _snap(int(lw * sobrep_frac), m)) if sobrep_frac > 0 else 0
    sy = min(lh // 2, _snap(int(lh * sobrep_frac), m)) if sobrep_frac > 0 else 0
    xs, ox = _posicoes(W, lw, sx, m)
    ys, oy = _posicoes(H, lh, sy, m)
    nc, nr = len(xs), len(ys)

    tiles = []
    for i, (r, c) in enumerate(_ordem(padrao, nr, nc)):
        x, y = xs[c], ys[r]
        w, h = min(lw, W - x), min(lh, H - y)
        if w > 0 and h > 0:
            tiles.append({"x": x, "y": y, "w": w, "h": h, "col": c, "row": r, "ordem": i})
    cfg = (MARCA, int(W), int(H), int(lw), int(lh), int(ox), int(oy), m,
           tuple((t["x"], t["y"], t["w"], t["h"], t["col"], t["row"], t["ordem"]) for t in tiles))
    return tiles, cfg, ox, oy


def ler_config(cfg):
    while isinstance(cfg, (list, tuple)) and len(cfg) == 1 and cfg[0] != MARCA:
        cfg = cfg[0]
    if not (isinstance(cfg, (list, tuple)) and len(cfg) == 9 and cfg[0] == MARCA):
        raise ValueError(
            "[Bruxos Tiler] 'tile_config' invalido. Ele tem que vir do 'Tile Slice (Bruxos)' -- "
            "nao e compativel com o TILE_CONFIG de outros pacotes de ladrilho."
        )
    _, W, H, lw, lh, ox, oy, m, td = cfg
    tiles = [{"x": t[0], "y": t[1], "w": t[2], "h": t[3],
              "col": t[4], "row": t[5], "ordem": t[6]} for t in td]
    return int(W), int(H), int(lw), int(lh), int(ox), int(oy), int(m), tiles


# ---------------------------------------------------------------------------
# fusao
# ---------------------------------------------------------------------------
def _rampa(w, h, x, y, W, H, fx, fy, dev):
    """Alfa do topo: distancia a borda INTERNA mais proxima, normalizada por
    eixo, o MINIMO das duas, e uma subida cosseno.

    O minimo (em vez do produto) e o detalhe que importa: multiplicar as
    rampas dos dois eixos cria um poco de peso nos CANTOS, onde as duas caem
    juntas -- e canto escuro na junta e o artefato classico de ladrilho."""
    if fx <= 0 and fy <= 0:
        return torch.ones((h, w), dtype=torch.float32, device=dev)
    ly = torch.arange(h, device=dev, dtype=torch.float32).view(-1, 1).expand(h, w)
    lx = torch.arange(w, device=dev, dtype=torch.float32).view(1, -1).expand(h, w)
    ns = []
    if x > 0 and fx > 0:
        ns.append(lx / fx)
    if y > 0 and fy > 0:
        ns.append(ly / fy)
    if x + w < W and fx > 0:
        ns.append(((w - 1) - lx) / fx)
    if y + h < H and fy > 0:
        ns.append(((h - 1) - ly) / fy)
    if not ns:
        return torch.ones((h, w), dtype=torch.float32, device=dev)
    t = torch.min(torch.stack(ns, 0), 0).values.clamp(0, 1)
    return 0.5 * (1.0 - torch.cos(math.pi * t))


def _curva(a, modo):
    if modo == "suave_entrada":
        return a * a
    if modo == "suave_saida":
        return 1.0 - (1.0 - a) * (1.0 - a)
    if modo == "suave_ambos":
        return a * a * (3.0 - 2.0 * a)
    return a


def _norm_tile(t):
    while isinstance(t, (list, tuple)) and len(t) > 0:
        t = t[0]
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t


def fundir(tiles, cfg, feather, curva="linear", modo="media_ponderada", dispositivo="auto"):
    W, H, lw, lh, ox, oy, m, specs = ler_config(cfg)
    tl = [_norm_tile(t) for t in tiles]
    if not tl:
        raise ValueError("[Bruxos Tiler] lista de ladrilhos VAZIA. O ramo de upscale nao devolveu nada.")
    if len(tl) != len(specs):
        raise ValueError(
            f"[Bruxos Tiler] chegaram {len(tl)} ladrilhos mas o tile_config descreve {len(specs)}.\n"
            f"Isso quase sempre e o ramo de upscale mudando o numero de itens da lista (algum node "
            f"que junta batch, ou um Preview no meio). A ordem TEM que ser a mesma do slicer."
        )
    p = tl[0]
    B, C = int(p.shape[0]), int(p.shape[3])
    dev = (torch.device("cpu") if dispositivo == "cpu" else
           (torch.device("cuda:0") if (dispositivo == "cuda" and torch.cuda.is_available()) else p.device))

    f = max(0.0, min(0.5, float(feather)))
    fx = min(ox, _snap(int(lw * f), m)) if (ox > 0 and f > 0) else 0
    fy = min(oy, _snap(int(lh * f), m)) if (oy > 0 and f > 0) else 0

    ordenados = sorted(enumerate(specs), key=lambda kv: kv[1]["ordem"])

    if modo == "media_ponderada":
        num = torch.zeros((B, H, W, C), dtype=torch.float32, device=dev)
        den = torch.zeros((B, H, W), dtype=torch.float32, device=dev)
        for k, (i, s) in enumerate(ordenados):
            t = tl[i].to(dev)
            x, y, w, h = s["x"], s["y"], s["w"], s["h"]
            a = (torch.ones((h, w), dtype=torch.float32, device=dev) if k == 0
                 else _curva(_rampa(w, h, x, y, W, H, fx, fy, dev), curva))
            num[:, y:y + h, x:x + w, :] += t.to(torch.float32) * a.unsqueeze(0).unsqueeze(-1)
            den[:, y:y + h, x:x + w] += a.unsqueeze(0)
        return (num / den.unsqueeze(-1).clamp(min=1e-8)).to(p.dtype)

    saida = torch.zeros((B, H, W, C), dtype=p.dtype, device=dev)
    coberto = torch.zeros((B, H, W), dtype=torch.bool, device=dev)
    for k, (i, s) in enumerate(ordenados):
        t = tl[i].to(dev)
        x, y, w, h = s["x"], s["y"], s["w"], s["h"]
        if k == 0:
            saida[:, y:y + h, x:x + w, :] = t
            coberto[:, y:y + h, x:x + w] = True
            continue
        a = _curva(_rampa(w, h, x, y, W, H, fx, fy, dev), curva)
        cov = coberto[:, y:y + h, x:x + w]
        # onde ninguem escreveu ainda, cola opaco: rampa so vale sobre pixel ja pintado
        a = torch.where(cov, a.unsqueeze(0).expand(B, h, w),
                        torch.ones((), dtype=torch.float32, device=dev)).unsqueeze(-1)
        reg = saida[:, y:y + h, x:x + w, :].to(torch.float32)
        saida[:, y:y + h, x:x + w, :] = (reg * (1 - a) + t.to(torch.float32) * a).to(p.dtype)
        coberto[:, y:y + h, x:x + w] = True
    return saida