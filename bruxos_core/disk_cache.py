# Cache de frames em disco — copia da implementacao do bruxos_disk_stream.py.
# NAO e registrada como node aqui (quem registra e o ComfyUI-Bruxos-MediaIO).
# Esta copia existe porque o Bernini SSD-81 e o H3 Contex Loop herdam/chamam
# estas classes ja em tempo de import.
"""
Bruxos do VFX - cache de frames em disco e leitura por janelas.

O formato guarda blocos uint8 lossless em .pt. O video e decodificado e gravado
sem jamais ser empilhado inteiro na RAM. Na leitura, somente os blocos que
intersectam a janela pedida viram IMAGE float32.

Limite importante: um consumidor ComfyUI que exige um IMAGE completo ainda vai
materializar esse lote. Para economia real, ligue a saida do Window em um ramo
que processe uma janela por execucao.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import shutil
import threading
import time
import uuid
from fractions import Fraction
from pathlib import Path

import torch

try:
    import folder_paths
except Exception:  # permite importar o modulo nos testes fora do ComfyUI
    folder_paths = None

from .media import (
    VIDEO_EXTS,
    _frame_iterator,
    _list_input_videos,
    _resize_frame,
    _resolve_path,
)


CAT = "Bruxos do VFX/Streaming em Disco"
MARCA = "BRUXOS_FRAME_CACHE_V1"
MANIFESTO = "manifesto.json"
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="bruxos-disk-prefetch"
)
_PREFETCH = {}
_PREFETCH_LOCK = threading.Lock()


def _nome_seguro(nome):
    nome = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(nome or "cache")).strip("._")
    return nome[:96] or "cache"


def _raiz_cache():
    if folder_paths is not None:
        base = folder_paths.get_temp_directory()
    else:
        base = os.path.join(os.getcwd(), ".bruxos_temp")
    raiz = os.path.abspath(os.path.join(base, "bruxos_frame_cache"))
    os.makedirs(raiz, exist_ok=True)
    return raiz


def _pasta_cache(nome):
    return os.path.join(_raiz_cache(), _nome_seguro(nome))


def _listar_caches():
    raiz = _raiz_cache()
    itens = []
    try:
        for nome in os.listdir(raiz):
            pasta = os.path.join(raiz, nome)
            if os.path.isdir(pasta) and os.path.isfile(os.path.join(pasta, MANIFESTO)):
                itens.append(nome)
    except OSError:
        pass
    return sorted(itens, key=lambda nome: os.path.getmtime(os.path.join(raiz, nome)), reverse=True)


def _resolver_cache_existente(cache_name, cache_path=""):
    bruto = str(cache_path or "").strip().strip('"')
    pasta = os.path.abspath(bruto) if bruto else _pasta_cache(cache_name)
    _ler_manifesto(pasta)
    return pasta


def _manifest_path(cache):
    return os.path.join(os.path.abspath(str(cache)), MANIFESTO)


def _ler_manifesto(cache):
    path = _manifest_path(cache)
    with open(path, "r", encoding="utf-8") as fp:
        man = json.load(fp)
    if man.get("marca") != MARCA:
        raise ValueError(f"[Bruxos Disk Stream] cache invalido ou versao desconhecida: {path}")
    return man


def _gravar_json_atomico(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _torch_save_atomico(tensor, path):
    tmp = path + ".tmp"
    torch.save(tensor, tmp)
    os.replace(tmp, path)


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch antigo
        return torch.load(path, map_location="cpu")


def _carregar_bloco(path):
    path = os.path.abspath(path)
    future = None
    with _PREFETCH_LOCK:
        future = _PREFETCH.pop(path, None)
    tensor = future.result() if future is not None else _torch_load_cpu(path)
    if not torch.is_tensor(tensor) or tensor.ndim != 4:
        raise ValueError(f"[Bruxos Disk Stream] bloco corrompido: {path}")
    return tensor


def _agendar_prefetch(paths):
    with _PREFETCH_LOCK:
        for path in paths:
            path = os.path.abspath(path)
            if path not in _PREFETCH:
                _PREFETCH[path] = _EXECUTOR.submit(_torch_load_cpu, path)


def _blocos_da_janela(man, inicio, quantidade):
    fim = min(int(man["frames"]), inicio + quantidade)
    return [b for b in man["blocos"] if int(b["inicio"]) < fim and int(b["fim"]) > inicio]


def _finalizar_bloco(pasta, indice, inicio, frames):
    # uint8 reduz o cache a 1/4 do tamanho de IMAGE float32 e e lossless para
    # frames que vieram de video/PNG.
    tensor = torch.stack(frames, dim=0).contiguous()
    nome = f"bloco_{indice:06d}.pt"
    _torch_save_atomico(tensor, os.path.join(pasta, nome))
    return {"arquivo": nome, "inicio": inicio, "fim": inicio + int(tensor.shape[0])}


def _preparar_pasta(nome, sobrescrever):
    pasta = _pasta_cache(nome)
    if os.path.exists(pasta):
        if not sobrescrever:
            try:
                _ler_manifesto(pasta)
                return pasta, False
            except Exception:
                raise ValueError(
                    f"[Bruxos Disk Stream] a pasta existe mas nao e um cache valido: {pasta}. "
                    "Ligue 'sobrescrever' ou use outro nome."
                )
        shutil.rmtree(pasta)
    os.makedirs(pasta, exist_ok=True)
    return pasta, True


class BruxosVideoParaDisco:
    @classmethod
    def INPUT_TYPES(cls):
        videos = _list_input_videos() or [""]
        return {"required": {
            "video": (videos,),
            "nome_cache": ("STRING", {"default": "h3_video_01"}),
            "frames_por_bloco": ("INT", {"default": 31, "min": 1, "max": 1024, "step": 1,
                                            "tooltip": "Bloco FISICO do SSD. 31 x 4 = uma janela H3 de 124 frames. "
                                                       "Isto nao muda o tamanho temporal da geracao."}),
            "pular_frames": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
            "limite_frames": ("INT", {"default": 0, "min": 0, "max": 1_000_000,
                                      "tooltip": "0 = ate o fim."}),
            "pegar_cada_n": ("INT", {"default": 1, "min": 1, "max": 1000}),
            "force_rate": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 240.0, "step": 0.01,
                                    "tooltip": "Reamostra temporalmente para este FPS. 24 e o correto para H3; 0 preserva."}),
            "largura": ("INT", {"default": 0, "min": 0, "max": 16384,
                                 "tooltip": "0 preserva. Se so uma dimensao for dada, preserva a proporcao."}),
            "altura": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "sobrescrever": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "video_path": ("STRING", {"default": "", "tooltip": "Caminho absoluto opcional."}),
        }}

    RETURN_TYPES = ("BRUXOS_FRAME_CACHE", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("cache", "frames", "fps", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Decodifica o video diretamente em blocos no SSD. Nunca cria um IMAGE com o video inteiro; "
        "use 'Disk Cache - Ler Janela' para alimentar apenas os proximos N frames."
    )

    def run(self, video, nome_cache, frames_por_bloco, pular_frames, limite_frames,
            pegar_cada_n, force_rate, largura, altura, sobrescrever, video_path=""):
        origem = _resolve_path(video, video_path)
        pasta, criar = _preparar_pasta(nome_cache, bool(sobrescrever))
        if not criar:
            man = _ler_manifesto(pasta)
            info = f"cache reutilizado | {man['frames']} frames | {pasta}"
            return pasta, int(man["frames"]), float(man.get("fps", 0.0)), info

        bloco_n = max(1, int(frames_por_bloco))
        nth = max(1, int(pegar_cada_n))
        skip = max(0, int(pular_frames))
        cap = max(0, int(limite_frames))
        blocos, buffer = [], []
        src_fps = 0.0
        mantidos = 0
        shape = None
        next_tick = 0.0
        rate_step = None
        inicio_t = time.perf_counter()
        try:
            for indice, (raw, fps) in enumerate(_frame_iterator(origem)):
                if fps:
                    src_fps = float(fps)
                if indice < skip:
                    continue
                relativo = indice - skip
                if force_rate and src_fps:
                    if rate_step is None:
                        rate_step = src_fps / float(force_rate)
                    if relativo < next_tick - 1e-9:
                        continue
                    next_tick += rate_step
                elif relativo % nth:
                    continue
                raw = _resize_frame(raw, int(largura), int(altura))
                frame = torch.from_numpy(raw[..., :3].copy()).to(torch.uint8)
                if shape is None:
                    shape = tuple(frame.shape)
                elif tuple(frame.shape) != shape:
                    raw = _resize_frame(raw, shape[1], shape[0])
                    frame = torch.from_numpy(raw[..., :3].copy()).to(torch.uint8)
                buffer.append(frame)
                mantidos += 1
                if len(buffer) >= bloco_n:
                    blocos.append(_finalizar_bloco(pasta, len(blocos), mantidos - len(buffer), buffer))
                    buffer = []
                if cap and mantidos >= cap:
                    break
            if buffer:
                blocos.append(_finalizar_bloco(pasta, len(blocos), mantidos - len(buffer), buffer))
            if not mantidos:
                raise RuntimeError("nenhum frame foi decodificado; confira caminho, pulo e limite")
            out_fps = float(force_rate) if force_rate else (src_fps / nth if src_fps else 0.0)
            man = {
                "marca": MARCA, "frames": mantidos, "fps": out_fps,
                "source_fps": src_fps, "altura": shape[0], "largura": shape[1],
                "canais": shape[2], "dtype": "uint8", "frames_por_bloco": bloco_n,
                "blocos": blocos,
                "origem": {"arquivo": os.path.abspath(origem),
                           "tamanho": os.path.getsize(origem),
                           "mtime_ns": os.stat(origem).st_mtime_ns},
            }
            _gravar_json_atomico(os.path.join(pasta, MANIFESTO), man)
        except Exception:
            shutil.rmtree(pasta, ignore_errors=True)
            raise
        dur = time.perf_counter() - inicio_t
        info = (f"{mantidos} frames {shape[1]}x{shape[0]} | {len(blocos)} blocos | "
                f"{out_fps:.3f} fps | {dur:.1f}s | {pasta}")
        print(f"[Bruxos Disk Stream] {info}", flush=True)
        return pasta, mantidos, float(out_fps), info

    @classmethod
    def IS_CHANGED(cls, video, nome_cache, frames_por_bloco, pular_frames,
                   limite_frames, pegar_cada_n, force_rate, largura, altura, sobrescrever,
                   video_path=""):
        try:
            path = _resolve_path(video, video_path)
            stat = os.stat(path)
            return (stat.st_mtime_ns, stat.st_size, nome_cache, frames_por_bloco,
                    pular_frames, limite_frames, pegar_cada_n, force_rate,
                    largura, altura, sobrescrever)
        except Exception:
            return float("nan")


class BruxosImagensParaDisco:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "imagens": ("IMAGE",),
            "nome_cache": ("STRING", {"default": "h3_checkpoint_01"}),
            "frames_por_bloco": ("INT", {"default": 31, "min": 1, "max": 1024,
                                            "tooltip": "Bloco FISICO do SSD; use 31 para quatro blocos por janela H3 de 124."}),
            "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 1000.0}),
            "sobrescrever": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("BRUXOS_FRAME_CACHE", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("cache", "frames", "fps", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Checkpoint lossless de um IMAGE em blocos no SSD. A entrada ja existe inteira em memoria; "
        "este node ajuda nas etapas seguintes, mas nao reduz o pico que ocorreu antes dele."
    )

    def run(self, imagens, nome_cache, frames_por_bloco, fps, sobrescrever):
        if not torch.is_tensor(imagens) or imagens.ndim != 4:
            raise ValueError("[Bruxos Disk Stream] imagens precisa ser [frames, altura, largura, canais]")
        pasta, criar = _preparar_pasta(nome_cache, bool(sobrescrever))
        if not criar:
            man = _ler_manifesto(pasta)
            info = f"cache reutilizado | {man['frames']} frames | {pasta}"
            return pasta, int(man["frames"]), float(man.get("fps", fps)), info
        total = int(imagens.shape[0])
        bloco_n = max(1, int(frames_por_bloco))
        blocos = []
        try:
            for inicio in range(0, total, bloco_n):
                # Quantizacao e por bloco; nao cria uma segunda copia do video inteiro.
                fatia = imagens[inicio:inicio + bloco_n, ..., :3].detach().to("cpu")
                u8 = fatia.mul(255.0).round_().clamp_(0, 255).to(torch.uint8).contiguous()
                nome = f"bloco_{len(blocos):06d}.pt"
                _torch_save_atomico(u8, os.path.join(pasta, nome))
                blocos.append({"arquivo": nome, "inicio": inicio, "fim": inicio + int(u8.shape[0])})
            man = {
                "marca": MARCA, "frames": total, "fps": float(fps),
                "altura": int(imagens.shape[1]), "largura": int(imagens.shape[2]),
                "canais": 3, "dtype": "uint8", "frames_por_bloco": bloco_n,
                "blocos": blocos, "origem": {"tipo": "IMAGE"},
            }
            _gravar_json_atomico(os.path.join(pasta, MANIFESTO), man)
        except Exception:
            shutil.rmtree(pasta, ignore_errors=True)
            raise
        info = f"{total} frames | {len(blocos)} blocos lossless-u8 | {pasta}"
        print(f"[Bruxos Disk Stream] {info}", flush=True)
        return pasta, total, float(fps), info


class BruxosDiscoLerJanela:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cache": ("BRUXOS_FRAME_CACHE", {"forceInput": True}),
            "inicio": ("INT", {"default": 0, "min": 0, "max": 10_000_000}),
            "quantidade": ("INT", {"default": 124, "min": 1, "max": 100_000,
                                  "tooltip": "124 = bloco padrao oficial do H3 (~5 s a 24 fps). "
                                             "Outros tamanhos H3 precisam obedecer length % 17 == 5."}),
            "sobreposicao": ("INT", {"default": 0, "min": 0, "max": 99_999,
                                      "tooltip": "O proximo inicio sera inicio + quantidade - sobreposicao."}),
            "prefetch": ("BOOLEAN", {"default": True,
                                      "tooltip": "Le os blocos da proxima janela em uma thread enquanto a atual e processada."}),
            "pin_memory": ("BOOLEAN", {"default": False,
                                       "tooltip": "Memoria RAM fixada pode acelerar CPU->GPU, mas consome RAM nao paginavel."}),
            "completar_ultimo": ("BOOLEAN", {"default": True,
                                             "tooltip": "Repete o ultimo frame para o bloco final tambem ter 124 frames. "
                                                        "Deixe ligado ao alimentar o H3."}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT", "BOOLEAN", "FLOAT", "STRING")
    RETURN_NAMES = ("imagens", "proximo_inicio", "total_frames", "terminou", "fps", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Carrega somente uma janela do cache. Com prefetch ligado, agenda no SSD/RAM os blocos da "
        "proxima janela. Mude 'inicio' (ou ligue um contador) a cada execucao."
    )

    def run(self, cache, inicio, quantidade, sobreposicao, prefetch, pin_memory,
            completar_ultimo=True):
        man = _ler_manifesto(cache)
        total = int(man["frames"])
        inicio = min(max(0, int(inicio)), max(0, total - 1))
        quantidade = max(1, int(quantidade))
        fim = min(total, inicio + quantidade)
        partes = []
        for bloco in _blocos_da_janela(man, inicio, quantidade):
            t = _carregar_bloco(os.path.join(cache, bloco["arquivo"]))
            a = max(inicio, int(bloco["inicio"])) - int(bloco["inicio"])
            b = min(fim, int(bloco["fim"])) - int(bloco["inicio"])
            partes.append(t[a:b])
        if not partes:
            raise RuntimeError(f"[Bruxos Disk Stream] janela {inicio}:{fim} nao encontrou blocos")
        u8 = partes[0] if len(partes) == 1 else torch.cat(partes, dim=0)
        reais = int(u8.shape[0])
        preenchidos = 0
        if completar_ultimo and reais < quantidade:
            preenchidos = quantidade - reais
            u8 = torch.cat([u8, u8[-1:].repeat(preenchidos, 1, 1, 1)], dim=0)
        imagens = u8.to(torch.float32).div_(255.0)
        if pin_memory and torch.cuda.is_available():
            imagens = imagens.pin_memory()
        passo = max(1, quantidade - min(max(0, int(sobreposicao)), quantidade - 1))
        proximo = min(total, inicio + passo)
        terminou = fim >= total
        if prefetch and not terminou:
            paths = [os.path.join(cache, b["arquivo"])
                     for b in _blocos_da_janela(man, proximo, quantidade)]
            _agendar_prefetch(paths)
        mb = imagens.numel() * imagens.element_size() / (1024 ** 2)
        grade = "H3-124" if quantidade == 124 else (
            "grade H3 valida" if quantidade >= 5 and quantidade % 17 == 5 else "fora da grade H3"
        )
        padding = f" | +{preenchidos} repetidos no fim" if preenchidos else ""
        info = (f"janela {inicio}:{fim} ({reais} reais) de {total} | {grade}{padding} | "
                f"{mb:.1f} MiB em RAM | proximo={proximo} | prefetch={'on' if prefetch else 'off'}")
        print(f"[Bruxos Disk Stream] {info}", flush=True)
        return imagens, proximo, total, bool(terminou), float(man.get("fps", 0.0)), info

    @classmethod
    def IS_CHANGED(cls, cache, inicio, quantidade, sobreposicao, prefetch, pin_memory,
                   completar_ultimo=True):
        try:
            stat = os.stat(_manifest_path(cache))
            return (stat.st_mtime_ns, inicio, quantidade, sobreposicao, prefetch,
                    pin_memory, completar_ultimo)
        except Exception:
            return float("nan")


class BruxosAbrirCacheDisco:
    @classmethod
    def INPUT_TYPES(cls):
        caches = _listar_caches() or ["(nenhum cache encontrado)"]
        return {
            "required": {
                "cache_name": (caches,),
            },
            "optional": {
                "cache_path": ("STRING", {
                    "default": "",
                    "tooltip": "Caminho absoluto da pasta que contem manifesto.json. Tem prioridade.",
                }),
            },
        }

    RETURN_TYPES = ("BRUXOS_FRAME_CACHE", "INT", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = ("cache", "frames", "fps", "width", "height", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    DESCRIPTION = (
        "Reabre um BRUXOS_FRAME_CACHE existente no temp usando o manifesto.json. "
        "Nao carrega os blocos na RAM."
    )

    def run(self, cache_name, cache_path=""):
        pasta = _resolver_cache_existente(cache_name, cache_path)
        man = _ler_manifesto(pasta)
        blocos = list(man.get("blocos") or [])
        if not blocos:
            raise RuntimeError(f"[Bruxos Disk Stream] cache sem blocos: {pasta}")
        faltando = [
            bloco.get("arquivo", "") for bloco in blocos
            if not os.path.isfile(os.path.join(pasta, bloco.get("arquivo", "")))
        ]
        if faltando:
            raise RuntimeError(
                f"[Bruxos Disk Stream] cache incompleto; faltam: {', '.join(faltando[:5])}"
            )
        frames = int(man["frames"])
        fps = float(man.get("fps", 0.0))
        width = int(man.get("largura", 0))
        height = int(man.get("altura", 0))
        info = (
            f"{frames} frames | {width}x{height} | {fps:.3f} fps | "
            f"{len(blocos)} blocos | {pasta}"
        )
        return pasta, frames, fps, width, height, info

    @classmethod
    def IS_CHANGED(cls, cache_name, cache_path=""):
        try:
            pasta = _resolver_cache_existente(cache_name, cache_path)
            stat = os.stat(_manifest_path(pasta))
            return stat.st_mtime_ns, stat.st_size
        except Exception:
            return float("nan")


class BruxosCacheParaVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache": ("BRUXOS_FRAME_CACHE", {"forceInput": True}),
                "filename_prefix": ("STRING", {
                    "default": "Bruxos/recovered_cache",
                    "tooltip": "Caminho relativo dentro de ComfyUI/output.",
                }),
                "fps_override": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "0 usa o fps do manifesto.",
                }),
                "crf": ("INT", {
                    "default": 19, "min": 0, "max": 51, "step": 1,
                    "tooltip": "Menor = mais qualidade e arquivo maior.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("video_path", "frames", "fps", "info")
    FUNCTION = "run"
    CATEGORY = CAT
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Recupera um cache de blocos .pt e grava MP4 H.264/yuv420p diretamente "
        "do SSD, sem materializar o video inteiro como IMAGE float32."
    )

    @staticmethod
    def _next_counter(out_dir, name):
        pattern = re.compile(rf"^{re.escape(name)}_(\d+)\.mp4$", re.IGNORECASE)
        maior = 0
        for arquivo in os.listdir(out_dir):
            match = pattern.match(arquivo)
            if match:
                maior = max(maior, int(match.group(1)))
        return maior + 1

    def run(self, cache, filename_prefix, fps_override, crf):
        man = _ler_manifesto(cache)
        total = int(man["frames"])
        fps = float(fps_override) if float(fps_override) > 0 else float(man.get("fps", 0.0))
        if fps <= 0:
            raise ValueError("[Bruxos Disk Stream] manifesto sem fps; informe fps_override.")
        width = int(man.get("largura", 0))
        height = int(man.get("altura", 0))
        if width <= 0 or height <= 0:
            raise ValueError("[Bruxos Disk Stream] dimensoes invalidas no manifesto.")
        if width % 2 or height % 2:
            raise ValueError(
                f"[Bruxos Disk Stream] H.264/yuv420p exige dimensoes pares; recebido {width}x{height}."
            )

        if folder_paths is not None:
            base = folder_paths.get_output_directory()
        else:
            base = os.path.abspath("output")
        subfolder = os.path.dirname(str(filename_prefix))
        name = os.path.basename(str(filename_prefix)) or "recovered_cache"
        out_dir = os.path.abspath(os.path.join(base, subfolder))
        if os.path.commonpath((os.path.abspath(base), out_dir)) != os.path.abspath(base):
            raise ValueError("[Bruxos Disk Stream] filename_prefix saiu da pasta output.")
        os.makedirs(out_dir, exist_ok=True)
        counter = self._next_counter(out_dir, name)
        final_path = os.path.join(out_dir, f"{name}_{counter:05d}.mp4")
        temp_path = os.path.join(
            out_dir, f".{name}_{counter:05d}.writing-{uuid.uuid4().hex[:8]}.mp4"
        )

        container = None
        gravados = 0
        inicio_t = time.perf_counter()
        try:
            import av
            container = av.open(temp_path, mode="w")
            stream = container.add_stream(
                "libx264", rate=Fraction(fps).limit_denominator(100000)
            )
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(int(crf)), "preset": "medium"}

            for bloco in sorted(man["blocos"], key=lambda item: int(item["inicio"])):
                tensor = _carregar_bloco(os.path.join(cache, bloco["arquivo"]))
                esperado = min(int(tensor.shape[0]), total - gravados)
                if esperado <= 0:
                    break
                if tensor.dtype != torch.uint8:
                    tensor = tensor.clamp(0, 255).to(torch.uint8)
                if tuple(tensor.shape[1:3]) != (height, width):
                    raise ValueError(
                        f"[Bruxos Disk Stream] bloco {bloco['arquivo']} tem "
                        f"{tensor.shape[2]}x{tensor.shape[1]}, esperado {width}x{height}."
                    )
                array = tensor[:esperado, ..., :3].contiguous().numpy()
                for frame_array in array:
                    frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    gravados += 1
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            container = None
            if gravados != total:
                raise RuntimeError(
                    f"[Bruxos Disk Stream] cache incompleto: codificados {gravados} de {total} frames."
                )

            from .optional import get_node_class
            _sv = get_node_class('BruxosSaveVideoV2', 'ComfyUI-Bruxos-MediaIO')
            if _sv is not None:
                _sv._validate_video(temp_path)
            os.replace(temp_path, final_path)
        except Exception:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
            try:
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            raise

        duracao = time.perf_counter() - inicio_t
        mb = os.path.getsize(final_path) / (1024 ** 2)
        info = (
            f"{gravados} frames | {width}x{height} | {fps:.3f} fps | "
            f"{mb:.1f} MiB | {duracao:.1f}s | sem audio | {final_path}"
        )
        rel_sub = os.path.relpath(out_dir, base)
        ui_ref = {
            "filename": os.path.basename(final_path),
            "subfolder": "" if rel_sub == "." else rel_sub,
            "type": "output",
            "format": "video/mp4",
        }
        print(f"[Bruxos Disk Stream] video recuperado | {info}", flush=True)
        return {
            "ui": {"images": [ui_ref], "animated": (True,)},
            "result": (final_path, gravados, fps, info),
        }


NODE_CLASS_MAPPINGS = {
    "BruxosVideoParaDisco": BruxosVideoParaDisco,
    "BruxosImagensParaDisco": BruxosImagensParaDisco,
    "BruxosDiscoLerJanela": BruxosDiscoLerJanela,
    "BruxosAbrirCacheDisco": BruxosAbrirCacheDisco,
    "BruxosCacheParaVideo": BruxosCacheParaVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosVideoParaDisco": "Video -> Disk Cache (Bruxos)",
    "BruxosImagensParaDisco": "Images -> Disk Cache (Bruxos)",
    "BruxosDiscoLerJanela": "Disk Cache -> Window (Bruxos)",
    "BruxosAbrirCacheDisco": "Open Disk Cache (Bruxos)",
    "BruxosCacheParaVideo": "Disk Cache -> Video (Bruxos)",
}
