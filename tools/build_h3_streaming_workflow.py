"""Build the Lyonir H3 124-frame rolling-memory workflow from the user's base."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def inp(name, typ, link=None, widget=False, shape=None, label=None):
    x = {"localized_name": name, "name": name, "type": typ, "link": link}
    if widget:
        x["widget"] = {"name": name}
    if shape is not None:
        x["shape"] = shape
    if label is not None:
        x["label"] = label
    return x


def out(name, typ, links=None, slot=None):
    x = {"localized_name": name, "name": name, "type": typ, "links": links}
    if slot is not None:
        x["slot_index"] = slot
    return x


def node(nid, typ, pos, size, inputs, outputs, widgets, title=None, mode=0):
    n = {
        "id": nid, "type": typ, "pos": list(pos), "size": list(size),
        "flags": {}, "order": 0, "mode": mode, "inputs": inputs,
        "outputs": outputs, "properties": {"Node name for S&R": typ},
        "widgets_values": widgets,
    }
    if title:
        n["title"] = title
    return n


def validate(data):
    nodes = {n["id"]: n for n in data["nodes"]}
    if len(nodes) != len(data["nodes"]):
        raise ValueError("duplicate node ids")
    links = {l[0]: l for l in data["links"]}
    if len(links) != len(data["links"]):
        raise ValueError("duplicate link ids")
    for lid, src, src_slot, dst, dst_slot, typ in data["links"]:
        if src not in nodes or dst not in nodes:
            raise ValueError(f"link {lid}: missing endpoint {src}->{dst}")
        if not 0 <= src_slot < len(nodes[src].get("outputs", [])):
            raise ValueError(f"link {lid}: invalid source slot {src}:{src_slot}")
        if not 0 <= dst_slot < len(nodes[dst].get("inputs", [])):
            raise ValueError(f"link {lid}: invalid destination slot {dst}:{dst_slot}")
        so = nodes[src]["outputs"][src_slot]
        di = nodes[dst]["inputs"][dst_slot]
        if lid not in (so.get("links") or []):
            raise ValueError(f"link {lid}: absent from source socket")
        if di.get("link") != lid:
            raise ValueError(f"link {lid}: absent from destination socket")
        if so.get("type") not in (typ, "*") or di.get("type") not in (typ, "*"):
            raise ValueError(
                f"link {lid}: type mismatch {so.get('type')} -> {typ} -> {di.get('type')}"
            )
    for n in data["nodes"]:
        for x in n.get("inputs", []):
            if x.get("link") is not None and x["link"] not in links:
                raise ValueError(f"node {n['id']}: dangling input link {x['link']}")
        for x in n.get("outputs", []):
            for lid in x.get("links") or []:
                if lid not in links:
                    raise ValueError(f"node {n['id']}: dangling output link {lid}")


def build(src: Path, dst: Path):
    data = json.loads(src.read_text(encoding="utf-8"))
    remove_nodes = {155, 170}
    remove_links = {325, 389, 395, 396}
    for link in data["links"]:
        if link[1] in remove_nodes or link[3] in remove_nodes:
            remove_links.add(link[0])

    # Remove stale references to deleted links from retained sockets.
    for n in data["nodes"]:
        for x in n.get("inputs", []):
            if x.get("link") in remove_links:
                x["link"] = None
        for x in n.get("outputs", []):
            if x.get("links") is not None:
                x["links"] = [lid for lid in x["links"] if lid not in remove_links]
    data["nodes"] = [n for n in data["nodes"] if n["id"] not in remove_nodes]
    data["links"] = [l for l in data["links"] if l[0] not in remove_links]

    nodes = {n["id"]: n for n in data["nodes"]}
    h3 = nodes[136]

    # Autogrow order: 3 images, 2 videos, optional audios, then widgets.
    h3["inputs"] = [
        inp("clip", "CLIP", 319),
        inp("vae", "VAE", 320),
        inp("audio_vae", "VAE", 321),
        inp("ref_images.ref_image_0", "IMAGE", 394, shape=7, label="ref_image_0"),
        inp("ref_images.ref_image_1", "IMAGE", shape=7, label="ref_image_1"),
        inp("ref_images.ref_image_2", "IMAGE", shape=7, label="ref_image_2"),
        inp("ref_videos.ref_video_0", "IMAGE", shape=7, label="ref_video_0"),
        inp("ref_videos.ref_video_1", "IMAGE", shape=7, label="ref_video_1"),
        inp("ref_video_audios.ref_video_audio_0", "AUDIO", shape=7, label="ref_video_audio_0"),
        inp("ref_audios.ref_audio_0", "AUDIO", shape=7, label="ref_audio_0"),
        inp("prompt", "STRING", widget=True),
        inp("width", "INT", 276, widget=True),
        inp("height", "INT", 277, widget=True),
        inp("length", "INT", widget=True),
        inp("ref_image_size", "COMBO", widget=True),
    ]
    h3["size"] = [430, 492]
    h3["title"] = "H3 Ref2VA · bloco atual + tail + memórias"

    # Existing links into H3 whose destination slots moved after Autogrow growth.
    for link in data["links"]:
        if link[0] == 276:
            link[4] = 11
        elif link[0] == 277:
            link[4] = 12

    # Prompt base: Picture 1 is no longer declared as a mandatory first frame
    # on every block. The rolling prompt assigns that role only on block zero.
    prompt_node = nodes[161]
    prompt_node["widgets_values"][0] = (
        "#previz The camera path, staging, geometry and timing follow @video1 exactly\n\n"
        "#cenario @img1 #estilo\n#invariancia"
    )
    prompt_node["widgets_values"][1] = 5.0
    prompt_node["title"] = "Direção base · sem reancorar Picture 1"

    # Save each queue as an individual block file as well as the rolling state.
    nodes[148]["widgets_values"][0] = "Bruxos/H3_Lyonir_Rolling/bloco"
    nodes[148]["title"] = "Salvar MP4 do bloco atual (áudio + vídeo)"
    nodes[168]["title"] = "Comparar fonte atual × bloco gerado"

    added = [
        node(185, "BruxosVideoParaDisco", (-1750, 6180), (350, 420), [
            inp("video", "COMBO", widget=True), inp("nome_cache", "STRING", widget=True),
            inp("frames_por_bloco", "INT", widget=True), inp("pular_frames", "INT", widget=True),
            inp("limite_frames", "INT", widget=True), inp("pegar_cada_n", "INT", widget=True),
            inp("force_rate", "FLOAT", widget=True), inp("largura", "INT", widget=True),
            inp("altura", "INT", widget=True), inp("sobrescrever", "BOOLEAN", widget=True, shape=7),
            inp("video_path", "STRING", widget=True),
        ], [
            out("cache", "BRUXOS_FRAME_CACHE"), out("frames", "INT"),
            out("fps", "FLOAT"), out("info", "STRING"),
        ], ["IMG_9016_2.mp4", "lyonir_source_24fps", 31, 0, 0, 1, 24.0, 720, 1280, False, ""],
             "1 · Vídeo fonte → cache SSD (uma vez)"),

        node(186, "BruxosDiscoLerJanela", (-1260, 6180), (390, 340), [
            inp("cache", "BRUXOS_FRAME_CACHE"), inp("inicio", "INT", widget=True),
            inp("quantidade", "INT", widget=True), inp("sobreposicao", "INT", widget=True),
            inp("prefetch", "BOOLEAN", widget=True, shape=7),
            inp("pin_memory", "BOOLEAN", widget=True, shape=7),
            inp("completar_ultimo", "BOOLEAN", widget=True, shape=7),
        ], [
            out("imagens", "IMAGE"), out("proximo_inicio", "INT"), out("total_frames", "INT"),
            out("terminou", "BOOLEAN"), out("fps", "FLOAT"), out("info", "STRING"),
        ], [0, 124, 0, True, False, True], "2 · Ler somente o bloco fonte de 124"),

        node(187, "BruxosH3Bloco124", (-1740, 5740), (350, 180), [
            inp("indice_bloco", "INT", widget=True), inp("sobreposicao_fonte", "INT", widget=True),
        ], [
            out("inicio_fonte", "INT"), out("length", "INT"),
            out("indice_bloco", "INT"), out("info", "STRING"),
        ], [0, 0], "CONTROLE · altere somente indice_bloco: 0, 1, 2…"),

        node(188, "BruxosH3MemoriaLer", (-790, 6180), (400, 300), [
            inp("fallback_atual", "IMAGE"), inp("nome_job", "STRING", widget=True),
            inp("indice_bloco", "INT"), inp("tail_frames", "INT", widget=True),
            inp("exigir_anterior", "BOOLEAN", widget=True, shape=7),
        ], [
            out("tail_video", "IMAGE"), out("memoria_inicial", "IMAGE"),
            out("memoria_marco", "IMAGE"), out("ultimo_frame", "IMAGE"), out("info", "STRING"),
        ], ["lyonir_rolling_01", 56, True], "3 · Ler tail 56 + memórias do bloco anterior"),

        node(189, "BruxosH3MemoriaPrompt", (-300, 6250), (420, 210), [
            inp("prompt_base", "STRING"), inp("indice_bloco", "INT"),
            inp("tail_frames", "INT", widget=True),
        ], [out("prompt", "STRING"), out("info", "STRING")], [56],
             "4 · Dar papéis às memórias (não são âncoras totais)"),

        node(190, "BruxosH3MemoriaSalvar", (710, 6200), (410, 300), [
            inp("imagens", "IMAGE"), inp("nome_job", "STRING", widget=True),
            inp("indice_bloco", "INT"), inp("tail_frames", "INT", widget=True),
            inp("sobrescrever_bloco", "BOOLEAN", widget=True, shape=7),
        ], [
            out("bloco", "IMAGE"), out("tail", "IMAGE"), out("memoria_inicial", "IMAGE"),
            out("memoria_marco", "IMAGE"), out("job", "STRING"), out("info", "STRING"),
        ], ["lyonir_rolling_01", 56, False], "5 · Salvar estado para a próxima fila"),

        node(191, "MarkdownNote", (-1770, 4680), (720, 820), [], [], [""],
             "COMO RODAR · H3 STREAMING 124 + MEMÓRIA ROLANTE"),
    ]

    added[-1]["widgets_values"] = [
        "# H3 Lyonir · blocos de 124 frames\n\n"
        "Este workflow NÃO carrega o vídeo-fonte inteiro. Ele mantém em memória apenas o bloco atual "
        "(124 frames), o tail anterior (56 frames) e duas imagens de memória.\n\n"
        "## Primeira execução\n"
        "1. Reinicie o ComfyUI depois de instalar/atualizar os nodes Bruxos.\n"
        "2. Em **Vídeo fonte → cache SSD**, selecione o vídeo e dê um `nome_cache` novo.\n"
        "3. Em **CONTROLE**, deixe `indice_bloco = 0`.\n"
        "4. Os nodes Ler/Salvar devem usar exatamente o mesmo `nome_job`.\n"
        "5. Enfileire e espere terminar. O estado do bloco 0 será salvo no SSD.\n\n"
        "## Próximos blocos\n"
        "Mude somente `indice_bloco` para 1, depois 2, 3… e execute UM por vez. Não enfileire vários "
        "índices antes do anterior terminar, porque cada bloco depende do tail recém-salvo.\n\n"
        "## Memória\n"
        "- Video 1: trecho atual do vídeo-fonte/previz.\n"
        "- Video 2: tail de 56 frames gerado no bloco anterior.\n"
        "- Picture 2: memória permanente do começo; identidade/aparência apenas.\n"
        "- Picture 3: marco do bloco anterior; somente mudanças que ainda são verdadeiras.\n"
        "As Pictures 2/3 NÃO mandam voltar à pose ou composição antiga.\n\n"
        "## Saída\n"
        "Cada fila salva um MP4 separado, com áudio, e um checkpoint float16 do bloco no SSD. "
        "O original não é alterado. Para começar outra sequência, troque `nome_job`."
    ]

    data["nodes"].extend(added)
    nodes = {n["id"]: n for n in data["nodes"]}
    next_link = max([l[0] for l in data["links"]] + [0]) + 1

    def add_link(src_id, src_slot, dst_id, dst_slot, typ):
        nonlocal next_link
        lid = next_link
        next_link += 1
        data["links"].append([lid, src_id, src_slot, dst_id, dst_slot, typ])
        links = nodes[src_id]["outputs"][src_slot].get("links")
        if links is None:
            links = []
            nodes[src_id]["outputs"][src_slot]["links"] = links
        links.append(lid)
        nodes[dst_id]["inputs"][dst_slot]["link"] = lid
        return lid

    # Cache/source block.
    add_link(185, 0, 186, 0, "BRUXOS_FRAME_CACHE")
    add_link(187, 0, 186, 1, "INT")
    add_link(187, 1, 186, 2, "INT")
    add_link(186, 0, 188, 0, "IMAGE")

    # Shared block index and exact H3 length.
    add_link(187, 2, 188, 2, "INT")
    add_link(187, 2, 189, 1, "INT")
    add_link(187, 2, 190, 2, "INT")
    add_link(187, 1, 136, 13, "INT")

    # Dynamic prompt roles.
    add_link(161, 0, 189, 0, "STRING")
    add_link(189, 0, 136, 10, "STRING")

    # Reference order documented in the dynamic prompt.
    add_link(188, 1, 136, 4, "IMAGE")   # Picture 2: permanent first memory
    add_link(188, 2, 136, 5, "IMAGE")   # Picture 3: recent milestone
    add_link(186, 0, 136, 6, "IMAGE")   # Video 1: current source block
    add_link(188, 0, 136, 7, "IMAGE")   # Video 2: previous generated tail

    # Compare current source against result and persist state after decode.
    add_link(186, 0, 168, 0, "IMAGE")
    add_link(122, 0, 190, 0, "IMAGE")

    # Recompute execution order conservatively: existing order remains useful,
    # new nodes are appended in dependency-friendly order.
    max_order = max(n.get("order", 0) for n in data["nodes"] if n["id"] < 185)
    for offset, nid in enumerate([191, 185, 187, 186, 188, 189, 190], start=1):
        nodes[nid]["order"] = max_order + offset

    data["last_node_id"] = max(n["id"] for n in data["nodes"])
    data["last_link_id"] = max(l[0] for l in data["links"])
    extra = data.setdefault("extra", {})
    extra["bruxos_h3_streaming"] = {
        "version": 1, "block_frames": 124, "tail_frames": 56,
        "source": str(src), "notes": "Manual sequential queue: block 0, then 1, 2, 3...",
    }

    validate(data)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("destination", type=Path)
    args = ap.parse_args()
    data = build(args.source, args.destination)
    print(f"wrote {args.destination} | nodes={len(data['nodes'])} links={len(data['links'])}")


if __name__ == "__main__":
    main()
