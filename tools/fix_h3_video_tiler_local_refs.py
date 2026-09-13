"""Corrige o workflow H3 VideoTiler para condicionamento local por ladrilho.

Preserva todos os modelos, arquivos e ajustes do usuario. O trecho externo que
empilhava tiles num unico IMAGE batch fica em bypass; os nodes Bruxos emitem
listas reais, permitindo ao ComfyUI mapear o ramo H3 uma vez por tile.
"""

import json
from pathlib import Path


SRC = Path(
    r"C:\Users\nyckm\Documents\c3\comfyEasy_01\ComfyUI-Easy-Install (2)\ComfyUI-Easy-Install"
    r"\ComfyUI\user\default\workflows\Bruxos_VideoTiler_MiniMaxH3 (2).json"
)
ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "workflows" / "Bruxos_VideoTiler_MiniMaxH3_LOCAL_REF_CORRIGIDO.json"

d = json.loads(SRC.read_text(encoding="utf-8-sig"))
nodes = d["nodes"]
links = d["links"]
by_id = {n["id"]: n for n in nodes}
next_link = max(x[0] for x in links) + 1


def remove_link(lid):
    if lid is None:
        return
    hit = next((x for x in links if x[0] == lid), None)
    if hit is None:
        return
    _, origin, origin_slot, target, target_slot, _ = hit
    links[:] = [x for x in links if x[0] != lid]
    source = by_id[origin]["outputs"][origin_slot]
    source["links"] = [x for x in (source.get("links") or []) if x != lid] or None
    by_id[target]["inputs"][target_slot]["link"] = None


def wire(origin, origin_slot, target, target_slot, typ):
    global next_link
    old = by_id[target]["inputs"][target_slot].get("link")
    remove_link(old)
    lid = next_link
    next_link += 1
    links.append([lid, origin, origin_slot, target, target_slot, typ])
    source = by_id[origin]["outputs"][origin_slot]
    if source.get("links") is None:
        source["links"] = []
    source["links"].append(lid)
    by_id[target]["inputs"][target_slot]["link"] = lid
    return lid


def inp(name, typ, widget=False, shape=None):
    x = {"name": name, "type": typ, "link": None}
    if widget:
        x["widget"] = {"name": name}
    if shape is not None:
        x["shape"] = shape
    return x


def out(name, typ):
    return {"name": name, "type": typ, "links": None}


def add(nid, typ, title, pos, size, inputs, outputs, widgets=None,
        color="#243447", bgcolor="#354d66"):
    n = {
        "id": nid, "type": typ, "pos": list(pos), "size": list(size),
        "flags": {}, "order": 90 + len(nodes), "mode": 0,
        "inputs": inputs, "outputs": outputs, "title": title,
        "properties": {"Node name for S&R": typ},
        "widgets_values": widgets or [], "color": color, "bgcolor": bgcolor,
    }
    nodes.append(n)
    by_id[nid] = n
    return n


def note(nid, title, text, pos, size=(760, 300)):
    return add(nid, "MarkdownNote", title, pos, size, [], [], [text], "#432", "#654")


# O caminho antigo empilhava todos os tiles num IMAGE batch. Isso e correto para
# um upscaler frame-a-frame, mas o H3 interpreta batch como tempo. Deixamos tudo
# visivel em bypass para comparacao, sem apagar o trabalho do usuario.
for nid in (7463, 7527, 7578, 7579, 7580, 7583, 7597, 7602, 7603,
            7683, 7684, 7722, 7854, 7926):
    if nid in by_id:
        by_id[nid]["mode"] = 4
        old_title = by_id[nid].get("title") or by_id[nid]["type"]
        if not old_title.startswith("LEGADO BYPASS"):
            by_id[nid]["title"] = "LEGADO BYPASS · " + old_title

# Remove especificamente os caminhos globais incorretos antes de ligar os novos.
remove_link(by_id[7832]["inputs"][0].get("link"))       # Get_VID_tiles -> encode
remove_link(by_id[7916]["inputs"][3].get("link"))       # imagem global -> Picture 1
remove_link(by_id[7916]["inputs"][9].get("link"))       # tamanho via batch externo
remove_link(by_id[7916]["inputs"][10].get("link"))
remove_link(by_id[7831]["outputs"][0].get("links", [None])[0] if by_id[7831]["outputs"][0].get("links") else None)
remove_link(by_id[7864]["inputs"][0].get("link"))       # merge externo -> saida

add(7950, "BruxosTileSlice", "1 · Tiles como LISTA real (um render H3 por tile)",
    (-3890, 6910), (430, 260), [
        inp("imagens", "IMAGE"), inp("ladrilho_w", "INT", True),
        inp("ladrilho_h", "INT", True), inp("multiplo", "INT", True),
        inp("sobreposicao", "FLOAT", True),
    ], [out("ladrilhos", "IMAGE"), out("tile_config", "TILE_CONFIG"),
        out("mapa", "IMAGE"), out("quantidade", "INT"), out("info", "STRING")],
    [1568, 672, 32, 0.25, "espiral"])

add(7951, "BruxosTileRefSlice", "2 · Recorta a referência na MESMA posição",
    (-3370, 7010), (430, 120), [
        inp("referencia", "IMAGE"), inp("tile_config", "TILE_CONFIG"),
    ], [out("ladrilhos_ref", "IMAGE"), out("quantidade", "INT")], [])

add(7952, "BruxosVideoTileMerge", "5 · Reúne os tiles com feather simétrico",
    (-1740, 6950), (430, 190), [
        inp("tile_config", "TILE_CONFIG"), inp("ladrilhos", "IMAGE"),
    ], [out("imagem", "IMAGE")], [0.20, "suave_ambos", "media_ponderada", "cpu"])

add(7953, "PreviewImage", "VALIDAR ANTES · mapa das regiões",
    (-3890, 7240), (400, 300), [inp("images", "IMAGE")], [], [])
add(7955, "PreviewImage", "VALIDAR · referências locais (devem ser diferentes)",
    (-3370, 7180), (430, 300), [inp("images", "IMAGE")], [], [])
add(7956, "PreviewImage", "VALIDAR · tiles do vídeo fonte",
    (-2890, 7180), (430, 300), [inp("images", "IMAGE")], [], [])

add(7957, "BruxosTileColorMatch", "6 · Uniformiza apenas cor/luz após a costura",
    (-1240, 6950), (450, 250), [
        inp("fundido", "IMAGE"), inp("referencia", "IMAGE"),
        inp("sigma", "FLOAT", True), inp("puxada", "FLOAT", True),
        inp("detalhe", "FLOAT", True), inp("travar_luminancia", "BOOLEAN", True, 7),
        inp("limite_luma", "FLOAT", True),
    ], [out("imagem", "IMAGE")], [18.0, 0.30, 1.0, True, 4.0])

note(7954, "CORREÇÃO · POR QUE A CENA SE REPETIA", (
    "O fluxo anterior juntava todos os tiles num único tensor IMAGE. O encoder H3 tratava esse "
    "batch como sequência temporal, e o Ref2Video recebia sempre a imagem inteira como Picture 1. "
    "Resultado: cada tile tentava reconstruir a cena completa.\n\n"
    "Agora Tile Slice e Tile Ref Slice emitem duas listas de mesmo tamanho. O ComfyUI faz zip: "
    "tile de vídeo 0 + referência 0, tile 1 + referência 1 etc. O tile fonte também entra como "
    "<Video 1>, tornando o @video1 do prompt verdadeiro.\n\n"
    "Antes de gastar GPU, confira as duas prévias: os tiles e as referências precisam mostrar "
    "partes diferentes e complementares da imagem. Para preservar mais a geometria, reduza o "
    "denoise do BasicScheduler de 0.70 para 0.45."
), (-2890, 7540), (1080, 380))

# Geometria. Reutiliza o Tile Grade 7940 e a imagem de trabalho 7533.
wire(7533, 0, 7950, 0, "IMAGE")
wire(7940, 0, 7950, 1, "INT")
wire(7940, 1, 7950, 2, "INT")
wire(7940, 2, 7950, 3, "INT")
wire(7940, 3, 7950, 4, "FLOAT")
wire(7946, 0, 7951, 0, "IMAGE")
wire(7950, 1, 7951, 1, "TILE_CONFIG")

# Um item de cada lista percorre o ramo H3. As duas listas tem a mesma ordem.
wire(7950, 0, 7832, 0, "IMAGE")
wire(7951, 0, 7916, 3, "IMAGE")       # Picture 1 local: estilo/aparencia
wire(7950, 0, 7916, 5, "IMAGE")       # Video 1 local: movimento/geometria
wire(7940, 0, 7916, 9, "INT")
wire(7940, 1, 7916, 10, "INT")

# Cada decode retorna um item; o merge declara INPUT_IS_LIST e recebe todos.
wire(7950, 1, 7952, 0, "TILE_CONFIG")
wire(7831, 0, 7952, 1, "IMAGE")
wire(7952, 0, 7957, 0, "IMAGE")
wire(7946, 0, 7957, 1, "IMAGE")
wire(7957, 0, 7864, 0, "IMAGE")

# Diagnosticos baratos para verificar a geometria antes do sampler.
wire(7950, 2, 7953, 0, "IMAGE")
wire(7951, 0, 7955, 0, "IMAGE")
wire(7950, 0, 7956, 0, "IMAGE")

# Mais conservador para video-to-video ladrilhado: mantem a estrutura local.
if by_id[7922].get("widgets_values") and len(by_id[7922]["widgets_values"]) >= 3:
    by_id[7922]["widgets_values"][0] = "simple"
    by_id[7922]["widgets_values"][1] = max(8, int(by_id[7922]["widgets_values"][1]))
    by_id[7922]["widgets_values"][2] = 0.45
    by_id[7922]["title"] = "BasicScheduler · V2V local (denoise 0.45)"

by_id[7916]["title"] = "3 · H3 recebe Picture local + Video local"
by_id[7832]["title"] = "3b · Encode somente deste tile"
by_id[299]["title"] = "4 · Sample somente deste tile"
by_id[7831]["title"] = "4b · Decode somente deste tile"

d["last_node_id"] = max(n["id"] for n in nodes)
d["last_link_id"] = max(x[0] for x in links)
d["revision"] = int(d.get("revision", 0)) + 1
d.setdefault("groups", []).append({
    "id": max([g.get("id", 0) for g in d.get("groups", [])] + [0]) + 1,
    "title": "CORREÇÃO H3 · REFERÊNCIA LOCAL POR TILE",
    "bounding": [-3970, 6810, 3260, 1160],
    "color": "#2b8a78", "flags": {},
})

DST.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"workflow corrigido: {DST}\nnodes={len(nodes)} links={len(links)}")
