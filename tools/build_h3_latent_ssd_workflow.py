"""Gera o laboratorio H3 Latent SSD a partir do workflow Streaming 124."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "workflows" / "Bruxos_H3_R2V_Lyonir_Streaming_124.json"
DST = ROOT / "workflows" / "Bruxos_H3_Laboratorio_Latent_SSD_124.json"

d = json.loads(SRC.read_text(encoding="utf-8-sig"))
nodes = d["nodes"]
links = d["links"]
by_id = {n["id"]: n for n in nodes}
next_link = max(x[0] for x in links) + 1


def link(origin, origin_slot, target, target_slot, typ):
    global next_link
    lid = next_link
    next_link += 1
    links.append([lid, origin, origin_slot, target, target_slot, typ])
    out = by_id[origin]["outputs"][origin_slot]
    if out.get("links") is None:
        out["links"] = []
    out["links"].append(lid)
    by_id[target]["inputs"][target_slot]["link"] = lid
    return lid


def remove_link(lid):
    global links
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


def inp(name, typ, widget=False, shape=None):
    x = {"name": name, "type": typ, "link": None}
    if widget:
        x["widget"] = {"name": name}
    if shape is not None:
        x["shape"] = shape
    return x


def out(name, typ):
    return {"name": name, "type": typ, "links": None}


def add(nid, typ, title, pos, size, inputs, outputs, widgets=None, mode=0, color="#243447", bgcolor="#354d66"):
    n = {
        "id": nid, "type": typ, "pos": list(pos), "size": list(size),
        "flags": {}, "order": 40 + len(nodes), "mode": mode,
        "inputs": inputs, "outputs": outputs, "title": title,
        "properties": {"Node name for S&R": typ}, "widgets_values": widgets or [],
        "color": color, "bgcolor": bgcolor,
    }
    nodes.append(n)
    by_id[nid] = n
    return n


def note(nid, title, text, pos, size=(620, 250)):
    return add(nid, "MarkdownNote", title, pos, size, [], [], [text], color="#432", bgcolor="#654")


# Desliga o antigo tail em pixels como Video 2. A memoria inicial e o marco
# continuam como referencias visuais fracas; a continuidade temporal passa a
# vir exclusivamente do tail latente.
old_tail_link = by_id[136]["inputs"][7].get("link")
remove_link(old_tail_link)

# Pictures 2/3 e o prompt de memoria eram parte da memoria em pixels. Neste
# laboratorio ficam fora: o tail latente nao e registrado no tokenizer e nao
# deve ser chamado de <Video 2> no texto.
remove_link(by_id[136]["inputs"][4].get("link"))
remove_link(by_id[136]["inputs"][5].get("link"))
remove_link(by_id[136]["inputs"][10].get("link"))

# Remove condicionamento direto 176 -> 126; o novo Latent->Ref entra no meio.
old_cond_link = by_id[126]["inputs"][1].get("link")
remove_link(old_cond_link)

# Titulos deixam evidente qual memoria esta ativa.
by_id[136]["title"] = "H3 Ref2VA · fonte + memórias visuais (sem tail pixel)"
by_id[188]["mode"] = 4
by_id[188]["title"] = "LEGADO BYPASS · ler memória em frames"
by_id[189]["mode"] = 4
by_id[189]["title"] = "LEGADO BYPASS · prompt para tail <Video 2>"
by_id[190]["mode"] = 4
by_id[190]["title"] = "LEGADO BYPASS · salvar tail em frames"

note(200, "LEIA · LABORATÓRIO H3 LATENT SSD", (
    "Este workflow substitui o tail temporal em PIXELS por tail LATENTE no SSD.\n\n"
    "Bloco 0: tem_memoria=false; Latent->Referencia fica automaticamente em bypass. "
    "Depois do sampler, Salvar SSD grava o tail do bloco 0.\n\n"
    "Bloco 1+: aumente somente indice_bloco no controlador. Ler SSD busca o bloco anterior "
    "e o injeta em minimax_refs sem decode/re-encode pelo VAE.\n\n"
    "O ramo Split->Pack->Info valida os eixos AV. O tail SSD é decodificado apenas no ramo de "
    "diagnóstico Divergencia->Juntar. O Join mostrado usa o tail anterior como A, portanto é um teste de costura, "
    "não o vídeo completo acumulado.\n\n"
    "Comece com carry_latents=12 (~39 frames) e ref_downscale=2x. O laboratório deixa "
    "sobrescrever=true para permitir repetir o mesmo índice durante a validação."
), (-1750, 7160), (900, 430))

add(201, "BruxosH3TailLatenteLerSSD", "A · Ler tail latente do bloco anterior", (-760, 5720), (410, 250), [
    inp("fallback_latent", "LATENT"), inp("nome_job", "STRING", True),
    inp("indice_bloco", "INT"), inp("carry_latents_fallback", "INT", True),
    inp("exigir_anterior", "BOOLEAN", True, 7),
], [out("tail_av", "LATENT"), out("tem_memoria", "BOOLEAN"), out("info", "STRING")],
    ["h3_latent_124_01", 12, True])

add(202, "BruxosH3LatentParaReferencia", "B · Tail latente → minimax_refs (2x)", (-260, 5660), (440, 370), [
    inp("conditioning", "CONDITIONING"), inp("latent_av", "LATENT"), inp("ativar", "BOOLEAN", shape=7),
    inp("carry_latents", "INT", True), inp("incluir_audio", "BOOLEAN", True, 7),
    inp("carry_video", "BOOLEAN", True, 7), inp("audio_latents", "INT", True),
    inp("ref_downscale", "COMBO", True),
], [out("conditioning", "CONDITIONING"), out("carried_frames", "INT"),
    out("factor_usado", "INT"), out("info", "STRING")], [12, True, True, 0, "2x"])

add(203, "BruxosH3TailLatenteSalvarSSD", "C · Salvar tail AV no SSD", (720, 5680), (420, 300), [
    inp("latent_av", "LATENT"), inp("nome_job", "STRING", True), inp("indice_bloco", "INT"),
    inp("carry_latents", "INT", True), inp("audio_latents", "INT", True),
    inp("sobrescrever", "BOOLEAN", True, 7),
], [out("pasta_job", "STRING"), out("info", "STRING")],
    ["h3_latent_124_01", 12, 0, True])

add(204, "BruxosH3SplitAV", "TESTE 1 · Split AV nos eixos corretos", (1220, 5680), (380, 190),
    [inp("latent_av", "LATENT")], [out("video_latent", "LATENT"), out("audio_latent", "LATENT"),
    out("video_t", "INT"), out("audio_t", "INT"), out("info", "STRING")])
add(205, "BruxosH3PackAV", "TESTE 2 · Pack AV + sincronizar áudio", (1660, 5680), (370, 150),
    [inp("video_latent", "LATENT"), inp("audio_latent", "LATENT")],
    [out("latent_av", "LATENT"), out("info", "STRING")])
add(206, "BruxosH3LatentInfoSSD", "TESTE 3 · Validar shapes/grid/tokens", (2090, 5680), (410, 190),
    [inp("latent", "LATENT")], [out("frames", "INT"), out("tokens_ref_frame", "INT"), out("info", "STRING")])

add(207, "BruxosH3EncontrarDivergencia", "TESTE 4 · Detectar repetição na emenda", (1210, 6120), (430, 310), [
    inp("anterior", "IMAGE"), inp("continuacao", "IMAGE"), inp("buscar_frames", "INT", True),
    inp("tail_anterior", "INT", True), inp("threshold", "FLOAT", True),
    inp("downsample", "INT", True), inp("comparar", "COMBO", True),
], [out("cortar_inicio_B", "INT"), out("erro", "FLOAT"), out("info", "STRING")],
    [96, 96, 0.05, 48, "estrutura"])

add(208, "BruxosH3JuntarAV", "TESTE 5 · Join em pixels + crossfade", (1710, 6120), (420, 250), [
    inp("imagens_a", "IMAGE"), inp("imagens_b", "IMAGE"), inp("cortar_inicio_b", "INT"),
    inp("crossfade_frames", "INT", True), inp("audio_a", "AUDIO"), inp("audio_b", "AUDIO"),
], [out("imagens", "IMAGE"), out("audio", "AUDIO"), out("info", "STRING")], [4])

add(209, "PreviewImage", "Prévia da emenda (tail anterior + bloco atual)", (2200, 6120), (360, 300),
    [inp("images", "IMAGE")], [])

add(210, "BruxosH3PlanejarFramesRef", "TESTE 6 · Planejar grade + downscale", (1210, 6620), (430, 260), [
    inp("segundos", "FLOAT", True), inp("largura", "INT", True), inp("altura", "INT", True),
    inp("downscale_desejado", "INT", True),
], [out("frames", "INT"), out("video_latents", "INT"), out("audio_latents", "INT"),
    out("ref_width", "INT"), out("ref_height", "INT"), out("info", "STRING")], [5.0, 1344, 768, 2])

note(211, "O QUE VALIDAR", (
    "1. Console: bloco 124 = 37V/207A.\n"
    "2. Tail padrão: 12V/65A = 39 frames.\n"
    "3. Canvas 1344x768: fatores válidos 1,2,3,6; pedido 4 deve ajustar para 3.\n"
    "4. Bloco 1+: Latent->Ref deve escrever ATIVA, fator usado 2x e 252 tokens/ref/frame.\n"
    "5. Split->Pack->Info deve retornar exatamente os shapes originais.\n"
    "6. Divergência: use o corte produzido no Join; mínimo fraco (<3x) pede inspeção visual.\n"
    "7. O tail e o bloco atual são decodificados separadamente; o Join une o áudio somente depois, "
    "no domínio waveform."
), (1210, 7080), (900, 340))

add(212, "VAEDecode", "Diagnóstico · decode somente do tail SSD", (710, 6120), (390, 90), [
    inp("samples", "LATENT"), inp("vae", "VAE"),
], [out("IMAGE", "IMAGE")], [], color="#232", bgcolor="#353")

add(213, "VAEDecodeAudio", "Diagnóstico · decode áudio do tail SSD", (710, 6250), (390, 90), [
    inp("samples", "LATENT"), inp("vae", "VAE"),
], [out("AUDIO", "AUDIO")], [], color="#232", bgcolor="#353")

# Wiring principal: latent vazio como fallback; estado anterior; condicionamento; sampler.
link(136, 1, 201, 0, "LATENT")
link(187, 2, 201, 2, "INT")
link(176, 0, 202, 0, "CONDITIONING")
link(201, 0, 202, 1, "LATENT")
link(201, 1, 202, 2, "BOOLEAN")
link(202, 0, 126, 1, "CONDITIONING")
link(161, 0, 136, 10, "STRING")

# Salvar e validar o resultado do sampler.
link(125, 0, 203, 0, "LATENT")
link(187, 2, 203, 2, "INT")
link(125, 0, 204, 0, "LATENT")
link(204, 0, 205, 0, "LATENT")
link(204, 1, 205, 1, "LATENT")
link(205, 0, 206, 0, "LATENT")

# Laboratorio de emenda em pixels.
link(201, 0, 212, 0, "LATENT")
link(160, 2, 212, 1, "VAE")
link(201, 0, 213, 0, "LATENT")
link(160, 3, 213, 1, "VAE")
link(212, 0, 207, 0, "IMAGE")
link(122, 0, 207, 1, "IMAGE")
link(212, 0, 208, 0, "IMAGE")
link(122, 0, 208, 1, "IMAGE")
link(207, 0, 208, 2, "INT")
link(213, 0, 208, 4, "AUDIO")
link(121, 0, 208, 5, "AUDIO")
link(208, 0, 209, 0, "IMAGE")

d["last_node_id"] = max(n["id"] for n in nodes)
d["last_link_id"] = max(x[0] for x in links)
d["revision"] = int(d.get("revision", 0)) + 1
d["groups"].append({
    "id": max(g["id"] for g in d["groups"]) + 1,
    "title": "H3 LATENT SSD · TESTES E VALIDAÇÃO",
    "bounding": [-820, 5600, 3420, 1880], "color": "#2b6f8a", "flags": {},
})

DST.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"workflow: {DST}\nnodes={len(nodes)} links={len(links)}")
