"""Ponte OPCIONAL entre pacotes Bruxos separados.

Depois da divisao do pacote unico em varios repositorios, alguns nodes ainda
querem usar codigo que ficou em OUTRO repositorio (ex.: o Bernini SSD-81 usa o
cache de frames em disco, que mora no ComfyUI-Bruxos-MediaIO).

Importar direto nao funciona: as pastas tem "-" no nome e o Python nao aceita
isso num import. Entao resolvemos em tempo de execucao, por dois caminhos:

  1. get_node_class("BruxosDiscoLerJanela")
     -> procura no NODE_CLASS_MAPPINGS global do ComfyUI (o jeito idiomatico;
        funciona com QUALQUER nome de pasta).

  2. import_from_pack("ComfyUI-Bruxos-MediaIO", "bruxos_disk_stream", ["ler_config"])
     -> acha a pasta do pacote dentro de custom_nodes e carrega o arquivo .py
        direto pelo caminho, pra pegar funcoes soltas (que nao sao nodes).

Se o outro pacote nao estiver instalado, os dois devolvem None / levantam um
erro em portugues dizendo exatamente o que instalar — nunca quebram o boot.
"""

import importlib.util
import logging
import os
import sys

log = logging.getLogger(__name__)

_CLS_CACHE = {}
_MOD_CACHE = {}


def get_node_class(node_id, pacote=None, obrigatorio=False):
    """Devolve a CLASSE de um node ja registrado no ComfyUI, pelo id dele.

    node_id     : id no NODE_CLASS_MAPPINGS (ex.: "BruxosDiscoLerJanela")
    pacote      : nome do repositorio que fornece esse node (so pra mensagem)
    obrigatorio : se True e nao achar, levanta RuntimeError em PT.
    """
    if node_id in _CLS_CACHE:
        cls = _CLS_CACHE[node_id]
        if cls is not None:
            return cls

    cls = None
    try:
        import nodes as _comfy_nodes
        cls = getattr(_comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(node_id)
    except Exception:
        cls = None

    if cls is not None:
        _CLS_CACHE[node_id] = cls
        return cls

    msg = (
        f"[Bruxos] o node '{node_id}' nao esta instalado."
        + (f" Ele vem do pacote {pacote}." if pacote else "")
    )
    if obrigatorio:
        raise RuntimeError(msg)
    log.info(msg + " (seguindo sem ele)")
    return None


def _custom_nodes_dirs():
    """Lista as pastas custom_nodes que o ComfyUI esta usando."""
    dirs = []
    try:
        import folder_paths
        for d in folder_paths.get_folder_paths("custom_nodes"):
            if os.path.isdir(d):
                dirs.append(d)
    except Exception:
        pass
    if not dirs:
        # fallback: sobe a partir deste arquivo ate achar "custom_nodes"
        here = os.path.abspath(os.path.dirname(__file__))
        while here and here != os.path.dirname(here):
            if os.path.basename(here) == "custom_nodes":
                dirs.append(here)
                break
            here = os.path.dirname(here)
    return dirs


def find_pack(*nomes_de_pasta):
    """Acha a pasta de um pacote irmao dentro de custom_nodes.

    Aceita varios nomes (o usuario pode ter clonado com outro nome). A
    comparacao ignora maiusculas, '-', '_' e sufixos tipo '-main'.
    """
    alvos = set()
    for n in nomes_de_pasta:
        alvos.add(n.lower().replace("-", "").replace("_", ""))
    for base in _custom_nodes_dirs():
        try:
            for item in os.listdir(base):
                p = os.path.join(base, item)
                if not os.path.isdir(p):
                    continue
                chave = item.lower().replace("-", "").replace("_", "")
                for suf in ("main", "master"):
                    if chave.endswith(suf):
                        chave_sem = chave[: -len(suf)]
                        if chave_sem in alvos:
                            return p
                if chave in alvos:
                    return p
        except Exception:
            continue
    return None


def import_from_pack(pacote, modulo, nomes=None, obrigatorio=False):
    """Carrega um modulo .py de um pacote Bruxos irmao, pelo caminho do arquivo.

    pacote  : nome (ou lista de nomes) da pasta em custom_nodes
    modulo  : nome do arquivo sem .py (ex.: "bruxos_video_tiler")
    nomes   : se passado, devolve uma TUPLA com esses atributos do modulo;
              senao devolve o proprio modulo.
    """
    if isinstance(pacote, str):
        pacote = [pacote]
    chave = (tuple(pacote), modulo)

    mod = _MOD_CACHE.get(chave)
    if mod is None:
        pasta = find_pack(*pacote)
        if pasta:
            caminho = os.path.join(pasta, modulo.replace(".", os.sep) + ".py")
            if os.path.isfile(caminho):
                try:
                    nome_unico = f"_bruxos_ext_{os.path.basename(pasta)}_{modulo}"
                    nome_unico = nome_unico.replace("-", "_").replace(".", "_")
                    if nome_unico in sys.modules:
                        mod = sys.modules[nome_unico]
                    else:
                        spec = importlib.util.spec_from_file_location(nome_unico, caminho)
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[nome_unico] = mod
                        spec.loader.exec_module(mod)
                    _MOD_CACHE[chave] = mod
                except Exception as e:
                    log.warning(f"[Bruxos] falhei ao carregar {modulo} de {pasta}: {e}")
                    mod = None

    if mod is None:
        msg = (
            f"[Bruxos] modulo '{modulo}' nao encontrado. "
            f"Instale o pacote: {' ou '.join(pacote)}"
        )
        if obrigatorio:
            raise RuntimeError(msg)
        log.info(msg + " (seguindo sem ele)")
        if nomes:
            return tuple(None for _ in nomes)
        return None

    if nomes:
        return tuple(getattr(mod, n, None) for n in nomes)
    return mod
