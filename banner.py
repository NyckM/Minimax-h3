# -*- coding: utf-8 -*-
"""Banner de inicializacao dos pacotes Bruxos do VFX.

Uma linha so: "BRUXOS DO VFX · <area>  v2 · N nodes".
Sem logo em ASCII — com 14 pacotes instalados, o console virava uma parede.
"""
import os
import sys

VERDE = (74, 222, 128)
ROXO = (168, 85, 247)


def _cor(texto, rgb, negrito=False):
    if not _tem_cor():
        return texto
    r, g, b = rgb
    pre = "\033[1m" if negrito else ""
    return f"{pre}\033[38;2;{r};{g};{b}m{texto}\033[0m"


def _fraco(texto):
    return texto if not _tem_cor() else f"\033[2m{texto}\033[0m"


def _tem_cor():
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("BRUXOS_BANNER", "").lower() in ("0", "off", "plain"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def print_banner(area="", node_count=0, version="v2"):
    """Imprime a linha de boot do pacote.

    area       : o que este pacote faz (ex.: "MiniMax-H3", "Media I/O")
    node_count : quantos nodes foram registrados
    version    : versao do pacote
    """
    if os.environ.get("BRUXOS_BANNER", "").lower() in ("0", "off", "none"):
        return
    v = version if str(version).startswith("v") else f"v{version}"
    marca = _cor("BRUXOS DO VFX", VERDE, negrito=True)
    nome = _cor(str(area), ROXO) if area else ""
    cauda = _fraco(f"{v} · {int(node_count)} nodes")
    meio = f" · {nome} " if nome else " "
    print(f"  {marca}{meio}{cauda}")
