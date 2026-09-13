"""Testes de CPU para bruxos_h3_row_chunk, sem torch e sem ComfyUI.

O node so pode ser exercitado de verdade numa GPU com um MiniMax H3 carregado.
Estes testes cobrem a parte que da para verificar sem isso: o calculo do
tamanho do chunk, o alinhamento, a deteccao de interop e — o mais importante —
o controle de fluxo dos pesos retidos, incluindo os caminhos de fallback e a
propagacao do cancelamento.

Rodar da raiz do pacote:

    python tests/test_h3_row_chunk.py
"""

import os
import sys
import types


# ------------------------------------------------- stubs de torch e comfy

def _install_stubs():
    """Injeta torch e comfy falsos suficientes para importar e exercitar o node."""

    torch = types.ModuleType("torch")

    class FakeT:
        """Tensor falso que so sabe fatiar linhas e receber atribuicao."""

        dtype = "bf16"
        device = "cpu"

        def __init__(self, n=0):
            self.n = n
            self.writes = []

        @property
        def shape(self):
            return (self.n,)

        def element_size(self):
            return 2

        def __setitem__(self, key, value):
            self.writes.append((key.start, key.stop))

        def __getitem__(self, key):
            start = key.start or 0
            stop = self.n if key.stop is None else min(key.stop, self.n)
            return FakeT(max(0, stop - start))

        def chunk(self, n, dim=-1):
            return tuple(FakeT(self.n) for _ in range(n))

        def mul_(self, other):
            return self

    torch.FakeT = FakeT
    torch.Tensor = FakeT
    torch.long = "long"
    torch.empty = lambda *a, **k: FakeT()
    torch.empty_like = lambda x, *a, **k: FakeT(getattr(x, "n", 0))
    torch.is_tensor = lambda x: isinstance(x, FakeT)

    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")
    functional.linear = lambda x, w, b=None: FakeT(getattr(x, "n", 0))
    functional.silu = lambda x: FakeT(getattr(x, "n", 0))
    nn.functional = functional
    torch.nn = nn

    comfy = types.ModuleType("comfy")
    ops = types.ModuleType("comfy.ops")
    mm = types.ModuleType("comfy.model_management")
    quant_ops = types.ModuleType("comfy.quant_ops")

    class InterruptProcessingException(Exception):
        pass

    mm.InterruptProcessingException = InterruptProcessingException

    class QuantizedTensor:
        pass

    class TensorWiseINT8Layout:
        pass

    quant_ops.QuantizedTensor = QuantizedTensor
    quant_ops.TensorWiseINT8Layout = TensorWiseINT8Layout

    comfy.ops = ops
    comfy.model_management = mm
    comfy.quant_ops = quant_ops

    sys.modules.update({
        "torch": torch,
        "torch.nn": nn,
        "torch.nn.functional": functional,
        "comfy": comfy,
        "comfy.ops": ops,
        "comfy.model_management": mm,
        "comfy.quant_ops": quant_ops,
    })
    return torch, ops, InterruptProcessingException


TORCH, OPS, Interrupt = _install_stubs()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bruxos_h3_row_chunk as M  # noqa: E402


FAILS = []
CALLS = {"module": 0, "cast": 0, "uncast": 0, "every": 0}
CAST_MODE = {"mode": "ok"}


def check(name, got, want):
    if got != want:
        FAILS.append("%s: %r != %r" % (name, got, want))
        print("  FALHA  %s: %r != %r" % (name, got, want))
    else:
        print("  ok  %s" % name)


def reset():
    CALLS.update(module=0, cast=0, uncast=0, every=0)


def _wire_ops():
    def linear_input_act(fc2, x, act):
        CALLS["module"] += 1
        return TORCH.FakeT(getattr(x, "n", 0))

    def run_every_op():
        CALLS["every"] += 1

    def cast_bias_weight(module, sample, **kwargs):
        CALLS["cast"] += 1
        if CAST_MODE["mode"] == "missing":
            raise TypeError("versao antiga do ComfyUI")
        stream = "MESMO" if CAST_MODE["mode"] == "alias" else object()
        return ("peso%d" % CALLS["cast"], None, (stream,))

    def uncast_bias_weight(module, weight, bias, handle):
        CALLS["uncast"] += 1

    OPS.linear_input_act = linear_input_act
    OPS.run_every_op = run_every_op
    OPS.cast_bias_weight = cast_bias_weight
    OPS.uncast_bias_weight = uncast_bias_weight
    return run_every_op


BASE_RUN_EVERY = _wire_ops()


class _FC:
    out_features = 8192

    def __call__(self, x):
        return TORCH.FakeT(getattr(x, "n", 0))


MLP = types.SimpleNamespace(fc1=_FC(), fc2=_FC())
LONGO = 200000   # forca chunking
CURTO = 500      # cabe em um chunk


def test_chunk_rows():
    print("== _chunk_rows: tetos e alinhamento ==")
    check("seq curta devolve a seq inteira", M._chunk_rows(1000, 8192, 2), 1000)
    check("teto por bytes", M._chunk_rows(500000, 8192, 2), 65536)

    bruto = (1 << 30) // (5000 * 2)
    check("teto bruto desalinhado", bruto, 107374)
    check("alinhado para baixo", M._chunk_rows(500000, 5000, 2), (bruto // 256) * 256)
    check("resultado e multiplo de 256", M._chunk_rows(500000, 5000, 2) % 256, 0)
    check("alinhamento=1 desliga", M._chunk_rows(500000, 5000, 2, 0, 1), bruto)

    check("override alinhado", M._chunk_rows(500000, 8192, 2, 5000), 4864)
    check("override maior que a seq", M._chunk_rows(100, 8192, 2, 9999), 100)

    largura = 3
    teto32 = int(M.INT32_ELEMENTS * M.INT32_MARGIN) // largura
    check("nunca ultrapassa o teto int32",
          M._chunk_rows(10 ** 12, largura, 2, 0, 256) <= teto32, True)
    check("alinhamento nunca zera o chunk",
          M._chunk_rows(10 ** 9, 10 ** 8, 2, 0, 256) >= 1, True)


def test_fatal():
    print("== _fatal: o que nunca pode virar fallback ==")
    check("cancelamento e fatal", M._fatal(Interrupt()), True)
    check("erro de kernel nao e fatal", M._fatal(RuntimeError("x")), False)


def test_held_weights():
    print("== pesos retidos: adquiridos uma vez, nao por chunk ==")
    CAST_MODE["mode"] = "ok"
    reset()
    M._mlp_forward(MLP, TORCH.FakeT(LONGO), 0, 256, True)
    check("dois casts (fc1 e fc2)", CALLS["cast"], 2)
    check("dois uncasts", CALLS["uncast"], 2)
    check("caminho de modulo nao foi usado", CALLS["module"], 0)
    check("run_every_op chamado por chunk", CALLS["every"] >= 3, True)


def test_stream_aliasing():
    print("== aliasing de buffer de cast e recusado ==")
    CAST_MODE["mode"] = "alias"
    reset()
    M._mlp_forward(MLP, TORCH.FakeT(LONGO), 0, 256, True)
    check("os dois pesos sao liberados", CALLS["uncast"], 2)
    check("volta ao caminho de modulo", CALLS["module"] >= 3, True)


def test_api_ausente():
    print("== ComfyUI sem a API de cast ==")
    CAST_MODE["mode"] = "missing"
    reset()
    M._mlp_forward(MLP, TORCH.FakeT(LONGO), 0, 256, True)
    check("usa o caminho de modulo", CALLS["module"] >= 3, True)
    check("nada a liberar", CALLS["uncast"], 0)


def test_desligado():
    print("== pesos_retidos = False ==")
    CAST_MODE["mode"] = "ok"
    reset()
    M._mlp_forward(MLP, TORCH.FakeT(LONGO), 0, 256, False)
    check("nem tenta adquirir", CALLS["cast"], 0)
    check("usa o caminho de modulo", CALLS["module"] >= 3, True)


def test_cancelamento_propaga():
    print("== cancelamento propaga em vez de repetir o trabalho ==")
    CAST_MODE["mode"] = "ok"
    reset()

    def cancelar():
        raise Interrupt("cancelado pelo usuario")

    OPS.run_every_op = cancelar
    try:
        M._mlp_forward(MLP, TORCH.FakeT(LONGO), 0, 256, True)
        FAILS.append("o cancelamento foi engolido pelo fallback")
        print("  FALHA  o cancelamento foi engolido pelo fallback")
    except Interrupt:
        check("a excecao sobe", True, True)
        check("nao repetiu pelo modulo", CALLS["module"], 0)
        check("os pesos sao liberados mesmo assim", CALLS["uncast"], 2)
    finally:
        OPS.run_every_op = BASE_RUN_EVERY


def test_erro_comum_vira_fallback():
    print("== erro de kernel cai para o caminho de modulo ==")
    reset()

    def quebrar():
        raise RuntimeError("kernel indisponivel")

    OPS.run_every_op = quebrar
    try:
        M._mlp_forward(MLP, TORCH.FakeT(LONGO), 0, 256, True)
        check("recuperou pelo modulo", CALLS["module"] >= 3, True)
        check("liberou os pesos", CALLS["uncast"], 2)
    finally:
        OPS.run_every_op = BASE_RUN_EVERY


def test_seq_curta():
    print("== sequencia curta nao chunka nem retem ==")
    reset()
    M._mlp_forward(MLP, TORCH.FakeT(CURTO), 0, 256, True)
    check("uma chamada de modulo, nenhum cast", (CALLS["module"], CALLS["cast"]), (1, 0))


def test_interop():
    print("== deteccao de patch externo no forward do bloco ==")

    class FakeModel:
        def __init__(self, patches):
            self.object_patches = patches

    def marcado():
        def f():
            pass
        f._h3_optimizations_memory = True
        return f

    check("sem patches", M._block_forward_patched(FakeModel({})), None)
    check("apenas os nossos patches", M._block_forward_patched(FakeModel({
        "diffusion_model.blocks.0.mlp.forward": lambda: None,
        "diffusion_model.blocks.0.attn.forward": lambda: None,
    })), None)
    check("H3-Optimizations identificado", M._block_forward_patched(FakeModel({
        "diffusion_model.blocks.0.forward": marcado(),
        "diffusion_model.blocks.1.forward": marcado(),
    })), ("H3-Optimizations (H3 Memory Optimization)", 2))
    check("pacote desconhecido", M._block_forward_patched(FakeModel({
        "diffusion_model.blocks.0.forward": lambda: None,
    })), ("outro pacote", 1))
    check("chave nao numerica ignorada", M._block_forward_patched(FakeModel({
        "diffusion_model.blocks.final.forward": lambda: None,
    })), None)


def test_validacao():
    print("== validacao de entrada do node ==")
    node = M.BruxosH3RowChunk()

    class NaoH3:
        model = types.SimpleNamespace(
            diffusion_model=types.SimpleNamespace()
        )

    try:
        node.apply(NaoH3(), 0)
        FAILS.append("deveria rejeitar modelo nao-H3")
        print("  FALHA  deveria rejeitar modelo nao-H3")
    except ValueError:
        check("rejeita modelo que nao e MiniMax H3", True, True)


def main():
    for test in (
        test_chunk_rows,
        test_fatal,
        test_held_weights,
        test_stream_aliasing,
        test_api_ausente,
        test_desligado,
        test_cancelamento_propaga,
        test_erro_comum_vira_fallback,
        test_seq_curta,
        test_interop,
        test_validacao,
    ):
        test()

    print()
    if FAILS:
        print("FALHAS (%d):" % len(FAILS))
        for item in FAILS:
            print(" -", item)
        return 1
    print("Todos os testes passaram.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
