"""Row chunking exato para o transformer MiniMax H3.

Adaptado de ComfyUI-NynxzH3, Copyright (c) 2026 Nynxz, sob licenca MIT.
O aviso MIT completo do projeto de origem e preservado em THIRD_PARTY_NOTICES.md.

As projecoes QKV e o MLP sao operacoes independentes por linha. Dividi-las
pela dimensao da sequencia reduz o pico de ativacao e evita o limite de
indexacao int32 dos kernels CUDA, sem aproximar a funcao do modelo.

Este node recebe `x` ja normalizado e modulado pelo forward do bloco H3, entao
o corte por linha e exato: nenhuma linha depende de outra. Os segmentos de
modulacao do H3 sao resolvidos antes de chegar aqui.

Duas otimizacoes alem do chunking basico:

1. Alinhamento. O tamanho do chunk e arredondado para baixo em multiplos de
   256 linhas, para que os chunks intermediarios caiam em tamanhos que os
   kernels CUDA executam bem. Apenas a cauda tem tamanho arbitrario.

2. Pesos retidos. Sem isso, chamar `camada(x[a:b])` N vezes faz o ComfyUI
   desquantizar/copiar o peso N vezes. O peso e adquirido uma vez por forward
   e reaproveitado em todos os chunks. Se a API do ComfyUI nao existir, se os
   dois pesos do MLP compartilharem o mesmo buffer assincrono de cast, ou se
   qualquer erro ocorrer, o node volta ao caminho de modulo sem mudar o
   resultado.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


INT32_ELEMENTS = 2**31
INT32_MARGIN = 0.85
TARGET_BYTES = 1 << 30
DEFAULT_ALIGNMENT = 256


# ---------------------------------------------------------------- identidade


def _diffusion_model(model):
    inner = getattr(model, "model", None)
    return getattr(inner, "diffusion_model", None)


def _is_h3(model):
    dit = _diffusion_model(model)
    return dit is not None and type(dit).__name__ == "MiniMaxH3Model"


def _out_features(linear):
    value = getattr(linear, "out_features", None)
    return int(value) if value is not None else int(linear.weight.shape[0])


# ------------------------------------------------------------------- chunks


def _chunk_rows(seq_len, width, itemsize, override=0, alignment=DEFAULT_ALIGNMENT):
    """Linhas por chunk, limitadas por int32 e por bytes, alinhadas para baixo."""
    if override > 0:
        rows = max(1, min(int(override), seq_len))
    else:
        by_int32 = int(INT32_ELEMENTS * INT32_MARGIN) // max(1, width)
        by_bytes = TARGET_BYTES // max(1, width * max(1, itemsize))
        rows = max(1, min(seq_len, by_int32, by_bytes))

    if rows >= seq_len:
        return seq_len

    # Arredondar para baixo nunca ultrapassa os tetos acima, so descarta a
    # cauda desalinhada do chunk intermediario.
    alignment = int(alignment)
    if alignment > 1 and rows > alignment:
        rows = (rows // alignment) * alignment
    return max(1, rows)


# ------------------------------------------------------------ pesos retidos


class _HeldLinear:
    """Segura peso e bias de uma camada por todos os chunks de um forward."""

    __slots__ = ("module", "weight", "bias", "handle", "_released")

    def __init__(self, module, weight, bias, handle):
        self.module = module
        self.weight = weight
        self.bias = bias
        self.handle = handle
        self._released = False

    def linear(self, x):
        return F.linear(x, self.weight, self.bias)

    def release(self):
        if self._released:
            return
        self._released = True
        try:
            import comfy.ops

            comfy.ops.uncast_bias_weight(
                self.module, self.weight, self.bias, self.handle
            )
        except Exception:
            pass


def _stream_of(handle):
    return handle[0] if isinstance(handle, tuple) and handle else None


def _acquire(module, sample):
    """Adquire peso/bias uma vez. None quando a API nao existe ou falha."""
    try:
        import comfy.ops

        weight, bias, handle = comfy.ops.cast_bias_weight(
            module,
            sample,
            offloadable=True,
            compute_dtype=sample.dtype,
            want_requant=True,
        )
    except Exception:
        # Inclui TypeError em versoes do ComfyUI sem esses argumentos.
        return None
    return _HeldLinear(module, weight, bias, handle)


def _run_every_op():
    """Preserva o cancelamento do ComfyUI ao pular o wrapper do modulo.

    So a ausencia da API e tolerada. A excecao de interrupcao levantada por
    `run_every_op` PRECISA subir: engoli-la desligaria o botao de cancelar.
    """
    try:
        import comfy.ops

        run = getattr(comfy.ops, "run_every_op", None)
    except ImportError:
        return
    if run is not None:
        run()


def _fatal(exc):
    """Excecoes que nunca podem virar fallback silencioso.

    O cancelamento do ComfyUI e uma Exception comum, entao sem esta checagem
    o `except` do fallback o transformaria numa repeticao inutil do trabalho.
    """
    try:
        import comfy.model_management

        interrupt = getattr(
            comfy.model_management, "InterruptProcessingException", None
        )
    except ImportError:
        return False
    return interrupt is not None and isinstance(exc, interrupt)


def _acquire_pair(module_a, module_b, sample):
    """Adquire dois pesos usados dentro do mesmo chunk.

    Retorna None quando ambos vem do mesmo buffer reutilizavel de cast
    assincrono: nesse caso o segundo cast sobrescreve o primeiro e o
    resultado seria silenciosamente errado.
    """
    held_a = _acquire(module_a, sample)
    if held_a is None:
        return None
    held_b = _acquire(module_b, sample)
    if held_b is None:
        held_a.release()
        return None

    stream_a = _stream_of(held_a.handle)
    stream_b = _stream_of(held_b.handle)
    if stream_a is not None and stream_a is stream_b:
        held_b.release()
        held_a.release()
        return None
    return held_a, held_b


# ----------------------------------------------------------------- SwiGLU


def _swiglu(x):
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)


def _fc2_swiglu(held_fc2, expanded):
    """fc2 sobre a ativacao SwiGLU, preservando o caminho INT8 fundido."""
    weight = held_fc2.weight
    bias = held_fc2.bias
    try:
        import comfy.quant_ops
        from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

        if (
            isinstance(weight, QuantizedTensor)
            and getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"
            and not getattr(weight._params, "transposed", False)
        ):
            qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
            return comfy.quant_ops.ck.int8_linear(
                expanded,
                qdata,
                scale,
                bias,
                expanded.dtype,
                convrot=getattr(weight._params, "convrot", False),
                convrot_groupsize=getattr(weight._params, "convrot_groupsize", 256),
                input_act="swiglu",
            )
    except ImportError:
        pass
    return F.linear(_swiglu(expanded), weight, bias)


# --------------------------------------------------------------------- MLP


def _mlp_loop(mlp, x, chunk, held):
    import comfy.ops

    rows = x.shape[0]
    out = torch.empty_like(x)
    for a in range(0, rows, chunk):
        b = min(a + chunk, rows)
        if held is None:
            out[a:b] = comfy.ops.linear_input_act(mlp.fc2, mlp.fc1(x[a:b]), "swiglu")
        else:
            _run_every_op()
            out[a:b] = _fc2_swiglu(held[1], held[0].linear(x[a:b]))
    return out


def _mlp_forward(mlp, x, override=0, alignment=DEFAULT_ALIGNMENT, held_weights=True):
    import comfy.ops

    rows = x.shape[0]
    chunk = _chunk_rows(
        rows, _out_features(mlp.fc1), x.element_size(), override, alignment
    )
    if chunk >= rows:
        return comfy.ops.linear_input_act(mlp.fc2, mlp.fc1(x), "swiglu")

    held = _acquire_pair(mlp.fc1, mlp.fc2, x[:1]) if held_weights else None
    if held is None:
        return _mlp_loop(mlp, x, chunk, None)

    try:
        return _mlp_loop(mlp, x, chunk, held)
    except Exception as exc:
        if _fatal(exc):
            raise
        # `out` e reescrito por completo, entao repetir pelo caminho de
        # modulo produz o resultado correto mesmo apos falha parcial.
        for item in held:
            item.release()
        held = None
        return _mlp_loop(mlp, x, chunk, None)
    finally:
        if held is not None:
            for item in held:
                item.release()


# --------------------------------------------------------------- atencao


def _linear_chunked(layer, x, chunk, held_weights=True):
    rows = x.shape[0]
    if chunk >= rows:
        return layer(x)

    held = _acquire(layer, x[:1]) if held_weights else None
    out = None
    try:
        for a in range(0, rows, chunk):
            b = min(a + chunk, rows)
            if held is None:
                piece = layer(x[a:b])
            else:
                _run_every_op()
                piece = held.linear(x[a:b])
            if out is None:
                out = torch.empty(
                    (rows, *piece.shape[1:]), dtype=piece.dtype, device=piece.device
                )
            out[a:b] = piece
    except Exception as exc:
        if held is None or _fatal(exc):
            raise
        held.release()
        held = None
        out = None
        for a in range(0, rows, chunk):
            b = min(a + chunk, rows)
            piece = layer(x[a:b])
            if out is None:
                out = torch.empty(
                    (rows, *piece.shape[1:]), dtype=piece.dtype, device=piece.device
                )
            out[a:b] = piece
    finally:
        if held is not None:
            held.release()
    return out


def _attention_forward(
    attn,
    x,
    rope_freqs=None,
    transformer_options={},
    override=0,
    alignment=DEFAULT_ALIGNMENT,
    held_weights=True,
):
    import comfy.model_management
    import comfy.quant_ops
    from comfy.ldm.modules.attention import optimized_attention

    seq = x.shape[0]
    heads, head_dim = attn.heads, attn.head_dim
    inner = heads * head_dim
    chunk = _chunk_rows(
        seq, _out_features(attn.qkv_proj), x.element_size(), override, alignment
    )

    if chunk >= seq:
        q, k, v = attn.qkv_proj(x).split(inner, dim=-1)
        q = q.view(1, seq, heads, head_dim)
        k = k.view(1, seq, heads, head_dim)
        v = v.view(seq, heads, head_dim)
    else:
        # qkv_proj e out_proj sao retidos em janelas separadas, nunca ao mesmo
        # tempo, entao nao ha risco de aliasing do buffer de cast.
        held = _acquire(attn.qkv_proj, x[:1]) if held_weights else None
        q = k = v = None
        try:
            for a in range(0, seq, chunk):
                b = min(a + chunk, seq)
                if held is None:
                    projected = attn.qkv_proj(x[a:b])
                else:
                    _run_every_op()
                    projected = held.linear(x[a:b])
                qc, kc, vc = projected.split(inner, dim=-1)
                if q is None:
                    opts = {"dtype": qc.dtype, "device": qc.device}
                    q = torch.empty(1, seq, heads, head_dim, **opts)
                    k = torch.empty(1, seq, heads, head_dim, **opts)
                    v = torch.empty(seq, heads, head_dim, **opts)
                q[0, a:b] = qc.view(-1, heads, head_dim)
                k[0, a:b] = kc.view(-1, heads, head_dim)
                v[a:b] = vc.view(-1, heads, head_dim)
        except Exception as exc:
            if held is None or _fatal(exc):
                raise
            held.release()
            held = None
            q = k = v = None
            for a in range(0, seq, chunk):
                b = min(a + chunk, seq)
                qc, kc, vc = attn.qkv_proj(x[a:b]).split(inner, dim=-1)
                if q is None:
                    opts = {"dtype": qc.dtype, "device": qc.device}
                    q = torch.empty(1, seq, heads, head_dim, **opts)
                    k = torch.empty(1, seq, heads, head_dim, **opts)
                    v = torch.empty(seq, heads, head_dim, **opts)
                q[0, a:b] = qc.view(-1, heads, head_dim)
                k[0, a:b] = kc.view(-1, heads, head_dim)
                v[a:b] = vc.view(-1, heads, head_dim)
        finally:
            if held is not None:
                held.release()

    if rope_freqs is not None:
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
            )
        q, k = q[0], k[0]
    else:
        q = attn.q_norm(q[0])
        k = attn.k_norm(k[0])

    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    out = optimized_attention(
        q, k, v, heads, mask=None, skip_reshape=True,
        transformer_options=transformer_options,
    ).squeeze(0)
    out_chunk = _chunk_rows(
        seq, _out_features(attn.out_proj), out.element_size(), override, alignment
    )
    return _linear_chunked(attn.out_proj, out, out_chunk, held_weights)


# -------------------------------------------------------------- interop


def _block_forward_patched(model):
    """Detecta um patch externo no forward do bloco (ex.: H3-Optimizations).

    Quando outro pacote substitui `blocks.N.forward`, ele normalmente chama
    `mlp.fc1`/`mlp.fc2` diretamente em vez de `mlp.forward`. Nesse caso o
    nosso chunking de MLP fica inativo, sem erro e sem aviso.
    """
    patches = getattr(model, "object_patches", None)
    if not patches:
        return None
    owner = None
    count = 0
    for key, value in patches.items():
        if not isinstance(key, str):
            continue
        if key.startswith("diffusion_model.blocks.") and key.endswith(".forward"):
            middle = key[len("diffusion_model.blocks."):-len(".forward")]
            if middle.isdigit():
                count += 1
                if owner is None and getattr(
                    value, "_h3_optimizations_memory", False
                ):
                    owner = "H3-Optimizations (H3 Memory Optimization)"
    if not count:
        return None
    return owner or "outro pacote", count


# ----------------------------------------------------------------- node


class BruxosH3RowChunk:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "chunk_rows": ("INT", {
                    "default": 0, "min": 0, "max": 1000000, "step": 1024,
                    "tooltip": "0 = automatico (~1 GiB por intermediario). Use manual apenas se ainda ocorrer CUDA illegal memory access.",
                }),
            },
            "optional": {
                "alinhamento": ("INT", {
                    "default": DEFAULT_ALIGNMENT, "min": 1, "max": 4096, "step": 64,
                    "tooltip": "Arredonda o tamanho do chunk para baixo neste multiplo. 256 mantem os chunks intermediarios em tamanhos que os kernels executam bem; so a cauda fica com tamanho livre. 1 desliga.",
                }),
                "pesos_retidos": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Adquire o peso da camada uma vez por forward em vez de re-castar a cada chunk. Em modelo quantizado (INT8 ConvRot) isso evita repetir a desquantizacao por chunk. Custo: no MLP os pesos de fc1 e fc2 ficam residentes ao mesmo tempo durante o laco, em vez de um de cada vez. Se voce estiver no limite exato da VRAM, desligue. Resultado identico nos dois modos; volta sozinho ao caminho normal se a API do ComfyUI nao permitir.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "relatorio")
    FUNCTION = "apply"
    CATEGORY = "Bruxos do VFX/MiniMax H3/Otimizacao"
    DESCRIPTION = "Divide QKV e MLP por linhas: menor pico de VRAM e evita o limite int32 em sequencias grandes, sem aproximacao."

    def apply(self, model, chunk_rows=0, alinhamento=DEFAULT_ALIGNMENT, pesos_retidos=True):
        if not _is_h3(model):
            wired = type(_diffusion_model(model)).__name__
            raise ValueError(
                "Bruxos H3 Row Chunk requer um MiniMax H3; o modelo conectado e "
                f"{wired!r}."
            )

        alinhamento = max(1, int(alinhamento))
        if chunk_rows and alinhamento > 1 and chunk_rows % alinhamento:
            raise ValueError(
                f"chunk_rows ({chunk_rows}) deve ser multiplo do alinhamento "
                f"({alinhamento}), ou use alinhamento = 1."
            )

        interop = _block_forward_patched(model)

        patched = model.clone()
        dit = _diffusion_model(patched)
        blocks = list(dit.blocks)
        for index, block in enumerate(blocks):
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.mlp.forward",
                lambda x, _m=block.mlp, _o=chunk_rows, _a=alinhamento, _h=pesos_retidos: (
                    _mlp_forward(_m, x, _o, _a, _h)
                ),
            )
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.attn.forward",
                lambda x, rope_freqs=None, transformer_options={}, _m=block.attn, _o=chunk_rows, _a=alinhamento, _h=pesos_retidos: (
                    _attention_forward(_m, x, rope_freqs, transformer_options, _o, _a, _h)
                ),
            )

        widest = max(_out_features(blocks[0].mlp.fc1), _out_features(blocks[0].attn.qkv_proj))
        ceiling = INT32_ELEMENTS // widest
        mode = f"manual: {chunk_rows:,} linhas" if chunk_rows else "automatico: alvo de ~1 GiB"
        linhas = [
            f"Bruxos H3 Row Chunk aplicado em {len(blocks)} blocos",
            f"modo: {mode}",
            f"alinhamento: {alinhamento} linhas" if alinhamento > 1 else "alinhamento: desligado",
            f"pesos retidos: {'sim' if pesos_retidos else 'nao'}",
            f"limite int32 estimado sem chunk: {ceiling:,} linhas",
            "qualidade: operacao exata por linha; pode ficar conectado em clips curtos",
            "Turbo LoRA/SolAttn: preservados porque as camadas e optimized_attention continuam sendo chamadas",
        ]
        if interop is not None:
            owner, count = interop
            linhas.append(
                f"AVISO: {owner} ja substituiu o forward de {count} bloco(s). "
                "Esse tipo de patch chama mlp.fc1/fc2 diretamente, entao o nosso "
                "chunking de MLP NAO sera executado nesses blocos; o de atencao "
                "continua valendo. Use o limitador de memoria de um pacote so."
            )
        return patched, "\n".join(linhas)


NODE_CLASS_MAPPINGS = {"BruxosH3RowChunk": BruxosH3RowChunk}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosH3RowChunk": "H3 Row Chunk Exato (Bruxos)"}
