# Bruxos MiniMax-H3 PDD Loader

Carrega diretamente os LoRAs oficiais `MiniMax-H3-*-Acc-8Step.safetensors`
da Alibaba PAI. O arquivo original permanece em `ComfyUI/models/loras`; a
tradução Diffusers → ComfyUI acontece somente em memória.

## Uso

1. Carregue o checkpoint MiniMax-H3 FL2VA ou Ref2VA correspondente.
2. Passe o `MODEL` pelo node **MiniMax H3 PDD Loader · Original LoRA (Bruxos)**.
3. Use `SamplerCustomAdvanced` com:
   - sampler: `euler`
   - sigmas: saída `sigmas` do loader (limites PDD exatos)
   - 8 avaliações (`steps` também é informado pelo node)
   - guider/CFG: `1.0`

Não use outro LoRA turbo/distill nem nodes de step cache junto com o PDD.

O loader configura automaticamente os shifts oficiais de vídeo/áudio em
`12.0 / 3.0`.

O mapeamento também converte o SwiGLU de `ff.net.0.proj` da ordem Diffusers
`[value; gate]` para a ordem MiniMax-H3 do ComfyUI `[gate; value]`.

## Compatibilidade

O PDD oficial inclui LoRA para as projeções AdaLN completas. Checkpoints
convertidos para a representação `adaln_t_table` (também chamados de
curve/pruned) têm outra forma matemática e são rejeitados com uma mensagem
clara, em vez de gerar um vídeo incorreto silenciosamente.

Base técnica: `alibaba-pai/MiniMax-H3-Acc-LoRAs/minimax_h3_pdd.py`, Apache-2.0.
O texto completo da licença upstream está em `LICENSE.upstream`.
