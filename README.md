# ComfyUI-Bruxos-MiniMaxH3

> Tudo de MiniMax-H3: loader, grade 17n+5, Context-IR, prompt rapido com macros, outpaint/mascara, upscale 2-pass e neural, rolling memory de 124, latente no SSD, autocontext, sampler unlimited, PDD e latent I/O.

Parte da familia **Bruxos do VFX** para ComfyUI. Este repositorio nasceu da
divisao do pacote unico `ComfyUI-Bruxos-do-VFX`, que ficou grande demais para
instalar e manter inteiro. Cada area agora tem repositorio, issues e instalacao
proprios — voce instala so o que usa.

## Instalacao

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/SEU-USUARIO/ComfyUI-Bruxos-MiniMaxH3
pip install -r ComfyUI-Bruxos-MiniMaxH3/requirements.txt
```

Depois reinicie o ComfyUI.

### Dependencias

- `safetensors`, `einops`, `huggingface_hub` — loader e PDD.
- `opencv-python` — upscale e refine.
- Para o Context-IR / Prompt Rapido com LLM local voce tambem vai querer `transformers` e `accelerate`.


## Nodes (55)

| id | nome no menu | arquivo |
|---|---|---|
| `BruxosMinimaxH3EncodeVideo` | MiniMax H3 · Encode Video -> Latente (Bruxos) | `minimax_h3_bruxos.py` |
| `BruxosMinimaxH3LatentInspect` | Inspetor de LATENT (Bruxos) | `minimax_h3_bruxos.py` |
| `BruxosH3MaskToLatent` | H3 Mascara -> Latente · inpaint (Bruxos) | `minimax_h3_mask_bruxos.py` |
| `BruxosH3Outpaint` | H3 Outpaint · Expandir Quadro (Bruxos) | `minimax_h3_mask_bruxos.py` |
| `MinimaxDaBruxosDecode` | MinimaxDaBruxos Decode H3 (AV) [legado] | `minimax_encode_bruxos.py` |
| `MinimaxDaBruxosEncode` | MinimaxDaBruxos Encode H3 (AV) [legado] | `minimax_encode_bruxos.py` |
| `BruxosH3Loader` | MiniMax H3 · Loader tudo-em-um (Bruxos) | `bruxos_h3_loader.py` |
| `BruxosH3Frames` | MiniMax H3 · Frames (grade válida) (Bruxos) | `bruxos_h3_frames.py` |
| `BruxosH3CondForca` | MiniMax H3 · Força do Condicionamento (Bruxos) | `bruxos_h3_cond_forca.py` |
| `BruxosQwen3PromptEnhancer` | Prompt Enhancer Qwen3 (Bruxos) | `bruxos_qwen3_enhancer.py` |
| `BruxosH3ContextIR` | MiniMax H3 · Context-IR local (Bruxos) | `bruxos_h3_context_ir.py` |
| `BruxosH3Referencia` | MiniMax H3 · Referência (Bruxos) | `bruxos_h3_context_ir.py` |
| `BruxosH3PromptRapido` | MiniMax H3 · Prompt Rápido @ # (Bruxos) | `bruxos_h3_prompt_rapido.py` |
| `BruxosH3LatentUpscale` | MiniMax H3 · Latent Upscale 2-pass (Bruxos) | `bruxos_h3_latent_upscale.py` |
| `BruxosH3LatentCondSync` | MiniMax H3 · Latent Cond Sync (Bruxos) | `bruxos_h3_neural_upscale.py` |
| `BruxosH3NeuralUpscaleMegapixels` | MiniMax H3 · Neural Latent Upscale (MP) (Bruxos) | `bruxos_h3_neural_upscale.py` |
| `BruxosH3UltimateInterpParams` | H3 Ultimate · Interpolação por MP (Bruxos) | `bruxos_h3_ultimate_upscale.py` |
| `BruxosH3UltimateModelParams` | H3 Ultimate · Upscale Neural por MP (Bruxos) | `bruxos_h3_ultimate_upscale.py` |
| `BruxosH3UltimateSpatialParams` | H3 Ultimate · Tiles Espaciais (Bruxos) | `bruxos_h3_ultimate_upscale.py` |
| `BruxosH3UltimateTemporalParams` | H3 Ultimate · Chunks Temporais (Bruxos) | `bruxos_h3_ultimate_upscale.py` |
| `BruxosH3UltimateUpscale` | MiniMax H3 · Upscale Latent Completo (Bruxos) | `bruxos_h3_ultimate_upscale.py` |
| `BruxosH3Bloco124` | H3 Bloco 124 (Bruxos) | `bruxos_h3_rolling_memory.py` |
| `BruxosH3BlocoFlex` | H3 Bloco Flex 17k+5 (Bruxos) | `bruxos_h3_rolling_memory.py` |
| `BruxosH3MemoriaLer` | H3 Memoria Rolante - Ler (Bruxos) | `bruxos_h3_rolling_memory.py` |
| `BruxosH3MemoriaLimpar` | H3 Memoria Rolante - Limpar Job (Bruxos) | `bruxos_h3_rolling_memory.py` |
| `BruxosH3MemoriaPrompt` | H3 Memoria Rolante - Prompt (Bruxos) | `bruxos_h3_rolling_memory.py` |
| `BruxosH3MemoriaSalvar` | H3 Memoria Rolante - Salvar (Bruxos) | `bruxos_h3_rolling_memory.py` |
| `BruxosH3ChainSourceWindow` | H3 Loop - Janela Fonte SSD (Bruxos) | `bruxos_h3_contex_loop.py` |
| `BruxosH3ChainTileContext` | H3 Loop - Motion Context por Tile (Bruxos) | `bruxos_h3_contex_loop.py` |
| `BruxosH3ChainTileContextPrep` | H3 Loop - Tail por Tile (Bruxos) | `bruxos_h3_contex_loop.py` |
| `BruxosPrimeiraImagem` | Lista - Primeira Imagem (Bruxos) | `bruxos_h3_contex_loop.py` |
| `BruxosPrimeiroAudio` | Lista - Primeiro Audio (Bruxos) | `bruxos_h3_contex_loop.py` |
| `BruxosPrimeiroInt` | Lista - Primeiro INT (Bruxos) | `bruxos_h3_contex_loop.py` |
| `BruxosPrimeiroLatent` | Lista - Primeiro Latent (Bruxos) | `bruxos_h3_contex_loop.py` |
| `BruxosVideoParaDiscoLazy` | Video -> Disk Cache Lazy (Bruxos) | `bruxos_h3_contex_loop.py` |
| `BruxosH3EncontrarDivergencia` | H3 Encontrar Divergencia (Bruxos) | `bruxos_h3_latent_ssd.py` |
| `BruxosH3JuntarAV` | H3 Juntar AV Decodificado (Bruxos) | `bruxos_h3_latent_ssd.py` |
| `BruxosH3LatentInfoSSD` | H3 Latent Info AV (Bruxos) | `bruxos_h3_latent_ssd.py` |
| `BruxosH3LatentParaReferencia` | H3 Tail Latente -> Referencia (Bruxos) | `bruxos_h3_latent_ssd.py` |
| `BruxosH3PackAV` | H3 Pack AV Correto (Bruxos) | `bruxos_h3_latent_ssd.py` |
| `BruxosH3PlanejarFramesRef` | H3 Planejar Frames + Ref (Bruxos) | `bruxos_h3_latent_ssd.py` |
| `BruxosH3SplitAV` | H3 Split AV Correto (Bruxos) | `bruxos_h3_latent_ssd.py` |
| `BruxosH3TailLatenteLerSSD` | H3 Tail Latente SSD - Ler (Bruxos) | `bruxos_h3_latent_ssd.py` |
| `BruxosH3TailLatenteSalvarSSD` | H3 Tail Latente SSD - Salvar (Bruxos) | `bruxos_h3_latent_ssd.py` |
| `BruxosH3ClayReferenceSetup` | MiniMax H3 Clay Reference Auto (Bruxos) | `bruxos_h3_clay_reference.py` |
| `BruxosH3RowChunk` | H3 Row Chunk Exato (Bruxos) | `bruxos_h3_row_chunk.py` |
| `BruxosH3ReferenceIsolate` | H3 Reference Isolate (Bruxos) | `bruxos_h3_reference_isolate.py` |
| `BruxosH3FacePerFrameDenoise` | H3 Face Denoise por Frame (Bruxos) | `bruxos_h3_face_refine.py` |
| `BruxosH3AutoContextParameter` | H3 Auto Chunks · Parâmetros (Bruxos) | `bruxos_autocontext/__init__.py` |
| `BruxosH3AutoContextSampler` | H3 Auto Chunks · I2V / Ref2V (Bruxos) | `bruxos_autocontext/__init__.py` |
| `BruxosH3AutoContextSeamCorrection` | H3 Auto Chunks · Corrigir Emenda (Bruxos) | `bruxos_autocontext/__init__.py` |
| `BruxosH3AutoContextSeamCorrectionV2` | H3 Auto Chunks · Corrigir Emenda 2.0 (Bruxos) | `bruxos_autocontext/__init__.py` |
| `BruxosH3SamplerCustomAdvancedUnlimited` | H3 Sampler Unlimited (Bruxos TESTE) | `bruxos_h3_sampler_unlimited/__init__.py` |
| `BruxosH3UnlimitedPreview` | H3 Preview Acumulado Unlimited (Bruxos TESTE) | `bruxos_h3_sampler_unlimited/__init__.py` |
| `BruxosMiniMaxH3PDDLoader` | MiniMax H3 PDD Loader · Original LoRA (Bruxos) | `bruxos_h3_pdd/nodes.py` |
## Workflows de exemplo

- `workflows/Bruxos_H3_Chromakey_TrocaFundo.json`
- `workflows/Bruxos_H3_FaceRefine_SAM3_TESTE.json`
- `workflows/Bruxos_H3_Laboratorio_Latent_SSD_124.json`
- `workflows/Bruxos_H3_Ltx2.3_Wan2.2_5B_T2V.json`
- `workflows/Bruxos_H3_Ltx2_Wan2.2_5B_Upscale.json`
- `workflows/Bruxos_H3_R2V_2pass.json`
- `workflows/Bruxos_H3_R2V_ContextIR.json`
- `workflows/Bruxos_H3_R2V_Lyonir_Streaming_124.json`
- `workflows/Bruxos_H3_R2V_Lyonir_Tiled_Hybrid.json`
- `workflows/Bruxos_H3_Ref2Video_UpscaleLTX_Wan.json`
- `workflows/Bruxos_MiniMaxH3_CONTEX_LOOP_TILED_SSD.json`
- `workflows/Bruxos_MiniMaxH3_CONTEX_LOOP_TILED_SSD_RAM.json`
- `workflows/Bruxos_VideoTiler_MiniMaxH3.json`
- `workflows/Bruxos_VideoTiler_MiniMaxH3_Disco.json`
- `workflows/Bruxos_VideoTiler_MiniMaxH3_FAST_124_SSD.json`
- `workflows/Bruxos_VideoTiler_MiniMaxH3_LOCAL_REF_CORRIGIDO.json`

## Observacoes da separacao

- Registra a rota HTTP `/bruxos/h3_macros`, que alimenta o popup de macros `#` e `@` do Prompt Rapido.
- `bruxos_h3_contex_loop.py` usa o cache em SSD (`bruxos_core/disk_cache.py`) e a config de
  ladrilho (`bruxos_core/tile_config.py`). Os *nodes* correspondentes estao no
  **ComfyUI-Bruxos-MediaIO** e no **ComfyUI-Bruxos-Tiler**; para os workflows de tiling completo,
  instale os tres.
- Subpacotes com licenca propria: `bruxos_autocontext/`, `bruxos_h3_pdd/`,
  `bruxos_h3_latent_io/`, `bruxos_h3_sampler_unlimited/`. Os arquivos `LICENSE.upstream`
  vieram junto.
- `docs_H3_macros.md` e `docs_H3_papeis.md` documentam as macros e os papeis das secoes.


## Sobre o `bruxos_core`

A pasta `bruxos_core/` e a biblioteca comum dos pacotes Bruxos (mascara, 4n+1,
encode/decode de video, latentes, limpeza de memoria, janelas de contexto,
loader do Qwen-VL). Ela vem **copiada dentro deste repositorio de proposito**:
assim o pacote instala sozinho, sem depender da ordem de instalacao dos outros e
sem precisar importar pastas com `-` no nome (o Python nao aceita).

Fonte canonica: [SEU-USUARIO/ComfyUI-Bruxos-Core](https://github.com/SEU-USUARIO/ComfyUI-Bruxos-Core).
Quando atualizar a lib la, copie a pasta por cima aqui.

## Licenca

Apache-2.0. Creditos de terceiros em `THIRD_PARTY_NOTICES.md`.
