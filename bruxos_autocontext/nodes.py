"""
nodes.py - H3 Auto Context Sampler 节点定义

Bruxos do VFX integration, adapted from
supElement/ComfyUI_MinimaxH3_AutoContext (Apache-2.0).

使用 io.ComfyNode 新 API，支持 io.Autogrow 动态端口：
- ref_image_0 ~ ref_image_9 (参考图片)
- ref_video_0 ~ ref_video_3 (参考视频)
- ref_video_audio_0 ~ ref_video_audio_3 (参考视频配对音轨)
- ref_audio_0 ~ ref_audio_3 (独立参考音频)

默认只显示 _0 端口，连接后才显示 _1，以此类推。

参数组 (分段/分辨率/音频等) 已拆分到 Minimax_H3_AutoContext_parameter 节点，
通过 parameter (Dict) 单端口传入本节点；主节点只保留采样/提示词/参考素材相关输入。
"""

from comfy_api.latest import io
from . import h3_sampler
from . import native_masked

_SAMPLERS = [
    "er_sde", "euler", "euler_ancestral", "euler_cfg_pp",
    "res_multistep", "res_multistep_cfg_pp", "dpmpp_2m", "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde", "dpmpp_3m_sde", "uni_pc", "uni_pc_bh2",
    "ddpm", "lms", "heun", "dpm_2", "dpm_2_ancestral",
]
_SCHEDULERS = ["simple", "normal", "karras", "exponential",
               "sgm_uniform", "beta", "linear_quadratic"]


def _validate_h3_model_patches(model):
    """Recusa cabeças PDD carregadas acidentalmente como uma LoRA comum."""
    patches = getattr(model, "patches", None) or {}
    unsafe = []
    expected_channels = {
        "diffusion_model.final_layer.video_out.weight": 96,
        "diffusion_model.final_layer.video_out.bias": 96,
        "diffusion_model.final_layer.audio_out.weight": 32,
        "diffusion_model.final_layer.audio_out.bias": 32,
    }

    for key, expected in expected_channels.items():
        for patch in patches.get(key, ()):
            if len(patch) < 2:
                continue
            adapter = patch[1]
            calculate_shape = getattr(adapter, "calculate_shape", None)
            if not callable(calculate_shape):
                continue
            try:
                shape = calculate_shape(key)
            except Exception:
                continue
            if shape and int(shape[0]) != expected:
                unsafe.append(f"{key}: {expected} -> {int(shape[0])}")

    if unsafe:
        details = "\n - ".join(unsafe)
        raise RuntimeError(
            "LoRA PDD carregada como LoRA comum antes do AutoContext.\n"
            f"Cabeças H3 incompatíveis:\n - {details}\n\n"
            "Remova ou ignore o node LoraLoaderModelOnly que carrega "
            "MiniMax-H3-Ref2VA-Acc-8Step*_comfy.safetensors. Esses arquivos "
            "contêm 32 bancos de saída PDD e não podem ser usados como uma LoRA comum.\n"
            "Para o checkpoint Turbo/curve/pruned, conecte o Load Diffusion Model "
            "diretamente ao AutoContext. Para usar PDD, use o node "
            "MiniMax H3 PDD Loader (Bruxos) com um checkpoint H3 de AdaLN completa."
        )

_PROMPT_MODE_TOOLTIP = (
    "Modo da timeline no Clip_Frame. auto reconhece marcações como 0-5s e repete "
    "trechos globais; timeline exige tempos; global usa o mesmo prompt em todos os "
    "chunks; sequential distribui os parágrafos em ordem. No Clip_Tag este campo é ignorado."
)
_CONTEXT_FRAMES_TOOLTIP = (
    "Frames do final anterior usados para continuar o próximo chunk. Valores maiores "
    "aumentam a continuidade e também o custo. Use a grade 5, 22, 39, 56, 73, 90, 107 ou 124."
)
_CLIP_MODE_TOOLTIP = (
    "Clip_Frame divide uniformemente usando chunk_frames. Clip_Tag cria um chunk para "
    "cada etiqueta numerada do prompt e calcula a duração pelo conteúdo daquela seção."
)
_CLIP_TAG_TOOLTIP = (
    "Modelo da etiqueta numerada usada pelo Clip_Tag, por exemplo Cena1, A01 ou "
    "[Trecho001]. A etiqueta deve terminar em número e ficar preferencialmente em uma linha própria."
)
_PROMPT_FORMAT_TOOLTIP = (
    "official usa o formato treinado pelo MiniMax H3 e é recomendado. legacy preserva "
    "a estrutura antiga. raw remove apenas as etiquetas de divisão e mantém o texto original."
)
_CROP_MODE_TOOLTIP = (
    "Ajuste espacial das referências: center preserva a proporção e recorta o centro; "
    "stretch ocupa todo o quadro e pode deformar; none mantém a proporção original e apenas alinha a 32."
)
_REF_SYNC_MODE_TOOLTIP = (
    "global envia a referência completa para todos os chunks. segmented envia apenas "
    "o trecho temporal correspondente, recomendado para manter movimento, fala e sincronismo labial."
)


class H3ParameterNode(io.ComfyNode):
    """H3 参数组节点：集中管理分段/分辨率/音频等参数，输出 parameter dict，并预览预计分段。"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BruxosH3AutoContextParameter",
            display_name="H3 Auto Chunks · Parâmetros (Bruxos)",
            category="Bruxos do VFX/MiniMax H3/Auto Chunks",
            description="Planeja chunks H3 17n+5, contexto, referências e segundo passe com latent ampliado.",
            inputs=[
                io.String.Input("long_prompt", multiline=True, dynamic_prompts=True,
                    socketless=False, default="",
                    tooltip="Prompt completo do vídeo. Também é usado para calcular e visualizar os chunks previstos."),
                io.Combo.Input("prompt_mode",
                    options=["auto", "timeline", "global", "sequential"],
                    default="auto", tooltip=_PROMPT_MODE_TOOLTIP),
                io.Combo.Input("clip_mode",
                    options=["Clip_Frame", "Clip_Tag"],
                    default="Clip_Frame", tooltip=_CLIP_MODE_TOOLTIP),
                io.String.Input("clip_tag", multiline=False, dynamic_prompts=True,
                    socketless=False, default="Cena1",
                    tooltip=_CLIP_TAG_TOOLTIP),
                io.Combo.Input("prompt_format",
                    options=["official", "legacy", "raw"],
                    default="official",
                    tooltip=_PROMPT_FORMAT_TOOLTIP),
                io.Combo.Input("crop_mode",
                    options=["center", "stretch", "none"],
                    default="stretch",
                    tooltip=_CROP_MODE_TOOLTIP),
                io.Combo.Input("ref_sync_mode",
                    options=["global", "segmented"],
                    default="segmented",
                    tooltip=_REF_SYNC_MODE_TOOLTIP),
                io.Int.Input("width", default=960, min=64, max=4096, step=32,
                    tooltip="Largura da primeira geração. No segundo passe, o tamanho vem do latent_input."),
                io.Int.Input("height", default=544, min=64, max=4096, step=32,
                    tooltip="Altura da primeira geração. No segundo passe, o tamanho vem do latent_input."),
                io.Int.Input("total_frames", default=362, min=5, max=2880, step=17,
                    tooltip="Total desejado na grade 17n+5: 5, 22, 39, 56, 73, 90... As marcações do prompt continuam em segundos."),
                io.Int.Input("fps", default=24, min=8, max=60, step=1,
                    tooltip="FPS usado para sincronizar áudio e converter os tempos escritos no prompt."),
                io.Int.Input("chunk_frames", default=90, min=5, max=2880, step=17,
                    tooltip="Frames por chunk na grade 17n+5. Se for igual ou maior que total_frames, o vídeo é processado em um único trecho."),
                io.Int.Input("context_frames", default=22, min=5, max=124, step=1,
                    tooltip=_CONTEXT_FRAMES_TOOLTIP),
                io.Combo.Input("continuation_method",
                    options=list(native_masked.CONTINUATION_METHODS),
                    default=native_masked.CONTINUATION_AUTO,
                    tooltip="Auto usa Native Masked quando o ComfyUI possui a API H3 por token e "
                            "recua para Guide em versões antigas. Native copia e protege o prefixo "
                            "latente anterior; Guide usa os anchors de movimento compatíveis."),
                io.Boolean.Input("lock_audio",
                    default=True,
                    tooltip="No segundo passe, mantém o áudio do latent_input e reamostra somente o vídeo."),
                io.Boolean.Input("audio_drive",
                    default=False,
                    tooltip="Quando ligado, codifica e protege drive_audio dentro do latent para dirigir fala, ritmo e movimento do vídeo."),
                io.Boolean.Input("decode_output",
                    default=False,
                    tooltip="Ligado decodifica imagem e áudio dentro do node. Desligado retorna somente latents e economiza tempo/VRAM."),
            ],
            outputs=[
                io.Dict.Output(display_name="parameter"),
            ],
        )

    @classmethod
    def execute(cls, long_prompt="", prompt_mode="auto",
                clip_mode="Clip_Frame", clip_tag="Cena1",
                prompt_format="official", crop_mode="stretch", ref_sync_mode="segmented",
                width=960, height=544, total_frames=362, fps=24, chunk_frames=90,
                context_frames=22, continuation_method=native_masked.CONTINUATION_AUTO,
                lock_audio=True, audio_drive=False,
                decode_output=False) -> io.NodeOutput:
        parameter = {
            "long_prompt": long_prompt,
            "prompt_mode": prompt_mode,
            "clip_mode": clip_mode, "clip_tag": clip_tag,
            "prompt_format": prompt_format, "crop_mode": crop_mode,
            "ref_sync_mode": ref_sync_mode,
            "width": width, "height": height, "total_frames": total_frames,
            "fps": fps, "chunk_frames": chunk_frames, "context_frames": context_frames,
            "continuation_method": continuation_method,
            "lock_audio": lock_audio, "audio_drive": audio_drive,
            "decode_output": decode_output,
        }
        return io.NodeOutput(parameter)


class H3AutoContextSampler(io.ComfyNode):
    """一键式 MiniMax H3 长视频自动化生成节点 (分段推理 + 续接锚定)"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BruxosH3AutoContextSampler",
            display_name="H3 Auto Chunks · I2V / Ref2V (Bruxos)",
            category="Bruxos do VFX/MiniMax H3/Auto Chunks",
            description="H3 longo para I2V/FL2V e Ref2V: chunks, anchors latentes, timeline e segundo passe/upscale.",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Clip.Input("clip"),

                io.Dict.Input("parameter",
                    tooltip="Saída do node H3 Auto Chunks · Parâmetros. Contém prompt, chunks, resolução, referências e áudio."),
                io.Sampler.Input("sampler", optional=True,
                    tooltip="Sampler externo opcional. Quando conectado, substitui sampler_name e scheduler internos."),
                io.Sigmas.Input("sigmas", optional=True,
                    tooltip="Sequência SIGMAS opcional. Tem prioridade sobre steps/scheduler; denoise multiplica os sigmas quando menor que 1."),
                io.Latent.Input("latent_input", optional=True,
                    tooltip="Conecte o latent da primeira geração, normalmente ampliado, para ativar o segundo passe. A resolução vem deste latent."),
                io.Dict.Input("info", optional=True,
                    tooltip="Info de outro Auto Chunks. Reutiliza exatamente a mesma divisão temporal em segundo ou terceiro passe."),
                io.Image.Input("first_frame", optional=True,
                    tooltip="Imagem que ancora o primeiro frame do vídeo."),
                io.Image.Input("last_frame", optional=True,
                    tooltip="Imagem opcional que ancora o último frame do vídeo."),

                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                    control_after_generate=True),
                io.Int.Input("steps", default=30, min=1, max=100, step=1),
                io.Float.Input("cfg", default=1.0, min=0.0, max=30.0, step=0.1),
                io.Combo.Input("sampler_name", options=_SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=_SCHEDULERS, default="simple"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Força geral do redesenho. 1 reamostra completamente; valores menores preservam mais do latent_input."),
                io.Float.Input("video_context_denoise", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Redesenho do contexto copiado entre chunks no modo Guide. 0 preserva exatamente o final anterior; 1 redesenha junto com o trecho novo. Com SplitSigmas no segundo passe, comece em 1 para evitar emendas quebradas."),

                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image",
                            tooltip="Imagem de referência. No prompt use image1, image 1 ou <Picture 1>."),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video",
                            tooltip="Frames de um vídeo de referência, preferencialmente entre 2 e 15 segundos."),
                        prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio",
                            tooltip="Áudio pertencente ao vídeo de referência com o mesmo número."),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio",
                            tooltip="Áudio de referência independente."),
                        prefix="ref_audio_", min=0, max=3)),
                io.Audio.Input("drive_audio", optional=True,
                    tooltip="Áudio que dirigirá fala, ritmo e movimento quando audio_drive estiver ligado."),
            ],
            outputs=[
                io.Image.Output(display_name="video_frames"),
                io.Audio.Output(display_name="audio"),
                io.Latent.Output(display_name="latent"),
                io.Latent.Output(display_name="denoised_latent"),
                io.Dict.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, model, vae, audio_vae, clip,
                parameter=None,
                sigmas=None, latent_input=None, info=None,
                first_frame=None, last_frame=None,
                denoise=1.0, video_context_denoise=0.0,
                steps=30, cfg=1.0,
                sampler_name="euler", scheduler="simple",
                sampler=None,
                seed=0,
                ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None,
                drive_audio=None) -> io.NodeOutput:

        # Evita gastar minutos inicializando um modelo cuja cabeça de áudio foi
        # expandida de 32 para 1024 canais por um PDD carregado no LoraLoader comum.
        _validate_h3_model_patches(model)

        # 从 parameter dict 解包参数 (缺失项用默认值兜底)
        p = parameter or {}
        long_prompt = p.get("long_prompt", "")
        prompt_mode = p.get("prompt_mode", "auto")
        clip_mode = p.get("clip_mode", "Clip_Frame")
        clip_tag = p.get("clip_tag", "Cena1")
        prompt_format = p.get("prompt_format", "official")
        crop_mode = p.get("crop_mode", "stretch")
        ref_sync_mode = p.get("ref_sync_mode", "segmented")
        width = int(p.get("width", 960))
        height = int(p.get("height", 544))
        total_frames = int(p.get("total_frames", 362))
        fps = int(p.get("fps", 24))
        chunk_frames = int(p.get("chunk_frames", 90))
        context_frames = int(p.get("context_frames", 22))
        continuation_method = p.get(
            "continuation_method", native_masked.CONTINUATION_GUIDE)
        lock_audio = p.get("lock_audio", True)
        audio_drive = p.get("audio_drive", False)
        decode_output = p.get("decode_output", False)

        # info 参数继承：info 中存在的分段参数覆盖 parameter 解包值 (保证多节点分段同步)
        if info:
            _inherited = {}
            for _k in ("total_frames", "chunk_frames", "context_frames",
                       "continuation_method",
                       "fps", "clip_mode", "clip_tag"):
                if _k in info and info[_k] is not None:
                    _inherited[_k] = info[_k]
            total_frames = int(_inherited.get("total_frames", total_frames))
            chunk_frames = int(_inherited.get("chunk_frames", chunk_frames))
            context_frames = int(_inherited.get("context_frames", context_frames))
            continuation_method = _inherited.get(
                "continuation_method", continuation_method)
            fps = int(_inherited.get("fps", fps))
            clip_mode = _inherited.get("clip_mode", clip_mode)
            clip_tag = _inherited.get("clip_tag", clip_tag)
            if _inherited:
                print(f"[H3 Auto Chunks/Bruxos] Parâmetros herdados do info: {list(_inherited.keys())}")

        resolved_continuation, native_support_issues = native_masked.resolve_method(
            continuation_method)
        audio_drive_enabled = (
            audio_drive == "enable" if isinstance(audio_drive, str) else bool(audio_drive)
        )
        if resolved_continuation == native_masked.CONTINUATION_NATIVE:
            context_frames = native_masked.normalize_context_frames(
                context_frames,
                fps=fps,
                carry_generated_audio=not (audio_drive_enabled and drive_audio is not None),
            )

        out_info = {
            "total_frames": int(total_frames),
            "chunk_frames": int(chunk_frames),
            "context_frames": int(context_frames),
            "continuation_method_requested": str(continuation_method),
            "continuation_method": str(resolved_continuation),
            "native_mask_contract_version": native_masked.NATIVE_MASK_CONTRACT_VERSION,
            "native_mask_supported": not bool(native_support_issues),
            "fps": int(fps),
            "clip_mode": str(clip_mode),
            "clip_tag": str(clip_tag),
        }

        if clip_mode != "Clip_Tag":
            if chunk_frames <= 0:
                chunk_frames = total_frames
        width = max(32, (width // 32) * 32)
        height = max(32, (height // 32) * 32)
        # 兼容旧版 combo 字符串值 ("disable"/"enable")，新版为布尔复选框
        if isinstance(decode_output, str):
            decode_output = (decode_output == "enable")
        if isinstance(audio_drive, str):
            audio_drive = (audio_drive == "enable")
        if isinstance(lock_audio, str):
            lock_audio = (lock_audio == "enable")

        def _autogrow_to_list(ag_dict, prefix, max_count):
            """将 autogrow dict 转为有序 list，按 ref_xxx_N 的 N 排序"""
            if not ag_dict:
                return []
            result = []
            for i in range(max_count):
                key = f"{prefix}{i}"
                val = ag_dict.get(key)
                if val is not None:
                    result.append(val)
            return result

        ref_image_list = _autogrow_to_list(ref_images, "ref_image_", 10)

        ref_video_audios = ref_video_audios or {}
        ref_video_list = []
        for i in range(4):
            vkey = f"ref_video_{i}"
            vval = (ref_videos or {}).get(vkey)
            if vval is None:
                continue
            akey = f"ref_video_audio_{i}"
            soundtrack = ref_video_audios.get(akey)
            ref_video_list.append({"video": vval, "audio": soundtrack})

        ref_audio_list = _autogrow_to_list(ref_audios, "ref_audio_", 4)

        frames, audio, latent, denoised_latent, seam_info = h3_sampler.run_auto_context_generation(
            model=model, vae=vae, audio_vae=audio_vae, clip=clip,
            first_frame=first_frame, last_frame=last_frame,
            ref_images=ref_image_list, ref_videos=ref_video_list,
            ref_audios=ref_audio_list,
            long_prompt=long_prompt, width=width, height=height,
            total_frames=total_frames, fps=fps,
            chunk_frames=chunk_frames, context_frames=int(context_frames),
            continuation_method=continuation_method,
            steps=steps, cfg=cfg, sampler_name=sampler_name,
            scheduler=scheduler, seed=seed, prompt_mode=prompt_mode,
            prompt_format=prompt_format,
            clip_mode=clip_mode, clip_tag=clip_tag,
            crop_mode=crop_mode, ref_sync_mode=ref_sync_mode,
            decode_output=decode_output,
            drive_audio=drive_audio, audio_drive=audio_drive,
            latent_input=latent_input, sigmas=sigmas,
            denoise=denoise, lock_audio=lock_audio,
            video_context_denoise=video_context_denoise,
            sampler=sampler,
        )
        out_info.update(seam_info or {})
        return io.NodeOutput(frames, audio, latent, denoised_latent, out_info)
