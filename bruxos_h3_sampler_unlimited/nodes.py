import logging
import math
import re

import torch

import comfy.nested_tensor
import comfy.sample
import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from .preview import begin_preview_execution


AUDIO_LATENT_FPS = 40
VIDEO_FPS = 24
CONTEXT_VIDEO_STEPS = 2
CANVAS_MULTIPLE = 32
DESCRIPTION_FIELD = re.compile(r"(?:integrated_multimodal_description|detailed_description)\s*:", re.IGNORECASE)
SHOT_MARKER = re.compile(r"\[Shot\s+(\d+)\](?:\s+At\s+(\d+):(\d{2})\.(\d{3}),)?", re.IGNORECASE)
DESCRIPTION_END = re.compile(r"\n\s*(?:overall_soundscape|non_diegetic_music)\s*:", re.IGNORECASE)
PICTURE_LABEL = re.compile(r"<Picture\s+\d+>", re.IGNORECASE)
SHOT_OPENING = re.compile(r"^(\s*)(?:the camera|the shot)\s+(?:cuts|transitions|changes|switches)\s+to\s+", re.IGNORECASE)
SHOT_BREAK = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'])\s+|\n\s*\n+")
SHOT_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+")


def _pixel_frames(latent_t):
    return sum(FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)] for index in range(latent_t))


def _audio_steps(frames):
    return round(frames * AUDIO_LATENT_FPS / VIDEO_FPS)


def _timestamp_frame(minutes, seconds, milliseconds, fps):
    return round((int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000.0) * fps)


def _frame_timestamp(frame, fps):
    total_milliseconds = round(frame / fps * 1000.0)
    minutes, milliseconds = divmod(total_milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _drop_picture_anchors(prompt):
    field = DESCRIPTION_FIELD.search(prompt)
    if field is None:
        return PICTURE_LABEL.sub("the established subject and scene", prompt)
    prefix = "\n".join(line for line in prompt[:field.start()].splitlines() if "picture" not in line.lower())
    if prefix:
        prefix += "\n"
    return prefix + PICTURE_LABEL.sub("the established subject and scene", prompt[field.start():])


def _shot_body_for_range(body, shot_start, shot_end, frame_start, frame_end):
    if frame_start <= shot_start and frame_end >= shot_end:
        return body

    units = []
    for sentence in SHOT_BREAK.split(body):
        sentence = sentence.strip()
        if not sentence:
            continue
        clauses = SHOT_CLAUSE_BREAK.split(sentence) if len(re.findall(r"\w+", sentence)) > 32 else [sentence]
        for clause in clauses:
            words = clause.split()
            units.extend(" ".join(words[index:index + 24]) for index in range(0, len(words), 24))
    weights = [max(1, len(re.findall(r"\w+", unit))) for unit in units]
    total_weight = sum(weights)
    start_weight = total_weight * max(frame_start, shot_start) - total_weight * shot_start
    end_weight = total_weight * min(frame_end, shot_end) - total_weight * shot_start
    duration = shot_end - shot_start
    start_weight /= duration
    end_weight /= duration

    selected = []
    offset = 0
    for unit, weight in zip(units, weights):
        if start_weight <= offset < end_weight:
            selected.append(unit)
        offset += weight
    return " " + " ".join(selected) if selected else " Continue the established shot and its ongoing action."


def _prompt_for_chunk(prompt, frame_start, frame_end, total_frames, fps, content_start=None, continuation=False, drop_picture_anchors=False):
    content_start = frame_start if content_start is None else content_start
    if drop_picture_anchors:
        prompt = _drop_picture_anchors(prompt)
    field = DESCRIPTION_FIELD.search(prompt)
    description_start = field.end() if field is not None else 0
    description_end_match = DESCRIPTION_END.search(prompt, description_start)
    description_end = description_end_match.start() if description_end_match is not None else len(prompt)
    markers = list(SHOT_MARKER.finditer(prompt, description_start, description_end))
    if not markers:
        return prompt

    shot_starts = []
    for index, marker in enumerate(markers):
        if int(marker.group(1)) != index + 1:
            raise ValueError("Os números [Shot N] do MiniMax devem começar em 1 e crescer em sequência.")
        if marker.group(2) is None:
            if index:
                raise ValueError("Depois do [Shot 1], cada marca deve usar 'At MM:SS.mmm,'.")
            shot_starts.append(0)
        else:
            if not index:
                raise ValueError("O [Shot 1] do MiniMax não deve ter tempo.")
            shot_starts.append(_timestamp_frame(marker.group(2), marker.group(3), marker.group(4), fps))
    if any(right <= left for left, right in zip(shot_starts, shot_starts[1:])):
        raise ValueError("Os tempos dos shots MiniMax devem crescer sem repetir ou voltar.")

    selected = []
    for index, marker in enumerate(markers):
        shot_end = shot_starts[index + 1] if index + 1 < len(markers) else total_frames
        if shot_starts[index] < frame_end and shot_end > content_start:
            segment_end = markers[index + 1].start() if index + 1 < len(markers) else description_end
            body = _shot_body_for_range(prompt[marker.end():segment_end], shot_starts[index], shot_end, content_start, frame_end)
            if body is not None:
                selected.append((marker, body, shot_starts[index]))
    if not selected:
        raise ValueError(f"Nenhum shot do prompt cobre os frames {content_start} a {frame_end - 1}.")

    rewritten = []
    for index, (marker, body, shot_start) in enumerate(selected):
        marker_text = f"[Shot {index + 1}]"
        if index:
            marker_text += f" At {_frame_timestamp(shot_start - frame_start, fps)},"
        elif continuation and shot_start < content_start:
            body = SHOT_OPENING.sub(r"\1the continuing shot shows ", body, count=1)
            marker_text += " Continue seamlessly from the provided opening frames at this point in the shot; do not restart or replay earlier actions."
        rewritten.append(marker_text + body.rstrip() + " ")
    return prompt[:markers[0].start()] + "".join(rewritten) + prompt[description_end:]


def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _reference_image(image, width, height):
    source_height, source_width = image.shape[1:3]
    scale = min(1.0, math.sqrt((width * height) / (source_width * source_height)))
    target_width = max(CANVAS_MULTIPLE, round(source_width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    target_height = max(CANVAS_MULTIPLE, round(source_height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return _resize(image, target_width, target_height, "disabled")


def _prompt_tokens(clip, prompt, images, positive, width, height, continuation):
    refs = positive[0].get("minimax_refs") if positive else None
    image_list = [] if images is None else [images[index:index + 1] for index in range(images.shape[0])]
    if refs:
        ref_items = []
        image_index = 0
        for ref in refs:
            kind = ref["kind"]
            if kind == "image":
                if image_index >= len(image_list):
                    raise ValueError("O H3 Sampler Unlimited precisa receber em images todas as imagens de referência Ref2VA, na ordem original.")
                ref_items.append({"type": "image", "data": _reference_image(image_list[image_index], width, height)})
                image_index += 1
            elif kind == "audio":
                ref_items.append({"type": "audio"})
            elif kind in ("video", "video_audio"):
                raise ValueError("O H3 Sampler Unlimited não consegue reconstruir uma referência Ref2VA de vídeo a partir da entrada images. Use H3 Auto Chunks.")
        if image_index != len(image_list):
            raise ValueError("A entrada images recebeu mais imagens do que o condicionamento Ref2VA utiliza.")
        return clip.tokenize(prompt, minimax_ref_items=ref_items)

    prompt_images = []
    for index, image in enumerate(() if continuation else image_list):
        prompt_images.append(_resize(image, width, height, "disabled" if index == 0 else "center"))
    return clip.tokenize(prompt, images=prompt_images)


def _encode_prompt(clip, prompt, images, positive, width, height, continuation):
    conditioning = clip.encode_from_tokens_scheduled(_prompt_tokens(clip, prompt, images, positive, width, height, continuation))
    if len(conditioning) != 1:
        raise ValueError("O H3 Sampler Unlimited espera apenas um segmento de condicionamento MiniMax H3.")
    return conditioning[0]


def _chunk_plan(video_t, audio_t, chunk_frames):
    max_chunk_frames = chunk_frames - (chunk_frames - 5) % 17
    max_chunk_t = ((max_chunk_frames - 5) // 17) * 5 + CONTEXT_VIDEO_STEPS

    if video_t < CONTEXT_VIDEO_STEPS or (video_t - CONTEXT_VIDEO_STEPS) % 5:
        raise ValueError("O latent de vídeo MiniMax H3 não está na grade válida 17k+5.")

    total_frames = _pixel_frames(video_t)
    if audio_t != _audio_steps(total_frames):
        raise ValueError("A duração do latent de áudio MiniMax H3 não corresponde à duração do vídeo.")

    plan = []
    video_end = 0
    audio_end = 0
    output_frames = 0
    remaining = video_t
    while remaining:
        if not plan:
            chunk_t = min(max_chunk_t, remaining)
            video_start = 0
            new_video_t = chunk_t
            chunk_frame_count = _pixel_frames(chunk_t)
        else:
            new_video_t = min(max_chunk_t - CONTEXT_VIDEO_STEPS, remaining)
            chunk_t = new_video_t + CONTEXT_VIDEO_STEPS
            video_start = video_end - CONTEXT_VIDEO_STEPS
            chunk_frame_count = _pixel_frames(chunk_t)

        output_frames += chunk_frame_count if not plan else chunk_frame_count - 5
        next_audio_end = _audio_steps(output_frames)
        chunk_audio_t = _audio_steps(chunk_frame_count)
        new_audio_t = next_audio_end - audio_end
        context_audio_t = 0 if not plan else chunk_audio_t - new_audio_t
        audio_start = 0 if not plan else audio_end - context_audio_t

        plan.append({
            "video_start": video_start,
            "video_end": video_start + chunk_t,
            "audio_start": audio_start,
            "audio_end": next_audio_end,
            "context_audio_t": context_audio_t,
            "frame_start": 0 if not plan else output_frames - chunk_frame_count,
            "frame_end": output_frames,
        })
        video_end += new_video_t
        audio_end = next_audio_end
        remaining -= new_video_t

    return plan


def _conditioning_for_chunk(original_conds, frame_start, frame_end, encoded_prompt, video_context=None,
                            audio_context=None, audio_end_frame=5.0):
    conds = {name: [item.copy() for item in values] for name, values in original_conds.items()}
    positive = conds.get("positive")
    if positive is None:
        raise ValueError("O H3 Sampler Unlimited precisa de um guider padrão com condicionamento positivo.")

    cross_attn, prompt_metadata = encoded_prompt
    for cond in positive:
        cond["cross_attn"] = cross_attn
        token_tags = prompt_metadata.get("minimax_token_tags")
        if token_tags is not None:
            cond["minimax_token_tags"] = token_tags
        else:
            cond.pop("minimax_token_tags", None)
        keyframes = []
        for keyframe in cond.get("minimax_keyframes", ()):
            position = keyframe["resolved_frame_index"]
            if frame_start <= position < frame_end:
                local_keyframe = keyframe.copy()
                local_keyframe["resolved_frame_index"] = position - frame_start
                keyframes.append(local_keyframe)

        if video_context is not None:
            keyframes.append({"resolved_frame_index": 0, "latent": video_context})
        if audio_context is not None:
            audio_start = audio_end_frame - audio_context.shape[-1] / FRAME_RESCALE
            keyframes.append({"resolved_frame_index": audio_start, "audio_latent": audio_context})
        if keyframes:
            cond["minimax_keyframes"] = keyframes
        else:
            cond.pop("minimax_keyframes", None)
    return conds


class _FixedNoise:
    def __init__(self, seed, samples):
        self.seed = seed
        self.samples = samples

    def generate_noise(self, _latent):
        return self.samples


class MiniMaxH3SamplerCustomAdvancedUnlimited(SamplerCustomAdvanced):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BruxosH3SamplerCustomAdvancedUnlimited",
            display_name="H3 Sampler Unlimited (Bruxos TESTE)",
            category="Bruxos do VFX/MiniMax H3/Chunks",
            description=("Substitui o SamplerCustomAdvanced para processar um latent AV longo do MiniMax H3 "
                         "em chunks. Reaproveita os 5 frames finais e a cauda de áudio no chunk seguinte, "
                         "ajusta os tempos [Shot N] por chunk e reúne tudo no final. Não use com noise_mask; "
                         "referências Ref2VA em vídeo não podem ser reconstruídas por este node."),
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
                io.Clip.Input("clip", tooltip="O mesmo CLIP MiniMax H3 usado para criar o condicionamento original."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                tooltip="Prompt MiniMax com [Shot 1] e [Shot N] At MM:SS.mmm. Os tempos absolutos serão recalculados dentro de cada chunk."),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001,
                               tooltip="FPS usado para converter os tempos absolutos do prompt em posições exatas de frame dentro de cada chunk."),
                io.Int.Input("chunk_frames", default=124, min=22, max=3600, step=17,
                             tooltip="Máximo de frames processados de uma vez. O valor é ajustado para baixo à grade válida 17k+5 do H3; use o maior que couber na VRAM."),
                io.Boolean.Input("debug", default=False,
                                 tooltip="Mostra no console e na saída chunk_prompts o prompt realmente usado em cada chunk."),
                io.Image.Input("images", optional=True,
                               tooltip="Batch das imagens do condicionamento original: primeiro frame, depois último frame opcional; ou todas as referências Ref2VA que sejam imagens, na ordem."),
            ],
            outputs=[
                io.Latent.Output(display_name="latent processado"),
                io.Latent.Output(display_name="latent sem ruído"),
                io.String.Output(display_name="prompts por chunk"),
            ],
        )

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image, clip, prompt, fps=24.0, chunk_frames=124, debug=False, images=None):
        samples = latent_image["samples"]
        if not samples.is_nested:
            sampled = super().execute(noise, guider, sampler, sigmas, latent_image)
            return io.NodeOutput(sampled[0], sampled[1], "")

        streams = samples.unbind()
        if len(streams) != 2 or streams[0].ndim != 5 or streams[0].shape[1] != 24 or streams[1].ndim != 4 or streams[1].shape[1] != 32:
            sampled = super().execute(noise, guider, sampler, sigmas, latent_image)
            return io.NodeOutput(sampled[0], sampled[1], "")

        video, audio = streams
        plan = _chunk_plan(video.shape[2], audio.shape[-1], chunk_frames)
        if len(plan) > 1 and "noise_mask" in latent_image:
            raise ValueError("H3 Sampler Unlimited (Bruxos): noise_mask não é suportado quando há mais de um chunk. Use o H3 Auto Chunks Bruxos para inpaint/denoise com máscara.")

        fixed_latent = latent_image.copy()
        fixed_latent["samples"] = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            samples,
            latent_image.get("downscale_ratio_spacial"),
            latent_image.get("downscale_ratio_temporal"),
        )
        full_noise = noise.generate_noise(fixed_latent)
        if not full_noise.is_nested or len(full_noise.unbind()) != 2:
            raise ValueError("O H3 Sampler Unlimited esperava ruído NestedTensor contendo vídeo e áudio.")
        video_noise, audio_noise = full_noise.unbind()

        original_conds = guider.original_conds
        positive = original_conds.get("positive")
        if positive is None:
            raise ValueError("O H3 Sampler Unlimited precisa de um guider padrão com condicionamento positivo.")
        ref2va = bool(positive[0].get("minimax_refs"))
        total_frames = plan[-1]["frame_end"]
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        output_video = []
        output_audio = []
        denoised_video = []
        denoised_audio = []
        previous_video = None
        previous_audio = None
        previous_frame_count = None
        output_template = None
        denoised_template = None
        debug_prompts = []
        preview_execution = begin_preview_execution(guider.model_patcher, len(plan))

        try:
            for index, chunk in enumerate(plan):
                vs, ve = chunk["video_start"], chunk["video_end"]
                aus, aue = chunk["audio_start"], chunk["audio_end"]
                context_audio_t = chunk["context_audio_t"]

                chunk_latent = fixed_latent.copy()
                chunk_latent["samples"] = comfy.nested_tensor.NestedTensor((
                    video[:, :, vs:ve],
                    audio[..., aus:aue],
                ))
                chunk_noise = comfy.nested_tensor.NestedTensor((
                    video_noise[:, :, vs:ve],
                    audio_noise[..., aus:aue],
                ))

                video_context = None if previous_video is None else previous_video[:, :, -CONTEXT_VIDEO_STEPS:].clone()
                audio_context = None if previous_audio is None else previous_audio[..., -context_audio_t:].clone()
                audio_end_frame = 5.0
                if previous_audio is not None:
                    overhang = previous_audio.shape[-1] - FRAME_RESCALE * previous_frame_count
                    audio_end_frame += overhang / FRAME_RESCALE
                continuation = index > 0
                content_start = chunk["frame_start"] + (5 if continuation else 0)
                chunk_prompt = _prompt_for_chunk(
                    prompt,
                    chunk["frame_start"],
                    chunk["frame_end"],
                    total_frames,
                    fps,
                    content_start=content_start,
                    continuation=continuation,
                    drop_picture_anchors=continuation and not ref2va,
                )
                if debug:
                    debug_prompt = (
                        f"=== Chunk {index + 1}: sampled frames {chunk['frame_start']}-{chunk['frame_end'] - 1}; "
                        f"output frames {content_start}-{chunk['frame_end'] - 1} ===\n{chunk_prompt}"
                    )
                    debug_prompts.append(debug_prompt)
                    logging.info("H3 Sampler Unlimited (Bruxos) — prompt do chunk:\n%s", debug_prompt)
                encoded_prompt = _encode_prompt(clip, chunk_prompt, images, positive, width, height, continuation)
                guider.original_conds = _conditioning_for_chunk(
                    original_conds,
                    chunk["frame_start"],
                    chunk["frame_end"],
                    encoded_prompt,
                    video_context,
                    audio_context,
                    audio_end_frame,
                )

                chunk_seed = (noise.seed + index) & 0xffffffffffffffff
                if preview_execution is not None:
                    preview_execution.set_chunk(
                        index,
                        chunk["frame_start"],
                        chunk["frame_end"] - 1,
                        content_start,
                        chunk["frame_end"] - 1,
                        0 if index == 0 else CONTEXT_VIDEO_STEPS,
                    )
                try:
                    sampled, denoised = super().execute(
                        _FixedNoise(chunk_seed, chunk_noise), guider, sampler, sigmas, chunk_latent
                    )
                finally:
                    if preview_execution is not None:
                        preview_execution.clear_chunk()
                output_template = sampled
                denoised_template = denoised
                previous_video, previous_audio = sampled["samples"].unbind()
                previous_frame_count = chunk["frame_end"] - chunk["frame_start"]
                denoised_chunk_video, denoised_chunk_audio = denoised["samples"].unbind()

                video_trim = 0 if index == 0 else CONTEXT_VIDEO_STEPS
                audio_trim = 0 if index == 0 else context_audio_t
                output_video.append(previous_video[:, :, video_trim:].clone())
                output_audio.append(previous_audio[..., audio_trim:].clone())
                denoised_video.append(denoised_chunk_video[:, :, video_trim:].clone())
                denoised_audio.append(denoised_chunk_audio[..., audio_trim:].clone())
        finally:
            guider.original_conds = original_conds
            if preview_execution is not None:
                preview_execution.close()

        output_template["samples"] = comfy.nested_tensor.NestedTensor((
            torch.cat(output_video, dim=2),
            torch.cat(output_audio, dim=-1),
        ))
        denoised_template["samples"] = comfy.nested_tensor.NestedTensor((
            torch.cat(denoised_video, dim=2),
            torch.cat(denoised_audio, dim=-1),
        ))
        return io.NodeOutput(output_template, denoised_template, "\n\n".join(debug_prompts))

    sample = execute
