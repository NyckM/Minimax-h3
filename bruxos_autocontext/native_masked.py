"""Continuação H3 exata usando a máscara nativa por token do ComfyUI.

Adaptado do ComfyUI-H3-Continuum (MIT), copyright 2026
ukr8b3g-cmyk. O prefixo aceito do chunk anterior é copiado para o
novo latent e marcado com 0 (preservar); a região nova recebe 1 (gerar).
"""

from __future__ import annotations

from typing import Any

import torch

import comfy.nested_tensor

from . import h3_conditioning


CONTINUATION_AUTO = "Auto (Native se disponível)"
CONTINUATION_GUIDE = "Guide / Motion Context"
CONTINUATION_NATIVE = "Native Masked / Exact"
CONTINUATION_METHODS = (
    CONTINUATION_AUTO,
    CONTINUATION_GUIDE,
    CONTINUATION_NATIVE,
)

NATIVE_MASK_CORE_PR = "ComfyUI PR #15375"
NATIVE_MASK_CONTRACT_VERSION = 1


class NativeMaskedContinuationError(RuntimeError):
    pass


def support_issues() -> list[str]:
    """Lista as capacidades do Core necessárias para a máscara H3 por token."""

    issues: list[str] = []
    try:
        import comfy.model_base as model_base
    except Exception as exc:
        return [f"não foi possível importar comfy.model_base: {exc}"]

    base_cls = getattr(model_base, "MiniMaxH3", None)
    if base_cls is None:
        issues.append("comfy.model_base.MiniMaxH3 ausente")
    else:
        for name in (
            "_denoise_mask_conds",
            "_denoise_mask_values",
            "_token_grid_masks",
            "scale_latent_inpaint",
        ):
            if not callable(getattr(base_cls, name, None)):
                issues.append(f"MiniMaxH3.{name} ausente")
    if not hasattr(comfy.nested_tensor, "NestedTensor"):
        issues.append("comfy.nested_tensor.NestedTensor ausente")
    return issues


def resolve_method(requested: str) -> tuple[str, list[str]]:
    """Resolve Auto sem alterar silenciosamente um Native solicitado."""

    requested = str(requested or CONTINUATION_GUIDE)
    if requested not in CONTINUATION_METHODS:
        requested = CONTINUATION_GUIDE
    issues = support_issues()
    if requested == CONTINUATION_AUTO:
        return (CONTINUATION_NATIVE if not issues else CONTINUATION_GUIDE), issues
    if requested == CONTINUATION_NATIVE and issues:
        detail = "; ".join(issues)
        raise NativeMaskedContinuationError(
            "Native Masked / Exact requer suporte de máscara por token do MiniMax H3 "
            f"({NATIVE_MASK_CORE_PR}) ou mais novo. Atualize o ComfyUI ou escolha "
            f"'{CONTINUATION_GUIDE}'. Capacidades ausentes: {detail}"
        )
    return requested, issues


def exact_audio_prefix_steps(context_frames: int, fps: int) -> int:
    """Converte o prefixo de vídeo para a grade H3 de áudio a 40 Hz."""

    frames = int(context_frames)
    fps = int(fps)
    if fps <= 0:
        raise NativeMaskedContinuationError("fps deve ser maior que zero")
    numerator = frames * 40
    if numerator % fps:
        raise NativeMaskedContinuationError(
            f"{frames} frames a {fps} fps não terminam em um token exato de áudio "
            "H3 a 40 Hz. Em 24 fps use 39 frames para continuidade AV exata."
        )
    return numerator // fps


def normalize_context_frames(
    context_frames: int,
    *,
    fps: int,
    carry_generated_audio: bool,
) -> int:
    """Seleciona o contrato AV seguro para novos workflows Native Masked."""

    frames = int(context_frames)
    if not carry_generated_audio:
        return frames
    if int(fps) == 24:
        return 39
    exact_audio_prefix_steps(frames, fps)
    return frames


def _streams(latent: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    video, audio = h3_conditioning.unpack_nested_latent(latent)
    if video is None or audio is None:
        raise NativeMaskedContinuationError(
            "Native Masked requer latent H3 completo com streams de vídeo e áudio"
        )
    return video, audio


def _existing_masks(latent: dict[str, Any]):
    raw = latent.get("noise_mask")
    if raw is None:
        return None, None
    if hasattr(raw, "is_nested") and raw.is_nested:
        masks = list(raw.unbind())
    elif isinstance(raw, (tuple, list)):
        masks = list(raw)
    else:
        masks = [raw]
    video = next((m for m in masks if torch.is_tensor(m) and m.ndim == 5), None)
    audio = next((m for m in masks if torch.is_tensor(m) and m.ndim == 4), None)
    return video, audio


def apply_native_prefix(
    latent: dict[str, Any],
    previous: dict[str, Any],
    *,
    context_frames: int,
    fps: int,
    carry_generated_audio: bool,
) -> dict[str, Any]:
    """Copia o tail aceito anterior e protege o prefixo do novo target."""

    if support_issues():
        # Mantém a mesma mensagem detalhada usada na resolução do modo.
        resolve_method(CONTINUATION_NATIVE)

    target_video, target_audio = _streams(latent)
    source_video, source_audio = _streams(previous)

    video_steps = h3_conditioning._latent_frames_for_pixel_frames(int(context_frames))
    video_steps = min(video_steps, int(source_video.shape[2]))
    if video_steps < 1 or int(target_video.shape[2]) <= video_steps:
        raise NativeMaskedContinuationError(
            "o chunk novo não possui região gerável depois do prefixo protegido"
        )
    if tuple(target_video.shape[-2:]) != tuple(source_video.shape[-2:]):
        raise NativeMaskedContinuationError(
            "a geometria do latent anterior não corresponde ao novo chunk"
        )

    target_video[:, :, :video_steps].copy_(
        source_video[:, :, -video_steps:].to(
            device=target_video.device, dtype=target_video.dtype
        )
    )

    existing_video_mask, existing_audio_mask = _existing_masks(latent)
    video_mask = torch.ones(
        (1, 1, int(target_video.shape[2]), int(target_video.shape[3]), int(target_video.shape[4])),
        device=target_video.device,
        dtype=torch.float32,
    )
    if existing_video_mask is not None:
        video_mask = existing_video_mask[:, :1].to(
            device=target_video.device, dtype=torch.float32
        ).clone()
    video_mask[:, :, :video_steps] = 0.0

    audio_mask = torch.ones(
        (1, 1, int(target_audio.shape[2]), int(target_audio.shape[3])),
        device=target_audio.device,
        dtype=torch.float32,
    )
    if existing_audio_mask is not None:
        audio_mask = existing_audio_mask[:, :1].to(
            device=target_audio.device, dtype=torch.float32
        ).clone()

    audio_steps = 0
    if carry_generated_audio:
        audio_steps = exact_audio_prefix_steps(context_frames, fps)
        audio_steps = min(audio_steps, int(source_audio.shape[-1]))
        if audio_steps < 1 or int(target_audio.shape[-1]) <= audio_steps:
            raise NativeMaskedContinuationError(
                "o áudio do chunk novo não possui região gerável depois do prefixo"
            )
        target_audio[..., :audio_steps].copy_(
            source_audio[..., -audio_steps:].to(
                device=target_audio.device, dtype=target_audio.dtype
            )
        )
        audio_mask[..., :audio_steps] = 0.0

    output = dict(latent)
    output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return output

