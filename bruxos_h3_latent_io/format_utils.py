from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence


CATALOG_SEPARATOR = "::"
CATALOG_LIBRARY_SUBFOLDER = "h3_latents"
NO_FILE = "(nenhum .latent encontrado; salve um primeiro)"


def sorted_shape_keys(keys: Iterable[str]) -> list[str]:
    """Return latent_shape_N keys in numeric order and reject missing indices."""
    shape_keys = [key for key in keys if key.startswith("latent_shape_")]
    try:
        shape_keys.sort(key=lambda key: int(key.rsplit("_", 1)[1]))
    except ValueError as exc:
        raise ValueError("Invalid latent shape key; expected latent_shape_N.") from exc

    indices = [int(key.rsplit("_", 1)[1]) for key in shape_keys]
    if indices and indices != list(range(len(indices))):
        raise ValueError(
            f"Latent shape keys must be contiguous from 0; found indices {indices}."
        )
    return shape_keys


def validate_packed_shapes(
    packed_shape: Sequence[int], stream_shapes: Sequence[Sequence[int]]
) -> None:
    if len(packed_shape) != 3 or packed_shape[1] != 1:
        raise ValueError(
            "Packed latent_tensor must have shape [batch, 1, flattened_streams]."
        )
    if not stream_shapes:
        raise ValueError("The file contains no latent stream shapes.")

    batch = int(packed_shape[0])
    expected_width = 0
    for index, shape in enumerate(stream_shapes):
        if len(shape) < 2:
            raise ValueError(f"Stream {index} has an invalid shape: {tuple(shape)}.")
        if any(int(size) < 0 for size in shape):
            raise ValueError(f"Stream {index} has a negative dimension: {tuple(shape)}.")
        if int(shape[0]) != batch:
            raise ValueError(
                f"Stream {index} batch {shape[0]} does not match packed batch {batch}."
            )
        expected_width += math.prod(int(size) for size in shape[1:])

    if int(packed_shape[2]) != expected_width:
        raise ValueError(
            "Packed latent size does not match its stream-shape metadata: "
            f"expected {expected_width}, found {packed_shape[2]}."
        )


def validate_h3_shapes(stream_shapes: Sequence[Sequence[int]]) -> None:
    if len(stream_shapes) != 2:
        raise ValueError(
            f"MiniMax H3 requires exactly 2 streams (video, audio); found {len(stream_shapes)}."
        )

    video = tuple(int(size) for size in stream_shapes[0])
    audio = tuple(int(size) for size in stream_shapes[1])
    if len(video) != 5 or video[1] != 24:
        raise ValueError(
            "H3 video latent must have shape [B, 24, T, H, W]; "
            f"found {video}."
        )
    if len(audio) != 4 or audio[1] != 32 or audio[2] != 2:
        raise ValueError(
            "H3 audio latent must have shape [B, 32, 2, T]; "
            f"found {audio}."
        )
    if video[0] != audio[0]:
        raise ValueError(
            f"Video and audio batch sizes differ: {video[0]} versus {audio[0]}."
        )
    if video[0] != 1:
        raise ValueError(
            f"MiniMax H3 supports batch size 1; found batch {video[0]}."
        )
    if video[2] < 1 or video[3] < 1 or video[4] < 1:
        raise ValueError(f"H3 video latent contains an invalid dimension: {video}.")
    if audio[3] < 1:
        raise ValueError(f"H3 audio latent contains an invalid dimension: {audio}.")


def make_catalog_label(source: str, relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    return f"{source}{CATALOG_SEPARATOR}{normalized}"


def split_catalog_label(label: str) -> tuple[str, str]:
    source, separator, relative_path = label.partition(CATALOG_SEPARATOR)
    if not separator or source not in {"input", "output"} or not relative_path:
        raise ValueError(
            "Latent path must be an input::relative/path.latent or "
            "output::relative/path.latent catalog entry."
        )
    if not relative_path.lower().endswith(".latent"):
        raise ValueError("Only .latent files are accepted.")
    return source, relative_path.replace("/", os.sep)


def resolve_catalog_path(label: str, roots: dict[str, str]) -> str:
    source, relative_path = split_catalog_label(label)
    root = os.path.realpath(roots[source])
    path = os.path.realpath(os.path.join(root, relative_path))
    try:
        common = os.path.commonpath((root, path))
    except ValueError as exc:
        raise ValueError("Latent path is outside the selected ComfyUI directory.") from exc
    if os.path.normcase(common) != os.path.normcase(root):
        raise ValueError("Latent path is outside the selected ComfyUI directory.")
    return path


def scan_latent_catalog(roots: dict[str, str]) -> list[str]:
    entries: list[str] = []
    for source in ("output", "input"):
        root = roots.get(source)
        if not root or not os.path.isdir(root):
            continue
        # Do not walk an entire ComfyUI output tree: video workflows may contain
        # hundreds of thousands of decoded frames. This plugin's saver and drag
        # uploader both use this dedicated persistent library by default.
        library_root = os.path.join(root, CATALOG_LIBRARY_SUBFOLDER)
        if not os.path.isdir(library_root):
            continue
        for directory, _, filenames in os.walk(library_root):
            for filename in filenames:
                if not filename.lower().endswith(".latent"):
                    continue
                path = os.path.join(directory, filename)
                relative_path = os.path.relpath(path, root)
                entries.append(make_catalog_label(source, relative_path))
    return sorted(entries, key=str.casefold)
