"""Standalone helpers for visualizing Triton linear layouts with tensor-viz."""

from __future__ import annotations

import colorsys
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "tensor-viz" / "python" / "src"))

from tensor_viz import SessionData, create_session_data, viz

LayoutColorAxes = Mapping[str, str]
LayoutColorRanges = Mapping[str, tuple[float, float]]
DEFAULT_COLOR_RANGES = {
    "H": (0.0, 0.85),
    "S": (0.35, 0.95),
    "L": (0.25, 0.95),
}


def _axis_by_name(names: list[str], needle: str, default: int | None) -> int | None:
    """Return the first axis whose name contains ``needle``."""

    lowered = needle.lower()
    return next(
        (axis for axis, name in enumerate(names) if lowered in name.lower()),
        default,
    )


def _dense_hs_values(colors: np.ndarray) -> list[float]:
    """Flatten one dense HS tensor into viewer manifest order."""

    return colors.reshape(-1, 2).ravel().tolist()


def _dense_rgba_values(colors: np.ndarray) -> list[float]:
    """Flatten one dense RGBA tensor into viewer manifest order."""

    return colors.reshape(-1, 4).ravel().tolist()


def _viewer_axis_labels(names: list[str]) -> list[str]:
    """Convert dim names into viewer-safe axis labels."""

    counts: dict[str, int] = {}
    labels: list[str] = []
    for name in names:
        base = next((char.upper() for char in name if char.isalpha()), "A")
        index = counts.get(base, 0)
        counts[base] = index + 1
        labels.append(base if index == 0 else f"{base}{index}")
    return labels


def _logical_output_dims(output_dims: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Return logical output dims in viewer order."""

    names = [name.lower() for name, _size in output_dims]
    return list(reversed(output_dims)) if names == ["x", "y"] else output_dims


def _logical_axis_labels(output_names: list[str]) -> list[str]:
    """Return viewer labels for logical output axes."""

    return ["Y", "X"] if len(output_names) == 2 else _viewer_axis_labels(output_names)


def _resolve_axis(names: list[str], axis_name: str) -> int:
    """Resolve one user-provided axis name against layout input dims."""

    lowered = axis_name.lower()
    exact = [axis for axis, name in enumerate(names) if name.lower() == lowered]
    if exact:
        return exact[0]
    partial = [axis for axis, name in enumerate(names) if lowered in name.lower()]
    if len(partial) == 1:
        return partial[0]
    raise ValueError(f"Unknown layout axis {axis_name!r}; expected one of {names!r}.")


def _normalize_color_axes(
    input_names: list[str],
    input_shape: tuple[int, ...],
    color_axes: LayoutColorAxes | None,
) -> dict[str, int | None]:
    """Resolve which input axes drive hue, saturation, and lightness."""

    channels = {
        "H": _axis_by_name(input_names, "thread", 1 if len(input_shape) >= 3 else None),
        "S": _axis_by_name(input_names, "warp", 0 if len(input_shape) >= 3 else None),
        "L": _axis_by_name(input_names, "reg", len(input_shape) - 1 if input_shape else None),
    }
    if color_axes is None:
        return channels
    for axis_name, channel_name in color_axes.items():
        channel = channel_name.upper()
        if channel not in channels:
            raise ValueError(f"Unknown color channel {channel_name!r}; expected H, S, or L.")
        channels[channel] = _resolve_axis(input_names, axis_name)
    used_axes = [axis for axis in channels.values() if axis is not None]
    if len(used_axes) != len(set(used_axes)):
        raise ValueError("Each color axis must map to at most one channel.")
    return channels


def _hue(value: int, size: int) -> int:
    """Return one hue value from an input axis coordinate."""

    return round((360 * value) / size) if size > 1 else 0


def _saturation(value: int, size: int) -> float:
    """Return one saturation value from an input axis coordinate."""

    return (value + 1) / size if size > 0 else 1.0


def _color_value(
    channel: str,
    coord: tuple[int, ...],
    shape: tuple[int, ...],
    axis: int | None,
    color_ranges: LayoutColorRanges | None,
) -> float:
    """Return one configured color-channel value."""

    if axis is None:
        defaults = {"H": 0.0, "S": 1.0, "L": 0.0}
        return defaults[channel]
    position = 0.0 if shape[axis] <= 1 else coord[axis] / (shape[axis] - 1)
    value = position
    ranges = DEFAULT_COLOR_RANGES if color_ranges is None else {
        **DEFAULT_COLOR_RANGES,
        **color_ranges,
    }
    start, stop = ranges[channel]
    if shape[axis] <= 1:
        return start
    return start + ((stop - start) * position)


def _rgba_color(hue: float, saturation: float, lightness: float) -> tuple[float, float, float, float]:
    """Convert the configured layout color channels into an RGBA tuple."""

    red, green, blue = colorsys.hsv_to_rgb(
        hue % 1.0,
        min(max(saturation, 0.0), 1.0),
        min(max(lightness, 0.0), 1.0),
    )
    return (
        red,
        green,
        blue,
        1.0,
    )


def create_layout_session_data(
    layout: Any,
    *,
    name: str | None = None,
    color_axes: LayoutColorAxes | None = None,
    color_ranges: LayoutColorRanges | None = None,
) -> SessionData:
    """Build a tensor-viz session for a linear-layout input/output mapping."""

    input_dims = list(layout.bases)
    input_names = [dim_name for dim_name, _bases in input_dims]
    input_shape = tuple(1 << len(bases) for _dim_name, bases in input_dims)
    output_dims = _logical_output_dims(list(layout.out_dims))
    output_names = [dim_name for dim_name, _size in output_dims]
    output_shape = tuple(size for _dim_name, size in output_dims)
    channel_axes = _normalize_color_axes(input_names, input_shape, color_axes)

    hardware_tensor = np.zeros(
        input_shape,
        dtype=np.float32 if color_ranges and "L" in color_ranges else np.int32,
    )
    hardware_colors = np.zeros((*input_shape, 2), dtype=np.float32)
    hardware_rgba = np.zeros((*input_shape, 4), dtype=np.float32)
    logical_tensor = np.full(
        output_shape,
        -1,
        dtype=np.float32 if color_ranges and "L" in color_ranges else np.int32,
    )
    logical_colors = np.zeros((*output_shape, 2), dtype=np.float32)
    logical_rgba = np.zeros((*output_shape, 4), dtype=np.float32)

    for input_coord in np.ndindex(input_shape):
        hue_axis = channel_axes["H"]
        saturation_axis = channel_axes["S"]
        lightness_axis = channel_axes["L"]
        hue = _color_value("H", input_coord, input_shape, hue_axis, color_ranges)
        saturation = _color_value(
            "S",
            input_coord,
            input_shape,
            saturation_axis,
            color_ranges,
        )
        lightness = _color_value(
            "L",
            input_coord,
            input_shape,
            lightness_axis,
            color_ranges,
        )
        hardware_tensor[input_coord] = lightness
        hardware_colors[input_coord] = (hue, saturation)
        # temporary hack: use direct rgba so lightness is absolute instead of
        # being renormalized by tensor-viz's heatmap path. remove this once
        # tensor-viz supports native custom color lightness.
        hardware_rgba[input_coord] = _rgba_color(hue, saturation, lightness)

        output_coord_map = layout.apply(dict(zip(input_names, input_coord, strict=True)))
        output_coord = tuple(output_coord_map[dim_name] for dim_name in output_names)
        logical_tensor[output_coord] = lightness
        logical_colors[output_coord] = hardware_colors[input_coord]
        logical_rgba[output_coord] = hardware_rgba[input_coord]

    return create_session_data(
        {
            "Hardware tensor": hardware_tensor,
            "Logical tensor": logical_tensor,
        },
        name=name or "Layout",
        labels={
            "Hardware tensor": _viewer_axis_labels(input_names),
            "Logical tensor": _logical_axis_labels(output_names),
        },
        color_instructions={
            "tensor-1": [{"mode": "rgba", "kind": "dense", "values": _dense_rgba_values(hardware_rgba)}],
            "tensor-2": [{"mode": "rgba", "kind": "dense", "values": _dense_rgba_values(logical_rgba)}],
        },
    )


def create_layouts_session_data(
    layouts: dict[str, Any] | list[tuple[str, Any]],
    *,
    color_axes: LayoutColorAxes | None = None,
    color_ranges: LayoutColorRanges | None = None,
) -> SessionData:
    """Build one multi-tab session for several linear layouts."""

    entries = layouts.items() if isinstance(layouts, dict) else layouts
    manifest = {"version": 1, "tabs": []}
    tensor_bytes: dict[str, bytes] = {}
    for tab_index, (name, layout) in enumerate(entries, start=1):
        session_data = create_layout_session_data(
            layout,
            name=name,
            color_axes=color_axes,
            color_ranges=color_ranges,
        )
        layout_manifest = json.loads(session_data.manifest_bytes)
        tab = layout_manifest["tabs"][0]
        tab_id = f"tab-{tab_index}"
        remapped_ids: dict[str, str] = {}
        for tensor_index, tensor in enumerate(tab["tensors"], start=1):
            tensor_id = f"tensor-{tensor_index}"
            remapped_ids[tensor["id"]] = tensor_id
            old_file = tensor["dataFile"]
            tensor["id"] = tensor_id
            tensor["dataFile"] = f"tabs/{tab_id}/tensors/{tensor_id}.bin"
            tensor_bytes[tensor["dataFile"]] = session_data.tensor_bytes[old_file]
        for tensor in tab["viewer"]["tensors"]:
            tensor["id"] = remapped_ids[tensor["id"]]
        tab["id"] = tab_id
        tab["viewer"]["activeTensorId"] = remapped_ids[tab["viewer"]["activeTensorId"]]
        manifest["tabs"].append(tab)
    return SessionData(
        manifest_bytes=json.dumps(manifest).encode("utf-8"),
        tensor_bytes=tensor_bytes,
    )


def visualize_layout(
    layout: Any,
    *,
    name: str | None = None,
    color_axes: LayoutColorAxes | None = None,
    color_ranges: LayoutColorRanges | None = None,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    keep_alive: bool = True,
):
    """Launch the viewer for one linear layout."""

    session_data = create_layout_session_data(
        layout,
        name=name,
        color_axes=color_axes,
        color_ranges=color_ranges,
    )
    return viz(
        np.zeros((1,), dtype=np.float32),
        session_data=session_data,
        open_browser=open_browser,
        host=host,
        port=port,
        keep_alive=keep_alive,
    )


def visualize_layouts(
    layouts: dict[str, Any] | list[tuple[str, Any]],
    *,
    color_axes: LayoutColorAxes | None = None,
    color_ranges: LayoutColorRanges | None = None,
    open_browser: bool = True,
    host: str = "127.0.0.1",
    port: int = 0,
    keep_alive: bool = True,
):
    """Launch the viewer for several linear layouts, one tab per layout."""

    session_data = create_layouts_session_data(
        layouts,
        color_axes=color_axes,
        color_ranges=color_ranges,
    )
    return viz(
        np.zeros((1,), dtype=np.float32),
        session_data=session_data,
        open_browser=open_browser,
        host=host,
        port=port,
        keep_alive=keep_alive,
    )
