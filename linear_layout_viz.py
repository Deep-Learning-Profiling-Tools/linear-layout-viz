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
LINEAR_LAYOUT_AXES = ("thread", "warp", "register")
LINEAR_LAYOUT_CHANNELS = ("H", "S", "L")
OUTPUT_AXIS_NAMES = (
    "x",
    "y",
    "z",
    "w",
    "v",
    "u",
    "t",
    "s",
    "r",
    "q",
    "p",
    "o",
    "n",
    "m",
    "l",
    "k",
    "j",
    "i",
    "h",
    "g",
    "f",
    "e",
    "d",
    "c",
    "b",
    "a",
)


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


def _dense_rgb_values(colors: np.ndarray) -> list[float]:
    """Flatten one dense RGB tensor into viewer manifest order."""

    return colors.reshape(-1, 3).ravel().tolist()


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


def _linear_layout_axis_label(name: str) -> str | None:
    """Return the sidebar axis label for one layout axis."""

    lowered = name.lower()
    if "thread" in lowered:
        return "thread"
    if "warp" in lowered:
        return "warp"
    if "reg" in lowered:
        return "register"
    return None


def _linear_layout_state(
    input_dims: list[tuple[str, Any]],
    color_axes: LayoutColorAxes | None,
    color_ranges: LayoutColorRanges | None,
) -> dict[str, Any]:
    """Return the sidebar state for the linear-layout editor."""

    input_names = [dim_name for dim_name, _bases in input_dims]
    bases = {axis: "[]" for axis in LINEAR_LAYOUT_AXES}
    axis_labels: dict[str, str] = {}
    for dim_name, dim_bases in input_dims:
        axis_label = _linear_layout_axis_label(dim_name)
        if axis_label is None:
            continue
        axis_labels[dim_name] = axis_label
        bases[axis_label] = json.dumps(dim_bases or [], separators=(",", ":"))
    present = set(axis_labels.values())
    mapping: dict[str, str] = {
        "H": "warp" if "warp" in present else "none",
        "S": "thread" if "thread" in present else "none",
        "L": "register" if "register" in present else "none",
    }
    if color_axes:
        for axis_name, channel_name in color_axes.items():
            channel = channel_name.upper()
            if channel not in LINEAR_LAYOUT_CHANNELS:
                continue
            axis_index = _resolve_axis(input_names, axis_name)
            axis_label = _linear_layout_axis_label(input_names[axis_index])
            mapping[channel] = axis_label if axis_label in LINEAR_LAYOUT_AXES else "none"
    ranges = DEFAULT_COLOR_RANGES if color_ranges is None else {
        **DEFAULT_COLOR_RANGES,
        **color_ranges,
    }
    range_state = {
        channel: [
            format(ranges[channel][0], "g"),
            format(ranges[channel][1], "g"),
        ]
        for channel in LINEAR_LAYOUT_CHANNELS
    }
    return {
        "bases": bases,
        "mapping": mapping,
        "ranges": range_state,
    }


def _logical_output_dims(output_dims: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Return logical output dims in viewer order."""

    names = [name.lower() for name, _size in output_dims]
    return list(reversed(output_dims)) if names == ["x", "y"] else output_dims


def _infer_output_dims(
    input_dims: list[tuple[str, Any]],
    output_names: list[str],
) -> list[tuple[str, int]]:
    """Infer power-of-two output sizes from the highest bit used on each axis."""

    sizes = [1] * len(output_names)
    for _dim_name, bases in input_dims:
        for basis in bases:
            for axis, value in enumerate(basis[: len(output_names)]):
                sizes[axis] = max(sizes[axis], 1 if int(value) <= 0 else 1 << int(value).bit_length())
    return [
        (dim_name, dim_size)
        for dim_name, dim_size in zip(output_names, sizes, strict=True)
    ]


def _default_output_names(input_dims: list[tuple[str, Any]]) -> list[str]:
    """Return the browser-default output axis names for one basis rank."""

    output_rank = max(
        (len(basis) for _dim_name, bases in input_dims for basis in bases),
        default=1,
    )
    if output_rank > len(OUTPUT_AXIS_NAMES):
        raise ValueError(
            f"Output rank {output_rank} exceeds supported axes {len(OUTPUT_AXIS_NAMES)}."
        )
    return list(reversed(OUTPUT_AXIS_NAMES[:output_rank]))


def _map_linear_layout_coord(
    input_coord: tuple[int, ...],
    input_dims: list[tuple[str, Any]],
    output_rank: int,
) -> tuple[int, ...]:
    """Map one input coordinate into output space using xor basis composition."""

    output_coord = [0] * output_rank
    for axis, (_dim_name, bases) in enumerate(input_dims):
        value = input_coord[axis] if axis < len(input_coord) else 0
        for bit, basis in enumerate(bases):
            if ((value >> bit) & 1) == 0:
                continue
            if len(basis) != output_rank:
                raise ValueError(
                    f"Basis {axis}[{bit}] has rank {len(basis)}, expected {output_rank}."
                )
            for out_axis, component in enumerate(basis):
                output_coord[out_axis] ^= component
    return tuple(output_coord)


def _hardware_input_dims(
    input_dims: list[tuple[str, Any]],
) -> tuple[list[tuple[str, int]], tuple[int, ...]]:
    """Return hardware dims reordered for contiguous 2D viewing."""

    dims = [(dim_name, 1 << len(bases)) for dim_name, bases in input_dims]
    if len(dims) != 3:
        return dims, tuple(range(len(dims)))
    axis_order = (1, 0, 2)
    return [dims[axis] for axis in axis_order], axis_order


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


def _rgb_color(hue: float, saturation: float, lightness: float) -> tuple[float, float, float]:
    """Convert the configured layout color channels into an RGB tuple."""

    red, green, blue = colorsys.hsv_to_rgb(
        hue % 1.0,
        min(max(saturation, 0.0), 1.0),
        min(max(lightness, 0.0), 1.0),
    )
    return (
        red,
        green,
        blue,
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
    hardware_dims, hardware_axis_order = _hardware_input_dims(input_dims)
    input_shape = tuple(1 << len(bases) for _dim_name, bases in input_dims)
    hardware_shape = tuple(size for _dim_name, size in hardware_dims)
    hardware_names = [dim_name for dim_name, _size in hardware_dims]
    raw_output_dims = _infer_output_dims(
        input_dims,
        _default_output_names(input_dims),
    )
    output_dims = _logical_output_dims(raw_output_dims)
    output_names = [dim_name for dim_name, _size in output_dims]
    output_axis_by_name = {dim_name: axis for axis, (dim_name, _size) in enumerate(raw_output_dims)}
    output_shape = tuple(size for _dim_name, size in output_dims)
    channel_axes = _normalize_color_axes(input_names, input_shape, color_axes)
    linear_layout_state = _linear_layout_state(input_dims, color_axes, color_ranges)
    linear_layout_spec = {
        "name": name or "Layout",
        "input_dims": [
            {
                "name": dim_name,
                "bases": [list(basis) for basis in dim_bases],
            }
            for dim_name, dim_bases in input_dims
        ],
        "output_dims": [
            {
                "name": dim_name,
                "size": int(dim_size),
            }
            for dim_name, dim_size in raw_output_dims
        ],
    }
    if color_axes:
        linear_layout_spec["color_axes"] = dict(color_axes)
    if color_ranges:
        linear_layout_spec["color_ranges"] = {
            channel: [float(value[0]), float(value[1])]
            for channel, value in color_ranges.items()
        }

    hardware_tensor = np.zeros(hardware_shape, dtype=np.int32)
    hardware_rgb = np.zeros((*hardware_shape, 3), dtype=np.float32)
    logical_tensor = np.full(output_shape, -1, dtype=np.int32)
    logical_rgb = np.zeros((*output_shape, 3), dtype=np.float32)

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
        value = 0 if lightness_axis is None else input_coord[lightness_axis]
        hardware_coord = tuple(input_coord[axis] for axis in hardware_axis_order)
        hardware_tensor[hardware_coord] = value
        hardware_rgb[hardware_coord] = _rgb_color(hue, saturation, lightness)

        output_coord = _map_linear_layout_coord(
            input_coord,
            input_dims,
            len(raw_output_dims),
        )
        logical_coord = tuple(
            output_coord[output_axis_by_name[dim_name]]
            for dim_name in output_names
        )
        logical_tensor[logical_coord] = value
        logical_rgb[logical_coord] = hardware_rgb[hardware_coord]

    session_data = create_session_data(
        {
            "Hardware Layout": hardware_tensor,
            "Logical Layout": logical_tensor,
        },
        name=name or "Layout",
        labels={
            "Hardware Layout": _viewer_axis_labels(hardware_names),
        },
        color_instructions={
            "tensor-1": [
                {"mode": "rgb", "kind": "dense", "values": _dense_rgb_values(hardware_rgb)}
            ],
            "tensor-2": [
                {"mode": "rgb", "kind": "dense", "values": _dense_rgb_values(logical_rgb)}
            ],
        },
    )
    manifest = json.loads(session_data.manifest_bytes)
    logical_marker_coords = np.argwhere(logical_tensor < 0).tolist()
    if logical_marker_coords:
        manifest["tabs"][0]["tensors"][1]["markerCoords"] = logical_marker_coords
    manifest["tabs"][0]["viewer"]["dimensionMappingScheme"] = "contiguous"
    manifest["tabs"][0]["viewer"]["linearLayoutState"] = linear_layout_state
    manifest["tabs"][0]["viewer"]["linearLayoutSpec"] = linear_layout_spec
    return SessionData(
        manifest_bytes=json.dumps(manifest).encode("utf-8"),
        tensor_bytes=session_data.tensor_bytes,
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
        tab["viewer"]["dimensionMappingScheme"] = "contiguous"
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
