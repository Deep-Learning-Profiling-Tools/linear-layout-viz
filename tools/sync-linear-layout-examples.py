"""Sync baked LL-viz linear-layout examples from demo_linear_layout.py."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

SYNC_START = "// sync-linear-layout-examples:start"
SYNC_END = "// sync-linear-layout-examples:end"
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
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CANDIDATES = (
    REPO_ROOT / "demo_linear_layout.py",
    REPO_ROOT.parent / "demo_linear_layout.py",
)
TARGET_PATH = (
    REPO_ROOT
    / "src"
    / "extensions"
    / "linear-layout"
    / "linear-layout.ts"
)


def compose_identifier(name: str) -> str:
    """Return one compose-layout-safe identifier."""

    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name).strip("_")
    if not cleaned:
        return "Layout_1"
    return cleaned if (cleaned[0].isalpha() or cleaned[0] == "_") else f"Layout_{cleaned}"


def viewer_axis_labels(names: list[str]) -> list[str]:
    """Convert dim names into viewer-safe axis labels."""

    counts: dict[str, int] = {}
    labels: list[str] = []
    for name in names:
        base = next((char.upper() for char in name if char.isalpha()), "A")
        index = counts.get(base, 0)
        counts[base] = index + 1
        labels.append(base if index == 0 else f"{base}{index}")
    return labels


def infer_output_dims(input_dims: list[tuple[str, list[list[int]]]]) -> list[tuple[str, int]]:
    """Infer power-of-two output sizes from the highest bit used on each axis."""

    output_names = list(OUTPUT_AXIS_NAMES[: max((len(basis) for _name, bases in input_dims for basis in bases), default=1)])
    sizes = [1] * len(output_names)
    for _dim_name, bases in input_dims:
        for basis in bases:
            for axis, value in enumerate(basis[: len(output_names)]):
                sizes[axis] = max(sizes[axis], 1 if int(value) <= 0 else 1 << int(value).bit_length())
    output_dims = list(zip(output_names, sizes, strict=True))
    return list(reversed(output_dims)) if [name for name, _size in output_dims] == ["x", "y"] else output_dims


def parse_demo_layouts(source_path: Path) -> list[tuple[str, list[tuple[str, list[list[int]]]], str, str | None]]:
    """Extract demo titles, bases, input names, and optional spec text overrides."""

    module = ast.parse(source_path.read_text())
    demos = next(
        node for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DEMOS" for target in node.targets)
    )
    if not isinstance(demos.value, ast.Dict):
        raise ValueError("DEMOS must be a dict literal.")
    layouts: list[tuple[str, list[tuple[str, list[list[int]]]], str, str | None]] = []
    for entry in demos.value.values:
        if not isinstance(entry, ast.Tuple) or len(entry.elts) not in {3, 4}:
            raise ValueError("Each DEMOS entry must be a 3-tuple or 4-tuple.")
        title = ast.literal_eval(entry.elts[0])
        layout_call = entry.elts[1]
        input_name = ast.literal_eval(entry.elts[2])
        specs_text = ast.literal_eval(entry.elts[3]) if len(entry.elts) == 4 else None
        if not isinstance(title, str) or not isinstance(input_name, str):
            raise ValueError("Demo title and input name must be strings.")
        if specs_text is not None and not isinstance(specs_text, str):
            raise ValueError("Optional specs text override must be a string.")
        if (
            not isinstance(layout_call, ast.Call)
            or not isinstance(layout_call.func, ast.Attribute)
            or layout_call.func.attr != "from_bases"
            or len(layout_call.args) < 1
        ):
            raise ValueError(f"{title} must call LinearLayout.from_bases(...).")
        input_dims = ast.literal_eval(layout_call.args[0])
        layouts.append((title, input_dims, input_name, specs_text))
    return layouts


def spec_lines(title: str, input_dims: list[tuple[str, list[list[int]]]], specs_text: str | None = None) -> list[str]:
    """Return the compose-layout specs text lines for one demo entry."""

    if specs_text is not None:
        return specs_text.splitlines()
    input_labels = viewer_axis_labels([dim_name for dim_name, _bases in input_dims])
    output_labels = viewer_axis_labels([dim_name for dim_name, _size in infer_output_dims(input_dims)])
    return [
        f"{compose_identifier(title)}: [{','.join(input_labels)}] -> [{','.join(output_labels)}]",
        *[
            f"{axis_label}: {json.dumps(dim_bases, separators=(',', ':'))}"
            for axis_label, (_dim_name, dim_bases) in zip(input_labels, input_dims, strict=True)
        ],
    ]


def const_name(title: str) -> str:
    """Return the generated ts constant name for one demo entry."""

    return f"{compose_identifier(title).upper()}_TEXT"


def format_block(layouts: list[tuple[str, list[tuple[str, list[list[int]]]], str, str | None]]) -> str:
    """Return the generated ts block for the baked examples."""

    const_blocks = [
        "\n".join([
            f"const {const_name(title)} = [",
            *[f"    {line!r}," for line in spec_lines(title, input_dims, specs_text)],
            "].join('\\n');",
        ])
        for title, input_dims, _input_name, specs_text in layouts
    ]
    example_lines = [
        "const BAKED_EXAMPLE_DEFINITIONS = [",
        *[
            "    { "
            f"title: {title!r}, specsText: {const_name(title)}, operationText: {compose_identifier(title)!r}, inputName: {input_name!r}"
            " },"
            for title, _input_dims, input_name, _specs_text in layouts
        ],
        "] as const;",
    ]
    return "\n".join([SYNC_START, *const_blocks, "", *example_lines, SYNC_END])


def main() -> None:
    """Rewrite the baked viewer-demo examples from demo_linear_layout.py."""

    source_path = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
    if source_path is None:
        # standalone tensor-viz checkouts do not carry the ll-viz python demo source
        print("No demo_linear_layout.py found; leaving baked examples unchanged.")
        return
    layouts = parse_demo_layouts(source_path)
    target_text = TARGET_PATH.read_text()
    pattern = re.compile(rf"{re.escape(SYNC_START)}.*?{re.escape(SYNC_END)}", re.DOTALL)
    replacement = format_block(layouts)
    if not pattern.search(target_text):
        raise ValueError("Missing sync markers in linear-layout.ts.")
    TARGET_PATH.write_text(pattern.sub(lambda _match: replacement, target_text, count=1))


if __name__ == "__main__":
    main()
