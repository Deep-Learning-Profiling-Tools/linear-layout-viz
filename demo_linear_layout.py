from triton.tools import LinearLayout

from linear_layout_viz import visualize_layouts

COLOR_AXES = {"warp": "L", "thread": "S", "reg": "H"}
COLOR_RANGES = {
    "H": (0.0, 0.8),
    "S": (0.25, 1),
    "L": (1.0, 0.25),
}

DEMOS = {
    "blocked": (
        "Blocked Layout",
        LinearLayout.from_bases(
            [
                ("warp", [[0, 8], [0, 16]]),
                ("thread", [[4, 0], [8, 0], [0, 1], [0, 2], [0, 4]]),
                ("register", [[1, 0], [2, 0]]),
            ],
            ["x", "y"],
        ),
    ),
    "mma": (
        "MMA A Layout (m16n8k16)",
        LinearLayout.from_bases(
            [
                ("warp", []),
                ("thread", [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]]),
                ("register", [[0, 1], [8, 0], [0, 8]]),
            ],
            ["row", "col"],
        ),
    ),
    "mma_b": (
        "MMA B Layout (m16n8k16)",
        LinearLayout.from_bases(
            [
                ("warp", []),
                ("thread", [[2, 0], [4, 0], [0, 1], [0, 2], [0, 4]]),
                ("register", [[1, 0], [8, 0]]),
            ],
            ["row", "col"],
        ),
    ),
    "mma_c": (
        "MMA C Layout (m16n8k16)",
        LinearLayout.from_bases(
            [
                ("warp", []),
                ("thread", [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]]),
                ("register", [[0, 1], [8, 0]]),
            ],
            ["row", "col"],
        ),
    ),
    "memory": (
        "Shared Memory 128B Swizzle",
        LinearLayout.from_bases(
            [
                ("warp", []),
                ("thread", [[1, 1], [2, 2], [4, 4]]),
                ("register", [[0, 1], [0, 2], [0, 4]]),
            ],
            ["row", "col_chunk"],
        ),
    ),
}


if __name__ == "__main__":
    visualize_layouts(
        list(DEMOS.values()),
        color_axes=COLOR_AXES,
        color_ranges=COLOR_RANGES,
    )
