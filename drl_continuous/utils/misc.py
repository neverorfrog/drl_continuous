from typing import Dict, List, Tuple, Union

import numpy as np


def parse_map_grid(
    map: Union[List[str], np.ndarray]
) -> Tuple[int, List[Tuple[int, int]], Dict[int, Tuple[int, int]]]:
    """
    Parses a string representation of a map to identify holes and goals.

    Args:
        map (grid): A list representation of the map where 'H' represents a hole
                          and digits represent goals.
    Returns:
        tuple: A tuple containing:
            - holes (list of tuple): A list of coordinates (x, y) for each hole.
            - goal (tuple): The coordinates (x, y) for the goal.
    """
    holes = []
    goals = {}

    height = 0
    old_width = 0

    for y, row in enumerate(map):
        x = -1
        width = 0
        height += 1
        for char in row.strip():
            if char != " ":
                x += 1
                width += 1
            else:
                continue
            if char == "O" or char == "⛔":
                holes.append((x, y))
            elif char.isdigit():
                goal_number = int(char)
                goals[goal_number] = (x, y)

        if old_width > 0:
            assert old_width == width, "The map is not rectangular."
        old_width = width

    assert len(goals.values()) > 0, "The map does not contain a goal."
    assert width == height, "The map is not square."
    size = width
    return size, holes, goals
