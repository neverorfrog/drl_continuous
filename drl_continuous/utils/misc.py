import os
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def project_root() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    max_iterations = 100  # Set a limit for the number of iterations
    for _ in range(max_iterations):
        if "requirements.txt" in os.listdir(current_dir):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError(
        "requirements.txt not found in any parent directories within the iteration limit"
    )


def generate_heatmap(
    q_table: np.ndarray, grid_size: Tuple[int, int], num_rm_states: int = 1
) -> None:

    # Extraction and Reshaping of Maximum Q Values
    q_table = q_table["q_table_a1"]  # TODO: Hardcoded for agent a1
    mean_q_values: np.ndarray = q_table.max(axis=1)
    reshaped_mean_q_values = mean_q_values.reshape(
        (*grid_size, num_rm_states + 1)
    )

    # Calculation of Optimal Actions
    optimal_actions: np.ndarray = np.argmax(q_table, axis=1)
    reshaped_optimal_actions = optimal_actions.reshape(
        (*grid_size, num_rm_states + 1)
    )

    # Map action codes to corresponding symbols
    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    # Visualization of Heatmaps
    fig, axes = plt.subplots(1, num_rm_states, figsize=(20, 6))

    # Ensure that axes is always an array
    if num_rm_states == 1:
        axes = [axes]  # Make axes an array if there is only one subplot

    for i in range(num_rm_states):

        data = reshaped_mean_q_values[:, :, i]

        minval = data.min()
        maxval = data.max()
        row, column = np.unravel_index(data.argmax(), data.shape)
        while maxval >= 99:
            datacopy: np.ndarray = data.copy()
            datacopy[row, column] = 0
            maxval = datacopy.max()

        sns.heatmap(
            data,
            annot=data,  # Annotate with numbers
            fmt=".2f",  # Format numbers to 2 decimal places
            cbar=True,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 15},
            cmap=sns.color_palette("coolwarm", as_cmap=True),
            vmin=minval - 0.05 * minval,
            vmax=maxval + 0.05 * maxval,
            ax=axes[i],
        )
        axes[i].set_title(f"RM State {i + 1}")
        axes[i].set_xlabel("Column")
        axes[i].set_ylabel("Row")

    plt.tight_layout()
    plt.show()


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
