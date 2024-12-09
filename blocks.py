# Looking at the the tasks from the Abstraction and Reasoning Corpus, while each task is unique, it is clear that there are certain concepts,
# such as operations like rotation or mirroring, that occur repeatedly throughout the corpus. What seems feasible is to think of a set of
# building blocks that encapsulate those concepts that can then be used to build solution programs, that is, task-specific programs that correctly
# transform each of the input grids of a given task into its corresponding output grid. Such a set of building blocks is a form of domain-specific
# language (DSL). A DSL defines a set of programs that it can express, and the process of finding or creating such a program solving a given task
# is a form of program synthesis. Building a good DSL that well captures the explicitly stated core knowledge priors of ARC in an abstract and
# combinable manner, combined with an adequate program synthesis approach is suggested as a possible way to tackling ARC by its creator François
# in On the Measure of Intelligence.

# This notebook presents a very simple such DSL tailored to ARC and how it can be used to perform program synthesis, intended to serve as a starting
# point for anyone new to the ARC benchmark. For the sake of demonstration, the DSL as well as the program synthesis are overly simplistic here,
# to the point of being naive, as will be demonstrated. First, the DSL is defined as some basic functions, also called primitives, that transform
# grids. Second, program synthesis as search over compositions of those primitives is performed. The search is naive in that it is a brute force
# search over all possible primitive compositions up to a certain depth that is done for each task and simply uses the first program it finds to
# solve the training examples to make predictions for the test examples.

# It is obvious why some tasks can't be solved by the above DSL, but one simple proof would be the following: No primitive ever increases the pixel
# count of a grid, hence neither can any composition of the primitives ever do so - and since for certain tasks, the output grids do have more pixels
# than the input grids, the DSL is incomplete. To increase the expressibity of the program space (disregarding the maximum program size), one will
# want to expand the set of the primitives and also extend the structure of the considered programs beyond mere composition. Maybe it is a good idea
# to have primitives which take more than one input argument, or primitives that operate on types other than only grids, such as objects or integers.
# Note that viewing the transformations from inputs to outputs as a linear function composition is very misleading, as many tasks can't be neatly
# squeezed into this form: Some tasks seem much better addressed on a pixel- or object-level than on a grid-level. A good DSL is probably concise and
# allows expressing solutions to many tasks as short programs. Such a DSL may best be built by bootstrapping, that is, building a minimal version of
# it and then iterating back and forth between using it to solve ARC tasks and expanding it to account for unsolvable ARC tasks, all while having
# abstractness and flexibility of the primitives and how they can interplay in mind.
#
# Objects
#     shirt objects
#     move two objects together
#     move objects to edge
#     extend
#     repeat an object
#     delete an object
#     count unique objects and select the object that appears the most times
#     create pattern based on imxage colors
#     overlay object
#     replace objects
# Coloring
#     select colors for objects
#     select dominant/smallest color in image
#     denoise
#     fill in empty spaces
# Lines
#     color edges
#     extrapolate a straight/diagonal line
#     draw a line between two dots / or inersections between such lines
#     draw a spiral
# Grids
#     select grid squares with most pixels
#     scale to amke the inout and output the same size?
# Patterns
#     complete a symmetrical/repeating pattern
# Subtasks
#     object detection
#     object cohesion
#     object seperation
#     object persistance
#     counting or sorting objects

import numpy as np

# region blocks extracted from arc-prize-2024


def object_detection(grid):
    """
    Input:
    A two-dimensional grid (list of lists) where each cell contains either zero or a positive integer.
    Zero represents an empty cell, and positive integers represent different objects (e.g., different colors or identifiers).
    The grid may contain multiple such objects composed of connected cells with the same non-zero value.

    Functionality:
    The `object_detection` function scans the input grid to identify and extract all distinct connected regions (objects).
    Cells are considered connected if they are adjacent horizontally, vertically, or diagonally (all eight directions).
    For each detected object, the function creates a subarray that captures the object's shape within its minimal bounding box,
    preserving the original values. The function returns a list of these subarrays, each representing an individual object extracted from the grid.

    Output:
    A list of two-dimensional subarrays, where each subarray corresponds to a detected object from the input grid.
    Each subarray contains the object's cells within its minimal bounding box, with zeros filling any empty spaces within that box.

    Example Input:
    grid = [
        [1, 1, 0, 0],
        [0, 1, 0, 2],
        [0, 0, 2, 2],
        [3, 0, 0, 0]
    ]

    Example Output:
    Object 1:
    [1, 1]
    [0, 1]

    Object 2:
    [0, 2]
    [2, 2]

    Object 3:
    [3]

    Explanation:
    - Object 1 consists of connected cells with the value '1' at positions (0,0), (0,1), and (1,1).
    The subarray captures these cells within their minimal bounding box.
    - Object 2 consists of connected cells with the value '2' at positions (1,3), (2,2), and (2,3).
    The subarray represents this cluster of '2's.
    - Object 3 is the single cell with the value '3' at position (3,0).
    Its subarray contains just this cell.
    """

    from collections import deque

    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    visited = [[False for _ in range(width)] for _ in range(height)]
    objects = []

    for i in range(height):
        for j in range(width):
            if grid[i][j] != 0 and not visited[i][j]:
                # Initialize variables for the new object
                color = grid[i][j]
                min_row, max_row = i, i
                min_col, max_col = j, j
                object_pixels = []
                queue = deque()
                queue.append((i, j))
                visited[i][j] = True

                # Perform BFS to find all connected pixels of the object
                while queue:
                    r, c = queue.popleft()
                    object_pixels.append((r, c))

                    # Update bounding box
                    min_row = min(min_row, r)
                    max_row = max(max_row, r)
                    min_col = min(min_col, c)
                    max_col = max(max_col, c)

                    # Check neighbors (including diagonals)
                    for dr, dc in [(-1, -1), (-1, 0), (-1, 1),
                                   (0, -1),          (0, 1),
                                   (1, -1),  (1, 0),  (1, 1)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < height and 0 <= nc < width and
                                grid[nr][nc] == color and not visited[nr][nc]):
                            visited[nr][nc] = True
                            queue.append((nr, nc))

                # Extract the object's subarray from the grid
                obj_height = max_row - min_row + 1
                obj_width = max_col - min_col + 1
                obj_array = [[0 for _ in range(obj_width)] for _ in range(obj_height)]

                for r, c in object_pixels:
                    obj_array[r - min_row][c - min_col] = grid[r][c]

                objects.append(obj_array)

    return objects


def find_loops(grid):
    """
    Input:
    A two-dimensional grid (list of lists) where each cell contains either zero or a positive integer.
    Zero represents an empty cell, and positive integers represent different objects (e.g., different colors or identifiers).

    Functionality:
    The `find_loops` function scans the input grid to identify and extract all distinct loops formed by connected cells with the same non-zero value.
    Cells are considered connected if they are adjacent horizontally, vertically, or diagonally (all eight directions).
    For each detected loop, the function creates a list of coordinates representing the loop's boundary and its interior elements.
    The function returns a list of dictionaries, each containing the boundary and interior of a detected loop.

    Output:
    A list of dictionaries, where each dictionary contains two keys:
    - 'boundary': A list of coordinates representing the boundary of the loop.
    - 'interior': A list of coordinates representing the interior elements of the loop.

    Example Input:
    grid = [
        [1, 1, 1, 0],
        [1, 0, 1, 2],
        [1, 1, 1, 2],
        [3, 0, 0, 2]
    ]

    Example Output:
    [
        {'boundary': [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)], 'interior': [(1, 1)]},
        {'boundary': [(1, 3), (2, 3), (3, 3)], 'interior': []}
    ]

    Explanation:
    - The first loop consists of connected cells with the value '1' forming a loop with an interior cell at position (1,1).
    - The second loop consists of connected cells with the value '2' at positions (1,3), (2,3), and (3,3).
    """

    rows = len(grid)
    cols = len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                  (-1, -1), (1, 1), (-1, 1), (1, -1)]  # Diagonal directions

    visited = set()

    def dfs(x, y, parent, path):
        if (x, y) in visited:
            if path[0] == (x, y) and len(path) >= 4:
                return path  # path forms a valid loop
            return None

        visited.add((x, y))
        path.append((x, y))

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 0:
                if (nx, ny) != parent:
                    result = dfs(nx, ny, (x, y), path)
                    if result:
                        return result

        path.pop()
        visited.remove((x, y))  # Backtrack
        return None

    loops = []

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and (i, j) not in visited:
                path = dfs(i, j, None, [])
                if path:
                    # Check if the detected loop is a new loop
                    path_set = set(path)
                    if not any(path_set == set(existing_loop['boundary']) for existing_loop in loops):
                        loops.append({'boundary': path})

    # Now, for each loop, find the interior elements
    for loop in loops:
        boundary = loop['boundary']
        boundary_set = set(boundary)
        # Get bounding box
        min_x = min(x for x, y in boundary)
        max_x = max(x for x, y in boundary)
        min_y = min(y for x, y in boundary)
        max_y = max(y for x, y in boundary)
        interior = []

        # Build polygon from boundary coordinates
        polygon = boundary.copy()

        # Function to check if a point is inside a polygon
        def point_in_polygon(point, polygon):
            x, y = point
            num = len(polygon)
            j = num - 1
            c = False
            for i in range(num):
                xi, yi = polygon[i]
                xj, yj = polygon[j]
                if ((yi > y) != (yj > y)) and \
                        (x < (xj - xi) * (y - yi) / ((yj - yi) if yj != yi else 1e-10) + xi):
                    c = not c
                j = i
            return c

        # Check each point inside bounding box
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) not in boundary_set:
                    if point_in_polygon((x, y), polygon):
                        interior.append((x, y))

        loop['interior'] = interior

    return loops


def array_and(array_1, array_2, x_step_size, y_step_size):
    """
    Input:
    Two two-dimensional arrays (list of lists or NumPy arrays) and two positive integer constants representing the step sizes.

    Functionality:
    The `array_and` function takes two 2D arrays and performs an element-wise logical 'and' operation between the two arrays.
    If the values in `array_1` and `array_2` are equal, the corresponding element in the result is set to the value from `array_1`.
    Otherwise, it is set to 0. The operation is carried out by sliding `array_1` over `array_2` in chunks of the same size,
    starting from the upper-left corner, moving horizontally by `x_step_size` and vertically by `y_step_size`.
    The result has the same dimensions as `array_2`.

    Output:
    A new two-dimensional array where each element represents the result of the logical 'and' operation between `array_1`
    and corresponding portions of `array_2`.

    Example Input:
    array_1 = [
        [1, 2],
        [3, 4]
    ]
    array_2 = [
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [5, 6, 3, 4]
    ]
    x_step_size = 2
    y_step_size = 1

    Example Output:
    [
        [1 2 0 0]
        [3 4 0 0]
        [0 0 0 0]
    ]

    Explanation:
    - The function slides `array_1` over `array_2` and performs an element-wise comparison.
    - If elements are equal, they are retained; otherwise, they are set to 0.
    """

    array_1 = np.array(array_1)
    array_2 = np.array(array_2)

    a1_height, a1_width = array_1.shape
    a2_height, a2_width = array_2.shape

    result = np.zeros_like(array_2)

    for y in range(0, a2_height - a1_height + 1, y_step_size):
        for x in range(0, a2_width - a1_width + 1, x_step_size):
            a2_chunk = array_2[y:y + a1_height, x:x + a1_width]
            and_result = np.where(array_1 == a2_chunk, array_1, 0)
            result[y:y + a1_height, x:x + a1_width] = and_result

    return result.tolist()


def change_elements_color(grid, elements, new_value):
    """
    Input:
    A two-dimensional grid (list of lists) where each cell contains either zero or a positive integer.
    Zero represents an empty cell, and positive integers represent different objects (e.g., different colors or identifiers).
    A list of elements (coordinates) to be changed and a new value to set for those elements.

    Functionality:
    The `change_elements_color` function scans the input grid and changes the color (value) of the specified elements to the new value.
    The elements are provided as a list of coordinates, and the new value is a positive integer.

    Output:
    The modified grid with the specified elements changed to the new value.

    Example Input:
    grid = [
        [1, 1, 0, 0],
        [0, 1, 0, 2],
        [0, 0, 2, 2],
        [3, 0, 0, 0]
    ]
    elements = [(0, 0), (1, 1), (2, 2)]
    new_value = 9

    Example Output:
    [
        [9, 1, 0, 0],
        [0, 9, 0, 2],
        [0, 0, 9, 2],
        [3, 0, 0, 0]
    ]

    Explanation:
    - The elements at coordinates (0, 0), (1, 1), and (2, 2) are changed to the new value '9'.
    """

    for x, y in elements:
        grid[x][y] = new_value

    return grid


# endregion


# region general blocks


def rotate_grid(grid, degrees):
    """
    Input:
    A two-dimensional grid (list of lists) where each cell contains either zero or a positive integer.
    Zero represents an empty cell, and positive integers represent different objects (e.g., different colors or identifiers).
    An integer representing the degrees to rotate the grid (90, 180, or 270).

    Functionality:
    The `rotate_grid` function rotates the input grid by the specified degrees in a clockwise direction.
    The function supports rotations of 90, 180, and 270 degrees.

    Output:
    A new two-dimensional grid that is the result of rotating the input grid by the specified degrees.

    Example Input:
    grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    degrees = 90

    Example Output:
    [
        [7, 4, 1],
        [8, 5, 2],
        [9, 6, 3]
    ]

    Explanation:
    - The input grid is rotated 90 degrees clockwise to produce the output grid.
    """

    if degrees not in [90, 180, 270]:
        raise ValueError("Degrees must be 90, 180, or 270")

    if degrees == 90:
        return [list(row) for row in zip(*grid[::-1])]
    elif degrees == 180:
        return [row[::-1] for row in grid[::-1]]
    elif degrees == 270:
        return [list(row) for row in zip(*grid)][::-1]


def mirror_grid(grid, direction):
    """
    Input:
    A two-dimensional grid (list of lists) where each cell contains either zero or a positive integer.
    Zero represents an empty cell, and positive integers represent different objects (e.g., different colors or identifiers).
    A string representing the direction to mirror the grid ('hor' for horizontal or 'ver' for vertical).

    Functionality:
    The `mirror_grid` function mirrors the input grid in the specified direction.
    If the direction is 'hor', the grid is mirrored horizontally (left to right).
    If the direction is 'ver', the grid is mirrored vertically (top to bottom).

    Output:
    A new two-dimensional grid that is the result of mirroring the input grid in the specified direction.

    Example Input:
    grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    direction = 'hor'

    Example Output:
    [
        [3, 2, 1],
        [6, 5, 4],
        [9, 8, 7]
    ]

    Explanation:
    - The input grid is mirrored horizontally to produce the output grid.
    """

    if direction == 'hor':
        return [row[::-1] for row in grid]
    elif direction == 'ver':
        return grid[::-1]
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")


def shift_grid(grid, direction, steps):
    """
    Input:
    A two-dimensional grid (list of lists) where each cell contains either zero or a positive integer.
    Zero represents an empty cell, and positive integers represent different objects (e.g., different colors or identifiers).
    A string representing the direction to shift the grid ('up', 'down', 'left', or 'right') and an integer representing the number of steps to shift.

    Functionality:
    The `shift_grid` function shifts the input grid in the specified direction by the given number of steps.
    If the direction is 'up', the grid is shifted upwards.
    If the direction is 'down', the grid is shifted downwards.
    If the direction is 'left', the grid is shifted to the left.
    If the direction is 'right', the grid is shifted to the right.
    Cells that are shifted out of the grid's bounds are replaced with zeros.

    Output:
    A new two-dimensional grid that is the result of shifting the input grid in the specified direction by the given number of steps.

    Example Input:
    grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    direction = 'up'
    steps = 1

    Example Output:
    [
        [4, 5, 6],
        [7, 8, 9],
        [0, 0, 0]
    ]

    Explanation:
    - The input grid is shifted upwards by 1 step to produce the output grid.
    - The top row is moved out of the grid and replaced with zeros at the bottom.
    """

    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    new_grid = [[0] * width for _ in range(height)]

    if direction == 'up':
        for i in range(height):
            for j in range(width):
                if i + steps < height:
                    new_grid[i][j] = grid[i + steps][j]
    elif direction == 'down':
        for i in range(height):
            for j in range(width):
                if i - steps >= 0:
                    new_grid[i][j] = grid[i - steps][j]
    elif direction == 'left':
        for i in range(height):
            for j in range(width):
                if j + steps < width:
                    new_grid[i][j] = grid[i][j + steps]
    elif direction == 'right':
        for i in range(height):
            for j in range(width):
                if j - steps >= 0:
                    new_grid[i][j] = grid[i][j - steps]
    else:
        raise ValueError("Direction must be 'up', 'down', 'left', or 'right'")

    return new_grid


def crop_grid(grid, top, bottom, left, right):
    """
    Input:
    A two-dimensional grid (list of lists) where each cell contains either zero or a positive integer.
    Zero represents an empty cell, and positive integers represent different objects (e.g., different colors or identifiers).
    Four integers representing the number of rows/columns to crop from the top, bottom, left, and right of the grid.

    Functionality:
    The `crop_grid` function crops the input grid by removing the specified number of rows and columns from the edges.
    The function removes `top` rows from the top, `bottom` rows from the bottom, `left` columns from the left, and `right` columns from the right.

    Output:
    A new two-dimensional grid that is the result of cropping the input grid by the specified amounts.

    Example Input:
    grid = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]
    top = 1
    bottom = 1
    left = 1
    right = 1

    Example Output:
    [
        [6, 7],
        [10, 11]
    ]

    Explanation:
    - The input grid is cropped by removing 1 row from the top, 1 row from the bottom, 1 column from the left, and 1 column from the right.
    - The resulting grid contains the remaining elements.
    """

    return [row[left:len(row)-right] for row in grid[top:len(grid)-bottom]]


def draw_border(grid, border_value):
    """
    Input:
    A two-dimensional grid (list of lists) where each cell contains either zero or a positive integer.
    Zero represents an empty cell, and positive integers represent different objects (e.g., different colors or identifiers).
    An integer representing the value to be used for the border.

    Functionality:
    The `draw_border` function adds a border around the input grid using the specified border value.
    The border is one cell wide and surrounds the entire grid.

    Output:
    A new two-dimensional grid with the border added.

    Example Input:
    grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    border_value = 9

    Example Output:
    [
        [9, 9, 9, 9, 9],
        [9, 1, 2, 3, 9],
        [9, 4, 5, 6, 9],
        [9, 7, 8, 9, 9],
        [9, 9, 9, 9, 9]
    ]

    Explanation:
    - The input grid is surrounded by a border of value '9' to produce the output grid.
    """

    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    # Create a new grid with the border
    new_grid = [[border_value] * (width + 2) for _ in range(height + 2)]

    # Copy the original grid into the new grid
    for i in range(height):
        for j in range(width):
            new_grid[i + 1][j + 1] = grid[i][j]

    return new_grid


def enlarge(array, constant):
    """
    Input:
    A two-dimensional array (list of lists or NumPy array) and a positive integer constant.

    Functionality:
    The `enlarge` function takes the input 2D array and enlarges it by repeating its elements based on the given constant.
    The resulting enlarged array has dimensions that are `constant` times the dimensions of the original array.
    Each element in the original array is expanded to cover a square block of size `constant x constant` in the enlarged array.

    Output:
    A new two-dimensional array where each element of the original array is repeated to form a larger grid.

    Example Input:
    original_array = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    constant = 3

    Example Output: [
        [1 1 1 2 2 2 3 3 3]
        [1 1 1 2 2 2 3 3 3]
        [1 1 1 2 2 2 3 3 3]
        [4 4 4 5 5 5 6 6 6]
        [4 4 4 5 5 5 6 6 6]
        [4 4 4 5 5 5 6 6 6]
    ]

    Explanation:
    - Each element of the original array is repeated in a block of `constant x constant` size.
    For example, the value '1' at position (0,0) in the original array is expanded to form a 3x3 block in the enlarged array.
    """

    # Convert the input to a NumPy array if it isn't already
    array = np.array(array)

    # Get the shape of the original array
    original_rows, original_cols = array.shape

    # Create a new enlarged array with the desired size
    enlarged_rows = original_rows * constant
    enlarged_cols = original_cols * constant

    # Initialize the enlarged array with zeros
    enlarged_array = np.zeros((enlarged_rows, enlarged_cols), dtype=array.dtype)

    # Fill the enlarged array by repeating the original values
    for i in range(enlarged_rows):
        for j in range(enlarged_cols):
            enlarged_array[i, j] = array[i // constant, j // constant]

    return enlarged_array.tolist()


def invert_colors(grid):
    """
    Input:
    A two-dimensional grid (list of lists) where each cell contains either zero or a positive integer.
    Zero represents an empty cell, and positive integers represent different colors.

    Functionality:
    The `invert_colors` function inverts each color value in the grid relative to the maximum color value.
    For each non-zero color value c, it is replaced with (max_color - c + 1), where max_color is the
    highest color value in the original grid. Zero values (empty cells) remain unchanged.

    Output:
    A new two-dimensional grid with inverted color values.

    Example Input:
    grid = [
        [1, 2, 0],
        [0, 3, 4],
        [5, 0, 2]
    ]

    Example Output:
    [
        [5, 4, 0],
        [0, 3, 2],
        [1, 0, 4]
    ]

    Explanation:
    - The maximum color value in the input grid is 5
    - Each non-zero value c is replaced with (5 - c + 1)
    - Zero values remain unchanged
    - For example, 1 becomes 5, 2 becomes 4, etc.
    """

    # Find the maximum color value in the grid
    max_color = max(max(row) for row in grid)

    # Create a new grid with the same dimensions
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    new_grid = [[0 for _ in range(width)] for _ in range(height)]

    # Invert each color value
    for i in range(height):
        for j in range(width):
            if grid[i][j] != 0:  # Only invert non-zero values
                new_grid[i][j] = max_color - grid[i][j] + 1
            else:
                new_grid[i][j] = 0  # Keep zero values unchanged

    return new_grid


def threshold(grid, threshold_value=1, above_value=1):
    """
    Input:
    A two-dimensional grid (list of lists) where each cell contains a numeric value,
    a threshold value, and an optional value to assign for elements above threshold.

    Functionality:
    The `threshold` function converts the input grid into a binary matrix by comparing
    each element with the threshold value. Elements below or equal to the threshold
    are set to 0, while elements above the threshold are set to the specified above_value
    (defaults to 1).

    Output:
    A new two-dimensional grid where elements have been thresholded.

    Example Input:
    grid = [
        [1, 4, 2],
        [5, 3, 6],
        [2, 1, 4]
    ]
    threshold_value = 3
    above_value = 1

    Example Output:
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 0, 1]
    ]

    Explanation:
    - Each value in the input grid is compared to the threshold (3)
    - Values <= 3 are set to 0
    - Values > 3 are set to above_value (1)
    """

    # Get the dimensions of the input grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    # Create a new grid with the same dimensions
    new_grid = [[0 for _ in range(width)] for _ in range(height)]

    # Apply thresholding
    for i in range(height):
        for j in range(width):
            if grid[i][j] > threshold_value:
                new_grid[i][j] = above_value
            else:
                new_grid[i][j] = 0

    return new_grid


def detect_edges(grid):
    """
    Input:
    A two-dimensional grid (list of lists) where each cell contains a numeric value.

    Functionality:
    The `detect_edges` function detects edges in the input grid by comparing each cell
    with its neighbors. A cell is considered part of an edge if it differs from any
    of its adjacent cells. The function uses a simple edge detection approach based
    on value differences between neighboring cells.

    Output:
    A new two-dimensional grid where edges are marked with 1's and non-edges with 0's.

    Example Input:
    grid = [
        [1, 1, 1, 5],
        [1, 1, 5, 5],
        [1, 5, 5, 5],
        [5, 5, 5, 5]
    ]

    Example Output:
    [
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ]

    Explanation:
    - Each cell is compared with its adjacent neighbors (up, down, left, right)
    - If any neighbor has a different value, the cell is marked as an edge (1)
    - Otherwise, the cell is marked as non-edge (0)
    """

    # Get dimensions of input grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    # Create output grid initialized with zeros
    edges = [[0 for _ in range(width)] for _ in range(height)]

    # Check each cell
    for i in range(height):
        for j in range(width):
            # Get current cell value
            current = grid[i][j]

            # Check neighbors
            is_edge = False

            # Check left neighbor
            if j > 0 and grid[i][j-1] != current:
                is_edge = True

            # Check right neighbor
            if j < width-1 and grid[i][j+1] != current:
                is_edge = True

            # Check top neighbor
            if i > 0 and grid[i-1][j] != current:
                is_edge = True

            # Check bottom neighbor
            if i < height-1 and grid[i+1][j] != current:
                is_edge = True

            # Mark as edge if any neighbor was different
            edges[i][j] = 1 if is_edge else 0

    return edges


def blur(grid, kernel_size=3):
    """
    Input:
    A two-dimensional grid (list of lists) where each cell contains a numeric value,
    and an optional kernel_size parameter (default 3) that determines the blur intensity.

    Functionality:
    The `blur` function applies an averaging blur filter to the input grid by replacing
    each cell's value with the average of its surrounding cells (including itself).
    The kernel_size parameter determines how many neighboring cells to include.

    Output:
    A new two-dimensional grid where each value has been blurred using its neighbors.

    Example Input:
    grid = [
        [1, 4, 2],
        [5, 3, 6],
        [2, 8, 1]
    ]
    kernel_size = 3

    Example Output:
    [
        [3, 3, 4],
        [4, 4, 4],
        [4, 4, 3]
    ]

    Explanation:
    - Each output value is the average of the surrounding values in a kernel_size x kernel_size window
    - Edge pixels use whatever neighbors are available
    - Values are rounded to nearest integer
    """
    import numpy as np

    # Convert input to numpy array for easier processing
    grid = np.array(grid)
    height, width = grid.shape

    # Create output grid of same size
    blurred = np.zeros((height, width), dtype=grid.dtype)

    # Calculate padding size
    pad = kernel_size // 2

    # Add padding to input grid
    padded = np.pad(grid, pad, mode='edge')

    # Apply blur filter
    for i in range(height):
        for j in range(width):
            # Extract neighborhood
            neighborhood = padded[i:i+kernel_size, j:j+kernel_size]
            # Calculate average
            blurred[i, j] = int(round(np.mean(neighborhood)))

    return blurred.tolist()


def sharpen(grid, amount=1):
    """
    Input:
    A two-dimensional grid (list of lists) where each cell contains a numeric value,
    and an optional amount parameter (default 1) that controls sharpening intensity.

    Functionality:
    The `sharpen` function enhances edges in the input grid by emphasizing differences
    between neighboring cells. It uses a sharpening kernel that increases central values
    while decreasing surrounding values.

    Output:
    A new two-dimensional grid where edges and details have been enhanced.

    Example Input:
    grid = [
        [1, 2, 1],
        [2, 3, 2],
        [1, 2, 1]
    ]
    amount = 1

    Example Output:
    [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ]

    Explanation:
    - Uses a 3x3 sharpening kernel to enhance central pixels
    - The amount parameter controls sharpening intensity
    - Values are clamped to maintain valid range
    """
    import numpy as np

    # Convert input to numpy array for easier processing
    grid = np.array(grid)
    height, width = grid.shape

    # Create output grid of same size
    sharpened = np.zeros((height, width), dtype=grid.dtype)

    # Sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 4 + amount, -1],
        [0, -1, 0]
    ])

    # Add padding to input grid
    padded = np.pad(grid, 1, mode='edge')

    # Apply sharpening filter
    for i in range(height):
        for j in range(width):
            # Extract neighborhood
            neighborhood = padded[i:i+3, j:j+3]
            # Apply kernel
            value = np.sum(neighborhood * kernel)
            # Clamp value to valid range
            sharpened[i, j] = max(0, min(9, value))

    return sharpened.tolist()


def add_noise(grid, intensity=1):
    """
    Input:
    A two-dimensional grid (list of lists) where each cell contains a numeric value,
    and an optional intensity parameter (default 1) that controls noise strength.

    Functionality:
    The `add_noise` function adds random variations to the color values in the grid.
    Each value is modified by adding or subtracting a random amount up to the 
    specified intensity level.

    Output:
    A new two-dimensional grid with random noise added to the values.

    Example Input:
    grid = [
        [1, 2, 1],
        [2, 3, 2],
        [1, 2, 1]
    ]
    intensity = 1

    Example Output:
    [
        [2, 1, 1],
        [2, 4, 1], 
        [1, 3, 2]
    ]

    Explanation:
    - Each value is modified by adding/subtracting a random amount
    - The intensity parameter controls maximum change
    - Values are clamped to valid range (0-9)
    """
    import numpy as np

    # Convert input to numpy array for easier processing
    grid = np.array(grid)
    height, width = grid.shape

    # Create output grid of same size
    noisy = np.zeros((height, width), dtype=grid.dtype)

    # Add random noise to each value
    for i in range(height):
        for j in range(width):
            if grid[i, j] > 0:  # Only add noise to non-zero values
                noise = np.random.randint(-intensity, intensity+1)
                value = grid[i, j] + noise
                # Clamp to valid range
                noisy[i, j] = max(0, min(9, value))
            else:
                noisy[i, j] = 0  # Keep zero values unchanged

    return noisy.tolist()


# endregion
