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

# Geometry
#     do nothing
#     rotate
#     mirror
#     shift image
#     crop image background
#     draw border
# Objects
#     rotate
#     mirror
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

# blocks extracted from arc-prize-2024


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


def find_loops(grid):
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


def change_elements_color(grid, elements, new_value):
    for x, y in elements:
        grid[x][y] = new_value

    return grid

# general blocks


def rotate_grid(grid, degrees):
    if degrees not in [90, 180, 270]:
        raise ValueError("Degrees must be 90, 180, or 270")

    if degrees == 90:
        return [list(row) for row in zip(*grid[::-1])]
    elif degrees == 180:
        return [row[::-1] for row in grid[::-1]]
    elif degrees == 270:
        return [list(row) for row in zip(*grid)][::-1]
