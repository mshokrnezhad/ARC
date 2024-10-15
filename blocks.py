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
