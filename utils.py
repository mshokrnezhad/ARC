import drawsvg
import svgwrite
import numpy as np
import json


def draw_matrix(matrix, cell_size=2, output_file='matrix.svg'):
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    dwg = svgwrite.Drawing(output_file, size=(cols * cell_size, rows * cell_size))
    unique_values = sorted(set(val for row in matrix for val in row))
    color_palette = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FFA500', '#800080', '#00FFFF', '#FFC0CB']
    colormap = {val: color_palette[i % len(color_palette)] for i, val in enumerate(unique_values)}
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            color = colormap.get(val, '#000000')
            dwg.add(dwg.rect(insert=(j * cell_size, i * cell_size),
                             size=(cell_size, cell_size), fill=color))
    dwg.save()


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(map(str, row)))


def pad_array(array, target_shape):
    """
    Pad a 2D array with zeros to match the target shape.
    """
    rows, cols = len(array), len(array[0])
    target_rows, target_cols = target_shape
    padded_array = np.zeros(target_shape, dtype=int)

    # Copy the original array into the top-left corner of the padded array
    for i in range(rows):
        for j in range(cols):
            padded_array[i][j] = array[i][j]

    return padded_array.tolist()


def extend_obj(obj):
    """
    Find the largest array shape in obj["train"] and extend all smaller arrays
    (including those in obj["test"]) to match that size by padding with zeros.
    """
    max_rows, max_cols = 0, 0

    # Find the largest shape in obj["train"] and obj["test"]
    for dataset in ["train", "test"]:
        for item in obj[dataset]:
            input_array = item["input"]
            max_rows = max(max_rows, len(input_array))
            max_cols = max(max_cols, len(input_array[0]))
            if dataset == "train" and "output" in item:
                output_array = item["output"]
                max_rows = max(max_rows, len(output_array))
                max_cols = max(max_cols, len(output_array[0]))

    # Pad all arrays to the largest shape found
    target_shape = (max_rows, max_cols)

    for dataset in ["train", "test"]:
        for item in obj[dataset]:
            item["input"] = pad_array(item["input"], target_shape)
            if dataset == "train" and "output" in item:
                item["output"] = pad_array(item["output"], target_shape)

    return obj


def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def pretty_print_json(data, indent=4):
    def _encode(o, level):
        indent_space = ' ' * (indent * level)
        if isinstance(o, dict):
            if not o:
                return '{}'
            items = []
            for k, v in o.items():
                key = json.dumps(k)
                if isinstance(v, (dict, list)):
                    value = _encode(v, level + 1)
                    # Add a newline after the key
                    item = f'{indent_space}{key}: \n{value}'
                else:
                    value = _encode(v, level)
                    item = f'{indent_space}{key}: {value}'
                items.append(item)
            return '{\n' + ',\n'.join(items) + '\n' + indent_space + '}'
        elif isinstance(o, list):
            if not o:
                return '[]'
            # Check if all elements are simple types
            if all(not isinstance(e, (dict, list)) for e in o):
                # Simple list, keep in one line
                return indent_space + '[' + ', '.join(json.dumps(e) for e in o) + ']'
            else:
                # List contains nested lists/dicts
                items = [_encode(e, level + 1) for e in o]
                return indent_space + '[\n' + ',\n'.join(items) + '\n' + indent_space + ']'
        else:
            return indent_space + json.dumps(o)

    print(_encode(data, 0))


def draw_grid(grid, xmax=10, ymax=10, padding=.5, extra_bottom_padding=0.5, group=False, add_size=True, label='', bordercol='#111111ff'):
    from config import cmap

    gridy, gridx = len(grid), len(grid[0])
    cellsize_x = xmax / gridx
    cellsize_y = ymax / gridy
    cellsize = min(cellsize_x, cellsize_y)

    xsize = gridx * cellsize
    ysize = gridy * cellsize

    line_thickness = 0.01
    border_width = 0.08
    lt = line_thickness / 2

    if group:
        drawing = drawsvg.Group()
    else:
        drawing = drawsvg.Drawing(xsize + padding, ysize + padding + extra_bottom_padding, origin=(-0.5 * padding, -0.5 * padding))
        drawing.set_pixel_scale(40)

    for j, row in enumerate(grid):
        for i, cell in enumerate(row):
            drawing.append(drawsvg.Rectangle(i * cellsize + lt, j * cellsize + lt, cellsize - lt, cellsize - lt, fill=cmap[cell]))

    bw = border_width / 3
    drawing.append(drawsvg.Rectangle(-bw, -bw, xsize + bw * 2, ysize + bw * 2, fill='none', stroke=bordercol, stroke_width=border_width))

    if not group:
        drawing.embed_google_font('Anuphan:wght@400;600;700', text=set('Input Output 0123456789x Test Task ABCDEFGHIJ? abcdefghjklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

    fontsize = (padding / 2 + extra_bottom_padding) / 2
    if add_size:
        drawing.append(drawsvg.Text(text=f'{gridx}x{gridy}', x=xsize, y=ysize + fontsize * 1.25, font_size=fontsize, fill='black', text_anchor='end', font_family='Anuphan'))
    if label:
        drawing.append(drawsvg.Text(text=label, x=-0.1 * fontsize, y=ysize + fontsize * 1.25, font_size=fontsize, fill='black', text_anchor='start', font_family='Anuphan', font_weight='600'))

    if group:
        return drawing, (-0.5 * padding, -0.5 * padding), (xsize + padding, ysize + padding + extra_bottom_padding)
    return drawing


def draw_task_v1(json_obj, index, id, width=30, height=18, include_test=False, label=True, bordercols=['#111111ff', '#111111ff', '#111111ff'], shortdesc=False):  # when the input includes three rows
    padding = 0.5
    bonus_padding = 0.25
    io_gap = 0.4
    ymax = (height - padding - bonus_padding - 2 * io_gap) / 3

    if include_test:
        examples = json_obj['train'] + json_obj['test']
    else:
        examples = json_obj['train']
    n_train = len(json_obj['train'])
    paddingless_width = width - padding * len(examples)

    max_widths = np.zeros(len(examples))
    for i, item in enumerate(examples):
        input_grid = item['input']
        output_grid = item.get('output', None)
        result_grid = item.get('result', None)
        input_grid_ratio = len(input_grid[0]) / len(input_grid)
        output_grid_ratio = len(output_grid[0]) / len(output_grid) if output_grid else 0
        result_grid_ratio = len(result_grid[0]) / len(result_grid) if result_grid else 0
        max_ratio = max(input_grid_ratio, output_grid_ratio, result_grid_ratio)
        xmax = ymax * max_ratio
        max_widths[i] = xmax

    allocation = np.zeros_like(max_widths)
    increment = 0.01
    for _ in range(int(paddingless_width // increment)):
        incr = (allocation + increment) <= max_widths
        if incr.sum() > 0:
            allocation[incr] += increment / incr.sum()

    drawlist = []
    x_ptr = 0
    y_ptr = 0
    for i, item in enumerate(examples):
        input_grid = item['input']
        output_grid = item.get('output', None)
        result_grid = item.get('result', None)

        if shortdesc:
            if i >= n_train:
                input_label = ''
                output_label = ''
                result_label = ''
            else:
                input_label = ''
                output_label = ''
                result_label = ''
        else:
            if i >= n_train:
                input_label = f'Test {i - n_train + 1}'
                output_label = f'Test {i - n_train + 1}' if output_grid else ''
                result_label = f'Test {i - n_train + 1}' if result_grid else ''
            else:
                input_label = f'Input {i + 1}'
                output_label = f'Output {i + 1}' if output_grid else ''
                result_label = f'Result {i + 1}' if result_grid else ''

        input_grid, offset, (input_x, input_y) = draw_grid(input_grid, padding=padding, xmax=allocation[i], ymax=ymax, group=True, label=input_label, extra_bottom_padding=0.5, bordercol=bordercols[0])
        drawlist.append(drawsvg.Use(input_grid, x=x_ptr + (allocation[i] + padding - input_x) / 2 - offset[0], y=-offset[1]))

        if output_grid:
            output_grid, offset, (output_x, output_y) = draw_grid(output_grid, padding=padding, xmax=allocation[i], ymax=ymax, group=True, label=output_label, extra_bottom_padding=0.5, bordercol=bordercols[1])
            drawlist.append(drawsvg.Use(output_grid, x=x_ptr + (allocation[i] + padding - output_x) / 2 - offset[0], y=input_y + io_gap - offset[1]))

        if result_grid:
            result_grid, offset, (result_x, result_y) = draw_grid(result_grid, padding=padding, xmax=allocation[i], ymax=ymax, group=True, label=result_label, extra_bottom_padding=0.5, bordercol=bordercols[2])
            drawlist.append(drawsvg.Use(result_grid, x=x_ptr + (allocation[i] + padding - result_x) / 2 - offset[0], y=input_y + output_y + 2 * io_gap - offset[1]))

        x_ptr += input_x

    x_ptr = round(x_ptr, 1)
    y_ptr2 = round(input_y * 3 + 2 * io_gap, 1)
    d = drawsvg.Drawing(x_ptr, y_ptr2 + 0.2, origin=(0, 0))
    d.append(drawsvg.Rectangle(0, 0, '100%', '100%', fill='#eeeff6'))
    d.embed_google_font('Anuphan:wght@400;600;700', text=set('Input Output Result 0123456789x Test Task ABCDEFGHIJ? abcdefghjklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    for item in drawlist:
        d.append(item)

    fontsize = 0.3
    d.append(drawsvg.Text(f"Task {index}: {id}", x=x_ptr - 0.1, y=y_ptr2 + 0.1, font_size=fontsize, font_family='Anuphan', font_weight='600', fill='#666666', text_anchor='end', alignment_baseline='bottom'))

    d.set_pixel_scale(40)
    return d


def draw_task_v2(json_obj, index, id, width=30, height=18, include_test=False, label=True, bordercols=['#111111ff', '#111111ff', '#111111ff'], shortdesc=False):  # when the input includes two rows
    padding = 0.5
    bonus_padding = 0.25
    io_gap = 0.4

    if include_test:
        examples = json_obj['train'] + json_obj['test']
    else:
        examples = json_obj['train']
    n_train = len(json_obj['train'])
    paddingless_width = width - padding * len(examples)

    drawlist = []
    x_ptr = 0
    max_y_ptr = 0

    for i, item in enumerate(examples):
        input_grid = item['input']
        output_grid = item.get('output', None)
        result_grid = item.get('result', None)

        # Determine the number of rows needed for this example
        num_rows = 1 + bool(output_grid) + bool(result_grid)
        ymax = (height - padding - bonus_padding - (num_rows - 1) * io_gap) / num_rows

        input_grid_ratio = len(input_grid[0]) / len(input_grid)
        output_grid_ratio = len(output_grid[0]) / len(output_grid) if output_grid else 0
        result_grid_ratio = len(result_grid[0]) / len(result_grid) if result_grid else 0
        max_ratio = max(input_grid_ratio, output_grid_ratio, result_grid_ratio)
        xmax = ymax * max_ratio

        if shortdesc:
            input_label = output_label = result_label = ''
        else:
            input_label = f'Input {i + 1}'
            output_label = f'Output {i + 1}' if output_grid else ''
            result_label = f'Result {i + 1}' if result_grid else ''

        input_grid, offset, (input_x, input_y) = draw_grid(input_grid, padding=padding, xmax=xmax, ymax=ymax, group=True, label=input_label, extra_bottom_padding=0.5, bordercol=bordercols[0])
        drawlist.append(drawsvg.Use(input_grid, x=x_ptr + (xmax + padding - input_x) / 2 - offset[0], y=-offset[1]))

        y_offset = input_y + io_gap

        if output_grid:
            output_grid, offset, (output_x, output_y) = draw_grid(output_grid, padding=padding, xmax=xmax, ymax=ymax, group=True, label=output_label, extra_bottom_padding=0.5, bordercol=bordercols[1])
            drawlist.append(drawsvg.Use(output_grid, x=x_ptr + (xmax + padding - output_x) / 2 - offset[0], y=y_offset - offset[1]))
            y_offset += output_y + io_gap

        if result_grid:
            result_grid, offset, (result_x, result_y) = draw_grid(result_grid, padding=padding, xmax=xmax, ymax=ymax, group=True, label=result_label, extra_bottom_padding=0.5, bordercol=bordercols[2])
            drawlist.append(drawsvg.Use(result_grid, x=x_ptr + (xmax + padding - result_x) / 2 - offset[0], y=y_offset - offset[1]))

        x_ptr += input_x
        max_y_ptr = max(max_y_ptr, y_offset)

    final_height = round(max_y_ptr, 1)
    d = drawsvg.Drawing(x_ptr, final_height + 0.2, origin=(0, 0))
    d.append(drawsvg.Rectangle(0, 0, '100%', '100%', fill='#eeeff6'))
    d.embed_google_font('Anuphan:wght@400;600;700', text=set('Input Output Result 0123456789x Test Task ABCDEFGHIJ? abcdefghjklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

    for item in drawlist:
        d.append(item)

    fontsize = 0.3
    d.append(drawsvg.Text(f"Task {index}: {id}", x=x_ptr - 0.1, y=final_height + 0.1, font_size=fontsize, font_family='Anuphan', font_weight='600', fill='#666666', text_anchor='end', alignment_baseline='bottom'))

    d.set_pixel_scale(40)
    return d


def output_drawing(d: drawsvg.Drawing, filename: str, context=None):
    if filename.endswith('.svg'):
        d.save_svg(filename)
    elif filename.endswith('.png'):
        d.save_png(filename)
    elif filename.endswith('.pdf'):
        buffer = io.StringIO()
        d.as_svg(output_file=buffer, context=context)
        import cairosvg
        cairosvg.svg2pdf(bytestring=buffer.getvalue(), write_to=filename)
    else:
        raise ValueError(f'Unknown file extension for {filename}')


def calculate_similarity(output, result):
    total_elements = len(output) * len(output[0])
    similar_elements = sum(
        1 for i in range(len(output)) for j in range(len(output[0])) if output[i][j] == result[i][j]
    )
    return round((similar_elements / total_elements) * 100, 2)


def analyze_task(task_json):
    if isinstance(task_json, dict):
        task_data = task_json
    else:
        task_data = json.loads(task_json)

    # Analyze Train Data
    print("Train:")
    for idx, sample in enumerate(task_data['train'], start=1):
        output = sample['output']
        result = sample['result']
        is_equal = 'Correct' if output == result else 'Wrong'
        similarity_ratio = calculate_similarity(output, result)
        print(f"- Sample {idx}: {is_equal}, Similarity Ratio: {similarity_ratio}%")

    # Analyze Test Data
    print("\nTest:")
    for test_sample in task_data['test']:
        output = test_sample['output']
        result = test_sample['result']
        is_equal = 'Correct' if output == result else 'Wrong'
        similarity_ratio = calculate_similarity(output, result)
        print(f"{is_equal}, Similarity Ratio: {similarity_ratio}%")


def store_json(id, transition, task_, filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    task = {}
    task["transition"] = transition
    task["task"] = task_

    data[id] = task

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def update_task_ids(ids_path, id):
    with open(ids_path, 'r') as file:
        task_ids = json.load(file)

    try:
        last_index = int(list(task_ids.keys())[-1])
    except IndexError:
        last_index = -1
    task_ids[str(last_index + 1)] = id

    with open(ids_path, 'w') as file:
        json.dump(task_ids, file, indent=4)


def check_duplication(ids_path, id):
    with open(ids_path, 'r') as file:
        task_ids = json.load(file)

    return any(id == sub_value for sub_value in task_ids.values()) == False
