import svgwrite


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
