from PIL import Image, ImageDraw, ImageFont
import textwrap

def create_perfect_table(headers, rows, output_path="table.png", total_width=1000):
    font_size = 20
    cell_padding_x = 15
    cell_padding_y = 12
    header_bg_color = (70, 130, 180)
    header_text_color = (255, 255, 255)
    row_colors = [(245, 245, 245), (255, 255, 255)]
    border_color = (0, 0, 0)

    try:
        font_path = ".\\fonts\\BRLNSR.TTF"
        font = ImageFont.truetype(font_path, font_size)
        print("Font loaded successfully")
    except:
        font = ImageFont.load_default()
        print("Font not found, using default font")

    def text_size(text):
        bbox = font.getbbox(text)
        return bbox[2], bbox[3]

    # Natural widths
    all_rows = [headers] + rows
    natural_widths = []
    for col in range(len(headers)):
        max_width = max(text_size(str(row[col]))[0] for row in all_rows) + 2*cell_padding_x
        natural_widths.append(max_width)
    
    total_natural_width = sum(natural_widths)
    scale = total_width / total_natural_width
    col_widths = [int(w * scale) for w in natural_widths]

    # Pre-wrap text based on pixel width
    wrapped_rows = []
    row_heights = []

    for r, row in enumerate(all_rows):
        wrapped_row = []
        max_height = 0
        for c, cell in enumerate(row):
            words = str(cell).split(" ")
            lines = []
            line = ""
            for word in words:
                test_line = line + (" " if line else "") + word
                w, _ = text_size(test_line)
                if w + 2*cell_padding_x <= col_widths[c]:
                    line = test_line
                else:
                    if line:
                        lines.append(line)
                    line = word
            if line:
                lines.append(line)
            wrapped_row.append(lines)
            cell_height = sum(text_size(line)[1] for line in lines) + 2*cell_padding_y
            if cell_height > max_height:
                max_height = cell_height
        wrapped_rows.append(wrapped_row)
        row_heights.append(max_height)

    img_width = sum(col_widths)
    img_height = sum(row_heights)
    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    # Draw table
    y = 0
    for r, wrapped_row in enumerate(wrapped_rows):
        x = 0
        for c, lines in enumerate(wrapped_row):
            bg_color = header_bg_color if r == 0 else row_colors[(r-1)%len(row_colors)]
            text_color = header_text_color if r == 0 else (0,0,0)
            draw.rectangle([x, y, x+col_widths[c], y+row_heights[r]], fill=bg_color, outline=border_color, width=2)
            
            text_y = y + cell_padding_y
            for line in lines:
                draw.text((x + cell_padding_x, text_y), line, fill=text_color, font=font)
                text_y += text_size(line)[1]
            x += col_widths[c]
        y += row_heights[r]

    img.save(output_path)
    print(f"Saved table to {output_path}")


# Example usage
headers = ["Model", "Recall@20", "95% CI", "Std Dev"]

rows = [
    ["Popularity baseline", "0.344", "[0.341, 0.347]", "-"],
    ["GFormer", "0.189", "[0.183, 0.194]", "0.344"],
    ["Hetero (SAGEConv)", "0.174", "[0.169, 0.179]", "0.320"],
    ["PinSAGE", "0.173", "[0.168, 0.178]", "0.321"],
    ["Hetero Attention", "0.162", "[0.157, 0.167]", "0.320"]
]




create_perfect_table(headers, rows, output_path=".\\Images_of_tables\\results_nice.png")