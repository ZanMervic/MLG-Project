from PIL import Image
from collections import Counter
import numpy as np
import json
import os
import tqdm
import sys

def top_colors_in_box(img, x, y, radius, top_n=6):
    width, height = img.size
    x_min = max(x - radius, 0)
    x_max = min(x + radius, width - 1)
    y_min = max(y - radius, 0)
    y_max = min(y + radius, height - 1)
    region = img.crop((x_min, y_min, x_max + 1, y_max + 1))
    pixels = np.array(region).reshape(-1, 3)
    counts = Counter(map(tuple, pixels))
    total = sum(counts.values())
    normalized = [(color, count / total) for color, count in counts.items()]
    normalized.sort(key=lambda x: x[1], reverse=True)
    return normalized[:top_n]

def color_distance(c1, c2):
    return np.sum(np.abs(np.array(c1) - np.array(c2)))

def process_image(path, max_color_dist=80):
    img = Image.open(path)
    img = img.convert("RGB")

    width, height = img.size

    # get corners of the yellow board
    yellow = (238, 223, 80) 
    # Loop through all pixels forward
    flag = False
    for y in range(height):
        for x in range(width):
            rgb = img.getpixel((x, y))
            if color_distance(rgb, yellow) < 100 and rgb[0] - rgb[1] < 25:
                upper_left = (x, y)
                flag = True 
                break
        if flag:
            break
    # Loop through all pixels backward
    flag = False
    for y in range(height)[::-1]:
        for x in range(width)[::-1]:
            rgb = img.getpixel((x, y))
            if color_distance(rgb, yellow) < 100 and rgb[0] - rgb[1] < 25:
                lower_right = (x, y)
                flag = True 
                break
        if flag:
            break
    print(upper_left, lower_right)
    # get middle of A18 and horizontal and vertical steps to next 
    left_len = (142 / 1015) * (lower_right[0] - upper_left[0])
    step_horizontal = (79 / 1015) * (lower_right[0] - upper_left[0])

    upper_len = (138 / 1568) * (lower_right[1] - upper_left[1])
    step_vertical = (78 / 1568) * (lower_right[1] - upper_left[1])

    start_x = upper_left[0] + left_len 
    start_y = upper_left[1] + upper_len 

    # get approx radius of the red/green/blue circles
    radius = int((61 / 1568) * (lower_right[1] - upper_left[1]))

    # process A-K 1-18
    holds = {"start":[], "end":[], "middle":[]}
    red = (244, 67, 54)
    blue = (41, 98, 255)
    green = (76, 175, 80)
    for i, a in enumerate("ABCDEFGHIJK"):
        for j, n in enumerate(range(1, 19)[::-1]):
            x = int(start_x + i * step_horizontal)
            y = int(start_y + j * step_vertical)
            top_n_colors = top_colors_in_box(img, x, y, radius)
            red_p = sum([p for color, p in top_n_colors if color_distance(color, red) < max_color_dist])
            blue_p = sum([p for color, p in top_n_colors if color_distance(color, blue) < max_color_dist])
            green_p = sum([p for color, p in top_n_colors if color_distance(color, green) < max_color_dist])
            if red_p > 0.15:
                holds["end"].append(f"{a}{n}")
            if blue_p > 0.15:
                holds["middle"].append(f"{a}{n}")
            if green_p > 0.15:
                holds["start"].append(f"{a}{n}")
    return holds

def process_images(images, f, t):
    # put screenshots in folder screenshots for this to work
    l = len(images)
    holds = {}
    for i in range(f, min(l, t)):
        scr = images[i]
        j = scr.rfind("_")
        name = scr[:j]
        scrpath = os.path.join("screenshots", scr)
        h = process_image(scrpath)
        holds[name] = h
    with open(f"holds_{f}_{t-1}.json", "w", encoding="utf-8") as f:
        json.dump(holds, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    data = os.listdir("./screenshots")
    screenshots = [x for x in data if x.endswith(".png")]
    if len(sys.argv) < 2:
        print("Usage: python screenshot_processer.py from to")
        sys.exit()

    f = int(sys.argv[1])
    t = int(sys.argv[2])
    process_images(screenshots, f, t)