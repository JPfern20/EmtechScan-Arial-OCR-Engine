#!/usr/bin/env python3
import os
import random
from PIL import Image, ImageFilter
import numpy as np
import subprocess

SPECIAL_CHAR_MAP = {
    '!': 'EXCLAMATION',
    '@': 'AT',
    '#': 'HASH',
    '$': 'DOLLAR',
    '%': 'PERCENT',
    '^': 'CARET',
    '&': 'AMPERSAND',
    '*': 'ASKTERISK',
    'PARENTHESIS_LEFT': '(',
    'PARENTHESIS_RIGHT': ')',
    'HYPHEN': '-',
    'UNDERSCORE': '_',
    'PLUS': '+',
    '=': 'EQUALS',
    'COMMA': ',',
    'DOT': '.',
    'QUESTION_MARK': '?',
    'SLASH': '/'
}

def load_character_images(input_dir):
    """
    Loads all character PNGs into a dict {char: Image}
    """
    char_images = {}

    for root, dirs, files in os.walk(input_dir):
        folder = os.path.basename(root).lower()
        for filename in files:
            if filename.endswith('.png'):
                char_name = filename.split('.')[0]
                # Determine actual character
                if folder == 'special_characters':
                    char_upper = char_name.upper()
                    if char_upper in SPECIAL_CHAR_MAP:
                        char = SPECIAL_CHAR_MAP[char_upper]
                    else:
                        print(f"Warning: Unknown special char '{char_name}' skipped.")
                        continue
                else:
                    if folder == 'upper_case':
                        char = char_name.upper()
                    elif folder == 'lower_case':
                        char = char_name.lower()
                    elif folder == 'digits':
                        char = char_name
                    else:
                        char = char_name

                # Load image and convert to grayscale ('L')
                img_path = os.path.join(root, filename)
                img = Image.open(img_path).convert('L')
                char_images[char] = img

    if not char_images:
        raise ValueError("EmtechScan: DataJob: No character images are there in the specified input directory.")
    if ' ' not in char_images:
        space_width = 20  # Adjust as needed for spacing
        space_height = 64  # Match your line_height
        space_img = Image.new('L', (space_width, space_height), color=255)
        char_images[' '] = space_img
    
    
    return char_images

def compose_text_line(text, char_images, line_height=64, spacing=5):
    """
    Create a PIL Image by horizontally concatenating character images
    """
    # Resize all chars to uniform height while keeping aspect ratio
    resized_chars = []
    for ch in text:
        if ch not in char_images:
            raise ValueError(f"EmtechScan: DataJob: Character '{ch}' not found in loaded images.")
        img = char_images[ch]
        w, h = img.size
        new_h = line_height
        new_w = int(w * (new_h / h))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        resized_chars.append(resized)


    total_width = sum(img.width for img in resized_chars) + spacing * (len(resized_chars) - 1)
    line_img = Image.new('L', (total_width, line_height), color=255)  # white background

    x_offset = 0
    for img in resized_chars:
        line_img.paste(img, (x_offset, 0))
        x_offset += img.width + spacing

    return line_img

def generate_training_data(input_dir="fonts/arial",
                           output_dir="tesseract_train_data",
                           num_lines=1000,
                           line_height=64,
                           spacing=5):

    with open("words_1000.txt", "r", encoding="utf-8") as f:
        word_list = f.read().split()

    os.makedirs(output_dir, exist_ok=True)


    print("Loading character images...")
    char_images = load_character_images(input_dir)
    characters = list(char_images.keys())
    print(f"Loaded {len(characters)} characters.")

    for i in range(num_lines):

        text = ' '.join(random.choices(word_list, k=random.randint(3, 8)))

        try:
            line_img = compose_text_line(text, char_images, line_height=line_height, spacing=spacing)
        except ValueError as e:
            print(f"Skipping line {i} due to missing character: {e}")
            continue
        

        if random.random() < 0.5:
            angle = random.uniform(-2, 2)
            line_img = line_img.rotate(angle, fillcolor=255)

        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            new_w = int(line_img.width * scale)
            new_h = int(line_img.height * scale)
            line_img = line_img.resize((new_w, new_h), Image.LANCZOS)
            canvas = Image.new('L', (line_img.width, line_img.height), color=255)
            offset_x = max((line_img.width - new_w) // 2, 0)
            offset_y = max((line_img.height - new_h) // 2, 0)
            canvas.paste(line_img, (offset_x, offset_y))
            line_img = canvas

        if random.random() < 0.3:
            line_img = line_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))


        img_path = os.path.join(output_dir, f"line_{i:04d}.tif")
        gt_path = os.path.join(output_dir, f"line_{i:04d}.gt.txt")
        line_img.save(img_path)
        with open(gt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        if i % 100 == 0:
            print(f"Generated {i} lines...")

    print(f"Emtech Datajob: generation is completed, Thank you. \n {num_lines} lines saved in '{output_dir}'")
    
    print("Emtech: Engine: Generating .box files")
    for fname in os.listdir(output_dir):
        if fname.endswith(".tif"):
         img_path = os.path.join(output_dir, fname)
         base = os.path.splitext(img_path)[0]
         tesseract_path = r".\engine\tesseract.exe"
         subprocess.run([tesseract_path, img_path, base, "--psm", "7", "-l", "eng", "batch.nochop", "makebox"])
    print("Emtech Engine: Box file generation complete!")
if __name__ == "__main__":
    generate_training_data()

