#!/usr/bin/env python3
import os
from PIL import Image, ImageDraw
import subprocess
import cv2
import numpy as np
import shutil

data_dir = "tesseract_train_data"
preview_dir = os.path.join(data_dir, "box_previews")
os.makedirs(preview_dir, exist_ok=True)
path="./engine/tesseract.exe"

def regenerate_box_file(image_path, gt_text, box_path, preview_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    if len(contours) != len(gt_text):
        print(f"⚠️ Mismatch: {len(contours)} contours vs {len(gt_text)} chars in {os.path.basename(image_path)}")
        return

    pil_img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    with open(box_path, 'w', encoding='utf-8') as f:
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            ch = gt_text[i]
            f.write(f"{ch} {x} {y} {x+w} {y+h} 0\n")
            draw.rectangle([x, y, x+w, y+h], outline="red", width=2)

    pil_img.save(preview_path)


def validate_and_fix_and_generate_lstmf():
    shutil.copy('./engine/myarial.traineddata', '.')
    print(f"Validating, fixing, and generating .lstmf files in '{data_dir}'...")
    fixed = 0
    lstmf_count = 0

    for fname in os.listdir(data_dir):
        if fname.endswith(".tif"):
            base = os.path.splitext(fname)[0]
            img_path = os.path.join(data_dir, fname)
            gt_path = os.path.join(data_dir, base + ".gt.txt")
            box_path = os.path.join(data_dir, base + ".box")
            preview_path = os.path.join(preview_dir, base + "_preview.png")
            lstmf_path = os.path.join(data_dir, base + ".lstmf")

            if not os.path.exists(gt_path):
                print(f"⚠️ Missing .gt.txt for {fname}")
                continue

            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_text = f.read().strip()

            regenerate = False
            if not os.path.exists(box_path):
                regenerate = True
            else:
                with open(box_path, 'r', encoding='utf-8') as f:
                    box_lines = [line.strip() for line in f if line.strip()]
                box_chars = [line.split()[0] for line in box_lines]
                if len(box_chars) != len(gt_text) or any(b != g for b, g in zip(box_chars, gt_text)):
                    regenerate = True

            if regenerate:
                regenerate_box_file(img_path, gt_text, box_path, preview_path)
                print(f"🔧 Regenerated .box and preview for {fname}")
                fixed += 1


            cmd = [
                path,
                img_path,
                os.path.join(data_dir, base),
                "--psm", "7",
                "-l", "eng",
                "lstm.train"
            ]
            try:
                subprocess.run(cmd, check=True)
                print(f"Created .lstmf: {base}.lstmf")
                lstmf_count += 1
            except subprocess.CalledProcessError as e:
                print(f"Error generating .lstmf for {fname}: {e}")

    print(f"\nDone! {fixed} .box files fixed, {lstmf_count} .lstmf files generated.")
    print(f"Previews saved in '{preview_dir}'.")

def run_lstmtraining():
    print("\n🚀 Starting lstmtraining...")

    lstmtraining_path = os.path.abspath("./engine/lstmtraining.exe")
    traineddata = os.path.abspath("./engine/tessdata/eng.traineddata")
    output_model = os.path.abspath("arial_checkpoint")
    train_list = os.path.abspath("train.list")

    cmd = [
        lstmtraining_path,
        "--model_output", output_model,
        "--traineddata", traineddata,
        "--train_listfile", train_list,
        "--max_iterations", "4000"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅ lstmtraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"lstmtraining failed: {e}")

if __name__ == "__main__":
    validate_and_fix_and_generate_lstmf()
    