#!/usr/bin/env python3
import os
import subprocess


# Configuration
data_dir = "tesseract_train_data"
tesseract_exe = "./engine/tesseract.exe"
lang = "eng" 
psm = "7"

def generate_lstmf_files():
    print(f"Generating .lstmf files in '{data_dir}'...")
    count = 0

    for fname in os.listdir(data_dir):
        if fname.endswith(".tif"):
            base = os.path.splitext(fname)[0]
            tif_path = os.path.join(data_dir, fname)
            gt_path = os.path.join(data_dir, base + ".gt.txt")

            if not os.path.exists(gt_path):
                print(f"⚠️ Skipping {fname}: missing ground truth file.")
                continue

            cmd = [
                tesseract_exe,
                tif_path,
                os.path.join(data_dir, base),
                "--psm", psm,
                "-l", lang,
                "lstm.train"
            ]

            try:
                subprocess.run(cmd, check=True)
                print(f"Created: {base}.lstmf")
                count += 1
            except subprocess.CalledProcessError as e:
                print(f"Error processing {fname}: {e}")

    print(f"\nDone! {count} .lstmf files generated. \n OCR Ready.")

if __name__ == "__main__":
    generate_lstmf_files()
