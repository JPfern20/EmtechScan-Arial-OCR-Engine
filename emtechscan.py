#!/usr/bin/env python3
"""
EmTechScan Notes

Our Team used the Tesseract with our dataset and we clarify that we did not used it on its deep learning capabilities,
but rather used with our custom-trained model from scratch.

All of the Detection Logic such as Gaussian Blur, Resizing Logics, 
and Detection was placed on the Dataset Generation and Training Phase.

Features:
- Used custom-trained ARIAL FONT ONLY for recognition
- Offers GUI for easy image selection and OCR execution without command line scripts.
- Supports image input with trained character recognition
- Saves recognized text to Word (.docx) or Text (.txt)

Please note that the Project is for printed documents only 
and does not support handwritten text recognition and does not support deep learning and multiple fonts.

A requirement for the course CPE 018 | Emerging Technologies in Computer Engineering
Technological University of the Philippines - Quezon City
Group Members:
Fernandez
Faustino
Jimenez
"""

import os
from PIL import Image, ImageTk
from tkinter import ttk, filedialog, messagebox
import tkinter as tk
from docx import Document
from datahander import ArialOCR

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EmTechScan")
        self.ocr = ArialOCR()
        self.image_path = None
        self.tk_img = None
        self.result_text = ""
        self.bounding_boxes = None
        self.setup_ui()

    def setup_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="EmTechScan", font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=4, pady=8)

        self.canvas = tk.Canvas(frm, width=640, height=480, bg="gray20")
        self.canvas.grid(row=1, column=0, columnspan=4, pady=5)

        # Styled buttons
        tk.Button(frm, text="Select Image", command=self.select_image, bg="yellow", fg="black").grid(row=2, column=0, columnspan=4, sticky="ew", pady=4)
        tk.Button(frm, text="Run OCR", command=self.run_ocr, bg="yellow", fg="black").grid(row=3, column=0, columnspan=4, sticky="ew", pady=4)
        tk.Button(frm, text="Show Debug Box", command=self.show_boxes, bg="yellow", fg="black").grid(row=4, column=0, columnspan=4, sticky="ew", pady=4)

        self.status = ttk.Label(frm, text="Status: Ready")
        self.status.grid(row=5, column=0, columnspan=4, sticky="w", pady=3)

        self.text_box = tk.Text(frm, wrap="word", width=70, height=12)
        self.text_box.grid(row=6, column=0, columnspan=4, pady=5)

        ttk.Label(frm, text="Output Format:").grid(row=7, column=0, sticky="e", padx=5)
        self.format_choice = ttk.Combobox(frm, values=[".txt", ".docx"], state="readonly", width=10)
        self.format_choice.set(".txt")
        self.format_choice.grid(row=7, column=1, sticky="w")

        tk.Button(frm, text="Save Output", command=self.save_output, bg="yellow", fg="black").grid(row=8, column=0, columnspan=4, sticky="ew", pady=5)

    def select_image(self):
        path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.image_path = path
            self.show_preview(path)
            self.status.config(text=f"Loaded: {os.path.basename(path)}")

    def show_preview(self, path):
        img = Image.open(path)
        img.thumbnail((640, 480))
        self.preview_width, self.preview_height = img.size
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(320, 240, anchor="center", image=self.tk_img)

    def run_ocr(self):
        if not self.image_path:
            messagebox.showwarning("No image", "Please select an image first")
            return
        try:
            self.status.config(text="Running OCR...")
            self.root.update()
            result, data = self.ocr.recognize(self.image_path)
            self.result_text = result
            self.bounding_boxes = data
            self.text_box.delete("1.0", tk.END)
            self.text_box.insert("1.0", result)
            self.status.config(text=f"OCR complete. {len(result.split())} words recognized.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_boxes(self):
        if not self.image_path or not self.bounding_boxes:
            messagebox.showwarning("No data", "Run OCR first to get bounding boxes")
            return

        self.canvas.delete("box")
        orig_img = Image.open(self.image_path)
        img_w, img_h = orig_img.size

        scale_x = self.preview_width / img_w
        scale_y = self.preview_height / img_h
        offset_x = (640 - self.preview_width) / 2
        offset_y = (480 - self.preview_height) / 2

        for i in range(len(self.bounding_boxes['text'])):
            if self.bounding_boxes['text'][i].strip():
                x = self.bounding_boxes['left'][i] * scale_x + offset_x
                y = self.bounding_boxes['top'][i] * scale_y + offset_y
                w = self.bounding_boxes['width'][i] * scale_x
                h = self.bounding_boxes['height'][i] * scale_y
                self.canvas.create_rectangle(x, y, x + w, y + h, outline="red", width=2, tags="box")

        self.status.config(text="Bounding boxes displayed")

    def save_output(self):
        if not self.result_text.strip():
            messagebox.showwarning("No text", "Please run OCR first")
            return

        selected_format = self.format_choice.get()
        filetypes = [("Text files", "*.txt"), ("Word Document", "*.docx")]
        default_ext = selected_format

        path = filedialog.asksaveasfilename(defaultextension=default_ext, filetypes=filetypes)
        if path:
            try:
                if selected_format == ".docx":
                    doc = Document()
                    doc.add_paragraph(self.result_text)
                    doc.save(path)
                else:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(self.result_text)
                self.status.config(text=f"Saved output to {path}")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))

def main():
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
