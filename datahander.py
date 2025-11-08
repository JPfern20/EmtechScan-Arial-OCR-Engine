import os
from PIL import Image
import pytesseract

class ArialOCR:
    def __init__(self, traineddata_path="myarial.traineddata"):
        os.environ['TESSDATA_PREFIX'] = os.path.abspath(".")
        pytesseract.pytesseract.tesseract_cmd = os.path.abspath("./engine/tesseract.exe")
        self.lang = "myarial"

    def recognize(self, img_path):
        img = Image.open(img_path)
        config = r'--psm 3'
        data = pytesseract.image_to_data(img, lang=self.lang, config=config, output_type=pytesseract.Output.DICT)
        result = ' '.join([word for word in data['text'] if word.strip()])
        return result, data
