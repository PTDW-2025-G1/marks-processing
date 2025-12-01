import cv2
import numpy as np
from PIL import Image, ImageOps

def enhance_for_yolo(image: Image.Image):
    # Convert PIL → OpenCV
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # CLAHE contrast boost (critical)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Edge enhancement: unsharp masking
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    # Back to 3-channel RGB for YOLO
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img)


def preprocess_for_embedding(image: Image.Image):
    img = image.convert("L")            # grayscale
    img = ImageOps.equalize(img)        # flatten lighting
    return img.convert("RGB")
