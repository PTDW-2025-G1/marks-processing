from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageOps
from io import BytesIO

from app.detector import MasonDetector
from app.embedding import EmbeddingExtractor
from app.preprocessor import enhance_for_yolo

app = FastAPI()

detector = MasonDetector("models/yolo11n.pt")
embedder = EmbeddingExtractor()


@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    img_bytes = await file.read()

    image = Image.open(BytesIO(img_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")

    image = enhance_for_yolo(image)
    cropped = detector.detect(image)

    print("isMasonMark:", cropped is not None)

    target_image = cropped if cropped is not None else image
    vector = embedder.vectorize(target_image)

    return {
        "isMasonMark": cropped is not None,
        "embedding": vector
    }
