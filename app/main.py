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

    if cropped is None:
        return {
            "isMasonMark": False,
            "embedding": None
        }

    vector = embedder.vectorize(cropped)

    return {
        "isMasonMark": True,
        "embedding": vector
    }
