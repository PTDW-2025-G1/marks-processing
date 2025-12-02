from fastapi import FastAPI, File, UploadFile
from PIL import Image
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
    image = Image.open(BytesIO(img_bytes)).convert("RGB")

    image = enhance_for_yolo(image)
    cropped = detector.detect(image)

    if cropped is None:
        return {
            "is_mark": False,
            "embedding": None
        }

    vector = embedder.vectorize(cropped)

    return {
        "is_mark": True,
        "embedding": vector
    }
