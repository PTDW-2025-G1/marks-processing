from ultralytics import YOLO
from PIL import Image

class MasonDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, image: Image.Image):
        results = self.model.predict(image, conf=0.05, iou=0.15, max_det=50)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best = boxes[boxes.conf.argmax()]
        x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
        return image.crop((x1, y1, x2, y2))
