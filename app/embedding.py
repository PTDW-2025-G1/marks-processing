import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

from app.preprocessor import preprocess_for_embedding


class EmbeddingExtractor:
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval()

        self.preprocess = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def vectorize(self, image: Image.Image):
        image = preprocess_for_embedding(image)
        x = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            vec = self.model(x)
        return vec.squeeze().numpy().tolist()
