"""Load the user's saved MobileNetV2 model and predict a leaf class."""
from pathlib import Path
import json

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "mobilenet_best.pth"
LABEL_PATH = BASE_DIR / "model" / "class_names.json"

# Identical to the final Kaggle notebook (no augmentation at inference time).
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    """Create MobileNetV2 with 38 output classes and load trained weights."""
    with LABEL_PATH.open(encoding="utf-8") as file:
        class_names = json.load(file)

    model = models.mobilenet_v2(weights=None)  # No download is needed.
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

    # weights_only=True keeps loading restricted to tensor state dictionaries.
    weights = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model, class_names


def predict_image(image: Image.Image, model, class_names):
    """Return the three highest-ranked (class name, softmax score) pairs."""
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)  # (1, 3, 160, 160)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        scores, indices = torch.topk(probabilities, k=3)

    return [
        (class_names[index], float(score))
        for score, index in zip(scores.tolist(), indices.tolist())
    ]


def format_class(class_name):
    """'Tomato___Early_blight' -> ('Tomato', 'Early blight')."""
    plant, condition = class_name.split("___", 1)
    plant = plant.replace("_", " ").replace(",", ", ").replace("  ", " ").strip()
    condition = condition.replace("_", " ").replace("  ", " ").strip()
    if condition.lower() == "healthy":
        condition = "Healthy"
    return plant, condition
