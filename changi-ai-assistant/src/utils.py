import json
from pathlib import Path
from torchvision import transforms

def load_json(path: str):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_cnn_transform(img_size: int = 224):
    # Match your training normalization (you used 0.5/0.5/0.5)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
