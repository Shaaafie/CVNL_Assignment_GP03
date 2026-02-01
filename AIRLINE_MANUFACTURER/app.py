from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Tuple

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

APP_ROOT = Path(__file__).parent
MODEL_PATH = Path(os.environ.get('MODEL_PATH', APP_ROOT / 'model3_best.pt'))
DEVICE_PREF = os.environ.get('DEVICE', 'cuda')

app = Flask(__name__)


def _pick_device() -> torch.device:
    if DEVICE_PREF == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _build_model(num_classes: int) -> torch.nn.Module:
    """Try common ResNet variants until state_dict loads cleanly."""
    candidates = [
        models.resnet18,
        models.resnet34,
        models.resnet50,
    ]
    last_err = None
    for ctor in candidates:
        model = ctor(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        try:
            model.load_state_dict(CKPT['model_state'], strict=True)
            return model
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"Failed to load checkpoint into known ResNet variants: {last_err}")


CKPT = torch.load(MODEL_PATH, map_location='cpu')
CLASS_NAMES: List[str] = CKPT.get('class_names', [])
IMG_SIZE = int(CKPT.get('img_size', 224))
BANNER_PX = int(CKPT.get('banner_px', 0))
DEVICE = _pick_device()

model = _build_model(len(CLASS_NAMES))
model.to(DEVICE)
model.eval()

# If your training used a custom normalization, update these values.
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _prepare_image(file_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    if BANNER_PX > 0:
        # Optional trim for any banner padding baked into the dataset.
        width, height = image.size
        image = image.crop((0, BANNER_PX, width, height))
    tensor = preprocess(image).unsqueeze(0)
    return tensor


@torch.no_grad()
def _predict(file_bytes: bytes, top_k: int = 5) -> Tuple[str, float, List[dict]]:
    tensor = _prepare_image(file_bytes).to(DEVICE)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    conf, idx = torch.max(probs, dim=0)

    topk_conf, topk_idx = torch.topk(probs, k=min(top_k, probs.numel()))
    topk = [
        {
            'class_name': CLASS_NAMES[i],
            'confidence': float(c),
        }
        for c, i in zip(topk_conf.cpu().tolist(), topk_idx.cpu().tolist())
    ]

    return CLASS_NAMES[int(idx)], float(conf), topk


@app.route('/')
def index():
    return render_template('index.html', class_count=len(CLASS_NAMES), img_size=IMG_SIZE)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'Empty file'}), 400
    try:
        label, confidence, topk = _predict(file.read())
        return jsonify({
            'label': label,
            'confidence': confidence,
            'topk': topk,
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
