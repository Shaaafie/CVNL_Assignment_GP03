import json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

from .cnn_model import AircraftCNN
from .rnn_model import TextRNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Config: paths for ALL models
# -----------------------------
CNN_CONFIG = {
    "Aircraft Family": {
        "weights": "models/cnn_family.pth",
        "idx_to_class": "label_maps/idx_to_class_family.json",
        "img_size": 224,
    },
    "Manufacturer": {
        "weights": "models/cnn_manufacturer.pth",
        "idx_to_class": "label_maps/idx_to_class_manufacturer.json",
        "img_size": 224,
    },
}

RNN_CONFIG = {
    "Intent": {
        "weights": "models/rnn_intent.pt",
        "word2idx": "label_maps/word2idx_intent.json",
        "id2label": "label_maps/id_to_label_intent.json",
        # these must match how your teammate trained
        "embed_dim": 128,
        "hidden_dim": 128,
        "max_len": 40,
    },
    "Sentiment": {
        "weights": "models/rnn_sentiment.pt",
        "word2idx": "label_maps/word2idx_sentiment.json",
        "id2label": "label_maps/id_to_label_sentiment.json",
        "embed_dim": 128,
        "hidden_dim": 128,
        "max_len": 40,
    },
}

# -----------------------------
# Utilities
# -----------------------------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def cnn_transform(img_size=224):
    # match your CNN training normalization
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
    ])

def simple_tokenize(text: str):
    # very basic tokenization; if your teammate used something else, match theirs
    return text.lower().strip().split()

def encode_text(tokens, word2idx, max_len):
    ids = [word2idx.get(tok, word2idx.get("<unk>", 0)) for tok in tokens]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

# -----------------------------
# Caching loaded models
# -----------------------------
_CNN_CACHE = {}
_RNN_CACHE = {}

def load_cnn_model(task_name: str):
    if task_name in _CNN_CACHE:
        return _CNN_CACHE[task_name]

    cfg = CNN_CONFIG[task_name]
    idx_to_class = load_json(cfg["idx_to_class"])
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    num_classes = len(idx_to_class)

    model = AircraftCNN(num_classes).to(DEVICE)
    state = torch.load(cfg["weights"], map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    tfm = cnn_transform(cfg["img_size"])
    _CNN_CACHE[task_name] = (model, idx_to_class, tfm)
    return _CNN_CACHE[task_name]

@torch.no_grad()
def predict_cnn(image_path: str, task_name: str, top_k: int = 3):
    model, idx_to_class, tfm = load_cnn_model(task_name)
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(top_k, probs.numel())
    top_probs, top_idxs = torch.topk(probs, k=k)

    results = []
    for p, idx in zip(top_probs.cpu().tolist(), top_idxs.cpu().tolist()):
        results.append({"label": idx_to_class[idx], "confidence": float(p)})
    return results

def load_rnn_model(task_name: str):
    if task_name in _RNN_CACHE:
        return _RNN_CACHE[task_name]

    cfg = RNN_CONFIG[task_name]
    word2idx = load_json(cfg["word2idx"])
    id2label = load_json(cfg["id2label"])
    id2label = {int(k): v for k, v in id2label.items()}

    vocab_size = max(word2idx.values()) + 1 if word2idx else 1
    num_classes = len(id2label)

    model = TextRNN(
        vocab_size=vocab_size,
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=num_classes
    ).to(DEVICE)

    state = torch.load(cfg["weights"], map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    _RNN_CACHE[task_name] = (model, word2idx, id2label, cfg["max_len"])
    return _RNN_CACHE[task_name]

@torch.no_grad()
def predict_rnn(text: str, task_name: str):
    model, word2idx, id2label, max_len = load_rnn_model(task_name)

    tokens = simple_tokenize(text)
    x = encode_text(tokens, word2idx, max_len).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())

    return {
        "label": id2label[pred_idx],
        "confidence": float(probs[pred_idx].item())
    }
