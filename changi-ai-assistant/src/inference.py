import json
import pickle
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

from src.aircraft_family_cnn import AircraftCNN
from src.rnn_model import TextRNN, TextLSTM
from src.resnet_model import build_resnet_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent.parent

# -----------------------------
# Config: paths for ALL models
# -----------------------------
CNN_CONFIG = {
    "Aircraft Family": {
        "weights": "models/aircraftcnn_family_best.pth",
        "idx_to_class": "label_maps/idx_to_class_aircraft_family.json",
        "img_size": 224,
        "type": "custom"
    },
}

# ResNet models (from Flask app)
RESNET_CONFIG = {
    "Manufacturer (ResNet)": {
        "weights": "models/resnet_manufacturer.pt",
        "img_size": 224,
        "type": "resnet"
    },
    "Airline (ResNet)": {
        "weights": "models/resnet_airline.pt",
        "img_size": 224,
        "type": "resnet"
    },
}

RNN_CONFIG = {
    "Intent Classification": {
        "weights": "models/RNN_Intent_Classifications.pth",
        "vocab_bundle": "label_maps/rnn_vocab_bundle.pkl",
        "id2label": "label_maps/intent10_label_map.json",
        # Match notebook training params
        "embed_dim": 128,
        "hidden_dim": 256,
        "max_len": 60,
        "dropout": 0.3,
    },
    "Sentiment": {
        "weights": "models/best_SentimentRNN_model.pth",
        "word2idx": "label_maps/word2idx_sentimentRNN.json",
        "id2label": "label_maps/id_to_label_sentimentRNN.json",
        "embed_dim": 128,
        "hidden_dim": 128,
        "max_len": 40,
        "dropout": 0.3,
    },
}

# -----------------------------
# Utilities
# -----------------------------
def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (BASE_DIR / p)

def load_json(path: str | Path):
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)

def load_vocab_bundle(path: str | Path):
    with open(resolve_path(path), "rb") as f:
        return pickle.load(f)

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
_RESNET_CACHE = {}

def resnet_transform(img_size=224, banner_px=0):
    """Transform for ResNet models with ImageNet normalization"""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_resnet_model(task_name: str):
    if task_name in _RESNET_CACHE:
        return _RESNET_CACHE[task_name]
    
    cfg = RESNET_CONFIG[task_name]
    model, class_names, img_size, banner_px = build_resnet_model(str(resolve_path(cfg["weights"])), DEVICE)
    tfm = resnet_transform(img_size, banner_px)
    
    _RESNET_CACHE[task_name] = (model, class_names, tfm, banner_px)
    return _RESNET_CACHE[task_name]

@torch.no_grad()
def predict_resnet(image_path: str, task_name: str, top_k: int = 5):
    model, class_names, tfm, banner_px = load_resnet_model(task_name)
    img = Image.open(image_path).convert("RGB")
    
    # Optional banner crop
    if banner_px > 0:
        width, height = img.size
        img = img.crop((0, banner_px, width, height))
    
    x = tfm(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    
    k = min(top_k, probs.numel())
    top_probs, top_idxs = torch.topk(probs, k=k)
    
    results = []
    for p, idx in zip(top_probs.cpu().tolist(), top_idxs.cpu().tolist()):
        results.append({"label": class_names[idx], "confidence": float(p)})
    return results

def load_cnn_model(task_name: str):
    if task_name in _CNN_CACHE:
        return _CNN_CACHE[task_name]

    cfg = CNN_CONFIG[task_name]
    idx_to_class = load_json(cfg["idx_to_class"])
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    num_classes = len(idx_to_class)

    model = AircraftCNN(num_classes).to(DEVICE)
    state = torch.load(resolve_path(cfg["weights"]), map_location=DEVICE)
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
    
    # Check for vocab_bundle or word2idx
    vocab_key = "vocab_bundle" if "vocab_bundle" in cfg else "word2idx"
    vocab_path = cfg.get(vocab_key, cfg.get("word2idx"))
    
    if not resolve_path(vocab_path).exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}. Please export it from your training notebook.")
    if not resolve_path(cfg["id2label"]).exists():
        raise FileNotFoundError(f"Label file not found: {cfg['id2label']}. Please export it from your training notebook.")
    if not resolve_path(cfg["weights"]).exists():
        raise FileNotFoundError(f"Model file not found: {cfg['weights']}. Please export it from your training notebook.")
    
    if vocab_key == "vocab_bundle":
        vocab_data = load_vocab_bundle(vocab_path)
        word2idx = vocab_data.get("word2idx", {})
    else:
        word2idx = load_json(vocab_path)
    
    id2label = load_json(cfg["id2label"])
    if isinstance(id2label, dict) and "id2label" in id2label:
        id2label = id2label["id2label"]
    id2label = {int(k): v for k, v in id2label.items()}

    state = torch.load(resolve_path(cfg["weights"]), map_location=DEVICE)
    if isinstance(state, dict) and "model_state" in state and isinstance(state["model_state"], dict):
        state_dict = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state_dict = state["state_dict"]
    else:
        state_dict = state

    vocab_size = max(word2idx.values()) + 1 if word2idx else 1
    num_classes = len(id2label)

    embedding_weight = state_dict.get("embedding.weight")
    fc_weight = state_dict.get("fc.weight")
    if embedding_weight is not None:
        vocab_size = int(embedding_weight.shape[0])
    if fc_weight is not None:
        num_classes = int(fc_weight.shape[0])

    if len(id2label) != num_classes:
        id2label = {i: id2label.get(i, f"class_{i}") for i in range(num_classes)}

    is_lstm = any(k.startswith("lstm.") or ".lstm." in k for k in state_dict.keys())
    if is_lstm:
        weight_ih = state_dict.get("lstm.weight_ih_l0")
        if weight_ih is not None:
            embed_dim = int(weight_ih.shape[1])
            hidden_dim = int(weight_ih.shape[0] // 4)
        else:
            embed_dim = cfg["embed_dim"]
            hidden_dim = cfg["hidden_dim"]

        layer_indices = set()
        for k in state_dict.keys():
            if k.startswith("lstm.weight_ih_l"):
                suffix = k.split("lstm.weight_ih_l", 1)[1]
                idx = "".join(ch for ch in suffix if ch.isdigit())
                if idx != "":
                    layer_indices.add(int(idx))
        num_layers = max(layer_indices) + 1 if layer_indices else 2

        model = TextLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=cfg.get("dropout", 0.3),
            num_layers=num_layers,
        ).to(DEVICE)
    else:
        weight_ih = state_dict.get("gru.weight_ih_l0")
        if weight_ih is not None:
            embed_dim = int(weight_ih.shape[1])
            hidden_dim = int(weight_ih.shape[0] // 3)
        else:
            embed_dim = cfg["embed_dim"]
            hidden_dim = cfg["hidden_dim"]

        layer_indices = set()
        for k in state_dict.keys():
            if k.startswith("gru.weight_ih_l"):
                suffix = k.split("gru.weight_ih_l", 1)[1]
                idx = "".join(ch for ch in suffix if ch.isdigit())
                if idx != "":
                    layer_indices.add(int(idx))
        num_layers = max(layer_indices) + 1 if layer_indices else 1

        model = TextRNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=cfg.get("dropout", 0.3),
            num_layers=num_layers,
        ).to(DEVICE)

    model.load_state_dict(state_dict)
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
