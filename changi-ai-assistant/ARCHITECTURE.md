# 🏗️ CHANGI AI ASSISTANT - ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB INTERFACE                       │
│                    http://localhost:8501                        │
│                                                                 │
│   ┌──────────────────────────┬──────────────────────────────┐  │
│   │  🖼️ IMAGE CLASSIFICATION │  💬 TEXT CLASSIFICATION     │  │
│   │                          │                              │  │
│   │ • Upload Image           │ • Enter Text                │  │
│   │ • Select Model           │ • Select Task               │  │
│   │ • Get Top-5              │ • Get Prediction            │  │
│   │   Predictions            │                              │  │
│   └──────────────────────────┴──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
           ↓                               ↓
    ┌──────────────┐            ┌──────────────┐
    │   INFERENCE  │            │   INFERENCE  │
    │   MODULE     │            │   MODULE     │
    │              │            │              │
    │ • Load Model │            │ • Load Model │
    │ • Preprocess │            │ • Tokenize   │
    │ • Predict    │            │ • Predict    │
    │ • Cache      │            │ • Cache      │
    └──────────────┘            └──────────────┘
           ↓                               ↓
    ┌──────────────────────┐      ┌──────────────────────┐
    │   MODEL LAYER        │      │   MODEL LAYER        │
    │                      │      │                      │
    │ ┌─────────────────┐  │      │ ┌─────────────────┐  │
    │ │ 1. Custom CNN   │  │      │ │ 4. BiGRU RNN    │  │
    │ │    (PyTorch)    │  │      │ │    (PyTorch)    │  │
    │ └─────────────────┘  │      │ └─────────────────┘  │
    │ ┌─────────────────┐  │      │ ┌─────────────────┐  │
    │ │ 2. ResNet-18/34 │  │      │ │ 5. BiGRU RNN    │  │
    │ │    (Transfer)   │  │      │ │    (Optional)   │  │
    │ └─────────────────┘  │      │ └─────────────────┘  │
    │ ┌─────────────────┐  │      └──────────────────────┘
    │ │ 3. ResNet-50    │  │
    │ │    (Transfer)   │  │      ┌──────────────────────┐
    │ └─────────────────┘  │      │ LABEL MAPPING        │
    │                      │      │                      │
    │ Status: ✅ Ready     │      │ • word2idx.json      │
    └──────────────────────┘      │ • id2label.json      │
          ↓                       │                      │
    ┌──────────────────────┐      │ Status: ⏳ Pending   │
    │ MODEL FILES          │      └──────────────────────┘
    │ (local storage)       │
    │                      │      ┌──────────────────────┐
    │ ✅ aircraftcnn_...   │      │ WORD EMBEDDINGS      │
    │ ✅ resnet_manu...    │      │                      │
    │ ✅ resnet_air...     │      │ • Vocab (60K words)  │
    │ ⏳ rnn_intent.pt     │      │                      │
    │                      │      │ Status: ✅ Ready     │
    └──────────────────────┘      └──────────────────────┘
```

---

## 🔄 Data Flow Examples

### IMAGE CLASSIFICATION FLOW
```
User uploads image
        ↓
Streamlit frontend
        ↓
Image preprocessing (resize, normalize)
        ↓
PyTorch model (CNN/ResNet)
        ↓
Softmax probabilities
        ↓
Top-5 predictions
        ↓
Display on frontend ✅
```

### TEXT CLASSIFICATION FLOW (After RNN Export)
```
User enters text
        ↓
Streamlit frontend
        ↓
Tokenization (split into words)
        ↓
Word-to-index mapping (vocab lookup)
        ↓
Padding/truncating to max_len
        ↓
Embedding layer
        ↓
BiGRU layers (forward + backward)
        ↓
Final hidden state
        ↓
Softmax classification
        ↓
Display prediction ✅
```

---

## 📊 Model Details

### CNN Model
```
Input: RGB Image (224 × 224 × 3)
   ↓
5 Conv Blocks (32→64→128→256→512 channels)
   ↓
Global Average Pooling
   ↓
Dropout (0.2)
   ↓
Linear (512 → num_classes)
   ↓
Output: Class probabilities
```

### ResNet Model
```
Input: RGB Image (224 × 224 × 3)
   ↓
ResNet backbone (18/34/50)
   ↓
Residual connections
   ↓
Average pooling
   ↓
Linear layer (fine-tuned)
   ↓
Output: Class probabilities
```

### RNN (BiGRU) Model
```
Input: Token IDs [max_len]
   ↓
Embedding Layer (vocab_size × embed_dim)
   ↓
BiGRU (embed_dim → hidden_dim)
   ↓ (processes forward AND backward)
Concatenate directions
   ↓
Dropout
   ↓
Linear (hidden_dim × 2 → num_classes)
   ↓
Output: Class probabilities
```

---

## ⚡ Performance

- **CNN**: ~100ms per image
- **ResNet**: ~200ms per image  
- **RNN**: ~50ms per text (once loaded)
- **Model Loading**: First time 2-5 seconds (cached after)

---

## 🔐 Caching Strategy

```
First request:
Model disk → Load into RAM → Cache

Subsequent requests:
Use cached model (instant)

Per-session cache:
_CNN_CACHE = {}      ← Stores 4 CNN models
_RESNET_CACHE = {}   ← Stores 2 ResNet models
_RNN_CACHE = {}      ← Stores 2 RNN models
```

---

## 📈 Scalability

**Current:** Works on CPU + minimal GPU

**For production:**
- Use GPU (CUDA) for ~5x speedup
- Add batch processing
- Deploy with TorchServe or FastAPI
- Add caching layer (Redis)
- Use model quantization for smaller size

---

## 🎯 What Each Model Does

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| CNN | Aircraft photo | Family (A320, Boeing, etc.) | Identify aircraft type |
| ResNet Manu | Aircraft photo | Manufacturer | Know aircraft maker |
| ResNet Air | Aircraft photo | Airline | Identify airline |
| RNN Intent | Passenger message | Query type | Route to right service |
| RNN Sentiment | Passenger message | Feeling (pos/neg) | Gauge satisfaction |

---

## 🚀 Deployment Readiness

✅ **Ready for deployment:**
- Modular architecture
- Error handling
- Model versioning
- Clear interfaces
- User-friendly UI

⏳ **To add before production:**
- Authentication
- Rate limiting
- Logging
- Monitoring
- Model A/B testing
- Data validation
