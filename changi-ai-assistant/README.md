# ✈️ Changi Airport AI Assistant

Multi-model prototype combining CNN and RNN models for aircraft recognition and passenger message analysis.

## 🎯 Features

### CNN Models (Image Classification)
- **Aircraft Family Classification**: Identifies aircraft types (A320, Boeing 737, etc.) - Custom CNN
- **Manufacturer Classification**: Recognizes aircraft manufacturers - Custom CNN

### ResNet Models (Transfer Learning)
- **Manufacturer (ResNet)**: ResNet-based manufacturer classification
- **Airline (ResNet)**: ResNet-based airline identification

### RNN Models (Text Classification)
- **Intent Classification**: Categorizes passenger queries (flight info, directions, baggage, etc.) - BiGRU
- **Sentiment Analysis**: Analyzes sentiment of passenger feedback - BiGRU

## 📁 Project Structure

```
changi-ai-assistant/
├── app/
│   └── streamlit_app.py       # Streamlit web interface
├── src/
│   ├── aircraft_family_cnn.py # CNN architecture
│   ├── rnn_model.py           # BiGRU + BiLSTM architectures
│   ├── resnet_model.py        # ResNet loader
│   └── inference.py           # Model loading & prediction
├── models/                     # Model files (.pth, .pt)
│   ├── aircraftcnn_family_best.pth
│   ├── resnet_manufacturer.pt
│   ├── resnet_airline.pt
│   ├── RNN_Intent_Classifications.pth
│   └── best_SentimentRNN_model.pth (optional)
├── label_maps/                 # Label mapping files
│   ├── idx_to_class_aircraft_family.json
│   ├── rnn_vocab_bundle.pkl
│   ├── intent10_label_map.json
│   ├── word2idx_sentimentRNN.json
│   └── id_to_label_sentimentRNN.json
└── requirements.txt

```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Model Files

After training your models in the notebook, transfer these files:

**From Colab Training (Intent RNN):**
- `RNN_Intent_Classifications.pth` → `models/`
- `rnn_vocab_bundle.pkl` → `label_maps/`
- `intent10_label_map.json` → `label_maps/`

**From Colab Training (Sentiment RNN - Optional):**
- `best_SentimentRNN_model.pth` → `models/`
- `word2idx_sentimentRNN.json` → `label_maps/`
- `id_to_label_sentimentRNN.json` → `label_maps/`

**CNN Models:**
- `aircraftcnn_family_best.pth` → `models/` (already present)
- Other CNN/RNN models as needed

### 3. Run the Application

**Windows PowerShell:**

1) Go to the project folder:

```
cd "C:\Documents\Year 2 Sem 2\CVNL\Assignment\CVNL_Assignment_GP03\changi-ai-assistant"
```

2) Start the app:

```
streamlit run "C:\Documents\Year 2 Sem 2\CVNL\Assignment\CVNL_Assignment_GP03\changi-ai-assistant\app\streamlit_app.py"
```

The app will open in your browser at `http://localhost:8501`

## 💻 Usage

### CNN - Aircraft Recognition
1. Select classification task (Aircraft Family or Manufacturer)
2. Upload an aircraft image (JPG/PNG)
3. Click "Analyze Image" to get predictions

### RNN - Text Analysis
1. Select analysis task (Intent Classification or Sentiment)
2. Enter passenger message or use quick examples
3. Click "Analyze Text" to get prediction and confidence

## 📊 Model Details

### Custom CNN Architecture
- 5 convolutional blocks with batch normalization
- Global average pooling
- Dropout for regularization
- Input: 224x224 RGB images

### ResNet Architecture (Transfer Learning)
- ResNet18/34/50 variants
- Pre-trained on ImageNet
- Fine-tuned for aircraft classification
- Input: 224x224 RGB images
- ImageNet normalization

### RNN Architecture (BiGRU)
- Bidirectional GRU layers
- Embedding dimension: 128
- Hidden dimension: 256 (Intent) / 128 (Sentiment)
- Max sequence length: 60 (Intent) / 40 (Sentiment)
- Dropout: 0.3

## 🔧 Integration Steps for Teammates

1. **Train your model** using the provided notebook
2. **Save model artifacts** (run the save cell in notebook)
3. **Transfer files** from Colab to local project:
   ```
   models/your_model.pt
   label_maps/your_vocab.json
   label_maps/your_labels.json
   ```
4. **Update configs** in `src/inference.py` if needed
5. **Test integration** by running the Streamlit app

## 📝 Example Queries

**Intent Classification:**
- "Is the wifi working in terminal 4?" → airport_directions
- "I lost my passport at immigration" → special_requests
- "Flight SQ321 delayed?" → flight_info

**Sentiment Analysis:**
- "Great service and friendly staff!" → positive
- "Terrible experience, long queues" → negative

## 🐛 Troubleshooting

**ModuleNotFoundError: No module named 'src'**
- Make sure you're running from the project root directory

**Model file not found**
- Check that `.pth`/`.pt` files are in `models/` directory
- Verify file names match those in `src/inference.py`

**JSON decode error in label maps**
- Ensure label map files are valid JSON
- Check that keys are properly quoted strings

## 👥 Team Members

CVNL Assignment - Group 03

## 📄 License

For educational purposes only.
