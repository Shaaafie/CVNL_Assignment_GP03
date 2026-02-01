# ✅ Integration Checklist

## Stage 1: Export Model from Notebook
- [ ] Open `CVNL (1).ipynb` in Colab
- [ ] Find the "QUICK EXPORT" cell (around cell 4)
- [ ] Run the cell
- [ ] Wait for "EXPORT COMPLETE!" message
- [ ] See `model_export/` folder created in Colab

## Stage 2: Download Files
- [ ] Click folder icon in Colab (left sidebar)
- [ ] Expand `model_export/` folder
- [ ] Download `rnn_intent.pt`
- [ ] Download `word2idx_intent.json`
- [ ] Download `id_to_label_intent.json`

## Stage 3: Place in Local Project
- [ ] Copy `rnn_intent.pt` to:
  ```
  C:\...\CVNL_Assignment_GP03\changi-ai-assistant\models\rnn_intent.pt
  ```

- [ ] Copy `word2idx_intent.json` to:
  ```
  C:\...\CVNL_Assignment_GP03\changi-ai-assistant\label_maps\word2idx_intent.json
  ```

- [ ] Copy `id_to_label_intent.json` to:
  ```
  C:\...\CVNL_Assignment_GP03\changi-ai-assistant\label_maps\id_to_label_intent.json
  ```

## Stage 4: Test Integration
- [ ] Reload Streamlit app (F5) or restart:
  ```
  streamlit run app/streamlit_app.py
  ```
- [ ] Go to Text Classification tab
- [ ] Select "Intent Classification"
- [ ] Enter a test message: "Is the wifi working?"
- [ ] Click "Run RNN Prediction"
- [ ] See prediction: "airport_directions" ✅

---

## 📋 Files Status

**CNN Models:** ✅ Ready
- aircraftcnn_family_best.pth

**ResNet Models:** ✅ Ready
- resnet_manufacturer.pt
- resnet_airline.pt

**RNN Models:** ⏳ Pending Export
- [ ] rnn_intent.pt
- [ ] word2idx_intent.json
- [ ] id_to_label_intent.json

---

## 🎯 Final Check

After completing all stages, your project should have:
```
changi-ai-assistant/
├── models/
│   ├── aircraftcnn_family_best.pth     ✅
│   ├── resnet_manufacturer.pt          ✅
│   ├── resnet_airline.pt               ✅
│   └── rnn_intent.pt                   ← Add this
└── label_maps/
    ├── idx_to_class_aircraft_family.json ✅
    ├── word2idx_intent.json            ← Add this
    └── id_to_label_intent.json         ← Add this
```

Once all 3 RNN files are in place, the app will be fully functional! 🚀
