# 🎯 Quick Start: Export RNN Model & Integrate

## Step 1: Run the Export Cell in Your Notebook

Open: `c:\Users\shafa\Downloads\CVNL (1).ipynb`

**Find and run cell 2** (titled "QUICK EXPORT - Run this cell to export your trained model")

This will create a `model_export/` folder with 3 files:
```
model_export/
├── rnn_intent.pt              ← Model weights
├── word2idx_intent.json       ← Vocabulary
└── id_to_label_intent.json    ← Labels
```

## Step 2: Download Files from Colab

1. Click the folder icon on the left in Colab
2. Expand `model_export/` folder
3. Download all 3 files to your computer

## Step 3: Place Files in Your Project

Copy the downloaded files to your local project:

```
C:\Documents\Year 2 Sem 2\CVNL\Assignment\CVNL_Assignment_GP03\
└── changi-ai-assistant\
    ├── models\
    │   └── rnn_intent.pt              ← Place here
    └── label_maps\
        ├── word2idx_intent.json       ← Place here
        └── id_to_label_intent.json    ← Place here
```

## Step 4: Refresh the App

1. Go to http://localhost:8501
2. If Streamlit crashed, restart with: `streamlit run app/streamlit_app.py`
3. Try the RNN model - it should now work! ✅

---

## 🆘 Troubleshooting

**Files still not found?**
- Make sure filenames match EXACTLY (case-sensitive on some systems)
- Reload the Streamlit page with F5
- Check the file paths exist

**Can't find cell 2?**
- Search for "QUICK EXPORT" in your notebook
- Or scroll to the end of the notebook - it should be near the bottom

**Still having issues?**
- Make sure the Streamlit app is running
- Check that files are in the correct directories
- Try restarting the app: Press Ctrl+C in terminal, then run streamlit again
