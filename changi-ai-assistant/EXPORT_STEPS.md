# 📋 STEP-BY-STEP EXPORT GUIDE

## ❓ Why do I need to export the model?

Your RNN model was trained in the Colab notebook, but it's only saved in Colab's memory. To use it in the Streamlit app on your computer, you need to:
1. **Save** it to a file in Colab
2. **Download** it to your computer
3. **Place** it in the project folder

---

## 🔴 **STEP 1: Open Your Notebook**

Go to: https://colab.research.google.com/

Find and open: `CVNL (1).ipynb`

---

## 🟠 **STEP 2: Find the Export Cell**

Scroll down to the **very end** of your notebook.

Look for a cell that starts with:
```
# QUICK EXPORT - Run this cell to export your trained model
```

**This should be near cell 4 or 5.**

---

## 🟡 **STEP 3: Run the Export Cell**

1. Click on the cell to select it
2. Click the **▶️ Play/Run button** on the left
3. Wait for it to complete (should be instant)

You should see output like:
```
✓ Model saved: model_export/rnn_intent.pt
✓ Vocab saved: model_export/word2idx_intent.json
✓ Labels saved: model_export/id_to_label_intent.json

📦 EXPORT COMPLETE!
```

---

## 🟢 **STEP 4: Open File Browser in Colab**

On the **LEFT SIDEBAR**, click the **folder icon** 📁

---

## 🔵 **STEP 5: Find the model_export Folder**

You should see a `model_export` folder.

Click the **expand arrow ▶️** next to it.

You should see 3 files inside:
```
model_export/
├── rnn_intent.pt
├── word2idx_intent.json
└── id_to_label_intent.json
```

---

## 🟣 **STEP 6: Download Each File**

For each file:
1. Right-click on the filename
2. Click **"Download"**
3. Wait for it to save to your computer

**Files to download:**
- ✅ rnn_intent.pt
- ✅ word2idx_intent.json  
- ✅ id_to_label_intent.json

---

## 📍 **STEP 7: Place Files in Your Project**

After downloading, copy each file to the correct location:

### File 1: rnn_intent.pt
```
From:  Downloads\rnn_intent.pt
To:    C:\Documents\Year 2 Sem 2\CVNL\Assignment\
       CVNL_Assignment_GP03\changi-ai-assistant\models\rnn_intent.pt
```

### File 2: word2idx_intent.json
```
From:  Downloads\word2idx_intent.json
To:    C:\Documents\Year 2 Sem 2\CVNL\Assignment\
       CVNL_Assignment_GP03\changi-ai-assistant\label_maps\word2idx_intent.json
```

### File 3: id_to_label_intent.json
```
From:  Downloads\id_to_label_intent.json
To:    C:\Documents\Year 2 Sem 2\CVNL\Assignment\
       CVNL_Assignment_GP03\changi-ai-assistant\label_maps\id_to_label_intent.json
```

---

## ✅ **STEP 8: Test It Works**

1. Go to: http://localhost:8501
2. Go to the **"💬 Text Classification"** tab
3. From dropdown, select **"Intent Classification"**
4. Type a test message: `"Is the wifi working?"`
5. Click **"Run RNN Prediction"**

**You should see:** `airport_directions` ✅

If you see this, **you're done!** 🎉

---

## 🆘 **Still Not Working?**

**Check 1:** Files are in the right place
```
changi-ai-assistant/
├── models/
│   └── rnn_intent.pt ✅
└── label_maps/
    ├── word2idx_intent.json ✅
    └── id_to_label_intent.json ✅
```

**Check 2:** File names are EXACTLY correct (case-sensitive)

**Check 3:** Reload the Streamlit app
- Press `Ctrl+C` in the terminal running Streamlit
- Run: `streamlit run app/streamlit_app.py`
- Reload the web page (F5)

**Check 4:** Make sure Streamlit is still running
- Terminal should show: `You can now view your Streamlit app in your browser`

If still stuck, check the error message at the bottom of the Streamlit page.
