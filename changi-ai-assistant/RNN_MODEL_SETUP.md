# Instructions to Export RNN Models

## You need to run the save cell in your training notebook to export:

1. **Model weights**: `rnn_intent.pt`
2. **Vocabulary**: `word2idx_intent.json` (currently exists but check if it matches your training)
3. **Label mapping**: `id_to_label_intent.json` (currently exists but check if it matches your training)

## Steps:

1. Open your training notebook: `CVNL (1).ipynb`
2. Run cell 2 (the save cell I added) - it will export to your Google Drive
3. Download the files from Colab:
   - `/content/drive/MyDrive/Colab Notebooks/Asg1/output/rnn_intent.pt`
   - `/content/drive/MyDrive/Colab Notebooks/Asg1/output/word2idx_intentRNN.json`
   - `/content/drive/MyDrive/Colab Notebooks/Asg1/output/id_to_label_intentRNN.json`

4. Place them in the project:
   - `rnn_intent.pt` → `models/`
   - Rename and place JSON files in `label_maps/`:
     - `word2idx_intentRNN.json` → `word2idx_intent.json` (replace existing if needed)
     - `id_to_label_intentRNN.json` → `id_to_label_intent.json` (replace existing if needed)

## Current Status:

✅ CNN Model: Working (aircraftcnn_family_best.pth)
✅ ResNet Models: Working (resnet_manufacturer.pt, resnet_airline.pt)
❌ RNN Intent Model: Missing `rnn_intent.pt`
⚠️  RNN Sentiment Model: Has vocab files but missing `rnn_sentiment.pt`

## Note:
The app will now show a clear error message if you try to use RNN models before exporting them from the notebook.
