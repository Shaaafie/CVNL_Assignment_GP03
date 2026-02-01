import torch
import torch.nn as nn
import torch.optim as optim
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==========================================
# 1. THE MODEL CLASS
# ==========================================
class ImprovedSentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Bidirectional LSTM as defined in your notebook
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        # hidden_dim * 2 because of bidirectional
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        dropped = self.dropout(hidden_cat)
        return self.fc(dropped)

# ==========================================
# 2. PREPROCESSING HELPERS
# ==========================================
def clean_tweet(tweet):
    """Exact cleaning logic from your assignment"""
    if not isinstance(tweet, str): return ""
    tweet = tweet.lower()
    tweet = re.sub(r'@\w+s*', '', tweet)        # Remove mentions
    tweet = re.sub(r'http\S+', '', tweet)       # Remove URLs
    tweet = re.sub(r'[^a-zA-Z\s]', ' ', tweet)  # Keep only letters
    tweet = re.sub(r'\s+', ' ', tweet).strip()  # Remove whitespace
    return tweet

def text_to_sequence(text, vocab, max_len=30):
    """Converts text to padded sequences"""
    tokens = text.split()
    # 0 is PAD, 1 is UNK based on your build_vocab logic
    seq = [vocab.get(token, 1) for token in tokens[:max_len]]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    return seq

# ==========================================
# 3. DATASET CLASS
# ==========================================
class TweetDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=30):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        sequence = text_to_sequence(text, self.vocab, self.max_len)
        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ==========================================
# 4. VISUALIZATION (YOUR 3 GRAPHS)
# ==========================================
def plot_history(history):
    """Generates the Loss and Accuracy graphs"""
    plt.figure(figsize=(12, 4))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['train_acc'], label='Train Acc', marker='o')
    plt.plot(history['epoch'], history['val_acc'], label='Val Acc', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def evaluate_model_performance(all_labels, all_preds):
    """
    Generates the Classification Report and the Confusion Matrix Heatmap.
    This fulfills the requirement for the final evaluation visual.
    """
    # 1. Print the text report
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=['Negative', 'Neutral', 'Positive']))

    # 2. Generate the Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
# ==========================================
# 5. INTEGRATION HELPERS
# ==========================================
def save_assets(model, vocab, model_name="sentiment_model.pt"):
    """Saves weights, word2idx, and id_to_label for the group"""
    torch.save(model.state_dict(), model_name)
    
    with open('word2idx_sentimentRNN.json', 'w') as f:
        json.dump(vocab, f)
        
    id_to_label = {0: "negative", 1: "neutral", 2: "positive"}
    with open('id_to_label_sentimentRNN.json', 'w') as f:
        json.dump(id_to_label, f)
    
    print(f"✅ Assets saved: {model_name}, word2idx_sentimentRNN.json, id_to_label_sentimentRNN.json")