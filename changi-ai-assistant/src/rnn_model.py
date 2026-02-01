import torch
import torch.nn as nn

class TextRNN(nn.Module):
    """BiGRU Intent Classifier - matches training notebook architecture"""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)
        _, h = self.gru(emb)
        # Concatenate both directions of final hidden state
        hidden = torch.cat([h[0], h[1]], dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)


class TextLSTM(nn.Module):
    """BiLSTM Sentiment Classifier - matches sentiment training notebook architecture"""
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        hidden = torch.cat([h[-2], h[-1]], dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

