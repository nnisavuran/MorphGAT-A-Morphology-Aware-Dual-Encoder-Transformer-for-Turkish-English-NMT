# transformer/embed.py
# DÜZELTME: PositionalEncoder'da batch-first uyumsuzluk hatası giderildi.
#   Eski: pe şekli [max_len, 1, d_model] → x.size(0) batch size'ı döndürüyor,
#         tüm pozisyonlar aynı PE değerini alıyordu (kritik hata).
#   Yeni: pe şekli [1, max_len, d_model] → x.size(1) ile seq_len indexleniyor,
#         her pozisyon doğru PE değerini alıyor.

import torch
import torch.nn as nn
import math


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx=0):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)          # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # *** DÜZELTME: [1, max_len, d_model] — batch_first tensörlerle uyumlu ***
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x  : [B, seq_len, d_model]
        # pe : [1, max_len, d_model]
        # pe[:, :seq_len, :] → [1, seq_len, d_model]
        #   broadcast ile → [B, seq_len, d_model]  ✓
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)