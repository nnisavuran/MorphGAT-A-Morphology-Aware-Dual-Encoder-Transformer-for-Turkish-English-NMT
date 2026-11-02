import torch
import torch.nn as nn

from .sublayer import FeedForward, MultiHeadAttention, Norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    """Standart decoder layer — tek bellek (baseline / use_dual_encoder=False)."""
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class DualCrossDecoderLayer(nn.Module):
    """
    Çift çapraz dikkatli decoder layer (önerilen — use_dual_encoder=True).

    Standart decoder'dan farkı: tek bir cross-attention yerine İKİ ayrı
    cross-attention var:
      - attn_bpe   : BPE (yüzey/anlamsal) belleğine dikkat
      - attn_graph : graf (morfolojik) belleğine dikkat

    İki cross-attention da sırayla, artıklı bağlantılarla uygulanır. Böylece
    decoder her iki bilgi kaynağını softmax ile bağımsız tartar; gate veya
    manuel füzyona gerek kalmaz, magnitude uyumsuzluğu otomatik çözülür.
    """
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)   # öz-dikkat öncesi
        self.norm_2 = Norm(d_model)   # BPE cross-attn öncesi
        self.norm_3 = Norm(d_model)   # graf cross-attn öncesi
        self.norm_4 = Norm(d_model)   # ileri besleme öncesi

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)

        self.attn_self  = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_bpe   = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_graph = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, bpe_memory, graph_memory, src_mask, graph_mask, trg_mask):
        # 1. Maskeli öz-dikkat (hedef dizi)
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_self(x2, x2, x2, trg_mask))

        # 2. BPE belleğine çapraz dikkat
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_bpe(x2, bpe_memory, bpe_memory, src_mask))

        # 3. Graf belleğine çapraz dikkat
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.attn_graph(x2, graph_memory, graph_memory, graph_mask))

        # 4. İleri besleme
        x2 = self.norm_4(x)
        x = x + self.dropout_4(self.ff(x2))
        return x


class ParallelCrossDecoderLayer(nn.Module):
    """Paralel çift çapraz dikkat (H1-#4 / H2-#6 ablation).
    Seri H2->H3 yerine iki cross-attn AYNI girdiden hesaplanıp toplanır."""
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)   # öz-dikkat öncesi
        self.norm_2 = Norm(d_model)   # paralel cross-attn öncesi (ortak girdi)
        self.norm_4 = Norm(d_model)   # ileri besleme öncesi

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_4 = nn.Dropout(dropout)

        self.attn_self  = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_bpe   = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_graph = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, bpe_memory, graph_memory, src_mask, graph_mask, trg_mask):
        # 1. Maskeli öz-dikkat
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_self(x2, x2, x2, trg_mask))
        # 2. İKİ cross-attn AYNI girdiden -> topla (paralel)
        x2 = self.norm_2(x)
        a_bpe   = self.attn_bpe(x2, bpe_memory, bpe_memory, src_mask)
        a_graph = self.attn_graph(x2, graph_memory, graph_memory, graph_mask)
        x = x + self.dropout_2(a_bpe + a_graph)
        # 3. İleri besleme
        x2 = self.norm_4(x)
        x = x + self.dropout_4(self.ff(x2))
        return x