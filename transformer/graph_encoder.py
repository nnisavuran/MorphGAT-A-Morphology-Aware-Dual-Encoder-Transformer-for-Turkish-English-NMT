# transformer/graph_encoder.py
# DÜZELTME: scatter_mean fonksiyonunda `out` parametresi artık gerçekten kullanılıyor.
#   Eski: `out` argümanı geçilse de fonksiyon onu yok sayıyor, kendi sıfır
#         tensörünü oluşturuyordu (gereksiz bellek tahsisi + yanıltıcı kod).
#   Yeni: `out` verilmişse sıfırlanarak doğrudan kullanılıyor.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data, Batch

from .embed import PositionalEncoder
from .sublayer import Norm


def scatter_mean(src, index, dim=0, out=None, dim_size=None):
    """
    src   : [N, d]   — düğüm/token özellik vektörleri
    index : [N]      — her düğümün hangi kelimeye ait olduğunu gösteren tamsayı endeks
    out   : [K, d]   — varsa, sonuç bu tensöre yazılır (K = kelime sayısı)
    """
    # dim_size'ı belirle
    if dim_size is None:
        dim_size = out.size(0) if out is not None else int(index.max().item()) + 1

    expanded_index = index.unsqueeze(-1).expand(-1, src.size(-1))  # [N, d]

    # *** DÜZELTME: `out` verilmişse sıfırla ve kullan; yoksa yeni tensör oluştur ***
    if out is not None:
        res = out
        res.zero_()
    else:
        res = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)

    res.scatter_add_(0, expanded_index, src)

    # Ortalama için sayaç
    count = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    count.scatter_add_(0, expanded_index, torch.ones_like(src))

    return res / count.clamp(min=1.0)


class GraphEncoder(nn.Module):
    def __init__(self, morph_vocab_size, d_model, heads, dropout=0.1, padding_idx=0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(morph_vocab_size, d_model, padding_idx=padding_idx)

        self.gat1 = GATv2Conv(d_model, d_model, heads=heads, concat=True,  dropout=dropout)
        self.gat2 = GATv2Conv(d_model * heads, d_model, heads=1, concat=False, dropout=dropout)

        self.norm = Norm(d_model)
        self.pe   = PositionalEncoder(d_model, dropout=dropout)

    def forward(self, graph_batch, expected_batch_size=None):
        device = graph_batch.x.device

        if isinstance(graph_batch, Data):
            x_ids               = graph_batch.x
            edge_index          = graph_batch.edge_index
            word_indices        = graph_batch.word_batch
            sentence_map        = torch.zeros(x_ids.size(0), dtype=torch.long, device=device)
            num_graphs          = 1
            output_batch_size   = expected_batch_size if expected_batch_size is not None else 1

        elif isinstance(graph_batch, Batch):
            x_ids               = graph_batch.x
            edge_index          = graph_batch.edge_index
            word_indices        = graph_batch.word_batch
            sentence_map        = graph_batch.batch
            num_graphs          = int(sentence_map.max().item()) + 1
            output_batch_size   = expected_batch_size if expected_batch_size is not None else num_graphs

        else:
            raise TypeError("graph_batch; torch_geometric Data veya Batch nesnesi olmalıdır.")

        # Boş graf kontrolü
        if x_ids.numel() == 0 or edge_index.numel() == 0 or word_indices.numel() == 0:
            empty_out  = torch.zeros(output_batch_size, 1, self.d_model, device=device)
            empty_mask = torch.zeros(output_batch_size, 1, dtype=torch.bool, device=device)
            return empty_out, empty_mask

        # --- GNN işleme ---
        x = self.embedding(x_ids.squeeze(-1)) * (self.d_model ** 0.5)
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)

        # --- Kelime düzeyine pooling ---
        word_level_outputs_list   = []
        word_level_batch_map_list = []

        for graph_idx in torch.unique(sentence_map):
            node_mask  = sentence_map == graph_idx
            node_feats = x[node_mask]
            word_ids   = word_indices[node_mask]

            if node_feats.numel() == 0:
                continue

            max_word_id = int(word_ids.max().item())
            out_buf     = torch.zeros(max_word_id + 1, self.d_model, device=device)
            pooled      = scatter_mean(node_feats, word_ids, dim=0, out=out_buf)

            word_level_outputs_list.append(pooled)
            word_level_batch_map_list.append(
                torch.full((pooled.size(0),), graph_idx, dtype=torch.long, device=device)
            )

        if not word_level_outputs_list:
            empty_out  = torch.zeros(output_batch_size, 1, self.d_model, device=device)
            empty_mask = torch.zeros(output_batch_size, 1, dtype=torch.bool, device=device)
            return empty_out, empty_mask

        word_flat      = torch.cat(word_level_outputs_list,   dim=0)
        batch_flat     = torch.cat(word_level_batch_map_list, dim=0)
        padded, mask   = to_dense_batch(word_flat, batch_flat, batch_size=output_batch_size)

        # PE ve normalizasyon (PositionalEncoder'daki düzeltme burayı da kapsar)
        out = self.norm(self.pe(padded))
        return out, mask