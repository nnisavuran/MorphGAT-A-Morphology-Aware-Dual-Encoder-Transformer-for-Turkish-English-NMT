# transformer/graph_encoder.py
#
# İki graf kodlayıcı tipi:
#   gnn_type="gatv2" → GATv2Conv (ana model, attention-tabanlı)
#   gnn_type="gcn"   → GCNConv   (ablation: attention yok)
#
# GCN ablation'ı "attention gerçekten gerekli mi?" sorusunu test eder.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data, Batch

from .embed import PositionalEncoder
from .sublayer import Norm


def scatter_mean(src, index, dim=0, out=None, dim_size=None):
    if dim_size is None:
        dim_size = out.size(0) if out is not None else int(index.max().item()) + 1
    expanded_index = index.unsqueeze(-1).expand(-1, src.size(-1))
    if out is not None:
        res = out
        res.zero_()
    else:
        res = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    res.scatter_add_(0, expanded_index, src)
    count = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    count.scatter_add_(0, expanded_index, torch.ones_like(src))
    return res / count.clamp(min=1.0)


class GraphEncoder(nn.Module):
    def __init__(self, morph_vocab_size, d_model, heads, dropout=0.1,
                 padding_idx=0, gnn_type="gatv2"):
        super().__init__()
        self.d_model  = d_model
        self.gnn_type = gnn_type
        self.embedding = nn.Embedding(morph_vocab_size, d_model, padding_idx=padding_idx)

        if gnn_type == "gatv2":
            # Ana model — attention-tabanlı
            self.conv1 = GATv2Conv(d_model, d_model, heads=heads, concat=True,  dropout=dropout)
            self.conv2 = GATv2Conv(d_model * heads, d_model, heads=1, concat=False, dropout=dropout)
        elif gnn_type == "gcn":
            # Ablation — attention YOK, basit komşu toplama
            self.conv1 = GCNConv(d_model, d_model)
            self.conv2 = GCNConv(d_model, d_model)
        else:
            raise ValueError(f"Bilinmeyen gnn_type: {gnn_type}")

        self.norm = Norm(d_model)
        self.pe   = PositionalEncoder(d_model, dropout=dropout)

    def forward(self, graph_batch, expected_batch_size=None):
        device = graph_batch.x.device

        if isinstance(graph_batch, Data):
            x_ids        = graph_batch.x
            edge_index   = graph_batch.edge_index
            word_indices = graph_batch.word_batch
            sentence_map = torch.zeros(x_ids.size(0), dtype=torch.long, device=device)
            num_graphs        = 1
            output_batch_size = expected_batch_size if expected_batch_size is not None else 1
        elif isinstance(graph_batch, Batch):
            x_ids        = graph_batch.x
            edge_index   = graph_batch.edge_index
            word_indices = graph_batch.word_batch
            sentence_map = graph_batch.batch
            num_graphs        = int(sentence_map.max().item()) + 1
            output_batch_size = expected_batch_size if expected_batch_size is not None else num_graphs
        else:
            raise TypeError("graph_batch; torch_geometric Data veya Batch nesnesi olmalıdır.")

        if x_ids.numel() == 0 or edge_index.numel() == 0 or word_indices.numel() == 0:
            empty_out  = torch.zeros(output_batch_size, 1, self.d_model, device=device)
            empty_mask = torch.zeros(output_batch_size, 1, dtype=torch.bool, device=device)
            return empty_out, empty_mask

        # --- GNN işleme (tip bağımsız) ---
        x = self.embedding(x_ids.squeeze(-1)) * (self.d_model ** 0.5)
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

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

        word_flat    = torch.cat(word_level_outputs_list,   dim=0)
        batch_flat   = torch.cat(word_level_batch_map_list, dim=0)
        padded, mask = to_dense_batch(word_flat, batch_flat, batch_size=output_batch_size)

        out = self.norm(self.pe(padded))
        return out, mask
