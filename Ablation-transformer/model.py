# transformer/model.py
#
# Modlar (fusion parametresi):
#   use_dual_encoder=False              → Standart Transformer (baseline)
#   fusion="cross_attn"                 → Çift çapraz dikkat (ANA MODEL)
#   fusion="gate"                       → Gated füzyon (ablation)
#   fusion="additive"                   → Toplama füzyonu (ablation)     ← YENİ
#   fusion="concat_cross"              → Birleştirilmiş tek cross-attn (ablation) ← YENİ
#
# gnn_type parametresi graph encoder'a geçer: "gatv2" (ana) | "gcn" (ablation)

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer  import EncoderLayer, DecoderLayer, DualCrossDecoderLayer
from .embed  import Embedder, PositionalEncoder
from .sublayer import Norm
from .graph_encoder import GraphEncoder


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def create_masks(src_bpe, trg_bpe, pad_idx, device):
    src_mask = (src_bpe != pad_idx).unsqueeze(1).unsqueeze(2) if src_bpe is not None else None
    if trg_bpe is not None:
        trg_pad_mask = (trg_bpe != pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len      = trg_bpe.size(1)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
        trg_mask     = trg_pad_mask & trg_sub_mask
    else:
        trg_mask = None
    return src_mask, trg_mask


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        max_len,
        heads,
        dropout,
        N,
        morph_vocab_size,
        use_dual_encoder=True,
        fusion="cross_attn",          # "cross_attn" | "gate" | "additive" | "concat_cross"
        gnn_type="gatv2",             # "gatv2" | "gcn"
        modality_dropout_prob=0.00,
    ):
        super().__init__()
        self.d_model               = d_model
        self.use_dual_encoder      = use_dual_encoder
        self.fusion                = fusion
        self.modality_dropout_prob = modality_dropout_prob

        self.bpe_embed = Embedder(vocab_size, d_model, padding_idx=0)
        self.pe        = PositionalEncoder(d_model, dropout=dropout, max_len=max_len)

        self.encoder_bpe  = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.encoder_norm = Norm(d_model)

        if use_dual_encoder:
            self.graph_encoder = GraphEncoder(
                morph_vocab_size, d_model, heads, dropout=dropout,
                padding_idx=0, gnn_type=gnn_type
            )
            if fusion == "gate":
                self.fusion_gate = nn.Linear(d_model * 2, d_model)
                nn.init.constant_(self.fusion_gate.bias, 1.0)
                nn.init.xavier_uniform_(self.fusion_gate.weight, gain=0.1)

        # Decoder katman tipi — cross_attn ve concat_cross farklı decoder kullanır
        if use_dual_encoder and fusion == "cross_attn":
            self.decoder_layers = get_clones(DualCrossDecoderLayer(d_model, heads, dropout), N)
        else:
            # baseline, gate, additive, concat_cross → standart tek-bellek decoder
            self.decoder_layers = get_clones(DecoderLayer(d_model, heads, dropout), N)

        self.norm = Norm(d_model)
        self.out  = nn.Linear(d_model, vocab_size)

    # ------------------------------------------------------------------
    def _gate_fuse(self, bpe_x, graph_memory, training):
        concat = torch.cat([bpe_x, graph_memory], dim=-1)
        gate   = torch.sigmoid(self.fusion_gate(concat))
        if training and torch.rand(1).item() < self.modality_dropout_prob:
            return graph_memory
        return gate * bpe_x + (1.0 - gate) * graph_memory

    def _align_graph_to_bpe(self, graph_out_dense, src_bpe_word_ids, B, Seq_Len_BPE):
        """Graf belleğini BPE token düzeyine hizala (gate/additive için)."""
        max_words    = graph_out_dense.size(1)
        safe_ids     = torch.clamp(src_bpe_word_ids, 0, max(0, max_words - 1))
        safe_ids_exp = safe_ids.unsqueeze(-1).expand(B, Seq_Len_BPE, self.d_model)
        return torch.gather(graph_out_dense, 1, safe_ids_exp)

    # ------------------------------------------------------------------
    def _encode(self, src_bpe, src_graph, src_bpe_word_ids, src_mask):
        """
        Döndürür: (bpe_mem, graph_mem, graph_mask)
          - Tek-bellek modları (baseline/gate/additive): graph_mem=None, graph_mask=None
          - cross_attn: bpe_mem + ayrı graph_mem + graph_mask
          - concat_cross: birleşik bellek bpe_mem'de, graph_mem=None
        """
        B           = src_bpe.size(0)
        Seq_Len_BPE = src_bpe.size(1)

        bpe_x = self.pe(self.bpe_embed(src_bpe))
        for layer in self.encoder_bpe:
            bpe_x = layer(bpe_x, src_mask)
        bpe_x = self.encoder_norm(bpe_x)

        if not self.use_dual_encoder:
            return bpe_x, None, None

        graph_out_dense, graph_valid_mask = self.graph_encoder(src_graph, expected_batch_size=B)

        # --- cross_attn: iki ayrı bellek ---
        if self.fusion == "cross_attn":
            graph_mask = graph_valid_mask.unsqueeze(1).unsqueeze(2)
            return bpe_x, graph_out_dense, graph_mask

        # --- concat_cross: iki belleği sekans ekseninde birleştir, tek bellek ---
        if self.fusion == "concat_cross":
            # bpe_x: [B, L_s, d], graph: [B, L_w, d] → concat [B, L_s+L_w, d]
            combined = torch.cat([bpe_x, graph_out_dense], dim=1)
            return combined, None, None

        # --- gate / additive: hizala sonra birleştir ---
        graph_memory = self._align_graph_to_bpe(graph_out_dense, src_bpe_word_ids, B, Seq_Len_BPE)

        if self.fusion == "gate":
            fused = self._gate_fuse(bpe_x, graph_memory, self.training)
        elif self.fusion == "additive":
            fused = bpe_x + graph_memory          # basit toplama
        else:
            raise ValueError(f"Bilinmeyen fusion: {self.fusion}")

        return fused, None, None

    # ------------------------------------------------------------------
    def _make_src_mask_for_decoder(self, src_bpe, pad_idx, device, combined_len=None):
        """
        cross_attn/gate/additive: normal src_mask (BPE uzunluğu).
        concat_cross: birleşik bellek için mask — BPE kısmı gerçek mask, graph kısmı hep açık.
        """
        base = (src_bpe != pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,L_s]
        if self.fusion == "concat_cross" and combined_len is not None:
            B = src_bpe.size(0)
            L_s = src_bpe.size(1)
            L_g = combined_len - L_s
            graph_part = torch.ones(B, 1, 1, L_g, dtype=torch.bool, device=device)
            return torch.cat([base, graph_part], dim=-1)
        return base

    # ------------------------------------------------------------------
    def _run_decoder(self, x, bpe_mem, graph_mem, graph_mask, src_mask, trg_mask):
        if self.use_dual_encoder and self.fusion == "cross_attn":
            for layer in self.decoder_layers:
                x = layer(x, bpe_mem, graph_mem, src_mask, graph_mask, trg_mask)
        else:
            for layer in self.decoder_layers:
                x = layer(x, bpe_mem, src_mask, trg_mask)
        return x

    # ------------------------------------------------------------------
    def forward(self, src_graph, trg_bpe, src_bpe, src_bpe_word_ids, pad_idx):
        device = trg_bpe.device
        _, trg_mask = create_masks(src_bpe, trg_bpe, pad_idx, device)

        base_src_mask, _ = create_masks(src_bpe, None, pad_idx, device)
        bpe_mem, graph_mem, graph_mask = self._encode(
            src_bpe, src_graph, src_bpe_word_ids, base_src_mask
        )

        # concat_cross için genişletilmiş src_mask
        if self.fusion == "concat_cross":
            src_mask = self._make_src_mask_for_decoder(src_bpe, pad_idx, device, bpe_mem.size(1))
        else:
            src_mask = base_src_mask

        x = self.pe(self.bpe_embed(trg_bpe))
        x = self._run_decoder(x, bpe_mem, graph_mem, graph_mask, src_mask, trg_mask)
        return self.out(self.norm(x))

    # ------------------------------------------------------------------
    def generate(self, src_graph, src_bpe, src_bpe_word_ids, sos_idx, eos_idx,
                 pad_idx, max_len=100, beam_size=3, repetition_penalty=1.3):
        self.eval()
        device = src_bpe.device
        src_real_len = int(src_bpe.ne(pad_idx).sum(dim=1).max().item())
        max_len      = min(max_len, int(src_real_len * 1.5) + 10)

        base_src_mask, _ = create_masks(src_bpe, None, pad_idx, device)

        with torch.no_grad():
            bpe_mem, graph_mem, graph_mask = self._encode(
                src_bpe, src_graph, src_bpe_word_ids, base_src_mask
            )
            if self.fusion == "concat_cross":
                src_mask = self._make_src_mask_for_decoder(src_bpe, pad_idx, device, bpe_mem.size(1))
            else:
                src_mask = base_src_mask

        beams = [(0.0, [sos_idx])]
        for _ in range(max_len):
            new_beams = []
            for score, tokens in beams:
                if tokens[-1] == eos_idx:
                    new_beams.append((score, tokens))
                    continue
                trg_tensor  = torch.LongTensor(tokens).unsqueeze(0).to(device)
                _, trg_mask = create_masks(None, trg_tensor, pad_idx, device)
                with torch.no_grad():
                    x = self.pe(self.bpe_embed(trg_tensor))
                    x = self._run_decoder(x, bpe_mem, graph_mem, graph_mask, src_mask, trg_mask)
                    logits = self.out(self.norm(x))[:, -1, :]
                    for tok in set(tokens[1:]):
                        if logits[0, tok] > 0:
                            logits[0, tok] /= repetition_penalty
                        else:
                            logits[0, tok] *= repetition_penalty
                    probs = F.log_softmax(logits, dim=-1).squeeze(0)
                topk_probs, topk_ids = torch.topk(probs, beam_size)
                for i in range(beam_size):
                    lp        = ((len(tokens) + 1) ** 0.7) / (6 ** 0.7)
                    new_score = score + topk_probs[i].item() / lp
                    new_beams.append((new_score, tokens + [topk_ids[i].item()]))
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
            if all(b[1][-1] == eos_idx for b in beams):
                break
        return beams[0][1]

    # ------------------------------------------------------------------
    def greedy_decode(self, src_graph, src_bpe, src_bpe_word_ids, max_len,
                      start_symbol, pad_idx, eos_idx=3):
        tokens = self.generate(
            src_graph=src_graph, src_bpe=src_bpe, src_bpe_word_ids=src_bpe_word_ids,
            sos_idx=start_symbol, eos_idx=eos_idx, pad_idx=pad_idx,
            max_len=max_len, beam_size=3,
        )
        return torch.tensor([tokens], dtype=torch.long, device=src_bpe.device)
