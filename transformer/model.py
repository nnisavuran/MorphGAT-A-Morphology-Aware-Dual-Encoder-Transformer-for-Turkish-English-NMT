# transformer/model.py
#
# Üç mod:
#   use_dual_encoder=False                    → Standart Transformer (baseline)
#   use_dual_encoder=True, fusion="gate"      → Gated füzyon (eski önerilen)
#   use_dual_encoder=True, fusion="cross_attn"→ Çift çapraz dikkat (yeni önerilen)
#
# Cross-attention modunda decoder, BPE ve graf belleklerine AYRI cross-attention
# uygular; gate/manuel füzyon yok, magnitude uyumsuzluğu softmax ile çözülür.

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer  import EncoderLayer, DecoderLayer, DualCrossDecoderLayer, ParallelCrossDecoderLayer
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
        fusion="cross_attn",          # "cross_attn" | "gate"
        modality_dropout_prob=0.00,
    ):
        super().__init__()
        self.d_model               = d_model
        self.use_dual_encoder      = use_dual_encoder
        self.fusion                = fusion
        self.modality_dropout_prob = modality_dropout_prob

        # BPE Embedding + PE
        self.bpe_embed = Embedder(vocab_size, d_model, padding_idx=0)
        self.pe        = PositionalEncoder(d_model, dropout=dropout, max_len=max_len)

        # Kodlayıcı 1 — BPE
        self.encoder_bpe  = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.encoder_norm = Norm(d_model)

        # Kodlayıcı 2 — Graf (sadece dual modda)
        if use_dual_encoder:
            self.graph_encoder = GraphEncoder(
                morph_vocab_size, d_model, heads, dropout=dropout, padding_idx=0
            )
            if fusion == "gate":
                self.fusion_gate = nn.Linear(d_model * 2, d_model)
                nn.init.constant_(self.fusion_gate.bias, 1.0)
                nn.init.xavier_uniform_(self.fusion_gate.weight, gain=0.1)

        # Kod Çözücü — moda göre farklı katman tipi
        if use_dual_encoder and fusion in ("cross_attn", "parallel_cross"):
            Layer = DualCrossDecoderLayer if fusion == "cross_attn" else ParallelCrossDecoderLayer
            self.decoder_layers = get_clones(Layer(d_model, heads, dropout), N)
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

    # ------------------------------------------------------------------
    def _encode(self, src_bpe, src_graph, src_bpe_word_ids, src_mask):
        """
        Döndürür:
          - Standart / gate modu: (memory, None, None)  → tek bellek
          - Cross-attn modu:      (bpe_x, graph_dense, graph_mask) → iki bellek
        """
        B           = src_bpe.size(0)
        Seq_Len_BPE = src_bpe.size(1)

        bpe_x = self.pe(self.bpe_embed(src_bpe))
        for layer in self.encoder_bpe:
            bpe_x = layer(bpe_x, src_mask)
        bpe_x = self.encoder_norm(bpe_x)

        if not self.use_dual_encoder:
            return bpe_x, None, None

        # Graf kodlayıcı
        graph_out_dense, graph_valid_mask = self.graph_encoder(src_graph, expected_batch_size=B)

                # --- cross_attn / parallel_cross: iki ayrı bellek ---
        if self.fusion in ("cross_attn", "parallel_cross"):
            # Graf belleğini kelime düzeyinde bırak (BPE'ye hizalama YOK)
            # graph_mask: [B, 1, 1, num_words] — cross-attention padding maskesi
            graph_mask = graph_valid_mask.unsqueeze(1).unsqueeze(2)
            return bpe_x, graph_out_dense, graph_mask

        # Gate modu — graf belleğini BPE token düzeyine hizala, sonra füzyon
        max_words    = graph_out_dense.size(1)
        safe_ids     = torch.clamp(src_bpe_word_ids, 0, max(0, max_words - 1))
        safe_ids_exp = safe_ids.unsqueeze(-1).expand(B, Seq_Len_BPE, self.d_model)
        graph_memory = torch.gather(graph_out_dense, 1, safe_ids_exp)
        fused        = self._gate_fuse(bpe_x, graph_memory, self.training)
        return fused, None, None

    # ------------------------------------------------------------------
    def _run_decoder(self, x, bpe_mem, graph_mem, graph_mask, src_mask, trg_mask):
        if self.use_dual_encoder and self.fusion in ("cross_attn", "parallel_cross"):
            for layer in self.decoder_layers:
                x = layer(x, bpe_mem, graph_mem, src_mask, graph_mask, trg_mask)
        else:
            for layer in self.decoder_layers:
                x = layer(x, bpe_mem, src_mask, trg_mask)
        return x

    # ------------------------------------------------------------------
    def forward(self, src_graph, trg_bpe, src_bpe, src_bpe_word_ids, pad_idx):
        device = trg_bpe.device
        src_mask, trg_mask = create_masks(src_bpe, trg_bpe, pad_idx, device)

        bpe_mem, graph_mem, graph_mask = self._encode(
            src_bpe, src_graph, src_bpe_word_ids, src_mask
        )

        x = self.pe(self.bpe_embed(trg_bpe))
        x = self._run_decoder(x, bpe_mem, graph_mem, graph_mask, src_mask, trg_mask)
        return self.out(self.norm(x))

    # ------------------------------------------------------------------
    def generate(
        self,
        src_graph,
        src_bpe,
        src_bpe_word_ids,
        sos_idx,
        eos_idx,
        pad_idx,
        max_len=100,
        beam_size=3,
        repetition_penalty=1.3,
    ):
        self.eval()
        device = src_bpe.device

        src_real_len = int(src_bpe.ne(pad_idx).sum(dim=1).max().item())
        max_len      = min(max_len, int(src_real_len * 1.5) + 10)

        src_mask, _ = create_masks(src_bpe, None, pad_idx, device)

        with torch.no_grad():
            bpe_mem, graph_mem, graph_mask = self._encode(
                src_bpe, src_graph, src_bpe_word_ids, src_mask
            )

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
    def greedy_decode(
        self, src_graph, src_bpe, src_bpe_word_ids, max_len, start_symbol, pad_idx, eos_idx=3
    ):
        tokens = self.generate(
            src_graph=src_graph,
            src_bpe=src_bpe,
            src_bpe_word_ids=src_bpe_word_ids,
            sos_idx=start_symbol,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            max_len=max_len,
            beam_size=3,
        )
        return torch.tensor([tokens], dtype=torch.long, device=src_bpe.device)
