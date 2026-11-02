# MorphGAT-A-Morphology-Aware-Dual-Encoder-Transformer-for-Turkish-English-NMT
This repository contains the code, data splits, and evaluation artifacts for the paper "MorphGAT: A Morphology-Aware Dual-Encoder Transformer for Turkish-to-English Neural Machine Translation"
<img width="5808" height="5344" alt="Dual_Encoder_En (2)" src="https://github.com/user-attachments/assets/aae9cab4-4c56-4b23-9d38-66644354f09c" />

MorphGAT pairs a standard BPE subword encoder with a graph attention encoder (GATv2) that operates over word-internal morphological graphs produced by the Zemberek analyzer. The decoder consults the two representations through separate (dual) cross-attention pathways. The study reports that, on the Tatoeba Turkish–English corpus, MorphGAT performs on par with an identically configured single-encoder baseline, and that the integration strategy (not the mere presence of morphological information) is the decisive design factor.

Environment
-------------------

Experiments were run with:

Python 3.13, PyTorch (CUDA 12.8)

PyTorch Geometric 2.8.0, torch-scatter 2.1.2

zemberek-python 0.2.3 (via JPype1 1.7.1) for Turkish morphological analysis

Hugging Face tokenizers (Metaspace pre-tokenizer/decoder) for BPE

sacrebleu 2.6.0 (BLEU, chrF, TER), bert-score 0.3.13 via evaluate 0.4.6 (BERTScore)


Single NVIDIA L4 GPU (24 GB), Google Colab Pro+

Data
----------------------
The Turkish–English parallel corpus is derived from the Tatoeba project.
Splits (train/val/test) and the morpheme vocabulary and BPE tokenizer are in data/.
BPE and the morpheme vocabulary are learned only on the training split; morphological units unseen in training are mapped to <UNK>.


Reproducing the experiments
----------------------
01_data_preparation.ipynb — clean the raw corpus and create the train/val/test split.

02_graph_construction.ipynb — run Zemberek analysis, build the morpheme vocabulary, train the BPE tokenizer, and construct the morphological graphs (.pt files).

03_training_evaluation.ipynb — train a model and evaluate it. The model variant is selected with flags (below). Final metrics (BLEU, chrF, METEOR, TER, BERTScore) are computed on the test split.


Model configurations
----------------------
The Transformer class is configured through flags:

Setting	Values	Meaning
----------------------
use_dual_encoder	→ False / True	single-encoder baseline / dual-encoder MorphGAT

fusion → cross_attn (proposed), parallel_cross, gate, additive, concat_cross	(how the two memories reach the decoder)

gnn_type →	gatv2 (proposed), gcn	morphological graph encoder


Checkpoints are selected on the validation set; decoding uses beam size 3.

Results
----------------------
Prediction and reference files for each configuration are in results/predictions/, and per-epoch training logs are in results/metrics/, so the reported scores and the paired bootstrap significance analysis can be reproduced without retraining.


Citation
----------------------

bibtex
@article{morphgat,
  title   = {MorphGAT: A Morphology-Aware Dual-Encoder Transformer for Turkish-to-English Neural Machine Translation},
  author  = {Vuran Sarı, Nisa and Acı, Mehmet and İnan Acı, Çiğdem},
  journal = {(under review)},
  year    = {2026}
}


Contact
----------------------

Nisa Vuran Sarı — nvuran@mersin.edu.tr — Department of Computer Engineering, Mersin University.
