# BERT Long Sequence Classification
 
> Fine-tuning `bert-base-uncased` for sentiment analysis on long documents using sliding window and hierarchical encoding strategies
 
---
 
## Overview
 
BERT is constrained to a maximum of 512 tokens per input. This is a well-documented limitation: long documents such as IMDb movie reviews are silently truncated, discarding content that may be critical for classification. This project investigates and benchmarks four strategies for overcoming this constraint on the IMDb sentiment analysis task.
 
| # | Model | Strategy |
|---|-------|---------|
| 1 | **Baseline BERT** | Hard truncation at 512 tokens |
| 2 | **Sliding Window — Mean Pool** | Overlapping chunks → mean-pooled CLS embeddings |
| 3 | **Sliding Window — Max Pool** | Overlapping chunks → max-pooled CLS embeddings |
| 4 | **Hierarchical BERT** *(Extension)* | Sentence-level BERT → document-level Transformer encoder |
 
---
 
## Results
 
| Model | Accuracy | F1 Score | Training Time | Parameters |
|-------|----------|----------|---------------|------------|
| Baseline BERT (Truncated) | 0.9140 | 0.9140 | 755s | 109,483,778 |
| Sliding Window — Mean Pool | 0.9020 | 0.9020 | 2805s | 109,483,778 |
| Sliding Window — Max Pool | 0.9020 | 0.9020 | 2803s | 109,483,778 |
| Hierarchical BERT | — | — | — | 120,511,746 |
 
> **Note:** The Hierarchical BERT model is implemented and architecturally complete but was not run to completion due to Colab session time constraints. Results for Models 1–3 are from a full 3-epoch run on a T4 GPU.
 
---
 
## Repository Structure
 
```
├── BERT_Long_Sequence_Classification.ipynb
├── results_summary.csv
├── token_distribution.png
├── comparison_plots.png
├── report
└── README.md
```
 
---
 
## Setup & Usage
 
### Requirements
 
This notebook is designed for **Google Colab (T4 GPU)**. All dependencies are installed within the first cell.
 
```
transformers
datasets
scikit-learn
pandas
matplotlib
seaborn
nltk
accelerate
torch (pre-installed on Colab)
``` 
---
 
## Methodology
 
### Token-Length Analysis
 
Of 300 randomly sampled IMDb training reviews:
- **Mean token length:** 304 tokens
- **Median token length:** 228 tokens
- **Maximum token length:** 1,181 tokens
- **Reviews exceeding 512 tokens:** 40 / 300 (13.3%)
 
This confirms that a non-trivial fraction of the dataset is affected by the 512-token limit.
 
### Sliding Window
 
Each review is encoded without special tokens, then split into overlapping chunks of 510 tokens with a stride of 256 (roughly 50% overlap). Each chunk receives `[CLS]` and `[SEP]` tokens and is processed independently by BERT. The resulting CLS embeddings are aggregated via **mean pooling** or **max pooling** before the classification head.
 
```
chunk_size = 510 tokens
stride     = 256 tokens
max_chunks = 4 per document
```
 
### Hierarchical BERT *(Extension)*
 
Reviews are first split into individual sentences using NLTK's sentence tokeniser. Each sentence (up to 20 sentences, 128 tokens each) is independently encoded by BERT. The resulting sentence embeddings are passed through a 2-layer Transformer encoder to produce a document-level representation, which is then classified.
 
---
 
## Hyperparameters
 
| Parameter | Value |
|-----------|-------|
| Base model | `bert-base-uncased` |
| Dataset | IMDb (2,000 train / 500 test) |
| Batch size | 4 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Warmup ratio | 10% |
| Optimizer | AdamW (weight decay 0.01) |
| Gradient clipping | 1.0 |
---
 
## References
 
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT.
- Yang, Z. et al. (2016). *Hierarchical Attention Networks for Document Classification.* NAACL-HLT.
- Beltagy, I., Peters, M. E., & Cohan, A. (2020). *Longformer: The Long-Document Transformer.* arXiv:2004.05150.
- Maas, A. et al. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL.
