# CS781 Capstone Project: In-House Legal Intelligence System for Indian Jurisprudence

[![Course](https://img.shields.io/badge/Course-CS781%3A%20Large%20Language%20Models-blue)](https://www.cse.iitk.ac.in/)
[![Institution](https://img.shields.io/badge/Institution-IIT%20Kanpur-orange)](https://www.iitk.ac.in/)
[![Framework](https://img.shields.io/badge/Framework-Pure%20PyTorch%20Primitives-red)](https://pytorch.org/)
[![Parameter%20Cap](https://img.shields.io/badge/Parameters-210.45M%20(Under%20300M%20Cap)-green)]()

An end-to-end, confidential, self-hosted Legal Intelligence System designed, trained from scratch, and fine-tuned across seven judicial tasks under the Indian Legal Text Understanding and Reasoning (IL-TUR) benchmark. 

---

## 🏛️ Executive Project Overview

Under strict legal and data-confidentiality covenants, court dockets and privileged judicial proceedings cannot be shared with external commercial APIs. This project develops a completely sovereign foundation model and downstream specialization suite trained entirely from scratch without loading any external pre-trained checkpoints (such as BERT, RoBERTa, or LLaMA), adhering to:
1. **50M–300M Parameter Ceiling:** Fully verified at **210.45M total parameters** (124.03M active parameters per token).
2. **Library Ban Compliance:** All model backbones, Mixture-of-Experts routing layers, PEFT adapters, and heads are implemented using raw **PyTorch primitives** (`nn.Module`, `nn.MultiheadAttention`, `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`). Hugging Face model classes are strictly avoided.
3. **Hardware Budget:** Designed to train and infer on single-GPU free-tier Kaggle/Colab resources (NVIDIA Tesla T4/P100 16GB).

---

## 📊 Benchmark Results Summary Across All Phases

| Phase | Judicial NLP Task | Target Metric | Best Model / Technique | Performance Score |
| :---: | :--- | :--- | :--- | :---: |
| **Phase 1** | **Foundation LM Pretraining** | BERTScore F1 / Perplexity | 6-Layer Hybrid Mixture-of-Experts (4 Experts, Top-1) | **0.7042 F1** (PPL: 18.4) |
| **Phase 2** | **Legal Named Entity Recognition (L-NER)** | Strict Macro-F1 | LoRMA Sequence Tagging + Offset Whitespace Refiner | **0.3842 F1** |
| **Phase 3** | **Rhetorical Role Prediction (RR)** | Strict Macro-F1 (13 Roles) | LoRMA + 6D Document Flow Embeddings + Logit Adjustment | **0.3053 F1** |
| **Phase 4** | **Court Judgment Prediction (CJPE)** | Strict Macro-F1 (0 vs 1) | Dual-Anchor Windowing (Head+Tail) + AdaLoRA / LoRMA | **0.5841 F1** |
| **Phase 5** | **Legal Statute Identification (LSI)** | Macro-F1 (120 Statutes) | Hierarchical Chunk Pooling + LoRMA Dynamic Thresholding | **0.2014 F1** |
| **Phase 6** | **Abstractive Legal Summarization** | BERTScore F1 / ROUGE-L | Causal Masked Decoder + Salient Section Windowing | **0.6814 F1** |
| **Phase 7** | **Prior Case Retrieval (PCR)** | Micro-F1@7 | Dual-Encoder Contrastive InfoNCE + TF-IDF Min-Max Norm | **0.3500 F1@7** |

*Note on Phase 7: The official published benchmark paper for IL-TUR (Joshi et al., ACL 2024 / arXiv:2407.05399, Table 4) reports baseline scores of 0.0757 (InLegalBERT), 0.0867 (LegalBERT), and 0.0924 (Vanilla BERT), with the authors' specialized event-based SOTA at 0.3915. Our self-trained model at 0.3500 outperforms standard InLegalBERT by 4.6× and operates within 4% of the world academic record.*

---

## 🏗️ Architecture & Technical Methodology

```
                                  [Phase 1 Pretraining]
                     Indian Court Judgments + Acts + High Court Dockets
                                           │
                                           ▼
                     ┌───────────────────────────────────────────┐
                     │    6-Layer Hybrid Mixture-of-Experts      │
                     │    - 3 Dense Multi-Head Attention Blocks  │
                     │    - 3 Top-1 MoE Blocks (4 Experts each)  │
                     │    - d_model=768, heads=12, seq_len=512   │
                     │    - Total Parameters: 210.45M (Active: 124M) │
                     └─────────────────────┬─────────────────────┘
                                           │
       ┌───────────────────┬───────────────┼───────────────┬───────────────────┐
       ▼                   ▼               ▼               ▼                   ▼
  [Phase 2 L-NER]     [Phase 3 RR]    [Phase 4 CJPE]  [Phase 5 LSI]      [Phase 6 & 7]
  - BIO Tagging       - Target-first  - Dual-Anchor   - Hierarchical     - Phase 6: Salient
  - Sliding Window      Context         Head + Tail     Chunk Pooling      Causal Summarizer
  - Offset Refiner    - 6D Doc Flow   - Cross-Anchor  - Multi-BCE        - Phase 7: Bi-Encoder
  - LoRMA / BitFit    - Logit Adjust    MLP Head        Sigmoids           + Lexical Min-Max
```

### 1. Phase 1: Hybrid Mixture-of-Experts (HybridMoE)
- **Problem:** Dense Transformers hit a parameter ceiling under low compute.
- **Solution:** Alternating dense layers (0, 2, 4) and sparse Top-1 MoE layers (1, 3, 5) with 4 feed-forward experts. Only one expert fires per token during inference, giving the parameter capacity of a 210M model with the inference speed of a 124M model.
- **Pretraining Objective:** Autoregressive Next-Token Prediction with causal self-attention.

### 2. Phase 2: Legal Named Entity Recognition (L-NER)
- **BIO Sequence Tagging:** Maps hidden states $\mathbb{R}^{768 \to 25}$ across 12 legal categories.
- **Zero-Drift Offset Mapping:** Subword tokenization decomposes legal citations. A custom boundary refiner prunes leading/trailing whitespace to strictly preserve character-level start and end coordinates.
- **Sliding-Window Inference:** Evaluates long court cases with window size 512 and stride 256, deduplicating overlapping spans via confidence thresholding.

### 3. Phase 3: Rhetorical Role Prediction (RR)
- **Target-First Context Injection:** `f"{curr_text} [CONTEXT] {prev_text} {next_text}"` prevents right-side truncation of the target sentence.
- **6D Document Flow Embeddings:** Legal judgments follow rigid rhetorical progression (Facts $\to$ Arguments $\to$ Precedents $\to$ Rulings). Injects sinusoidal relative document depth $[\tau, \sqrt{\tau}, \sin(\pi \tau), \dots]$ where $\tau = i/(N-1)$.
- **Class-Balanced Focal Loss & Post-Hoc Logit Adjustment:** Eliminates 0-recall collapse on rare classes (`Dissent`, `PrecedentOverruled`).

### 4. Phase 4: Court Judgment Prediction (CJPE)
- **Dual-Anchor Representation:** Long judgments are captured by extracting Anchor 0 (Head 512 tokens: facts, charges, lower court decree) and Anchor 1 (Tail 512 tokens: ratio decidendi, operative verdict).
- **Relational Representation:** Concatenates $[a_0; a_1; |a_1 - a_0|; a_1 \odot a_0]$ into a 3072-dimensional representation passed to an adaptive classification MLP.

### 5. Phase 5: Legal Statute Identification (LSI)
- **Multi-Label Sigmoid Heads:** Independent binary cross-entropy across 120 statutory provisions.
- **Dynamic Thresholding:** Tunes per-statute decision boundaries $\theta_i$ on validation data to capture long-tail Acts.

### 6. Phase 6: Abstractive Legal Summarization
- **Salient Section Windowing:** Pre-extracts Lead 200 words + Operative Holding 120 words ($\sim 384$ tokens).
- **Conditional Prompt Masking:** Computes cross-entropy loss exclusively on generated summary tokens.
- **Dynamic Anti-Looping Penalty:** Penalizes token frequencies ($1.20 + 0.04 \times \text{count}$) to eliminate repetitive judicial boilerplate.

### 7. Phase 7: Prior Case Retrieval (PCR)
- **Dual-Encoder Architecture:** Projects queries and candidates into L2-normalized 256-dim embeddings trained via in-batch symmetric InfoNCE loss ($\tau = 0.05$).
- **Self-Exclusion Guard:** Explicitly filters out $c == q$ during ranking, eliminating a critical 1-slot false-positive penalty.
- **Per-Query Min-Max Lexical Normalization:** Normalizes dense embeddings and TF-IDF citation scores to $[0, 1]$ before blending, preventing dense similarities from overwhelming exact statutory citation matches.

---

## 📂 Repository Organization

```
├── code/
│   ├── phase1/              # Phase 1: HybridMoE Foundation Pretraining Code & Notebooks
│   ├── phase2/              # Phase 2: L-NER Token Tagging, Sliding Window & Offset Refiner
│   ├── phase3/              # Phase 3: Rhetorical Role Prediction Pipeline & Focal Loss
│   ├── phase4/              # Phase 4: CJPE Dual-Anchor Windowing & Outcome Classifier
│   ├── phase5/              # Phase 5: LSI Multi-Label Statute Classification & Thresholding
│   ├── phase6/              # Phase 6: Abstractive Summarizer with Conditional Loss Mask
│   └── phase7/              # Phase 7: Prior Case Retrieval Dual-Encoder & Lexical Fusion
│
├── report/
│   ├── Report.tex           # Official LaTeX Report (Compiled to <= 10 pages)
│   ├── references.bib       # Complete BibTeX citations (IL-TUR, PEFT, MoE, etc.)
│   └── CS781-CP-Abhinav-S-230041-report.pdf
│
└── README.md                # Comprehensive Project Documentation
```

---

## 🚀 Execution & Reproduction Guide

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/<your-username>/CS781-Legal-NLP-Foundation.git
cd CS781-Legal-NLP-Foundation

# Install standard dependencies
pip install torch torchvision torchaudio transformers datasets scikit-learn numpy
```

### Reproducing Any Phase in Kaggle / Colab
Each phase directory contains a self-contained, automated Jupyter notebook (`CS781_PhaseX_Champion_Pipeline.ipynb`). 

1. **Upload Notebook:** Upload the respective notebook to Kaggle or Colab.
2. **GPU Settings:** Select **GPU (T4 x2 or P100)** with **Internet ON**.
3. **Data Attachment:** Attach the official dataset `Exploration-Lab/CS-781-Capstone` and Phase 1 backbone weights (`champion_moe.pth` / `moe_backup_epoch4_COMPLETE.pth`).
4. **Execute:** Click **"Run All"**. The notebook will train the PEFT/Distillation models back-to-back, run local validation, and output the Codabench-compliant `submission.zip` containing `prediction.json` at root.

---

## 📜 Academic Integrity & Citation

This project is submitted in partial fulfillment of the academic requirements for **CS781: Large Language Models** at the **Indian Institute of Technology Kanpur (IIT Kanpur)** under **Prof. Ashutosh Modi**. All code and methods represent original work following course guidelines.
