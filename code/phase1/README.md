# Phase 1: Foundation Legal Language Model Pre-Training

## 1. Overview & Objectives
- **Task:** Unsupervised causal language model pre-training from scratch on Indian legal texts (Supreme Court, High Courts, Central Acts, Tribunals).
- **Primary Metric:** BERTScore F1, ROUGE-L F1, BLEU-4.
- **Parameter Ceiling:** Strictly constrained between 50M and 300M parameters.
- **Prohibitions:** External pre-trained weights (e.g. Hugging Face BERT, GPT-2 weights, LLaMA) and external closed APIs are strictly banned.

---

## 2. Model Architecture: Hybrid Mixture-of-Experts (HybridMoE)
To maximize knowledge capacity while staying within low-compute limits, we designed a **6-Layer Hybrid Mixture-of-Experts (HybridMoE)** causal language model in raw PyTorch:
- **Total Parameters:** **210.45 Million** (Strictly within 50M–300M limit).
- **Active Parameters per Token:** **124.03 Million**.
- **Hidden Dimension ($d_{\text{model}}$):** 768.
- **Attention Heads:** 12 ($d_{\text{head}} = 64$).
- **Context Length ($T$):** 512 tokens.
- **Layer Stacking:**
  - Layers 0, 2, 4: Standard Dense Multi-Head Attention + MLP.
  - Layers 1, 3, 5: Top-1 Mixture-of-Experts (4 feed-forward experts per block, routing with noisy softmax gating).
- **Tokenizer:** Byte-level BPE (GPT-2 vocabulary of 50,257 tokens).

---

## 3. Files in this Directory
- `CS781_Phase1_New_Champion_Base_Model.ipynb`: Automated, self-contained Jupyter notebook to train the HybridMoE base model from scratch on Kaggle/Colab GPU.
- `build_phase1_base_model.py`: Standalone script to assemble and verify parameter counts of the HybridMoE architecture.

---

## 4. How to Run
1. Open `CS781_Phase1_New_Champion_Base_Model.ipynb` in Kaggle or Google Colab (GPU accelerator enabled).
2. Attach the dataset `Exploration-Lab/CS-781-Capstone`.
3. Execute **Run All**. The notebook will train the base model, perform evaluation sweeps, and serialize the champion weights (`champion_moe.pth`).
