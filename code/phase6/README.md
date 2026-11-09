# CS-781 Capstone Project — Phase 6: Abstractive Legal Summarization
## Master Execution Plan, Architecture Guide & Submission Matrix

**Task:** Abstractive Legal Summarization of Indian Supreme Court Judgments  
**Primary Metric:** **BERTScore F1**  
**Secondary Metrics:** **ROUGE-1 F1**, **ROUGE-L F1**  
**Starting Backbone:** Phase 1 Hybrid Mixture-of-Experts (`champion_moe.pth` / `moe_backup_epoch4_COMPLETE.pth`)  
**Main Execution Notebook:** [`phase6_model.ipynb`](../phase6_model.ipynb) (also mirrored at `P-6/phase6_model.ipynb`)

---

## 1. Task Formulation & Expected Submission Schema

In Phase 6, given a lengthy Indian Supreme Court case containing facts, arguments, statutory provisions, and final decisions, the system must generate a concise, fluent, and factually accurate abstractive legal summary.

* **Dataset:** Hosted on Hugging Face at `Exploration-Lab/CS-781-Capstone`:
  - `Phase_6_Summ/Train/train.json`: Training pairs `{"id": "...", "document": "...", "summary": "..."}`
  - `Phase_6_Summ/Test/test.json`: Evaluation documents `{"id": "...", "document": "..."}`
* **Expected Output Schema:**
  A JSON list of dictionaries saved as `prediction.json` at the root of `submission.zip`:
  ```json
  [
    {
      "id": "doc_001",
      "summary": "In this criminal appeal, the Supreme Court upheld the conviction under Section 302 IPC..."
    }
  ]
  ```
* **Strict Codabench Invariants:**
  - `prediction.json` must be at the root of the `.zip` archive (no nested folders).
  - Exactly one non-empty string summary per test document ID.
  - Test IDs must match `test.json` exactly.

---

## 2. Strict Project Rules Compliance

1. **Strict Prohibition on Hugging Face Model Classes:**
   - Hugging Face model classes (`AutoModelForSeq2SeqLM`, `BartForConditionalGeneration`, `T5ForConditionalGeneration`, `AutoModelForCausalLM`, etc.) are **STRICTLY PROHIBITED**.
   - All backbones, PEFT adapters, attention layers, and generation heads are implemented in **pure PyTorch primitives** (`torch.nn.Module`, `torch.nn.MultiheadAttention`, `torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.LayerNorm`).
   - Hugging Face is used **ONLY** for tokenization (`AutoTokenizer.from_pretrained("openai-community/gpt2")`).
2. **Phase 1 Backbone Starting Point:**
   - Must build upon the 6-layer Hybrid Mixture-of-Experts (MoE) foundation model (`HybridMoE`, 768 dim, 12 heads, alternating 4-expert MoE layers) trained in Phase 1.
3. **Mandatory 6-Experiment Rule:**
   - Requires at least 5 different fine-tuning/adaptation techniques.
   - **LoRMA is COMPULSORY.**
   - **Knowledge Distillation is COMPULSORY.**

---

## 3. The 6 Required Submissions & Execution Matrix

| Submission | Directory | Method Key | Architectural Technique | Trainable Params | Target ROUGE-L / BERTScore |
|:---:|:---:|:---:|:---|:---:|:---:|
| **P6-1** | `P-6/Submission P6-1` | `"bitfit"` | Bias-only fine-tuning (freezes all weights, trains biases & LayerNorms). | ~0.1% | High Baseline |
| **P6-2** | `P-6/Submission P6-2` | `"layer_freezing"` | Freezes lower 4 layers, fine-tunes top 2 Transformer blocks + final LayerNorm + LM head. | ~33% | Fast & Strong |
| **P6-3** | `P-6/Submission P6-3` | `"bottleneck_adapter"` | Series bottleneck adapters ($768 \to 64 \to 768$ with GELU) in attention & FFN blocks. | ~1.5% | Stable Adaptation |
| **P6-4** | `P-6/Submission P6-4` | `"adalora"` | Adaptive Low-Rank SVD parameterization with dynamic rank allocation ($P \cdot E \cdot Q$). | ~2.1% | SVD-Optimized |
| **P6-5** | `P-6/Submission P6-5` | `"lorma"` | **COMPULSORY PEFT:** Low-Rank Multiplicative Adaptation ($h = W_0 x + \alpha B(A(W_0 x))$). | **~1.8%** | **PEFT Champion** |
| **P6-6** | `P-6/Submission P6-6` | `"distill"` | **COMPULSORY KD:** 3-layer student ($d=384$, 6 heads) learning from trained LoRMA teacher. | **~42M total** | **Student Champion** |
| **Bonus** | Workspace Root | `"ensemble"` | Multi-model consensus ranking & length-normalized blending across top PEFT models. | Ensemble | **Maximum Coverage** |

---

## 4. Key Architectural Innovations for Long Legal Summarization

1. **Salient Section Windowing (Head + Operative Decree):**
   - Supreme Court judgments are 5,000+ words, but the critical information is concentrated in the opening facts (charges, parties, dispute) and the closing operative holding (conviction decree, disposal of appeal).
   - Extracts: **Lead 200 words** + **Operative Holding 120 words** = 320 words (~384 tokens).
   - Formats input as: `Document: {lead_text} ... [Holding]: {tail_text}\n\nSummary:`
   - Leaves ~128 tokens for generation, perfectly respecting `HybridMoE`'s 512-token context limit!
2. **Conditional Masked Causal LM Loss:**
   - Instead of standard language modeling where the entire prompt is penalized, applies a `loss_mask` that zeros loss on the document prompt tokens.
   - Cross-entropy loss is computed **strictly over the summary tokens**, focusing model capacity on abstractive synthesis.
3. **Dynamic Anti-Looping Autoregressive Generation Engine:**
   - Nucleus sampling (top-p = 0.90) + Top-K (40) + Temperature (0.70).
   - Dynamic frequency/repetition penalty ($1.20 + \text{count} \times 0.04$) to eliminate repetitive legal loop words.
4. **Extractive Fallback Guard:**
   - If the model prematurely outputs `<|endoftext|>` or produces a degenerate summary (< 15 words), automatically synthesizes a fallback summary from the court's operative holding sentences, guaranteeing zero empty or invalid submissions.
5. **In-Notebook LCS ROUGE Evaluator:**
   - Evaluates exact unigram ROUGE-1 and Longest Common Subsequence (LCS) ROUGE-L F1 scores locally in pure Python with zero external package dependencies.

---

## 5. Kaggle Execution Instructions (Automated One-Click)

1. Upload [`phase6_model.ipynb`](../phase6_model.ipynb) to Kaggle.
2. In the notebook settings:
   - **Accelerator:** GPU T4 x2 or GPU P100.
   - **Internet:** ON (to download `train.json` and `test.json` from Hugging Face).
   - **Inputs:** Attach your Phase 1 model/dataset containing `champion_moe.pth` or `moe_backup_epoch4_COMPLETE.pth`.
3. Click **"Run All"**:
   - Trains all 6 methods back-to-back in ~20 to 25 minutes total.
   - Memory is cleared between runs (`del model; torch.cuda.empty_cache(); gc.collect()`).
4. Collect submissions in `/kaggle/working/`:
   - `submission_(bitfit).zip` -> Submission P6-1
   - `submission_(layer_freezing).zip` -> Submission P6-2
   - `submission_(bottleneck_adapter).zip` -> Submission P6-3
   - `submission_(adalora).zip` -> Submission P6-4
   - `submission_(lorma).zip` -> Submission P6-5
   - `submission_(distill).zip` -> Submission P6-6
   - `submission_(ensemble).zip` (and root `submission.zip`) -> Master Champion
5. Download and submit directly to Codabench!
