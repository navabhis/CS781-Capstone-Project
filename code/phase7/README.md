# CS-781 Capstone Project — Phase 7: Prior Case Retrieval (PCR)
## Master Execution Plan, Architecture Guide & Submission Matrix

**Task:** Prior Case Retrieval (PCR) from Indian Supreme Court Judgments  
**Primary Metric:** **Micro-F1@7**  
**Secondary Metrics:** **Micro-Precision@7**, **Micro-Recall@7**  
**Starting Backbone:** Phase 1 Hybrid Mixture-of-Experts (`champion_moe.pth` / `moe_backup_epoch4_COMPLETE.pth`)  
**Main Execution Notebook:** [`phase7_model.ipynb`](../phase7_model.ipynb) (also mirrored at `P-7/phase7_model.ipynb`)

---

## 1. Task Formulation & Expected Submission Schema

In Phase 7, given a query legal case, the system must understand its legal issues, statutory provisions, and reasoning to retrieve and rank the **top 7 most relevant prior legal cases** from the corresponding candidate-case pool.

* **Dataset:** Hosted on Hugging Face at `Exploration-Lab/CS-781-Capstone`:
  - `Phase_7_PCR/Train/train.json` + `Phase_7_PCR/Train/train_candidates.json`
  - `Phase_7_PCR/Dev/dev.json` + `Phase_7_PCR/Dev/dev_candidates.json`
  - `Phase_7_PCR/Test/test.json` + `Phase_7_PCR/Test/test_candidates.json`
* **Expected Output Schema (`prediction.json`):**
  A JSON object mapping each test query case ID to a list of **EXACTLY 7 UNIQUE** candidate case IDs, ordered from highest to lowest predicted relevance:
  ```json
  {
    "0000021652": [
      "0000590455",
      "0000916840",
      "0001481813",
      "0001907199",
      "0000000987",
      "0000001566",
      "0000002886"
    ],
    "0000030721": [
      "0000004354",
      "0000320809",
      "0000516439",
      "0000611866",
      "0000685234",
      "0000785119",
      "0000849101"
    ]
  }
  ```
* **Strict Codabench Invariants:**
  - `prediction.json` must be placed directly at the root of `submission.zip`.
  - Exactly 7 candidate IDs per query.
  - Candidate IDs must be unique and belong to `test_candidates.json`.
  - Every test query must have predictions.

---

## 2. Strict Course Rules Compliance

1. **Strict Prohibition on Hugging Face Model Classes:**
   - Hugging Face model classes (`AutoModel`, `BertModel`, `SentenceTransformer`, `DPRQuestionEncoder`, etc.) are **STRICTLY PROHIBITED**.
   - All backbones, adapters, projection heads, and contrastive loss functions are implemented in **pure PyTorch primitives** (`torch.nn.Module`, `torch.nn.MultiheadAttention`, `torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.LayerNorm`).
   - Hugging Face is used **ONLY** for tokenization (`AutoTokenizer.from_pretrained("openai-community/gpt2")`).
2. **Phase 1 Backbone Starting Point:**
   - Adapts the Phase 1 6-layer Hybrid Mixture-of-Experts (MoE) foundation model (`HybridMoE`) via bidirectional attention and attention-masked mean pooling into a dual-encoder representation model (`HybridMoEForPCR`).
3. **Mandatory 6-Experiment Rule (Rubric Section 13.2):**
   - At least 5 different fine-tuning/adaptation techniques.
   - **LoRMA is COMPULSORY.**
   - **Knowledge Distillation is COMPULSORY.**

---

## 3. The 6 Required Submissions & Execution Matrix

| Submission | Directory | Method Key | Architectural Technique | Trainable Params | Target Micro-F1@7 |
|:---:|:---:|:---:|:---|:---:|:---:|
| **P7-1** | `P-7/Submission P7-1` | `"bitfit"` | Bias-only fine-tuning (freezes all weights, trains biases & LayerNorms). | ~0.1% | ~0.46 – 0.50 |
| **P7-2** | `P-7/Submission P7-2` | `"layer_freezing"` | Freezes lower 4 layers, fine-tunes top 2 blocks + projection head. | ~33% | ~0.49 – 0.53 |
| **P7-3** | `P-7/Submission P7-3` | `"bottleneck_adapter"` | Series bottleneck adapters ($768 \to 64 \to 768$ with GELU) in attention & FFN blocks. | ~1.5% | ~0.50 – 0.54 |
| **P7-4** | `P-7/Submission P7-4` | `"adalora"` | Adaptive Low-Rank SVD parameterization with dynamic rank allocation ($P \cdot E \cdot Q$). | ~2.1% | ~0.52 – 0.56 |
| **P7-5** | `P-7/Submission P7-5` | `"lorma"` | **COMPULSORY PEFT CHAMPION:** Low-Rank Multiplicative Adaptation ($h = W_0 x + \alpha B(A(W_0 x))$). | **~1.8%** | **0.55 – 0.60+** |
| **P7-6** | `P-7/Submission P7-6` | `"distill"` | **COMPULSORY KD CHAMPION:** 3-layer student ($d=384$, 6 heads) learning from trained LoRMA dual-encoder. | **~42M total** | **0.54 – 0.58+** |
| **Bonus** | Workspace Root | `"ensemble"` | Reciprocal Rank Fusion (RRF) across top models: $\text{RRF}(c) = \sum \frac{1}{60 + \text{rank}(c)}$. | Ensemble | **0.58 – 0.65+** |

---

## 4. Key Architectural Innovations for Legal Prior Case Retrieval

1. **Dual-Encoder Architecture with Pre-Encoded Candidate Caching:**
   - Passing all query-candidate pairs through a cross-encoder would take hours.
   - Dual-encoder encodes all test candidates **once** into $\mathbf{V} \in \mathbb{R}^{M \times 256}$.
   - Query retrieval is computed via dense cosine similarity $\mathbf{S} = \mathbf{U} \mathbf{V}^\top$ in under **10 milliseconds**!
2. **Salient Section Windowing (Head + Operative Decree):**
   - Extracts Lead 200 words (charges, parties, dispute) + Tail 120 words (operative decree, cited precedents, holding) = 320 words (~384 tokens), fitting within the 512-token context limit while preserving essential legal signals.
3. **Hybrid Neural-Lexical Representation Fusion:**
   - Fuses dense semantic similarity with sublinear TF-IDF lexical overlap:
     $$\mathbf{S}_{\text{final}} = 0.65 \cdot \mathbf{S}_{\text{dense}} + 0.35 \cdot \mathbf{S}_{\text{lexical}}$$
   - Captures both conceptual legal relevance and explicit statutory citations (*"Section 302 IPC"*, *"Article 226"*).
4. **Symmetric In-Batch InfoNCE Contrastive Loss:**
   - Leverages in-batch negatives ($\tau = 0.05$) to optimize bidirectional metric alignment between queries and positive prior cases.
5. **Reciprocal Rank Fusion (RRF) Booster:**
   - Smoothly combines rankings from top diverse PEFT models without score-scale distortion.

---

## 5. Kaggle Execution Instructions (Automated One-Click)

1. Upload [`phase7_model.ipynb`](../phase7_model.ipynb) to Kaggle.
2. Under Kaggle settings:
   * **Accelerator:** GPU T4 x2 or GPU P100.
   * **Internet:** ON (downloads `train`, `dev`, `test` splits from Hugging Face).
   * **Input Checkpoint:** Ensure your Phase 1 pretraining backbone (`champion_moe.pth` or `moe_backup_epoch4_COMPLETE.pth`) is attached in `/kaggle/input/`.
3. Click **"Run All"**:
   * Automatically executes all 6 methods back-to-back in **~20 to 25 minutes total**.
   * Memory is cleaned between runs (`del model; gc.collect(); torch.cuda.empty_cache()`).
4. Collect submissions in `/kaggle/working/`:
   * `submission_(bitfit).zip` -> Submission P7-1
   * `submission_(layer_freezing).zip` -> Submission P7-2
   * `submission_(bottleneck_adapter).zip` -> Submission P7-3
   * `submission_(adalora).zip` -> Submission P7-4
   * `submission_(lorma).zip` -> Submission P7-5 (Compulsory PEFT)
   * `submission_(distill).zip` -> Submission P7-6 (Compulsory Distillation)
   * `submission_(ensemble).zip` and root `submission.zip` -> Champion RRF Ensemble
5. Download and submit directly to Codabench!
