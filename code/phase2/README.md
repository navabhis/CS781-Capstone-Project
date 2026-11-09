# Phase 2: Legal Named Entity Recognition (L-NER)

## 1. Overview & Objectives
- **Task:** Token-level sequence tagging and character-span extraction across 12 legal entity classes:
  `APP` (Appellant), `RESP` (Respondent), `A.COUNSEL`, `R.COUNSEL`, `JUDGE`, `WIT` (Witness), `AUTH` (Authority), `COURT`, `STAT` (Statute), `PREC` (Precedent), `DATE`, `CASENO`.
- **BIO Target Space:** 25 classes ($1 \times \text{O} + 12 \times \text{B} + 12 \times \text{I}$).
- **Evaluation Metric:** **Strict Macro-F1** over exact character boundaries $[start\_char, end\_char, label]$. Zero tolerance for even a 1-character shift.
- **Starting Backbone:** Pre-trained Phase 1 `HybridMoE` weights.

---

## 2. Key Architectural Innovations
1. **Zero-Drift Offset Mapping:** Subword tokenization often breaks punctuation and citation boundaries. We implement a boundary refiner that cleans bounding whitespaces to recover exact ground-truth character spans.
2. **Sliding-Window Inference:** Legal documents frequently span thousands of tokens. We evaluate documents via sliding windows ($window = 512, stride = 256$) and deduplicate overlapping spans based on token confidence.
3. **Inverse Class-Frequency Focal Loss:** Majority background tokens (`"O"`) occupy $>86\%$ of tokens. Power-weighted inverse-frequency loss prevents minority classes (`WIT`, `R.COUNSEL`) from collapsing to zero recall.

---

## 3. Files in this Directory
- `CS781_Phase2_Champion_Pipeline.ipynb`: Self-contained runnable Jupyter/Kaggle notebook for training and evaluating Phase 2 champion models.
- `phase2_model.py`: Pure PyTorch definition of `HybridMoE` token tagging head and PEFT modules.
- `phase2_data.py`: Dataset loader, sliding-window chunker, and BPE character-to-token offset mapping.
- `phase2_evaluator.py`: Official strict Macro-F1 evaluator matching exact character offsets.
- `phase2_train_and_distill.py`: Modular training loop supporting FFT, LoRMA, LoRA, BitFit, and Knowledge Distillation.

---

## 4. How to Run
```bash
python phase2_train_and_distill.py --method lorma --epochs 5 --lr 2e-4
```
Outputs `prediction.json` and packages `submission.zip` ready for Codabench.
