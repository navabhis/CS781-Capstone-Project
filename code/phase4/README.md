# CS-781 Phase 4: Court Judgment Prediction (CJPE)
## Ultra-Fast Dual-Anchor Legal NLP Pipeline (~8–10 Min / Submission)

### 1. Task & Data Overview
* **Objective:** Predict the binary outcome (`0` vs `1`) of court cases from full legal judgment documents.
* **Corpus Size:** **37,387 cases** (Label 0: 22,067, Label 1: 15,320).
* **Evaluation Metric:** **Strict Macro-F1** across classes `0` and `1`:
  $$\text{Macro-F1} = \frac{\text{F1}_0 + \text{F1}_1}{2}$$
* **Official Dataset:** `Exploration-Lab/CS-781-Capstone` (`Phase_4_CJPE/`):
  * `Train/train.json`
  * `Train/dev.json`
  * `Test/test.json`
* **Output Schema:** A JSON list of objects inside `submission.zip` with `prediction.json` at root:
  ```json
  [
    { "id": "case_001", "pred": 1 },
    { "id": "case_002", "pred": 0 }
  ]
  ```
* **Deadline:** October 21, 2026, 11:59 PM.

---

### 2. The 6 Required Submissions (Capstone Section 10.2.1)

All 6 submissions use the exact same notebook ([`CS781_Phase4_Champion_Pipeline.ipynb`](file:///c:/Users/HP/Documents/CS781%20Project/CS781_Phase4_Champion_Pipeline.ipynb)). You only change **one variable** (`EXPERIMENT_METHOD`) in **Cell 9**:

| Run | Folder | `EXPERIMENT_METHOD` | Trainable Params | Description & Rationale | Runtime |
|:---:|:---:|:---:|:---:|:---|:---:|
| **P4-1** | `P-4/Submission P4-1` | `"bitfit"` | ~0.1% | Parameter-efficient bias-only fine-tuning. Lightweight baseline. | **~7 mins** |
| **P4-2** | `P-4/Submission P4-2` | `"layer_freezing"` | ~0.3% | Backbone completely frozen; trains only the dual-anchor attention aggregator and classification head. | **~6 mins** |
| **P4-3** | `P-4/Submission P4-3` | `"bottleneck_adapter"` | ~0.6% | Series bottleneck adapter (Houlsby/Pfeiffer style) inserted into attention and FFN blocks. | **~8 mins** |
| **P4-4** | `P-4/Submission P4-4` | `"adalora"` | ~0.8% | Adaptive low-rank adaptation via SVD parameterization (Zhang et al., ICLR 2023) allocating rank dynamically across layers. | **~8 mins** |
| **P4-5** | `P-4/Submission P4-5` | `"distill"` | 3 Layers, Hidden 384 | **Upgraded Champion Student:** Siamese contrastive legal representation `[a0, a1, a1-a0, a1*a0]` with 100% focused Focal Loss & dense threshold sweep. | **~8.5 mins** |
| **P4-6** | `P-4/Submission P4-6` | `"lorma"` | ~0.8% | **Compulsory Champion:** Low-Rank Multiplicative Adaptation ($h = W_0 x + \alpha B A W_0 x, \alpha=1.0$) with cross-anchor attention. | **~8 mins** |
| **Bonus** | `P-4/` | *Cell 12 (Ensemble)* | Multi-Model | **Soft-Voting Ensemble:** Blends calibrated probabilities across trained runs (`probs_*.json`). | **~30 secs** |

*Total runtime across all 6 runs: ~45 to 55 minutes total (well within your 5-hour budget!)*

---

### 3. Why This Pipeline Runs in 8–10 Minutes (The Fast Architecture)

1. **Dual-Anchor Representation (`MAX_CHUNKS = 2`):**
   * Instead of wasting compute encoding 4 or 6 chunks, we extract the two decisive anchors of a court judgment:
     - **Anchor 0 (Head):** Opening 512 tokens (Facts, parties, lower court decision being appealed).
     - **Anchor 1 (Tail):** Closing 512 tokens (Court's reasoning, ratio decidendi, and verdict application).
   * In legal NLP benchmarks (ILDC / ECHR), Head + Tail carries >95% of the outcome prediction signal, while running **2x faster**.

2. **Stratified Balanced Subsampling (12,000 cases):**
   * 6,000 Label 0 + 6,000 Label 1 cases = 12,000 cases.
   * Eliminates the majority-class verdict bias at the data level.
   * With `BATCH_SIZE = 16`, 1 epoch is **only 750 batches** (vs 9,347 batches previously).
   * 2 full epochs = 1,500 optimizer updates = **~6 minutes of training**!

3. **Automated Validation & Threshold Sweep:**
   * Automatically sweeps classification threshold $\theta^* \in [0.25, 0.75]$ to maximize Macro-F1 on `dev.json` in **~45 seconds**.

---

### 4. Step-by-Step Kaggle Execution Guide (Fresh Start)

#### Step 1: Stop the Running Kaggle Session
1. In your Kaggle notebook, click the **Stop (■)** button in the top toolbar to terminate the slow run.
2. In the top right menu, click **Restart Session** (or refresh the page).

#### Step 2: Upload the Updated Notebook
1. Open [`CS781_Phase4_Champion_Pipeline.ipynb`](file:///c:/Users/HP/Documents/CS781%20Project/CS781_Phase4_Champion_Pipeline.ipynb).
2. In Kaggle, click **File -> Upload notebook** and upload this file.
3. Verify:
   - **Accelerator:** GPU (T4 x2 or P100) is turned ON.
   - **Data input:** `moe_backup_epoch4_COMPLETE` is attached in `/kaggle/input/`.
   - **Internet:** ON (if downloading from Hugging Face).

#### Step 3: Run Submission 1 (P4-1: BitFit)
1. By default, Cell 9 is set to `EXPERIMENT_METHOD = "bitfit"`.
2. Click **Run All**.
3. Training will take **~7 minutes** (750 steps per epoch, with live updates printed every 100 steps).
4. Cell 10 will validate the schema and generate `/kaggle/working/submission.zip`.
5. Download `submission.zip`, place it in `P-4/Submission P4-1/`, and upload to Codabench.

#### Step 4: Run Submissions 2 through 6
For each subsequent submission:
1. Scroll down to **Cell 9**.
2. Change `EXPERIMENT_METHOD` to the next method:
   - For P4-2: `EXPERIMENT_METHOD = "layer_freezing"`
   - For P4-3: `EXPERIMENT_METHOD = "bottleneck_adapter"`
   - For P4-4: `EXPERIMENT_METHOD = "adalora"`
   - For P4-5: `EXPERIMENT_METHOD = "distill"`
   - For P4-6: `EXPERIMENT_METHOD = "lorma"`
3. Run **Cell 9** and **Cell 10**.
4. Download the newly created `submission.zip` and place it in the respective `P-4/Submission P4-x/` folder.
5. Upload to Codabench!
