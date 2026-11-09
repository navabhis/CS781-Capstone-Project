"""
CS-781 Capstone Project: Phase 7 Prior Case Retrieval (PCR)
Calibration & Multi-Model Consensus Generator

Fixes the self-retrieval bug (c == qid) and performs weighted Reciprocal Rank Fusion (RRF)
across the 5 trained architectures (Distill, AdaLoRA, Bottleneck Adapter, BitFit, Layer Freezing).
Generates ready-to-submit packages for all Phase 7 submission slots (P7-1 to P7-6) and root submission.zip.
"""

import os
import json
import zipfile
import shutil
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
P7_DIR = BASE_DIR
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# 1. Load predictions from all 5 distinct model runs
models = ["distill", "adalora", "bottleneck_adapter", "bitfit", "layer_freezing"]
weights = {
    "distill": 2.5,             # Highest performing model (0.33 even with self-retrieval penalty)
    "adalora": 1.4,             # Dynamic rank SVD allocation
    "bottleneck_adapter": 1.2,  # Residual bottleneck projection
    "bitfit": 1.0,              # Bias-only adaptation
    "layer_freezing": 0.9       # Selective top-layer adaptation
}

all_preds = {}
for m in models:
    zip_path = os.path.join(P7_DIR, f"submission_({m}).zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing submission archive: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_preds[m] = json.loads(zf.read("prediction.json").decode("utf-8"))

test_q_ids = sorted(list(all_preds["distill"].keys()))
print(f"[1/5] Loaded {len(models)} model runs across {len(test_q_ids)} test queries.")

# 2. Forensic verification of self-retrieval
for m in models:
    self_count = sum(1 for qid in test_q_ids if all_preds[m][qid][0] == qid)
    print(f"  - Model {m:20s}: {self_count}/{len(test_q_ids)} queries have qid at Rank 1 (100% False Positive penalty)")

# 3. Build Calibrated Consensus & Calibrated Single-Model Packages
consensus_preds = {}
distill_calib = {}
adalora_calib = {}
bottleneck_calib = {}
bitfit_calib = {}
layer_calib = {}
peft_consensus_preds = {}

# Weights for PEFT-only consensus (Methods 1-4)
peft_weights = {
    "adalora": 1.5,
    "bottleneck_adapter": 1.3,
    "bitfit": 1.1,
    "layer_freezing": 1.0
}

for qid in test_q_ids:
    # Weighted RRF across all models (Distill + PEFT)
    rrf_all = Counter()
    for m in models:
        w = weights[m]
        # Filter out self-query immediately
        cands = [c for c in all_preds[m][qid] if c != qid]
        for rank, c in enumerate(cands):
            rrf_all[c] += w / (60.0 + rank)
    
    top7_all = [c for c, _ in rrf_all.most_common(7)]
    assert len(top7_all) == 7, f"Query {qid} has fewer than 7 consensus candidates!"
    consensus_preds[qid] = top7_all

    # Weighted RRF across PEFT methods only (for Submission P7-5)
    rrf_peft = Counter()
    for m in ["adalora", "bottleneck_adapter", "bitfit", "layer_freezing"]:
        w = peft_weights[m]
        cands = [c for c in all_preds[m][qid] if c != qid]
        for rank, c in enumerate(cands):
            rrf_peft[c] += w / (60.0 + rank)
    
    peft_cands = [c for c, _ in rrf_peft.most_common()]
    # If fewer than 7, backfill from rrf_all
    for c, _ in rrf_all.most_common():
        if c not in peft_cands:
            peft_cands.append(c)
        if len(peft_cands) == 7:
            break
    top7_peft = peft_cands[:7]
    assert len(top7_peft) == 7, f"Query {qid} has fewer than 7 PEFT candidates!"
    peft_consensus_preds[qid] = top7_peft

    # Single-model calibrated: keep the model's own top 6 non-self candidates, fill 7th from consensus
    def calibrate_single(model_name):
        cands = [c for c in all_preds[model_name][qid] if c != qid]
        for c, _ in rrf_all.most_common():
            if c not in cands:
                cands.append(c)
            if len(cands) == 7:
                break
        return cands

    distill_calib[qid] = calibrate_single("distill")
    adalora_calib[qid] = calibrate_single("adalora")
    bottleneck_calib[qid] = calibrate_single("bottleneck_adapter")
    bitfit_calib[qid] = calibrate_single("bitfit")
    layer_calib[qid] = calibrate_single("layer_freezing")

print("[2/5] Successfully generated calibrated predictions for all models and consensus.")

# 4. Strict Schema Verification Function
def verify_predictions(pred_dict, name):
    assert len(pred_dict) == len(test_q_ids), f"{name}: Query count mismatch {len(pred_dict)} vs {len(test_q_ids)}"
    for qid, preds in pred_dict.items():
        assert len(preds) == 7, f"{name}: Query {qid} does not have exactly 7 candidates ({len(preds)})"
        assert len(set(preds)) == 7, f"{name}: Query {qid} has duplicate candidates: {preds}"
        assert qid not in preds, f"{name}: Query {qid} contains self-query: {preds}"
    print(f"  [VERIFIED] {name:30s} -> Exactly 7 unique candidates, ZERO self-retrieval.")

# 5. Package submissions
packages = {
    # Method 1 (Submission P7-1): BitFit Calibrated
    "Submission P7-1": (bitfit_calib, "submission_(bitfit_calibrated).zip", "Method 1: BitFit Calibrated"),
    # Method 2 (Submission P7-2): Layer Freezing Calibrated
    "Submission P7-2": (layer_calib, "submission_(layer_freezing_calibrated).zip", "Method 2: Layer Freezing Calibrated"),
    # Method 3 (Submission P7-3): Bottleneck Adapter Calibrated
    "Submission P7-3": (bottleneck_calib, "submission_(bottleneck_adapter_calibrated).zip", "Method 3: Bottleneck Adapter Calibrated"),
    # Method 4 (Submission P7-4): AdaLoRA Calibrated
    "Submission P7-4": (adalora_calib, "submission_(adalora_calibrated).zip", "Method 4: AdaLoRA Calibrated"),
    # Method 5 (Submission P7-5): Multi-PEFT High-Confidence Consensus
    "Submission P7-5": (peft_consensus_preds, "submission_(peft_consensus).zip", "Method 5: Multi-PEFT Consensus"),
    # Method 6 (Submission P7-6): Distillation Calibrated (Previous Highest 0.33)
    "Submission P7-6": (distill_calib, "submission_(distill_calibrated).zip", "Method 6: Distillation Calibrated"),
}

print("[3/5] Verifying and writing all submission packages...")
for folder_name, (pred_data, zip_name, desc) in packages.items():
    verify_predictions(pred_data, desc)
    folder_path = os.path.join(P7_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Write prediction.json in folder
    pred_path = os.path.join(folder_path, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_data, f, indent=2)
    
    # Create submission.zip in folder
    sub_zip_path = os.path.join(folder_path, "submission.zip")
    with zipfile.ZipFile(sub_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.json")
        
    # Also write named zip in P7_DIR
    p7_zip_path = os.path.join(P7_DIR, zip_name)
    shutil.copy(sub_zip_path, p7_zip_path)
    print(f"  [SAVED] {desc} -> {folder_name}/submission.zip & {zip_name} ({os.path.getsize(sub_zip_path)/1024:.1f} KB)")

# 6. Master Grand Consensus (Root submission.zip)
print("[4/5] Building Master Grand Consensus (Distill + PEFT Fusion)...")
verify_predictions(consensus_preds, "Master Grand Consensus")

master_json_path = os.path.join(P7_DIR, "prediction_consensus.json")
with open(master_json_path, "w", encoding="utf-8") as f:
    json.dump(consensus_preds, f, indent=2)

master_zip_path = os.path.join(P7_DIR, "submission_(consensus).zip")
with zipfile.ZipFile(master_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(master_json_path, arcname="prediction.json")

# Copy to root submission.zip for immediate Codabench upload
root_submission = os.path.join(PROJECT_ROOT, "submission.zip")
shutil.copy(master_zip_path, root_submission)
# Also copy to P-7/submission.zip
shutil.copy(master_zip_path, os.path.join(P7_DIR, "submission.zip"))

print(f"[5/5] SUCCESS! Created Master Grand Consensus:")
print(f"  -> {root_submission} ({os.path.getsize(root_submission)/1024:.1f} KB)")
print(f"  -> {master_zip_path} ({os.path.getsize(master_zip_path)/1024:.1f} KB)")
print(f"  -> {os.path.join(P7_DIR, 'submission.zip')}")
print("ALL SUBMISSION PACKAGES ARE 100% READY FOR CODABENCH!")
