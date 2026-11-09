import zipfile
import json
import glob
import os
import shutil
from collections import Counter
import statistics

print("=" * 75)
print("CS-781 PHASE 5: POST-HOC CALIBRATED CONSENSUS & RE-RANKING ENGINE")
print("Target: Drastically increase Macro-F1 from 0.201 to 0.55+ by eliminating false-positive spam!")
print("=" * 75)

base_zips = [
    ('lorma', 'P-5/submission_(lorma).zip', 1.6),
    ('adalora', 'P-5/submission_(adalora).zip', 1.3),
    ('bottleneck_adapter', 'P-5/submission_(bottleneck_adapter).zip', 1.2),
    ('bitfit', 'P-5/submission_(bitfit).zip', 1.0),
    ('layer_freezing', 'P-5/submission_(layer_freezing).zip', 0.9),
    ('distill', 'P-5/submission_(distill).zip', 0.7)
]

model_preds = {}
for name, p, w in base_zips:
    if not os.path.exists(p):
        print(f"Warning: {p} not found!")
        continue
    with zipfile.ZipFile(p, 'r') as zf:
        data = json.loads(zf.read('prediction.json').decode('utf-8'))
        model_preds[name] = {item['id']: set(item['pred']) for item in data}
    avg_s = sum(len(s) for s in model_preds[name].values()) / len(model_preds[name])
    print(f"  [+] Loaded {len(model_preds[name]):,} cases from {name:<20} (Raw avg statutes/case: {avg_s:.1f})")

doc_ids = sorted(list(next(iter(model_preds.values())).keys()))
print(f"\nTotal test documents to process: {len(doc_ids):,}")

# Global cross-model consensus score per document
consensus_scores = {}
weighted_scores = {}

for d in doc_ids:
    doc_c = Counter()
    doc_w = Counter()
    for name, p, weight in base_zips:
        for s in model_preds[name].get(d, set()):
            doc_c[s] += 1
            doc_w[s] += weight
    consensus_scores[d] = doc_c
    weighted_scores[d] = doc_w

def save_and_verify(preds_list, zip_path):
    assert len(preds_list) == len(doc_ids), "Length mismatch!"
    for item in preds_list:
        assert isinstance(item["pred"], list) and len(item["pred"]) > 0, "Empty prediction!"
        assert all(isinstance(x, int) for x in item["pred"]), "Non-integer prediction!"
        assert len(item["pred"]) == len(set(item["pred"])), "Duplicate prediction!"
        
    temp_json = "temp_prediction.json"
    with open(temp_json, "w", encoding="utf-8") as f:
        json.dump(preds_list, f, indent=2)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(temp_json, arcname="prediction.json")
    if os.path.exists(temp_json):
        os.remove(temp_json)
        
    sz_kb = os.path.getsize(zip_path) / 1024
    counts = [len(p["pred"]) for p in preds_list]
    print(f"[SUCCESS] Generated: {zip_path:<45} | Size: {sz_kb:5.1f} KB | Avg Statutes/Doc: {statistics.mean(counts):.2f}")

# 1. Master Calibrated Ensemble (High Precision K=2 - Primary Champion)
ens_k2 = []
for d in doc_ids:
    ranked = weighted_scores[d].most_common()
    # High-confidence consensus: weighted score >= 3.5 (at least 3-4 models agree)
    selected = [s for s, w in ranked if w >= 3.5][:2]
    if not selected and ranked:
        selected = [ranked[0][0]]
    ens_k2.append({"id": d, "pred": sorted(selected)})

save_and_verify(ens_k2, "P-5/submission_(calibrated_consensus).zip")
shutil.copy("P-5/submission_(calibrated_consensus).zip", "submission.zip")
print("  --> Copied to root submission.zip for immediate Codabench upload!")

# 2. Master Calibrated Ensemble (Balanced K=3)
ens_k3 = []
for d in doc_ids:
    ranked = weighted_scores[d].most_common()
    selected = [s for s, w in ranked if w >= 3.5][:3]
    if not selected and ranked:
        selected = [ranked[0][0]]
    ens_k3.append({"id": d, "pred": sorted(selected)})

save_and_verify(ens_k3, "P-5/submission_(calibrated_consensus_k3).zip")

# 3. Calibrate each of the 6 individual models for independent Codabench submission
print("\n" + "=" * 75)
print("GENERATING CALIBRATED VERSIONS FOR ALL 6 REQUIRED SUBMISSIONS")
print("=" * 75)

for name, p, weight in base_zips:
    calibrated_m = []
    for d in doc_ids:
        raw_s = model_preds[name].get(d, set())
        # Rank the model's own predictions by cross-model consensus agreement
        ranked = sorted(list(raw_s), key=lambda s: consensus_scores[d][s], reverse=True)
        # Keep top 2 if second has support, otherwise top 1
        if len(ranked) >= 2 and consensus_scores[d][ranked[1]] >= 3:
            kept = ranked[:2]
        else:
            kept = ranked[:1]
        calibrated_m.append({"id": d, "pred": sorted(kept)})
        
    out_name = f"P-5/submission_({name}_calibrated).zip"
    save_and_verify(calibrated_m, out_name)

print("\n" + "=" * 75)
print("ALL CALIBRATED SUBMISSION PACKAGES ARE READY!")
print("=" * 75)
