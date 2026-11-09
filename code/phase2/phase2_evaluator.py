"""
CS-781 Capstone Project: Phase 2 Legal Named Entity Recognition (L-NER)
Official Strict Macro-F1 Evaluator & Validation Suite

Evaluation Criteria (Capstone Document Section 8.4 & 15.1):
- Primary Metric: Strict Macro-F1
- Secondary Metrics: Strict Macro-Precision, Strict Macro-Recall
- Protocol: Strict Span Matching (start offset, end offset, label must match EXACTLY).
- Covers all 12 official classes:
  APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO.
"""

import json
import argparse
from typing import Dict, List, Tuple, Any, Set

OFFICIAL_LABELS = [
    "APP",        # Appellant
    "RESP",       # Respondent
    "A.COUNSEL",  # Appellant's Counsel
    "R.COUNSEL",  # Respondent's Counsel
    "JUDGE",      # Judge
    "WIT",        # Witness
    "AUTH",       # Authority
    "COURT",      # Court
    "STAT",       # Statute
    "PREC",       # Precedent
    "DATE",       # Date
    "CASENO"      # Case Number
]


def validate_prediction_format(preds: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Checks that the prediction dictionary adheres strictly to the competition schema:
    {
        "<doc_id>": [
            [start_char, end_char, "LABEL"],
            ...
        ]
    }
    """
    errors = []
    if not isinstance(preds, dict):
        return False, ["Predictions root must be a JSON object mapping doc_id -> list of spans."]

    allowed_labels = set(OFFICIAL_LABELS)

    for doc_id, spans in preds.items():
        if not isinstance(spans, list):
            errors.append(f"Doc {doc_id}: Spans must be a list of lists.")
            continue
        for idx, span in enumerate(spans):
            if not isinstance(span, (list, tuple)) or len(span) < 3:
                errors.append(f"Doc {doc_id}, span #{idx}: Must contain at least [start, end, label]. Got: {span}")
                continue
            start, end, label = span[0], span[1], span[2]
            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"Doc {doc_id}, span #{idx}: Offsets must be integers. Got: ({type(start)}, {type(end)})")
            elif start < 0 or end <= start:
                errors.append(f"Doc {doc_id}, span #{idx}: Invalid interval bounds [{start}, {end}].")
            if label not in allowed_labels:
                errors.append(f"Doc {doc_id}, span #{idx}: Invalid label '{label}'. Allowed: {allowed_labels}")

    return len(errors) == 0, errors


def evaluate_lner_strict(
    gold_data: Dict[str, List[List[Any]]],
    pred_data: Dict[str, List[List[Any]]]
) -> Dict[str, Any]:
    """
    Computes Strict Macro-F1, Strict Macro-Precision, and Strict Macro-Recall.
    Matches predictions against gold references with exact boundary and label equivalence.
    """
    # Trackers per class
    tp_per_class = {c: 0 for c in OFFICIAL_LABELS}
    fp_per_class = {c: 0 for c in OFFICIAL_LABELS}
    fn_per_class = {c: 0 for c in OFFICIAL_LABELS}

    all_doc_ids = set(gold_data.keys()).union(set(pred_data.keys()))

    for doc_id in all_doc_ids:
        raw_golds = gold_data.get(doc_id, [])
        raw_preds = pred_data.get(doc_id, [])

        # Filter to official labels and tuple format (start, end, label)
        gold_spans: List[Tuple[int, int, str]] = [
            (g[0], g[1], g[2]) for g in raw_golds if g[2] in OFFICIAL_LABELS
        ]
        pred_spans: List[Tuple[int, int, str]] = [
            (p[0], p[1], p[2]) for p in raw_preds if p[2] in OFFICIAL_LABELS
        ]

        # Use set matching with multiplicity handling
        matched_golds = set()
        matched_preds = set()

        for p_idx, p in enumerate(pred_spans):
            for g_idx, g in enumerate(gold_spans):
                if g_idx in matched_golds:
                    continue
                # Exact match requirement
                if p[0] == g[0] and p[1] == g[1] and p[2] == g[2]:
                    matched_golds.add(g_idx)
                    matched_preds.add(p_idx)
                    tp_per_class[p[2]] += 1
                    break

        # False positives: predictions that did not match any gold span
        for p_idx, p in enumerate(pred_spans):
            if p_idx not in matched_preds:
                fp_per_class[p[2]] += 1

        # False negatives: gold spans that were not matched by any prediction
        for g_idx, g in enumerate(gold_spans):
            if g_idx not in matched_golds:
                fn_per_class[g[2]] += 1

    # Calculate per-class metrics
    class_metrics = {}
    f1_list = []
    prec_list = []
    rec_list = []

    for c in OFFICIAL_LABELS:
        tp = tp_per_class[c]
        fp = fp_per_class[c]
        fn = fn_per_class[c]

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        class_metrics[c] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1": round(f1, 4),
            "Gold_Count": tp + fn,
            "Pred_Count": tp + fp
        }
        f1_list.append(f1)
        prec_list.append(p)
        rec_list.append(r)

    # Macro averages across all 12 classes
    strict_macro_f1 = sum(f1_list) / len(f1_list)
    strict_macro_precision = sum(prec_list) / len(prec_list)
    strict_macro_recall = sum(rec_list) / len(rec_list)

    # Micro totals
    total_tp = sum(tp_per_class.values())
    total_fp = sum(fp_per_class.values())
    total_fn = sum(fn_per_class.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "Strict_Macro_F1": round(strict_macro_f1, 4),
        "Strict_Macro_Precision": round(strict_macro_precision, 4),
        "Strict_Macro_Recall": round(strict_macro_recall, 4),
        "Micro_F1": round(micro_f1, 4),
        "Per_Class_Metrics": class_metrics
    }


def print_evaluation_report(results: Dict[str, Any]):
    print("=" * 80)
    print("CS-781 PHASE 2: STRICT LEGAL NER (L-NER) EVALUATION REPORT")
    print("=" * 80)
    print(f"PRIMARY METRIC   - Strict Macro-F1:        {results['Strict_Macro_F1']:.4f}")
    print(f"SECONDARY METRIC - Strict Macro-Precision: {results['Strict_Macro_Precision']:.4f}")
    print(f"SECONDARY METRIC - Strict Macro-Recall:    {results['Strict_Macro_Recall']:.4f}")
    print(f"MICRO METRIC     - Micro-F1:               {results['Micro_F1']:.4f}")
    print("-" * 80)
    print(f"{'Class':<12} | {'Gold':<6} | {'Pred':<6} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Prec':<7} | {'Rec':<7} | {'F1':<7}")
    print("-" * 80)
    for c, m in results["Per_Class_Metrics"].items():
        print(f"{c:<12} | {m['Gold_Count']:<6} | {m['Pred_Count']:<6} | {m['TP']:<5} | {m['FP']:<5} | {m['FN']:<5} | {m['Precision']:<7.4f} | {m['Recall']:<7.4f} | {m['F1']:<7.4f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 L-NER Predictions")
    parser.add_argument("--gold", type=str, required=True, help="Path to gold JSON file (e.g. fold1-gold.json)")
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction JSON file")
    args = parser.parse_args()

    with open(args.gold, "r", encoding="utf-8") as f:
        gold_dict = json.load(f)
    with open(args.pred, "r", encoding="utf-8") as f:
        pred_dict = json.load(f)

    valid, errs = validate_prediction_format(pred_dict)
    if not valid:
        print(f"[ERROR] Found {len(errs)} format validation errors in predictions:")
        for err in errs[:10]:
            print("  ", err)
        if len(errs) > 10:
            print(f"  ... and {len(errs) - 10} more.")
        exit(1)

    res = evaluate_lner_strict(gold_dict, pred_dict)
    print_evaluation_report(res)
