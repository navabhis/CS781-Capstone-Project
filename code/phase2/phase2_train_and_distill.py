"""
CS-781 Capstone Project: Phase 2 Legal Named Entity Recognition (L-NER)
End-to-End Training, PEFT Adaptation, Knowledge Distillation & Inference Pipeline

Supports:
1. Full Fine-Tuning (Baseline)
2. LoRMA (Compulsory Low-Rank Multiplicative Adaptation)
3. LoRA (Additive Low-Rank Adaptation)
4. BitFit (Bias-only tuning)
5. Knowledge Distillation (Teacher -> Compact Student)
6. Dual GPU (T4) DataParallel Acceleration
7. Class-Weighted Loss (Tackles severe legal class imbalance)
8. Zero-Drift Prediction Formatting & Codabench submission.zip Packaging
"""

import os
import json
import zipfile
import argparse
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from phase2_model import (
    HybridMoE,
    HybridMoEForTokenClassification,
    CompactLegalStudent,
    load_phase1_backbone,
    apply_peft_adaptation
)
from phase2_data import (
    OFFICIAL_LABELS,
    TAG2ID,
    ID2TAG,
    NUM_CLASSES,
    LegalNERDataset,
    collate_ner_batch,
    tokens_to_char_spans,
    deduplicate_and_merge_spans
)
from phase2_evaluator import (
    evaluate_lner_strict,
    validate_prediction_format,
    print_evaluation_report
)


# ============================================================================
# 1. Class-Weighted Loss for Heavy Legal Entity Imbalance
# ============================================================================

def compute_class_weights(
    train_dataset: LegalNERDataset,
    num_classes: int = NUM_CLASSES,
    device: torch.device = torch.device("cpu"),
    smoothing: float = 0.5
) -> torch.Tensor:
    """
    Computes class weights inversely proportional to token frequency.
    Ensures rare legal entities (WIT, COUNSEL, AUTH) contribute fairly to Macro-F1.
    """
    counts = torch.zeros(num_classes, dtype=torch.float)
    for chunk in train_dataset.chunks:
        for lbl in chunk["labels"]:
            if 0 <= lbl < num_classes:
                counts[lbl] += 1.0

    counts = torch.clamp(counts, min=1.0)
    # Inverse frequency with power smoothing
    weights = (counts.max() / counts) ** smoothing
    # Moderate the "O" class weight so model doesn't over-predict entities
    weights[0] = 0.2
    # Normalize weights so mean is 1.0
    weights = weights / weights.mean()
    return weights.to(device)


# ============================================================================
# 2. Knowledge Distillation Loss
# ============================================================================

class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        # Hard label CrossEntropy
        ce_loss = self.ce(student_logits.view(-1, student_logits.shape[-1]), labels.view(-1))

        # Soft teacher distribution matching
        student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)

        # Mask padding positions (-100)
        mask = (labels != -100).unsqueeze(-1)
        student_log_probs = student_log_probs.masked_select(mask).view(-1, student_logits.shape[-1])
        teacher_probs = teacher_probs.masked_select(mask).view(-1, teacher_logits.shape[-1])

        if student_log_probs.numel() > 0:
            kl_loss = self.kl(student_log_probs, teacher_probs) * (self.temperature ** 2)
        else:
            kl_loss = torch.tensor(0.0, device=student_logits.device)

        return (1.0 - self.alpha) * ce_loss + self.alpha * kl_loss


# ============================================================================
# 3. Validation & Evaluation Loop
# ============================================================================

def run_evaluation(
    model: nn.Module,
    val_dataset: LegalNERDataset,
    val_gold_dict: Dict[str, List[List[Any]]],
    val_text_dict: Dict[str, str],
    device: torch.device,
    batch_size: int = 16
) -> Dict[str, Any]:
    model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ner_batch
    )

    doc_spans: Dict[str, List[List[Any]]] = {doc_id: [] for doc_id in val_text_dict.keys()}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            # Use module if wrapped in DataParallel
            outputs = model(input_ids)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            for doc_id, pred_seq, offsets, raw_text in zip(
                batch["doc_ids"], preds, batch["offset_mappings"], batch["raw_texts"]
            ):
                spans = tokens_to_char_spans(pred_seq.tolist(), offsets, raw_text)
                doc_spans[doc_id].extend(spans)

    # Deduplicate and merge overlapping chunks
    final_preds = {}
    for doc_id, spans in doc_spans.items():
        final_preds[doc_id] = deduplicate_and_merge_spans(spans)

    # Compute strict metrics
    results = evaluate_lner_strict(val_gold_dict, final_preds)
    return results, final_preds


# ============================================================================
# 4. Test Set Inference & Submission Packaging
# ============================================================================

def generate_test_predictions(
    model: nn.Module,
    test_dataset: LegalNERDataset,
    test_text_dict: Dict[str, str],
    device: torch.device,
    output_pred_path: str = "prediction.json",
    output_zip_path: str = "submission.zip",
    batch_size: int = 16
) -> Dict[str, List[List[Any]]]:
    print(f"\n[Inference] Running test inference on {len(test_text_dict)} test documents...")
    model.eval()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ner_batch
    )

    doc_spans: Dict[str, List[List[Any]]] = {doc_id: [] for doc_id in test_text_dict.keys()}

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            for doc_id, pred_seq, offsets, raw_text in zip(
                batch["doc_ids"], preds, batch["offset_mappings"], batch["raw_texts"]
            ):
                spans = tokens_to_char_spans(pred_seq.tolist(), offsets, raw_text)
                doc_spans[doc_id].extend(spans)

    final_predictions = {}
    for doc_id, spans in doc_spans.items():
        final_predictions[doc_id] = deduplicate_and_merge_spans(spans)

    # Validate output schema before packaging
    is_valid, errors = validate_prediction_format(final_predictions)
    if not is_valid:
        print("[WARNING] Prediction validation found errors:", errors[:5])
    else:
        print("[Inference] Predictions strictly valid according to competition schema!")

    # Write prediction.json
    with open(output_pred_path, "w", encoding="utf-8") as f:
        json.dump(final_predictions, f, indent=2)
    print(f"[Inference] Saved predictions to {output_pred_path}")

    # Package into submission.zip with prediction.json at root
    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_pred_path, arcname="prediction.json")
    print(f"[Inference] Generated Codabench submission archive: {output_zip_path}")

    return final_predictions


# ============================================================================
# 5. Main Training Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CS-781 Phase 2 L-NER Training & Adaptation Pipeline")
    parser.add_argument("--train_text_1", type=str, default="Phase_2_LNER/Train/fold1-text.json")
    parser.add_argument("--train_gold_1", type=str, default="Phase_2_LNER/Train/fold1-gold.json")
    parser.add_argument("--train_text_2", type=str, default="Phase_2_LNER/Train/fold2-text.json")
    parser.add_argument("--train_gold_2", type=str, default="Phase_2_LNER/Train/fold2-gold.json")
    parser.add_argument("--test_text", type=str, default="Phase_2_LNER/Test/fold3-text.json")
    parser.add_argument("--phase1_ckpt", type=str, default=None, help="Path to Phase 1 Champion .pth checkpoint")
    parser.add_argument("--method", type=str, default="lorma", choices=["full", "lorma", "lora", "bitfit", "linear_head_only", "distill"])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./phase2_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase2 Setup] Device: {device} (Count: {torch.cuda.device_count()})")

    # Load GPT-2 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Data Folds
    # Fold 1 as primary training, Fold 2 as validation (or 5-fold cross-validation)
    print("[Data] Loading fold 1 and fold 2 datasets...")
    with open(args.train_text_1, "r", encoding="utf-8") as f:
        train_text = json.load(f)
    with open(args.train_gold_1, "r", encoding="utf-8") as f:
        train_gold = json.load(f)

    with open(args.train_text_2, "r", encoding="utf-8") as f:
        val_text = json.load(f)
    with open(args.train_gold_2, "r", encoding="utf-8") as f:
        val_gold = json.load(f)

    has_test = os.path.exists(args.test_text)
    if has_test:
        with open(args.test_text, "r", encoding="utf-8") as f:
            test_text = json.load(f)
    else:
        test_text = {}

    train_dataset = LegalNERDataset(train_text, train_gold, tokenizer, max_length=args.max_length, stride=args.stride)
    val_dataset = LegalNERDataset(val_text, val_gold, tokenizer, max_length=args.max_length, stride=args.stride)
    test_dataset = LegalNERDataset(test_text, None, tokenizer, max_length=args.max_length, stride=args.stride) if has_test else None

    # Compute class weights to tackle extreme imbalance
    class_weights = compute_class_weights(train_dataset, device=device)

    # Initialize Model
    if args.method == "distill":
        # Knowledge Distillation: Teacher is pre-trained + adapted model; Student is CompactLegalStudent
        print("[Distillation] Initializing Teacher and Student models...")
        if args.phase1_ckpt and os.path.exists(args.phase1_ckpt):
            teacher_backbone = load_phase1_backbone(args.phase1_ckpt, device="cpu")
        else:
            teacher_backbone = HybridMoE()
        teacher = HybridMoEForTokenClassification(teacher_backbone, num_labels=NUM_CLASSES)
        teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        student = CompactLegalStudent(num_labels=NUM_CLASSES).to(device)
        model = student
        criterion = DistillationLoss(temperature=2.0, alpha=0.5, class_weights=class_weights)
        is_kd = True
    else:
        # Standard Adaptation / Fine-Tuning
        if args.phase1_ckpt and os.path.exists(args.phase1_ckpt):
            backbone = load_phase1_backbone(args.phase1_ckpt, device="cpu")
        else:
            print("[Warning] No Phase 1 checkpoint provided or path does not exist. Initializing fresh backbone.")
            backbone = HybridMoE()

        model = HybridMoEForTokenClassification(backbone, num_labels=NUM_CLASSES, causal=False)
        model = apply_peft_adaptation(model, method=args.method, r=args.lora_r, alpha=args.lora_alpha)
        model.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        is_kd = False

    # Multi-GPU DataParallel wrapper
    if torch.cuda.device_count() > 1:
        print(f"[Training] Utilizing {torch.cuda.device_count()} GPUs with nn.DataParallel!")
        model = nn.DataParallel(model)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_ner_batch
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )

    best_f1 = 0.0
    best_ckpt_path = os.path.join(args.output_dir, f"best_model_{args.method}.pth")

    print(f"\n[Training] Beginning training: Method={args.method.upper()}, Epochs={args.epochs}, LR={args.lr}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if is_kd:
                with torch.no_grad():
                    teacher_out = teacher(input_ids)
                    teacher_logits = teacher_out["logits"]
                student_out = model(input_ids)
                student_logits = student_out["logits"]
                loss = criterion(student_logits, teacher_logits, labels)
            else:
                outputs = model(input_ids)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                loss = criterion(logits.view(-1, NUM_CLASSES), labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if step % 20 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch} complete. Average Loss: {avg_loss:.4f}. Running Validation...")

        eval_res, _ = run_evaluation(model, val_dataset, val_gold, val_text, device)
        print_evaluation_report(eval_res)

        val_f1 = eval_res["Strict_Macro_F1"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"--> NEW BEST STRICT MACRO-F1: {best_f1:.4f}! Saved checkpoint to {best_ckpt_path}")

    # Generate Test Predictions with Best Model
    if test_dataset is not None:
        print(f"\nLoading best checkpoint ({best_ckpt_path}) for final test inference...")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        pred_path = os.path.join(args.output_dir, "prediction.json")
        zip_path = os.path.join(args.output_dir, "submission.zip")
        generate_test_predictions(model, test_dataset, test_text, device, pred_path, zip_path)


if __name__ == "__main__":
    main()
