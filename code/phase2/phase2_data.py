"""
CS-781 Capstone Project: Phase 2 Legal Named Entity Recognition (L-NER)
Data Pipeline, Precise Character-Token Alignment, and Chunking Engine

Features:
- Handles the 12 official L-NER legal entities + BIO tagging scheme (25 classes).
- Handles long legal documents via sliding-window chunking (context <= 512 tokens).
- Sub-character precision offset mapping with ZERO boundary drift.
- Fast PyTorch Dataset and DataLoader collation.
- Clean span decoding from token predictions to [[start, end, label]] format.
"""

import json
from typing import List, Dict, Tuple, Any, Optional
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


# ============================================================================
# 1. Official L-NER Label Taxonomy & BIO Tag Mapping
# ============================================================================

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

def build_bio_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds the 25-class BIO tag set:
    'O' + 12 * 'B-<LABEL>' + 12 * 'I-<LABEL>'
    """
    tag2id = {"O": 0}
    id2tag = {0: "O"}
    idx = 1
    for label in OFFICIAL_LABELS:
        b_tag = f"B-{label}"
        i_tag = f"I-{label}"
        tag2id[b_tag] = idx
        id2tag[idx] = b_tag
        idx += 1
        tag2id[i_tag] = idx
        id2tag[idx] = i_tag
        idx += 1
    return tag2id, id2tag

TAG2ID, ID2TAG = build_bio_vocab()
NUM_CLASSES = len(TAG2ID)  # 25


# ============================================================================
# 2. Strict Span Decoding & Boundary Refinement (Zero Character Drift)
# ============================================================================

def tokens_to_char_spans(
    token_tag_ids: List[int],
    offset_mapping: List[Tuple[int, int]],
    raw_text: str,
    id2tag: Dict[int, str] = ID2TAG
) -> List[List[Any]]:
    """
    Decodes predicted token tags into character-level entity spans:
    Returns [[start_char, end_char, label], ...]
    
    Includes boundary refinement:
    If a BPE token incorporates leading/trailing whitespace, trim offsets
    to point precisely to the non-whitespace characters of the entity span,
    avoiding the 1-character shift penalty in Strict Macro-F1.
    """
    spans = []
    current_label = None
    span_start_char = None
    span_end_char = None

    for idx, (tag_id, (char_start, char_end)) in enumerate(zip(token_tag_ids, offset_mapping)):
        # Ignore special/padding tokens
        if char_start == 0 and char_end == 0:
            continue

        tag_str = id2tag.get(tag_id, "O")

        if tag_str.startswith("B-"):
            # Close existing open span if any
            if current_label is not None and span_start_char is not None and span_end_char is not None:
                cleaned = refine_span_offsets(raw_text, span_start_char, span_end_char, current_label)
                if cleaned is not None:
                    spans.append(cleaned)

            current_label = tag_str[2:]
            span_start_char = char_start
            span_end_char = char_end

        elif tag_str.startswith("I-"):
            label = tag_str[2:]
            if current_label == label and span_start_char is not None:
                # Continuation of current span
                span_end_char = char_end
            else:
                # Malformed I- tag without matching B-: treat as new span or close previous
                if current_label is not None and span_start_char is not None:
                    cleaned = refine_span_offsets(raw_text, span_start_char, span_end_char, current_label)
                    if cleaned is not None:
                        spans.append(cleaned)
                current_label = label
                span_start_char = char_start
                span_end_char = char_end

        else:  # "O"
            if current_label is not None and span_start_char is not None:
                cleaned = refine_span_offsets(raw_text, span_start_char, span_end_char, current_label)
                if cleaned is not None:
                    spans.append(cleaned)
                current_label = None
                span_start_char = None
                span_end_char = None

    # Flush final open span
    if current_label is not None and span_start_char is not None and span_end_char is not None:
        cleaned = refine_span_offsets(raw_text, span_start_char, span_end_char, current_label)
        if cleaned is not None:
            spans.append(cleaned)

    return spans


def refine_span_offsets(raw_text: str, start: int, end: int, label: str) -> Optional[List[Any]]:
    """
    Trims leading/trailing whitespace from character offsets to ensure
    exact match with gold standard boundaries.
    """
    if start >= end or start >= len(raw_text) or end > len(raw_text):
        return None

    span_str = raw_text[start:end]
    
    # Trim leading whitespace
    stripped_left = span_str.lstrip()
    left_diff = len(span_str) - len(stripped_left)
    refined_start = start + left_diff

    # Trim trailing whitespace
    stripped_full = stripped_left.rstrip()
    if not stripped_full:
        return None
    refined_end = refined_start + len(stripped_full)

    return [refined_start, refined_end, label]


def deduplicate_and_merge_spans(spans: List[List[Any]]) -> List[List[Any]]:
    """
    Deduplicates spans from multiple overlapping sliding-window chunks.
    Sorts by start offset and resolves overlapping span collisions.
    """
    if not spans:
        return []

    # Sort primarily by start offset, secondarily by longest span
    sorted_spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    merged = []
    
    for s in sorted_spans:
        # Check if already present
        if any(m[0] == s[0] and m[1] == s[1] and m[2] == s[2] for m in merged):
            continue
        
        # Check if overlaps with last admitted span
        if merged and s[0] < merged[-1][1]:
            # Overlapping spans: keep the longer or earlier one
            continue
        else:
            merged.append(s)

    return merged


# ============================================================================
# 3. Document Chunking & Token-Label Alignment
# ============================================================================

def normalize_gold_spans(raw_spans: Any) -> List[List[Any]]:
    """
    Normalizes any gold annotations format into clean [[start, end, label], ...].
    Handles 4-item lists [start, end, label, text], dicts, or nested wrappers.
    """
    if not raw_spans:
        return []
    if isinstance(raw_spans, dict):
        for k in ["spans", "entities", "annotations"]:
            if k in raw_spans and isinstance(raw_spans[k], list):
                raw_spans = raw_spans[k]
                break
    normalized = []
    for item in raw_spans:
        if isinstance(item, (list, tuple)):
            if len(item) >= 3:
                start, end, label = item[0], item[1], item[2]
            else:
                continue
        elif isinstance(item, dict):
            start = item.get("start", item.get("start_char", None))
            end = item.get("end", item.get("end_char", None))
            label = item.get("label", item.get("entity", item.get("tag", None)))
        else:
            continue
        if start is not None and end is not None and label is not None:
            try:
                s, e, l = int(start), int(end), str(label).strip()
                if s < e and l in OFFICIAL_LABELS:
                    normalized.append([s, e, l])
            except (ValueError, TypeError):
                continue
    return sorted(normalized, key=lambda x: x[0])


def align_gold_spans_to_tokens(
    token_offsets: List[Tuple[int, int]],
    gold_spans: List[List[Any]],
    tag2id: Dict[str, int] = TAG2ID
) -> List[int]:
    """
    Maps character-level gold spans [start, end, label] to token-level BIO tag IDs.
    Returns a list of integer tag IDs matching the length of token_offsets.
    """
    num_tokens = len(token_offsets)
    labels = [tag2id["O"]] * num_tokens

    # Normalize gold spans
    sorted_golds = normalize_gold_spans(gold_spans)

    for g_start, g_end, g_label in sorted_golds:
        b_tag = tag2id.get(f"B-{g_label}", 0)
        i_tag = tag2id.get(f"I-{g_label}", 0)

        first_token_marked = False
        for tok_idx, (t_start, t_end) in enumerate(token_offsets):
            # Skip special tokens with 0, 0 offsets
            if t_start == 0 and t_end == 0:
                continue

            # Check overlap between token character interval and gold entity span
            # A token belongs to the entity if it has significant character overlap
            if max(t_start, g_start) < min(t_end, g_end):
                if not first_token_marked:
                    labels[tok_idx] = b_tag
                    first_token_marked = True
                else:
                    labels[tok_idx] = i_tag

    return labels


def chunk_document(
    doc_id: str,
    raw_text: str,
    tokenizer: Any,
    gold_spans: Optional[List[List[Any]]] = None,
    max_length: int = 512,
    stride: int = 256,
    tag2id: Dict[str, int] = TAG2ID
) -> List[Dict[str, Any]]:
    """
    Splits a document of arbitrary length into overlapping chunks of max_length tokens.
    Guarantees strict compliance with the model's 512-token context window limit.
    """
    # Fast tokenization with character offsets mapping
    enc = tokenizer(
        raw_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False
    )
    all_input_ids = enc["input_ids"]
    all_offsets = enc["offset_mapping"]

    # Align gold labels across the entire document first (if gold is provided)
    if gold_spans is not None:
        all_labels = align_gold_spans_to_tokens(all_offsets, gold_spans, tag2id=tag2id)
    else:
        all_labels = None

    chunks = []
    total_tokens = len(all_input_ids)

    if total_tokens == 0:
        return chunks

    for start_idx in range(0, total_tokens, stride):
        end_idx = min(start_idx + max_length, total_tokens)
        
        chunk_ids = all_input_ids[start_idx:end_idx]
        chunk_offsets = all_offsets[start_idx:end_idx]
        chunk_labels = all_labels[start_idx:end_idx] if all_labels is not None else [-100] * len(chunk_ids)

        # Pad to max_length for uniform tensor batching
        pad_len = max_length - len(chunk_ids)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 50256

        input_ids_padded = chunk_ids + [pad_token_id] * pad_len
        labels_padded = chunk_labels + [-100] * pad_len
        # Offsets for padded tokens are (0, 0)
        offsets_padded = chunk_offsets + [(0, 0)] * pad_len
        attention_mask = [1] * len(chunk_ids) + [0] * pad_len

        chunks.append({
            "doc_id": doc_id,
            "chunk_start_token": start_idx,
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "offset_mapping": offsets_padded,
            "raw_text": raw_text
        })

        if end_idx == total_tokens:
            break

    return chunks


# ============================================================================
# 4. PyTorch Dataset & DataLoader Collation
# ============================================================================

class LegalNERDataset(Dataset):
    def __init__(
        self,
        text_dict: Dict[str, str],
        gold_dict: Optional[Dict[str, List[List[Any]]]] = None,
        tokenizer: Any = None,
        max_length: int = 512,
        stride: int = 256
    ):
        self.tokenizer = tokenizer
        self.chunks: List[Dict[str, Any]] = []

        print(f"[Dataset] Processing {len(text_dict)} documents with max_length={max_length}, stride={stride}...")
        for doc_id, raw_text in text_dict.items():
            golds = gold_dict.get(doc_id, []) if gold_dict is not None else None
            doc_chunks = chunk_document(
                doc_id=doc_id,
                raw_text=raw_text,
                tokenizer=tokenizer,
                gold_spans=golds,
                max_length=max_length,
                stride=stride
            )
            self.chunks.extend(doc_chunks)

        print(f"[Dataset] Generated {len(self.chunks)} total training/inference chunks.")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.chunks[idx]
        return {
            "doc_id": item["doc_id"],
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "offset_mapping": item["offset_mapping"],
            "raw_text": item["raw_text"]
        }


def collate_ner_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "doc_ids": [b["doc_id"] for b in batch],
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "offset_mappings": [b["offset_mapping"] for b in batch],
        "raw_texts": [b["raw_text"] for b in batch]
    }
