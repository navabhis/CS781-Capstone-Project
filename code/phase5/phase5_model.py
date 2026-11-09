# Step 0: Environment Setup & Deterministic Seeds
import os
import sys
import gc
import json
import math
import time
import random
import re
import zipfile
import shutil
import glob
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch Version: {torch.__version__}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")


#======================================================================

# Step 1: Pretrained Phase 1 Backbone Checkpoint Discovery
SAVED_CKPT_PATH = "/kaggle/working/champion_moe.pth"

def prepare_checkpoint():
    if os.path.exists(SAVED_CKPT_PATH) and os.path.getsize(SAVED_CKPT_PATH) > 100 * 1024 * 1024:
        return SAVED_CKPT_PATH

    data_pkl_matches = glob.glob("/kaggle/input/**/data.pkl", recursive=True)
    if data_pkl_matches:
        ckpt_dir = os.path.dirname(data_pkl_matches[0])
        print(f"[Safeguard] Detected unpacked checkpoint directory: {ckpt_dir}")
        print(f"[Safeguard] Packaging into PyTorch format at {SAVED_CKPT_PATH}...")
        with zipfile.ZipFile(SAVED_CKPT_PATH, "w", compression=zipfile.ZIP_STORED) as zf:
            for root, dirs, files in os.walk(ckpt_dir):
                for f in files:
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, ckpt_dir).replace(os.sep, "/")
                    arc_p = rel_p if rel_p.startswith("archive/") else f"archive/{rel_p}"
                    zf.write(full_p, arcname=arc_p)
        return SAVED_CKPT_PATH

    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            full_p = os.path.join(root, f)
            if not f.endswith((".json", ".csv", ".txt", ".py", ".md", ".png")):
                if os.path.isfile(full_p) and os.path.getsize(full_p) > 100 * 1024 * 1024:
                    print(f"[Safeguard] Found model checkpoint at {full_p}. Copying to {SAVED_CKPT_PATH}...")
                    shutil.copy(full_p, SAVED_CKPT_PATH)
                    return SAVED_CKPT_PATH

    return None

ckpt_src = prepare_checkpoint()
if ckpt_src:
    size_mb = os.path.getsize(ckpt_src) / (1024 * 1024)
    print("=" * 65)
    print(f"✅ PHASE 1 BACKBONE CHECKPOINT READY: {ckpt_src} ({size_mb:.1f} MB)")
    print("=" * 65)
else:
    print("=" * 65)
    print("⚠️ Checkpoint not found yet! Please attach Phase 1 dataset/model to Kaggle.")
    print("=" * 65)


#======================================================================

# Step 2: Acquire Official Phase 5 LSI Dataset (Local Input or Hugging Face Hub)
HF_REPO_ID = "Exploration-Lab/CS-781-Capstone"

hf_token = os.environ.get("HF_TOKEN", None)
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    print("🔑 Detected HF_TOKEN from Kaggle Secrets.")
except Exception:
    pass

if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF")
        print("🔑 Detected HF token from Kaggle Secrets.")
    except Exception:
        pass

if hf_token:
    try:
        from huggingface_hub import login
        login(token=hf_token)
        print("✅ Logged into Hugging Face successfully!")
    except Exception as e:
        print(f"HF login note: {e}")

files_to_download = [
    "Phase_5_LSI/Train/train.json",
    "Phase_5_LSI/Train/dev.json",
    "Phase_5_LSI/Test/test.json"
]

data_paths = {}
for rel_path in files_to_download:
    fname = os.path.basename(rel_path)
    local_matches = glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
    if local_matches:
        data_paths[rel_path] = local_matches[0]
        print(f"[Dataset] Found local file in /kaggle/input/: {local_matches[0]}")
        continue
        
    if os.path.exists(fname):
        data_paths[rel_path] = fname
        print(f"[Dataset] Found in working directory: {fname}")
        continue

    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=rel_path, repo_type="dataset", token=hf_token)
        data_paths[rel_path] = local_path
        print(f"[Dataset] Downloaded from HF: {rel_path} -> {local_path}")
    except Exception as e:
        print(f"[Dataset] Download notice for {rel_path}: {e}")


#======================================================================

# Step 3: Parse Dataset & Discover Statute Universe
def parse_lsi_file(file_path):
    if not file_path or not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    docs = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                d_id = str(item.get("id", item.get("case_id", len(docs))))
                text = item.get("text", item.get("case_text", ""))
                if isinstance(text, list):
                    text = " ".join(text)
                raw_lbls = item.get("labels", item.get("pred", []))
                if isinstance(raw_lbls, (int, str)):
                    raw_lbls = [raw_lbls]
                clean_lbls = [int(x) for x in raw_lbls if str(x).lstrip("-").isdigit()]
                docs.append({"id": d_id, "text": text.strip(), "labels": clean_lbls})
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                d_id = str(v.get("id", k))
                text = v.get("text", v.get("case_text", ""))
                if isinstance(text, list):
                    text = " ".join(text)
                raw_lbls = v.get("labels", v.get("pred", []))
                if isinstance(raw_lbls, (int, str)):
                    raw_lbls = [raw_lbls]
                clean_lbls = [int(x) for x in raw_lbls if str(x).lstrip("-").isdigit()]
                docs.append({"id": d_id, "text": text.strip(), "labels": clean_lbls})
            elif isinstance(v, str):
                docs.append({"id": str(k), "text": v.strip(), "labels": []})
    return docs

raw_train_docs = parse_lsi_file(data_paths.get("Phase_5_LSI/Train/train.json", ""))
dev_docs = parse_lsi_file(data_paths.get("Phase_5_LSI/Train/dev.json", ""))
test_docs = parse_lsi_file(data_paths.get("Phase_5_LSI/Test/test.json", ""))

all_statute_set = set()
for d in raw_train_docs + dev_docs:
    for sid in d["labels"]:
        all_statute_set.add(int(sid))

ALL_STATUTES = sorted(list(all_statute_set))
NUM_STATUTES = len(ALL_STATUTES)
statute_to_idx = {sid: i for i, sid in enumerate(ALL_STATUTES)}
idx_to_statute = {i: sid for i, sid in enumerate(ALL_STATUTES)}

with open("statute_vocab.json", "w", encoding="utf-8") as vf:
    json.dump({"statute_to_idx": statute_to_idx, "idx_to_statute": idx_to_statute}, vf, indent=2)

# Full Multi-Label Coverage Training Set (up to 22,000 cases for deep training)
def get_high_coverage_subset(docs, max_samples=22000, seed=42):
    if len(docs) <= max_samples:
        return docs
    random.seed(seed)
    statute_doc_map = {}
    for idx, d in enumerate(docs):
        for sid in d["labels"]:
            statute_doc_map.setdefault(sid, []).append(idx)
            
    mandatory_indices = set()
    for sid, doc_indices in statute_doc_map.items():
        chosen = random.sample(doc_indices, min(8, len(doc_indices)))
        mandatory_indices.update(chosen)
        
    remaining_indices = list(set(range(len(docs))) - mandatory_indices)
    needed = max_samples - len(mandatory_indices)
    if needed > 0 and remaining_indices:
        sampled_additional = random.sample(remaining_indices, min(needed, len(remaining_indices)))
        mandatory_indices.update(sampled_additional)
        
    final_docs = [docs[i] for i in sorted(list(mandatory_indices))]
    random.shuffle(final_docs)
    return final_docs

train_docs = get_high_coverage_subset(raw_train_docs, max_samples=22000, seed=42)

# Calibrated Positive Weights: bounded between [1.0, 3.0] to prevent false-positive explosion!
statute_counts = Counter()
for d in train_docs:
    for sid in d["labels"]:
        statute_counts[sid] += 1

total_train_cases = len(train_docs)
pos_weights_list = []
for sid in ALL_STATUTES:
    cnt = statute_counts.get(sid, 0)
    w = math.sqrt(float(total_train_cases) / float(max(1, cnt) + 100.0))
    w = max(1.0, min(3.0, w))  # Moderated scaling
    pos_weights_list.append(w)

pos_weights_tensor = torch.tensor(pos_weights_list, dtype=torch.float).to(device)

print("=" * 65)
print("PHASE 5 SINGLE CHAMPION DATASET SUMMARY:")
print(f"Total Available Training Cases: {len(raw_train_docs):,}")
print(f"Active Champion Training Set:   {len(train_docs):,}")
print(f"Validation Cases (dev.json):    {len(dev_docs):,}")
print(f"Test Cases (test.json):          {len(test_docs):,}")
print(f"Statute Universe:               {NUM_STATUTES} classes (IDs {min(ALL_STATUTES)} to {max(ALL_STATUTES)})")
print("=" * 65)


#======================================================================

# Step 4: Statute-Salient 3-Anchor Windowing (Head + Dense Mid + Tail = 1,536 tokens)
MAX_CHUNKS = 3
CHUNK_LEN = 512

STATUTE_KW_REGEX = re.compile(r'\b(section|sec|act|article|art|order|rule|code|penal|procedure|statute|schedule)\b', re.IGNORECASE)

def find_most_salient_statute_window(words, window_size=520, step=250):
    if len(words) <= window_size:
        return words
    best_start = 0
    max_kw_count = -1
    for i in range(250, len(words) - window_size, step):
        chunk_words = words[i : i + window_size]
        text_snippet = " ".join(chunk_words)
        kw_count = len(STATUTE_KW_REGEX.findall(text_snippet))
        if kw_count > max_kw_count:
            max_kw_count = kw_count
            best_start = i
    if max_kw_count <= 0:
        best_start = len(words) // 3
    return words[best_start : best_start + window_size]

class StatuteSalientDataset(Dataset):
    def __init__(self, doc_list, tokenizer, statute_map, max_chunks=MAX_CHUNKS, chunk_len=CHUNK_LEN):
        self.samples = []
        self.tokenizer = tokenizer
        self.statute_map = statute_map
        self.num_statutes = len(statute_map)
        self.max_chunks = max_chunks
        self.chunk_len = chunk_len
        pad_id = tokenizer.pad_token_id or 0
        
        for d in doc_list:
            doc_id = d["id"]
            text = d["text"]
            raw_labels = d.get("labels", [])
            
            target = np.zeros(self.num_statutes, dtype=np.float32)
            for sid in raw_labels:
                if sid in self.statute_map:
                    target[self.statute_map[sid]] = 1.0
            
            words = text.split()
            if len(words) > 1200:
                head_w = words[:520]
                mid_w = find_most_salient_statute_window(words, window_size=520, step=250)
                tail_w = words[-520:]
                
                c0 = tokenizer.encode(" ".join(head_w), add_special_tokens=True, truncation=True, max_length=chunk_len)
                c1 = tokenizer.encode(" ".join(mid_w), add_special_tokens=False, truncation=True, max_length=chunk_len)
                c2 = tokenizer.encode(" ".join(tail_w), add_special_tokens=False, truncation=True, max_length=chunk_len)
                chunks = [c0, c1, c2]
            elif len(words) > 600:
                c0 = tokenizer.encode(" ".join(words[:520]), add_special_tokens=True, truncation=True, max_length=chunk_len)
                c1 = tokenizer.encode(" ".join(words[-520:]), add_special_tokens=False, truncation=True, max_length=chunk_len)
                chunks = [c0, c1, c1]
            else:
                toks = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=chunk_len)
                chunks = [toks, toks, toks]
                
            chunk_ids = np.full((max_chunks, chunk_len), pad_id, dtype=np.int64)
            chunk_masks = np.zeros((max_chunks, chunk_len), dtype=np.int64)
            valid_mask = np.ones(max_chunks, dtype=bool)
            
            for c_i in range(max_chunks):
                ch = chunks[c_i][:chunk_len]
                chunk_ids[c_i, :len(ch)] = ch
                chunk_masks[c_i, :len(ch)] = 1
                
            self.samples.append({
                "doc_id": doc_id,
                "chunk_ids": chunk_ids,
                "chunk_masks": chunk_masks,
                "valid_mask": valid_mask,
                "target": target
            })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "doc_id": item["doc_id"],
            "chunk_input_ids": torch.tensor(item["chunk_ids"], dtype=torch.long),
            "chunk_attention_mask": torch.tensor(item["chunk_masks"], dtype=torch.long),
            "chunk_valid_mask": torch.tensor(item["valid_mask"], dtype=torch.bool),
            "target": torch.tensor(item["target"], dtype=torch.float)
        }

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 100000

print("Tokenizing datasets with 3-Anchor Statute-Salient Windowing...")
t0_tok = time.time()
train_ds = StatuteSalientDataset(train_docs, tokenizer, statute_to_idx, max_chunks=MAX_CHUNKS, chunk_len=CHUNK_LEN)
dev_ds = StatuteSalientDataset(dev_docs, tokenizer, statute_to_idx, max_chunks=MAX_CHUNKS, chunk_len=CHUNK_LEN)
test_ds = StatuteSalientDataset(test_docs, tokenizer, statute_to_idx, max_chunks=MAX_CHUNKS, chunk_len=CHUNK_LEN)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, pin_memory=True)
print(f"Tokenization complete in {time.time()-t0_tok:.1f}s! Multi-anchor DataLoaders ready.")


#======================================================================

# Step 5: Fast Statistical Lexical Prior (Catches Explicit Statute Citations)
print("Training Statistical Lexical Prior on statutory vocabulary...")
t0_lex = time.time()

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    
    # Train text snippets (opening 1,500 chars + statutory matches)
    def extract_lexical_snippet(d):
        t = d["text"]
        words = t.split()
        kw_matches = " ".join(STATUTE_KW_REGEX.findall(t))
        return " ".join(words[:400]) + " " + kw_matches + " " + " ".join(words[-400:])

    train_texts = [extract_lexical_snippet(d) for d in train_docs]
    test_texts = [extract_lexical_snippet(d) for d in test_docs]
    
    # Build Multi-Hot Y matrix for train
    Y_train = np.zeros((len(train_docs), NUM_STATUTES), dtype=np.float32)
    for i, d in enumerate(train_docs):
        for sid in d["labels"]:
            if sid in statute_to_idx:
                Y_train[i, statute_to_idx[sid]] = 1.0

    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(train_texts)
    X_test_vec = vectorizer.transform(test_texts)
    
    clf = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=150, solver="liblinear"), n_jobs=-1)
    clf.fit(X_train_vec, Y_train)
    
    lexical_test_probs = clf.predict_proba(X_test_vec)  # (N_test, 100)
    has_lexical_prior = True
    print(f"✅ Lexical Prior trained successfully in {time.time()-t0_lex:.1f}s! Ready for hybrid fusion.")
except Exception as e:
    print(f"Lexical Prior Note: {e} -> Continuing with pure neural champion.")
    has_lexical_prior = False
    lexical_test_probs = None


#======================================================================

# Step 6: Deep Legal Statute Champion Architecture (Pure PyTorch)
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim)
        )
    def forward(self, x, mask=None):
        n_x = self.norm1(x)
        attn_out, _ = self.attn(n_x, n_x, n_x, attn_mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class DeepLegalStatuteChampion(nn.Module):
    """
    Deep Legal Statute Identification Champion Model.
    4 Transformer Blocks (d=512, 8 heads), Pretrained Legal Embedding Warm-Start,
    Inter-Anchor Multi-Head Attention, and Attentive Document Pooling.
    """
    def __init__(self, vocab_count=50257, embed_size=512, heads=8, layers=4, num_classes=NUM_STATUTES, max_chunks=MAX_CHUNKS, dropout=0.15):
        super().__init__()
        self.embed_size = embed_size
        self.max_chunks = max_chunks
        self.num_classes = num_classes
        
        self.tok_emb = nn.Embedding(vocab_count, embed_size)
        self.pos_emb = nn.Embedding(512, embed_size)
        self.blocks = nn.ModuleList([TransformerBlock(embed_size, heads=heads, ffn_dim=2048, dropout=0.1) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(embed_size)
        
        # Inter-Anchor Positional Embeddings
        self.chunk_pos_emb = nn.Embedding(max_chunks, embed_size)
        self.inter_chunk_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        self.inter_norm = nn.LayerNorm(embed_size)
        
        # Attentive Anchor Scorer
        self.chunk_scorer = nn.Sequential(
            nn.Linear(embed_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Multi-Label Classification Head: [Weighted_Anchor, Max_Anchor] -> 2*d
        self.classifier = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, num_classes)
        )
        
    def forward(self, chunk_input_ids, chunk_attention_mask, chunk_valid_mask=None):
        b, k, seq_len = chunk_input_ids.shape
        flat_input_ids = chunk_input_ids.view(b * k, seq_len)
        flat_attn_mask = chunk_attention_mask.view(b * k, seq_len)
        
        pos = torch.arange(seq_len, device=flat_input_ids.device)
        out = self.tok_emb(flat_input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            out = block(out, mask=None)
        h = self.final_norm(out)
        
        # Masked mean pooling across token length
        mask_exp = flat_attn_mask.unsqueeze(-1).expand_as(h).float()
        chunk_embs = (torch.sum(h * mask_exp, dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-9)).view(b, k, self.embed_size)
        
        # Add anchor positional embeddings
        c_pos = torch.arange(k, device=chunk_input_ids.device)
        chunk_embs = chunk_embs + self.chunk_pos_emb(c_pos).unsqueeze(0)
        
        # Inter-Anchor Attention
        attn_out, _ = self.inter_chunk_attn(chunk_embs, chunk_embs, chunk_embs)
        chunk_embs = self.inter_norm(chunk_embs + attn_out)
        
        # Attentive Document Representation
        scores = self.chunk_scorer(chunk_embs)
        weights = F.softmax(scores, dim=1)
        weighted_doc = torch.sum(chunk_embs * weights, dim=1)
        max_doc, _ = torch.max(chunk_embs, dim=1)
        
        doc_rep = torch.cat([weighted_doc, max_doc], dim=-1)
        return self.classifier(doc_rep)

print("Initializing Deep Legal Statute Champion Model (4 Layers, Hidden 512, 8 Heads)...")
model = DeepLegalStatuteChampion(vocab_count=50257, embed_size=512, heads=8, layers=4, num_classes=NUM_STATUTES, max_chunks=MAX_CHUNKS)

# Warm-start token embeddings from Phase 1 Pretrained Backbone
try:
    p1_data = torch.load(SAVED_CKPT_PATH, map_location="cpu")
    if isinstance(p1_data, dict) and "state_dict" in p1_data: p1_data = p1_data["state_dict"]
    elif isinstance(p1_data, dict) and "model" in p1_data: p1_data = p1_data["model"]
    p1_data = {k.replace("module.", ""): v for k, v in p1_data.items()}
    if "tok_emb.weight" in p1_data:
        with torch.no_grad():
            model.tok_emb.weight.copy_(p1_data["tok_emb.weight"][:, :512])
        print("🔥 [Warm-Start] Transferred 50,257 pretrained legal token embeddings from Phase 1!")
except Exception as e:
    print(f"[Warm-Start Note] Default initialization: {e}")

model.to(device)
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Trainable Parameters: {trainable_p:,}")


#======================================================================

# Step 7: Calibrated Asymmetric Loss & Precision-Guarded Metric Evaluator
class MultiLabelAsymmetricLoss(nn.Module):
    """
    Calibrated Asymmetric Loss (ASL) with Margin Thresholding.
    Hard-zeros negative gradients below margin m=0.05 to eliminate false-positive probability bleed.
    """
    def __init__(self, pos_weights=None, gamma_neg=2.0, margin=0.05):
        super().__init__()
        self.pos_weights = pos_weights
        self.gamma_neg = gamma_neg
        self.margin = margin

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Positive loss with moderated class weighting
        pos_loss = targets * torch.log(torch.clamp(probs, min=1e-8))
        if self.pos_weights is not None:
            pos_loss = pos_loss * self.pos_weights
            
        # Negative loss with probability margin
        p_m = torch.clamp(probs - self.margin, min=0.0, max=1.0)
        neg_loss = (1.0 - targets) * (p_m ** self.gamma_neg) * torch.log(torch.clamp(1.0 - p_m, min=1e-8))
        
        return - (pos_loss + neg_loss).mean()

def precision_guarded_predict(probs_matrix, max_k=3, min_p2=0.30, min_p3=0.35, min_p4=0.45):
    """
    Precision-Guarded Dynamic Top-K Decision Rule.
    Prevents false-positive explosion by capping predictions to the true empirical court prior (1.5-2.2 statutes/case).
    """
    all_preds = []
    for doc_probs in probs_matrix:
        sorted_indices = np.argsort(doc_probs)[::-1]
        top1 = int(sorted_indices[0])
        p1 = float(doc_probs[top1])
        selected = [top1]
        
        # Add Top-2 if confident
        top2 = int(sorted_indices[1])
        p2 = float(doc_probs[top2])
        if p2 >= min_p2 and p2 >= 0.40 * p1:
            selected.append(top2)
            
            # Add Top-3 only if strongly supported
            top3 = int(sorted_indices[2])
            p3 = float(doc_probs[top3])
            if max_k >= 3 and p3 >= min_p3 and p3 >= 0.55 * p1:
                selected.append(top3)
                
                # Add Top-4 only in rare multi-statute cases
                if max_k >= 4:
                    top4 = int(sorted_indices[3])
                    p4 = float(doc_probs[top4])
                    if p4 >= min_p4 and p4 >= 0.70 * p1:
                        selected.append(top4)
                        
        all_preds.append(sorted(selected))
    return all_preds

def evaluate_lsi_macro_f1(gold_targets, pred_probs):
    gold = np.array(gold_targets)
    probs = np.array(pred_probs)
    N, C = gold.shape
    
    discrete_preds = precision_guarded_predict(probs, max_k=3)
    
    pred_matrix = np.zeros_like(gold)
    for i, p_list in enumerate(discrete_preds):
        for c in p_list:
            if c < C: pred_matrix[i, c] = 1
            
    per_class_f1 = []
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for c in range(C):
        g_c = gold[:, c]
        p_c = pred_matrix[:, c]
        tp = np.sum((p_c == 1) & (g_c == 1))
        fp = np.sum((p_c == 1) & (g_c == 0))
        fn = np.sum((p_c == 0) & (g_c == 1))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1.append(f1)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f1 = float(np.mean(per_class_f1))
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    
    avg_predicted_statutes = float(np.mean([len(p) for p in discrete_preds]))
    return macro_f1, micro_f1, avg_predicted_statutes


#======================================================================

# Step 8: High-Performance Champion Training Loop (~25–30 mins total)
EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 4e-4

criterion = MultiLabelAsymmetricLoss(pos_weights=pos_weights_tensor, gamma_neg=2.0, margin=0.05)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.08 * total_steps)

def lr_lambda(step):
    if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
    prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.05, 0.5 * (1.0 + math.cos(math.pi * prog)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

try:
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    autocast_ctx = lambda: torch.amp.autocast('cuda', enabled=torch.cuda.is_available())
except Exception:
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=torch.cuda.is_available())

best_macro_f1 = 0.0
best_micro_f1 = 0.0
best_model_path = "best_model_lsi_champion.pth"

print("=" * 70)
print(f"🚀 STARTING SINGLE HIGH-SCORE CHAMPION TRAINING: {EPOCHS} EPOCHS")
print(f"Batches per Epoch: {len(train_loader):,} | Batch Size: {BATCH_SIZE} | Expected Time: ~25-30 mins")
print("=" * 70)

t0_train = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    t0_ep = time.time()
    
    for step, batch in enumerate(train_loader):
        chunk_ids = batch["chunk_input_ids"].to(device)
        chunk_mask = batch["chunk_attention_mask"].to(device)
        valid_mask = batch["chunk_valid_mask"].to(device)
        targets = batch["target"].to(device)
        
        with autocast_ctx():
            logits = model(chunk_ids, chunk_mask, valid_mask)
            loss = criterion(logits, targets)
                
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        total_loss += loss.item()
        
        if (step + 1) % 250 == 0 or (step + 1) == len(train_loader):
            elapsed = time.time() - t0_ep
            pct = 100.0 * (step + 1) / len(train_loader)
            rem_batches = len(train_loader) - (step + 1)
            eta_min = (elapsed / (step + 1)) * rem_batches / 60.0
            print(f"  [Epoch {epoch:02d}/{EPOCHS:02d}] Step {step+1:4d}/{len(train_loader):4d} ({pct:5.1f}%) | Loss: {loss.item():.4f} | Elapsed: {elapsed/60:4.1f}m | ETA: {eta_min:4.1f}m")
            
    avg_loss = total_loss / len(train_loader)
    
    # Validation Evaluation
    model.eval()
    val_probs, val_gold = [], []
    with torch.no_grad():
        for batch in dev_loader:
            c_ids = batch["chunk_input_ids"].to(device)
            c_mask = batch["chunk_attention_mask"].to(device)
            v_mask = batch["chunk_valid_mask"].to(device)
            with autocast_ctx():
                l = model(c_ids, c_mask, v_mask)
                p = torch.sigmoid(l).cpu().numpy()
            val_probs.extend(p)
            val_gold.extend(batch["target"].numpy())
            
    val_macro_f1, val_micro_f1, avg_pred_s = evaluate_lsi_macro_f1(val_gold, val_probs)
    print(f"\n>>> Epoch [{epoch:02d}/{EPOCHS:02d}] COMPLETE | Train Loss: {avg_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f} | Val Micro-F1: {val_micro_f1:.4f} | Avg Statutes/Doc: {avg_pred_s:.2f} ({time.time()-t0_ep:.1f}s)\n")
    
    if val_macro_f1 > best_macro_f1:
        best_macro_f1 = float(val_macro_f1)
        best_micro_f1 = float(val_micro_f1)
        torch.save({
            "state_dict": model.state_dict(),
            "best_macro_f1": best_macro_f1,
            "best_micro_f1": best_micro_f1,
            "epoch": epoch
        }, best_model_path)
        print(f"🔥 Best Model Saved to {best_model_path}! Macro-F1: {best_macro_f1:.4f}")

print("=" * 70)
print(f"CHAMPION TRAINING COMPLETE in {(time.time()-t0_train)/60:.1f} mins! Best Validation Macro-F1: {best_macro_f1:.4f}")
print("=" * 70)


#======================================================================

# Step 9: Hybrid Neural-Lexical Inference & Codabench Packaging
print("Generating Final Predictions for Codabench Submission...")

if os.path.exists(best_model_path):
    saved_bundle = torch.load(best_model_path, map_location=device, weights_only=False)
    if isinstance(saved_bundle, dict) and "state_dict" in saved_bundle:
        model.load_state_dict(saved_bundle["state_dict"])
        best_macro_f1 = saved_bundle.get("best_macro_f1", best_macro_f1)
    else:
        model.load_state_dict(saved_bundle)
    print(f"Loaded best checkpoint (Validation Macro-F1: {best_macro_f1:.4f})")
model.eval()

test_doc_ids = []
neural_test_probs = []

with torch.no_grad():
    for batch in test_loader:
        c_ids = batch["chunk_input_ids"].to(device)
        c_mask = batch["chunk_attention_mask"].to(device)
        v_mask = batch["chunk_valid_mask"].to(device)
        with autocast_ctx():
            l = model(c_ids, c_mask, v_mask)
            p = torch.sigmoid(l).cpu().numpy()
        test_doc_ids.extend([str(d) for d in batch["doc_id"]])
        neural_test_probs.extend(p)

neural_test_probs_np = np.array(neural_test_probs)

# Hybrid Fusion: Blend Neural probabilities with Lexical Prior
if has_lexical_prior and lexical_test_probs is not None:
    print("Applying Hybrid Neural-Lexical Probability Fusion (0.75 Neural + 0.25 Lexical)...")
    final_test_probs = 0.75 * neural_test_probs_np + 0.25 * lexical_test_probs
else:
    print("Using Pure Neural Probabilities...")
    final_test_probs = neural_test_probs_np

# Precision-Guarded Dynamic Top-K Decision Rule
discrete_preds = precision_guarded_predict(final_test_probs, max_k=3, min_p2=0.30, min_p3=0.35)

test_predictions = []
pred_counts = []
for d_id, selected_indices in zip(test_doc_ids, discrete_preds):
    statute_ids = sorted(list(set(int(idx_to_statute[idx]) for idx in selected_indices)))
    test_predictions.append({"id": d_id, "pred": statute_ids})
    pred_counts.append(len(statute_ids))

# Strict Codabench Schema Verification
expected_ids = {d["id"] for d in test_docs}
actual_ids = {p["id"] for p in test_predictions}

assert len(test_predictions) == len(test_docs), f"Length mismatch! {len(test_predictions)} vs {len(test_docs)}"
assert expected_ids == actual_ids, "Test ID mismatch!"
for p in test_predictions:
    assert isinstance(p["pred"], list) and len(p["pred"]) > 0, f"Empty prediction: {p}"
    assert all(isinstance(x, int) for x in p["pred"]), f"Non-integer prediction: {p}"
    assert len(p["pred"]) == len(set(p["pred"])), f"Duplicate prediction: {p}"

print("\n" + "=" * 70)
print("✅ STRICT SCHEMA VALIDATION PASSED:")
print(f"  Total Test Cases:            {len(test_predictions):,}")
print(f"  Average Statutes per Case:   {np.mean(pred_counts):.2f} (Target prior: 1.7–2.1)")
print(f"  Median Statutes per Case:    {np.median(pred_counts)}")
print(f"  Cases with 1 statute:        {pred_counts.count(1):,} ({100*pred_counts.count(1)/len(test_predictions):.1f}%)")
print(f"  Cases with 2 statutes:       {pred_counts.count(2):,} ({100*pred_counts.count(2)/len(test_predictions):.1f}%)")
print(f"  Cases with 3 statutes:       {pred_counts.count(3):,} ({100*pred_counts.count(3)/len(test_predictions):.1f}%)")
print("=" * 70)

# Save prediction.json & Package submission.zip
with open("prediction.json", "w", encoding="utf-8") as f:
    json.dump(test_predictions, f, indent=2)

with zipfile.ZipFile("submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("prediction.json", arcname="prediction.json")

shutil.copy("submission.zip", "submission_champion_high_score.zip")
zip_kb = os.path.getsize("submission.zip") / 1024

print(f"\n🎉 SUCCESS! Created submission.zip ({zip_kb:.1f} KB) ready for Codabench upload!")
print(f"Sample Predictions (first 5 cases):")
for s_i, p_item in enumerate(test_predictions[:5]):
    print(f"  [{s_i+1:02d}] Case ID: {p_item['id']} -> Predicted Statutes: {p_item['pred']}")
