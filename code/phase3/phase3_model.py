# Step 0: Environment Setup & Deterministic Seeds
import os
import sys
import gc
import json
import math
import random
import zipfile
import shutil
import glob
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

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

    # 2. Check for unpacked checkpoint directory (contains data.pkl)
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

    # 3. Check for any checkpoint file (>100 MB) anywhere in /kaggle/input/
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

# Step 2: Acquire Official Phase 3 RR Dataset (Local Input or Hugging Face Hub)
HF_REPO_ID = "Exploration-Lab/CS-781-Capstone"

# Retrieve Hugging Face Access Token from Kaggle Secrets if available
hf_token = os.environ.get("HF_TOKEN", None)
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    print("🔑 Detected HF_TOKEN from Kaggle Secrets.")
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
    "Phase_3_RR/Train/CL_train.json",
    "Phase_3_RR/Train/CL_dev.json",
    "Phase_3_RR/Train/IT_train.json",
    "Phase_3_RR/Train/IT_dev.json",
    "Phase_3_RR/Train/label_vocab.json",
    "Phase_3_RR/Test/CL_test.json",
    "Phase_3_RR/Test/IT_test.json"
]

data_paths = {}
for rel_path in files_to_download:
    fname = os.path.basename(rel_path)
    # 1. Search locally in /kaggle/input/
    local_matches = glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
    if local_matches:
        data_paths[rel_path] = local_matches[0]
        print(f"[Dataset] Found local file in /kaggle/input/: {local_matches[0]}")
        continue
        
    # 2. Check current working directory
    if os.path.exists(fname):
        data_paths[rel_path] = fname
        print(f"[Dataset] Found in working directory: {fname}")
        continue

    # 3. Fallback to Hugging Face Hub download
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=rel_path, repo_type="dataset", token=hf_token)
        data_paths[rel_path] = local_path
        print(f"[Dataset] Downloaded from HF: {rel_path} -> {local_path}")
    except Exception as e:
        print(f"[Dataset] Download notice for {rel_path}: {e}")


#======================================================================

# Step 3: Parse Multi-Domain Datasets (CL & IT) and Build Label Vocab
OFFICIAL_RR_LABELS = [
    "Fact",
    "Issue",
    "ArgumentPetitioner",
    "ArgumentRespondent",
    "Statute",
    "Dissent",
    "PrecedentReliedUpon",
    "PrecedentNotReliedUpon",
    "PrecedentOverruled",
    "RulingByLowerCourt",
    "RatioOfTheDecision",
    "RulingByPresentCourt",
    "None"
]

# Check if official label_vocab.json exists
vocab_path = data_paths.get("Phase_3_RR/Train/label_vocab.json", "")
if os.path.exists(vocab_path):
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            v_data = json.load(f)
            if isinstance(v_data, list):
                OFFICIAL_RR_LABELS = v_data
            elif isinstance(v_data, dict):
                OFFICIAL_RR_LABELS = list(v_data.keys())
    except Exception as e:
        print(f"Using default official labels: {e}")

LABEL2ID = {lbl: idx for idx, lbl in enumerate(OFFICIAL_RR_LABELS)}
ID2LABEL = {idx: lbl for idx, lbl in enumerate(OFFICIAL_RR_LABELS)}
NUM_CLASSES = len(OFFICIAL_RR_LABELS)

print(f"Total Rhetorical Role Classes: {NUM_CLASSES}")
for idx, lbl in ID2LABEL.items():
    print(f"  [{idx:02d}] {lbl}")

def parse_rr_file(file_path):
    """
    Parses Phase 3 document files robustly across list-of-dicts and dict-of-dicts formats.
    Returns: list of dicts: [{'id': doc_id, 'domain': domain, 'sentences': [...], 'labels': [...]}]
    """
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    docs = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                d_id = str(item.get("id", item.get("doc_id", len(docs))))
                sents = item.get("text", item.get("sentences", []))
                lbls = item.get("labels", [])
                docs.append({"id": d_id, "sentences": sents, "labels": lbls})
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                d_id = str(v.get("id", k))
                sents = v.get("text", v.get("sentences", []))
                lbls = v.get("labels", [])
                docs.append({"id": d_id, "sentences": sents, "labels": lbls})
            elif isinstance(v, list):
                docs.append({"id": str(k), "sentences": v, "labels": []})
    return docs

cl_train_docs = parse_rr_file(data_paths["Phase_3_RR/Train/CL_train.json"])
it_train_docs = parse_rr_file(data_paths["Phase_3_RR/Train/IT_train.json"])
cl_dev_docs = parse_rr_file(data_paths["Phase_3_RR/Train/CL_dev.json"])
it_dev_docs = parse_rr_file(data_paths["Phase_3_RR/Train/IT_dev.json"])
cl_test_docs = parse_rr_file(data_paths["Phase_3_RR/Test/CL_test.json"])
it_test_docs = parse_rr_file(data_paths["Phase_3_RR/Test/IT_test.json"])

# Combined Cross-Domain Datasets for Maximum Representation Learning
train_docs = cl_train_docs + it_train_docs
dev_docs = cl_dev_docs + it_dev_docs

train_sents = sum(len(d["sentences"]) for d in train_docs)
dev_sents = sum(len(d["sentences"]) for d in dev_docs)
cl_test_sents = sum(len(d["sentences"]) for d in cl_test_docs)
it_test_sents = sum(len(d["sentences"]) for d in it_test_docs)

print("=" * 65)
print(f"PHASE 3 DATASET SUMMARY:")
print(f"Training:   {len(train_docs)} documents ({train_sents:,} sentences) [CL: {len(cl_train_docs)}, IT: {len(it_train_docs)}]")
print(f"Validation: {len(dev_docs)} documents ({dev_sents:,} sentences) [CL: {len(cl_dev_docs)}, IT: {len(it_dev_docs)}]")
print(f"Test (CL):  {len(cl_test_docs)} documents ({cl_test_sents:,} sentences)")
print(f"Test (IT):  {len(it_test_docs)} documents ({it_test_sents:,} sentences)")
print("=" * 65)


#======================================================================

# Step 4: Phase 1 Hybrid Mixture-of-Experts Architecture
class StandardExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
    def forward(self, x): 
        return self.net(x)

class MoERouter(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
    def forward(self, x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.max(probs, dim=-1)
        return weights, indices

class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.router = MoERouter(dim, num_experts)
        self.experts = nn.ModuleList([StandardExpert(dim) for _ in range(num_experts)])
        
    def forward(self, x):
        b, seq, d = x.shape
        flat_x = x.view(-1, d)
        weights, indices = self.router(flat_x)
        out = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            mask = (indices == i)
            if mask.any():
                exp_out = expert(flat_x[mask])
                out[mask] = exp_out * weights[mask].unsqueeze(-1)
        return out.view(b, seq, d)

class HybridBlock(nn.Module):
    def __init__(self, dim, heads, is_moe=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        if is_moe:
            self.ffn = MoELayer(dim, num_experts=4)
        else:
            self.ffn = StandardExpert(dim)
        self.attn_adapter = None
            
    def forward(self, x, mask=None):
        n_x = self.norm1(x)
        attn_out, _ = self.attn(n_x, n_x, n_x, attn_mask=mask)
        if hasattr(self, "attn_adapter") and self.attn_adapter is not None:
            attn_out = self.attn_adapter(attn_out)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class HybridMoE(nn.Module):
    def __init__(self, vocab_count=50257, embed_size=768, heads=12, layers=6):
        super().__init__()
        self.embed_size = embed_size
        self.tok_emb = nn.Embedding(vocab_count, embed_size)
        self.pos_emb = nn.Embedding(512, embed_size)
        
        self.blocks = nn.ModuleList()
        for i in range(layers):
            is_moe = (i % 2 != 0)
            self.blocks.append(HybridBlock(embed_size, heads, is_moe))
            
        self.final_norm = nn.LayerNorm(embed_size)
        self.output_head = nn.Linear(embed_size, vocab_count, bias=False)
        self.tok_emb.weight = self.output_head.weight 
        
    def get_hidden_states(self, x, causal=False):
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, device=x.device)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float("-inf"), diagonal=1) if causal else None
        out = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            out = block(out, mask)
        return self.final_norm(out)

    def forward(self, x, causal=True):
        h = self.get_hidden_states(x, causal=causal)
        return self.output_head(h)


#======================================================================

# Step 5: Sentence-Level Classifier & PEFT Modules (LoRA, rsLoRA, LoRMA, BitFit)

class HybridMoEForSentenceClassification(nn.Module):
    """
    Sentence-level classification model leveraging Phase 1 pretrained representations.
    Combines masked mean-pooled sentence representations with normalized document-level
    positional features (relative depth & sinusoidal projections) to model canonical legal judgment flow.
    """
    def __init__(self, backbone, num_labels=13, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        
        # 6 document position features -> 64-dim structural embedding
        self.pos_proj = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )
        self.classifier = nn.Sequential(
            nn.Linear(backbone.embed_size + 64, backbone.embed_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(backbone.embed_size, num_labels)
        )
        
    def forward(self, input_ids, attention_mask=None, pos_features=None):
        h = self.backbone.get_hidden_states(input_ids, causal=False)
        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).expand_as(h).float()
            sum_h = torch.sum(h * mask_exp, dim=1)
            sum_m = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
            pooled = sum_h / sum_m
        else:
            pooled = torch.mean(h, dim=1)
            
        if pos_features is not None:
            pos_emb = self.pos_proj(pos_features)
            combined = torch.cat([pooled, pos_emb], dim=-1)
        else:
            dummy_pos = torch.zeros(pooled.shape[0], 64, device=pooled.device)
            combined = torch.cat([pooled, dummy_pos], dim=-1)
            
        logits = self.classifier(self.dropout(combined))
        return logits

class CompactLegalStudentForSentenceClassification(nn.Module):
    """
    Mandatory Knowledge Distillation Student (Section 9.2.2).
    A lightweight 3-layer dense architecture with hidden size 384 (3x smaller than Teacher).
    """
    def __init__(self, vocab_count=50257, embed_size=384, heads=6, layers=3, num_labels=13, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.tok_emb = nn.Embedding(vocab_count, embed_size)
        self.pos_emb = nn.Embedding(512, embed_size)
        self.blocks = nn.ModuleList([HybridBlock(embed_size, heads, is_moe=False) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.pos_proj = nn.Sequential(
            nn.Linear(6, 48),
            nn.GELU(),
            nn.Linear(48, 48)
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_size + 48, embed_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, num_labels)
        )
        
    def forward(self, input_ids, attention_mask=None, pos_features=None):
        seq_len = input_ids.shape[1]
        pos = torch.arange(seq_len, device=input_ids.device)
        out = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            out = block(out, mask=None)
        h = self.final_norm(out)
        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).expand_as(h).float()
            pooled = torch.sum(h * mask_exp, dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        else:
            pooled = torch.mean(h, dim=1)
            
        if pos_features is not None:
            pos_emb = self.pos_proj(pos_features)
            combined = torch.cat([pooled, pos_emb], dim=-1)
        else:
            dummy_pos = torch.zeros(pooled.shape[0], 48, device=pooled.device)
            combined = torch.cat([pooled, dummy_pos], dim=-1)
            
        return self.classifier(self.dropout(combined))

# --- PEFT Adaptation Modules ---
class LoRMAAdapter(nn.Module):
    def __init__(self, dim, r=16, alpha=1.0, dropout=0.05):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.lorma_A = nn.Linear(dim, r, bias=False)
        self.lorma_B = nn.Linear(r, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.normal_(self.lorma_A.weight, std=1.0 / math.sqrt(r))
        nn.init.zeros_(self.lorma_B.weight)
        
    def forward(self, x):
        return x + self.lorma_B(self.lorma_A(self.dropout(x))) * self.alpha

class LoRMALinear(nn.Module):
    """LoRMA: Low-Rank Multiplicative Adaptation (Bihany et al., ACL 2025)"""
    def __init__(self, original_linear, r=16, alpha=1.0, dropout=0.05):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.lorma_A = nn.Linear(original_linear.out_features, r, bias=False)
        self.lorma_B = nn.Linear(r, original_linear.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.normal_(self.lorma_A.weight, std=1.0 / math.sqrt(r))
        nn.init.zeros_(self.lorma_B.weight)

    @property
    def weight(self): return self.original_linear.weight
    @property
    def bias(self): return self.original_linear.bias
        
    def forward(self, x):
        y0 = self.original_linear(x)
        return y0 + self.lorma_B(self.lorma_A(self.dropout(y0))) * self.alpha

class LoRAAdapter(nn.Module):
    def __init__(self, dim, r=16, alpha=32.0, dropout=0.05, rank_stabilized=False):
        super().__init__()
        self.scaling = (alpha / math.sqrt(r)) if rank_stabilized else (alpha / r)
        self.lora_A = nn.Linear(dim, r, bias=False)
        self.lora_B = nn.Linear(r, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        return x + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

class LoRALinear(nn.Module):
    """Standard LoRA and rsLoRA Linear Adapter"""
    def __init__(self, original_linear, r=16, alpha=32.0, dropout=0.05, rank_stabilized=False):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        self.scaling = (alpha / math.sqrt(r)) if rank_stabilized else (alpha / r)
        self.lora_A = nn.Linear(original_linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, original_linear.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self): return self.original_linear.weight
    @property
    def bias(self): return self.original_linear.bias
        
    def forward(self, x):
        return self.original_linear(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

def load_phase1_backbone(checkpoint_path, device="cpu"):
    backbone = HybridMoE(vocab_count=50257, embed_size=768, heads=12, layers=6)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    backbone.load_state_dict(clean_state_dict, strict=False)
    print(f"Successfully loaded Phase 1 backbone weights from: {checkpoint_path}")
    return backbone

def configure_adaptation(model, method="lorma", r=16, alpha=None):
    """
    Configures PEFT adaptation across the 6 required methodologies:
    'bitfit', 'linear_probe', 'lora', 'rslora', 'distill', 'lorma'
    """
    method = method.lower()
    if method == "full":
        for p in model.parameters(): p.requires_grad = True
    elif method in ["linear_probe", "linear_head_only"]:
        for p in model.backbone.parameters(): p.requires_grad = False
        for p in model.classifier.parameters(): p.requires_grad = True
    elif method == "bitfit":
        for n, p in model.named_parameters():
            p.requires_grad = ("bias" in n or "classifier" in n)
    elif method in ["lorma", "lora", "rslora"]:
        for p in model.backbone.parameters(): p.requires_grad = False
        for p in model.classifier.parameters(): p.requires_grad = True
        
        is_rslora = (method == "rslora")
        if alpha is None:
            alpha = 1.0 if method == "lorma" else (16.0 if is_rslora else 32.0)
            
        dim = model.backbone.embed_size
        for block in model.backbone.blocks:
            if method == "lorma":
                block.attn_adapter = LoRMAAdapter(dim, r=r, alpha=alpha)
                if isinstance(block.ffn, MoELayer):
                    for expert in block.ffn.experts:
                        expert.net[0] = LoRMALinear(expert.net[0], r=r, alpha=alpha)
                        expert.net[2] = LoRMALinear(expert.net[2], r=r, alpha=alpha)
                else:
                    block.ffn.net[0] = LoRMALinear(block.ffn.net[0], r=r, alpha=alpha)
                    block.ffn.net[2] = LoRMALinear(block.ffn.net[2], r=r, alpha=alpha)
            else:
                block.attn_adapter = LoRAAdapter(dim, r=r, alpha=alpha, rank_stabilized=is_rslora)
                if isinstance(block.ffn, MoELayer):
                    for expert in block.ffn.experts:
                        expert.net[0] = LoRALinear(expert.net[0], r=r, alpha=alpha, rank_stabilized=is_rslora)
                        expert.net[2] = LoRALinear(expert.net[2], r=r, alpha=alpha, rank_stabilized=is_rslora)
                else:
                    block.ffn.net[0] = LoRALinear(block.ffn.net[0], r=r, alpha=alpha, rank_stabilized=is_rslora)
                    block.ffn.net[2] = LoRALinear(block.ffn.net[2], r=r, alpha=alpha, rank_stabilized=is_rslora)
                    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[PEFT Config: {method.upper()}] Trainable Parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


#======================================================================

# Step 6: Dataset & Batch Collation
class RRSentenceDataset(Dataset):
    """
    Dataset for sentence-level Rhetorical Role prediction.
    Flattens documents into individual sentence samples while preserving document metadata.
    Computes 6-dim structural positional features (canonical document flow)
    and places target sentence first to eliminate truncation loss.
    """
    def __init__(self, doc_list, tokenizer, max_length=160, use_context=False):
        self.samples = []
        for d in doc_list:
            doc_id = d["id"]
            sentences = d["sentences"]
            labels = d.get("labels", [])
            n_sents = max(len(sentences), 1)
            
            for i, sent in enumerate(sentences):
                lbl_id = LABEL2ID.get(labels[i], -100) if i < len(labels) else -100
                curr_text = sent.strip()
                
                if use_context:
                    prev_text = sentences[i-1].strip() if i > 0 else ""
                    next_text = sentences[i+1].strip() if i < len(sentences) - 1 else ""
                    # Target sentence is placed FIRST so right-side truncation NEVER drops the target!
                    ctx = (prev_text + " " + next_text).strip()
                    input_text = f"{curr_text} [CONTEXT] {ctx}".strip()
                else:
                    input_text = curr_text
                    
                # 6-dim normalized positional features: [tau, sqrt(tau), sin(pi*tau), cos(pi*tau), sin(2pi*tau), cos(2pi*tau)]
                tau = i / (n_sents - 1) if n_sents > 1 else 0.5
                pos_feat = [
                    tau,
                    math.sqrt(tau),
                    math.sin(math.pi * tau),
                    math.cos(math.pi * tau),
                    math.sin(2.0 * math.pi * tau),
                    math.cos(2.0 * math.pi * tau)
                ]
                
                self.samples.append({
                    "doc_id": doc_id,
                    "sent_idx": i,
                    "total_sents": n_sents,
                    "text": input_text,
                    "label": lbl_id,
                    "pos_features": pos_feat
                })
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "doc_id": item["doc_id"],
            "sent_idx": item["sent_idx"],
            "total_sents": item["total_sents"],
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pos_features": torch.tensor(item["pos_features"], dtype=torch.float),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


#======================================================================

# Step 7: ClassBalancedFocalLoss & Strict Macro-F1 Evaluator
class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss (Lin et al. 2017; Cui et al. 2019)
    Combines inverse class frequency weighting with focal modulation gamma=1.5
    to penalize easy frequent examples and focus gradients on underrepresented rhetorical roles.
    """
    def __init__(self, weights=None, gamma=1.5, ignore_index=-100):
        super().__init__()
        self.weights = weights
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        valid_mask = (targets != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        v_logits = logits[valid_mask]
        v_targets = targets[valid_mask]
        
        ce_loss = F.cross_entropy(v_logits, v_targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_term = (1.0 - pt) ** self.gamma
        
        if self.weights is not None:
            alpha = self.weights[v_targets]
            loss = alpha * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss
            
        return loss.mean()

def evaluate_rr_macro_f1(gold_dict, pred_dict):
    """
    Computes official Macro-F1 across the 13 rhetorical role categories.
    Validates that every test document has exactly the required number of predictions.
    """
    tp = {c: 0 for c in OFFICIAL_RR_LABELS}
    fp = {c: 0 for c in OFFICIAL_RR_LABELS}
    fn = {c: 0 for c in OFFICIAL_RR_LABELS}
    
    for doc_id, g_lbls in gold_dict.items():
        p_lbls = pred_dict.get(doc_id, [])
        for g, p in zip(g_lbls, p_lbls):
            if g == p:
                tp[p] += 1
            else:
                fp[p] += 1
                fn[g] += 1
        # Account for length mismatch if any
        if len(p_lbls) < len(g_lbls):
            for g in g_lbls[len(p_lbls):]:
                fn[g] += 1

    f1_scores = {}
    for c in OFFICIAL_RR_LABELS:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1_scores[c] = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
    macro_f1 = sum(f1_scores.values()) / len(f1_scores)
    return macro_f1, f1_scores


#======================================================================

# Step 8: Configure & Execute Selected Experiment
# =============================================================================
# SELECT YOUR EXPERIMENT METHOD (Choose 1 of the 6 required):
#   - 'bitfit'       : Submission P3-1 (Bias-only tuning)
#   - 'linear_probe' : Submission P3-2 (Frozen backbone baseline)
#   - 'lora'         : Submission P3-3 (Standard LoRA, r=16, alpha=32.0)
#   - 'rslora'       : Submission P3-4 (Rank-Stabilized LoRA, alpha/sqrt(r))
#   - 'distill'      : Submission P3-5 (Mandatory Knowledge Distillation)
#   - 'lorma'        : Submission P3-6 (LoRMA Champion, r=16, alpha=1.0 - COMPULSORY)
# =============================================================================
EXPERIMENT_METHOD = "lorma"  # <-- Change this to run different submissions!

EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 6e-4 if EXPERIMENT_METHOD in ["lorma", "lora", "rslora"] else 1e-3
USE_CONTEXT = True

gc.collect()
torch.cuda.empty_cache()

PHASE1_CKPT = SAVED_CKPT_PATH
if not os.path.exists(PHASE1_CKPT):
    raise FileNotFoundError(f"Missing Phase 1 checkpoint at {PHASE1_CKPT}!")

# Datasets
train_ds = RRSentenceDataset(train_docs, tokenizer, max_length=160, use_context=USE_CONTEXT)
dev_ds = RRSentenceDataset(dev_docs, tokenizer, max_length=160, use_context=USE_CONTEXT)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, pin_memory=True)

# Compute Empirical Class Priors and Balanced Weights
label_counts = Counter([s["label"] for s in train_ds.samples if s["label"] >= 0])
total_labeled = sum(label_counts.values())
priors = torch.zeros(NUM_CLASSES, dtype=torch.float)
weights = torch.ones(NUM_CLASSES, dtype=torch.float)
max_c = max(label_counts.values()) if label_counts else 1

for lbl_id in range(NUM_CLASSES):
    cnt = label_counts.get(lbl_id, 1)
    priors[lbl_id] = cnt / total_labeled
    weights[lbl_id] = (max_c / (cnt + 5.0)) ** 0.50

weights = torch.clamp(weights, min=1.0, max=15.0)
weights = (weights / weights.mean()).to(device)
priors_device = priors.to(device)

print("Class Distribution & Calibrated Weights:")
for idx, lbl in ID2LABEL.items():
    c_n = label_counts.get(idx, 0)
    print(f"  [{idx:02d}] {lbl:<22} Count: {c_n:5d} ({100*c_n/total_labeled:5.1f}%) | Weight: {weights[idx].item():.2f}")

# Model Initialization
if EXPERIMENT_METHOD == "distill":
    print("[Distillation] Initializing Pretrained MoE Teacher...")
    teacher_backbone = load_phase1_backbone(PHASE1_CKPT, device="cpu")
    teacher = HybridMoEForSentenceClassification(teacher_backbone, num_labels=NUM_CLASSES)
    teacher = configure_adaptation(teacher, method="lorma", r=16, alpha=1.0)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    
    print("[Distillation] Initializing Compact Legal Student (3 Layers, Hidden 384)...")
    model = CompactLegalStudentForSentenceClassification(vocab_count=50257, embed_size=384, heads=6, layers=3, num_labels=NUM_CLASSES)
    model.to(device)
else:
    backbone = load_phase1_backbone(PHASE1_CKPT, device="cpu")
    model = HybridMoEForSentenceClassification(backbone, num_labels=NUM_CLASSES)
    model = configure_adaptation(model, method=EXPERIMENT_METHOD, r=16)
    model.to(device)

criterion = ClassBalancedFocalLoss(weights=weights, gamma=1.5, ignore_index=-100)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.01)
total_steps = EPOCHS * len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

best_f1 = 0.0
best_beta = 0.0
best_model_path = f"best_model_{EXPERIMENT_METHOD}.pth"

print("=" * 65)
print(f"STARTING TRAINING: METHOD={EXPERIMENT_METHOD.upper()} | EPOCHS={EPOCHS} | LR={LEARNING_RATE:.2e}")
print("=" * 65)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        pos_feat = batch["pos_features"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=attn_mask, pos_features=pos_feat)
        
        if EXPERIMENT_METHOD == "distill":
            with torch.no_grad():
                t_logits = teacher(input_ids, attention_mask=attn_mask, pos_features=pos_feat)
            T = 2.0
            kd_loss = F.kl_div(
                F.log_softmax(logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean"
            ) * (T * T)
            ce_loss = criterion(logits, labels)
            loss = 0.5 * ce_loss + 0.5 * kd_loss
        else:
            loss = criterion(logits, labels)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    
    # Validation: Collect raw logits to evaluate optimal Logit Adjustment beta
    model.eval()
    all_val_logits = []
    val_doc_ids = []
    val_sent_indices = []
    
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            pos_feat = batch["pos_features"].to(device)
            logits = model(input_ids, attention_mask=attn_mask, pos_features=pos_feat)
            all_val_logits.append(logits)
            val_doc_ids.extend(batch["doc_id"])
            val_sent_indices.extend(batch["sent_idx"].cpu().tolist())
            
    all_val_logits = torch.cat(all_val_logits, dim=0)
    gold_dev = {d["id"]: d["labels"] for d in dev_docs}
    
    # Search optimal beta in [0.0, 0.15, 0.25, 0.35, 0.45]
    candidate_betas = [0.0, 0.15, 0.25, 0.35, 0.45]
    epoch_best_beta = 0.0
    epoch_best_f1 = 0.0
    epoch_best_breakdown = {}
    
    for beta_val in candidate_betas:
        adj_logits = all_val_logits - beta_val * torch.log(priors_device + 1e-6)
        preds = torch.argmax(adj_logits, dim=-1).cpu().numpy()
        
        temp_preds = {}
        for d_id, s_idx, p in zip(val_doc_ids, val_sent_indices, preds):
            if d_id not in temp_preds: temp_preds[d_id] = []
            temp_preds[d_id].append((s_idx, ID2LABEL[p]))
            
        sorted_preds = {d: [p[1] for p in sorted(p_list, key=lambda x: x[0])] for d, p_list in temp_preds.items()}
        m_f1, f1_dict = evaluate_rr_macro_f1(gold_dev, sorted_preds)
        
        if m_f1 > epoch_best_f1:
            epoch_best_f1 = m_f1
            epoch_best_beta = beta_val
            epoch_best_breakdown = f1_dict
            
    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] Train Loss: {avg_loss:.4f} | Val Strict Macro-F1: {epoch_best_f1:.4f} (Optimal Beta: {epoch_best_beta:.2f})")
    
    if epoch_best_f1 > best_f1:
        best_f1 = epoch_best_f1
        best_beta = epoch_best_beta
        torch.save({
            "state_dict": model.state_dict(),
            "best_f1": best_f1,
            "best_beta": best_beta
        }, best_model_path)
        print(f"  🔥 New Best Model Saved! Macro-F1: {best_f1:.4f} (Beta: {best_beta:.2f})")
        print(f"     Key Roles F1: Fact={epoch_best_breakdown.get('Fact',0):.2f}, Ratio={epoch_best_breakdown.get('RatioOfTheDecision',0):.2f}, Overruled={epoch_best_breakdown.get('PrecedentOverruled',0):.2f}, Dissent={epoch_best_breakdown.get('Dissent',0):.2f}")

print("=" * 65)
print(f"TRAINING COMPLETE! Best Validation Macro-F1: {best_f1:.4f} with Logit Adjustment Beta: {best_beta:.2f}")
print("=" * 65)


#======================================================================

# Step 9: Final Test Inference across CL & IT Domains and submission.zip Packaging
print(f"Generating Test Predictions for {EXPERIMENT_METHOD.upper()}...")

if os.path.exists(best_model_path):
    saved_bundle = torch.load(best_model_path, map_location=device)
    if isinstance(saved_bundle, dict) and "state_dict" in saved_bundle:
        model.load_state_dict(saved_bundle["state_dict"])
        best_beta = saved_bundle.get("best_beta", best_beta)
        best_f1 = saved_bundle.get("best_f1", best_f1)
    else:
        model.load_state_dict(saved_bundle)
    print(f"Loaded checkpoint from {best_model_path} (Val F1: {best_f1:.4f}, Beta: {best_beta:.2f})")
model.eval()

# Combine CL Test and IT Test documents
all_test_docs = cl_test_docs + it_test_docs
test_ds = RRSentenceDataset(all_test_docs, tokenizer, max_length=160, use_context=USE_CONTEXT)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, pin_memory=True)

raw_test_preds = {}
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        pos_feat = batch["pos_features"].to(device)
        
        logits = model(input_ids, attention_mask=attn_mask, pos_features=pos_feat)
        adj_logits = logits - best_beta * torch.log(priors_device + 1e-6)
        preds = torch.argmax(adj_logits, dim=-1).cpu().numpy()
        
        for doc_id, s_idx, p in zip(batch["doc_id"], batch["sent_idx"], preds):
            if doc_id not in raw_test_preds: raw_test_preds[doc_id] = []
            raw_test_preds[doc_id].append((s_idx.item(), ID2LABEL[p]))

final_test_predictions = {
    doc_id: [p[1] for p in sorted(preds, key=lambda x: x[0])]
    for doc_id, preds in raw_test_preds.items()
}

# --- Strict Validation of Output Schema ---
doc_map = {d["id"]: d for d in all_test_docs}
validation_passed = True
total_sents_predicted = 0

for doc_id, pred_list in final_test_predictions.items():
    expected_len = len(doc_map[doc_id]["sentences"])
    actual_len = len(pred_list)
    total_sents_predicted += actual_len
    if expected_len != actual_len:
        print(f"❌ Length mismatch in Doc {doc_id}: Expected {expected_len}, got {actual_len}!")
        validation_passed = False
    for p in pred_list:
        if p not in OFFICIAL_RR_LABELS:
            print(f"❌ Invalid label in Doc {doc_id}: {p}")
            validation_passed = False

if validation_passed:
    print(f"✅ STRICT VALIDATION PASSED:")
    print(f"  Total Test Documents: {len(final_test_predictions)} (CL: {len(cl_test_docs)}, IT: {len(it_test_docs)})")
    print(f"  Total Sentences Predicted: {total_sents_predicted:,}")
    
    # Save prediction.json
    with open("prediction.json", "w", encoding="utf-8") as f:
        json.dump(final_test_predictions, f, indent=2)
        
    # Compress into submission.zip
    with zipfile.ZipFile("submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write("prediction.json", arcname="prediction.json")
        
    zip_size_kb = os.path.getsize("submission.zip") / 1024
    print(f"\n🎉 SUCCESS! Created submission.zip ({zip_size_kb:.1f} KB) with prediction.json at root.")
    
    # Sample Output Inspection
    sample_id = list(final_test_predictions.keys())[0]
    print(f"\nSample Predictions for Doc {sample_id} (first 5 sentences):")
    for s_i, lbl in enumerate(final_test_predictions[sample_id][:5]):
        print(f"  [Sentence {s_i+1:02d}]: {lbl}")
else:
    raise ValueError("Submission validation failed! Please check sentence lengths.")


#======================================================================

# Step 10: Benchmark Table Entry Generator
print(f"============================================================")
print(f"SUBMISSION SUMMARY FOR CAPSTONE REPORT (PHASE 3)")
print(f"============================================================")
print(f"Method:            {EXPERIMENT_METHOD.upper()}")
print(f"Validation F1:     {best_f1:.4f}")
print(f"Epochs Trained:    {EPOCHS}")
print(f"Batch Size:        {BATCH_SIZE}")
print(f"Learning Rate:     {LEARNING_RATE:.2e}")
print(f"Context-Aware:     {USE_CONTEXT}")
print(f"File Produced:     submission.zip (contains prediction.json)")
print(f"============================================================")
