# MIT License
#
# Copyright (c) 2025 Songnian Qian
# Multi-MLP Training Script: Simplified training script for Multi-Layer Perceptron model

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Tokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
from tqdm import tqdm
from torch import amp
import sys
import time
import signal
import argparse
import json
import math
from transformers import GPT2LMHeadModel
from context_readers_model import SingleMLP, GatingMLP, MultiMLPGPT2Model, MultiMLPTrainer, expert_kinds

import argparse
from torch.serialization import add_safe_globals
add_safe_globals([argparse.Namespace])  # allowlist for safe loads in PyTorch ≥2.6

# Import your existing dataset utility
project_root = Path(__file__).parent.parent 
src_path = project_root  
if src_path.exists():
    sys.path.insert(0, str(src_path))
    print(f"Added to Python path: {src_path}")
else:
    print(f"Warning: src folder not found at: {src_path}")

try:
    from dataset import WikiTextDataset
except ImportError:
    print("Warning: WikiTextDataset not found. Please ensure utils.dataset is available.")
    WikiTextDataset = None

# Global interrupt flag
training_interrupted = False

def signal_handler(signum, frame):
    global training_interrupted
    print("\nTraining interruption requested...")
    print("Will save checkpoint and exit after current batch...")
    training_interrupted = True

signal.signal(signal.SIGINT, signal_handler)

def _add_group(groups, params, lr, wd, name):
    if params:
        groups.append({
            "params": params,
            "lr": float(lr),
            "weight_decay": float(wd),
            "name": name,
        })

def parse_kind_bias(s: str):
    out = {}
    if not s: return out
    for item in s.split(","):
        k, v = item.split(":")
        out[k.strip()] = float(v.strip())
    return out

@torch.no_grad()
def _init_rank2_from_hf(expert, W1, b1, W2, b2):
    """
    Initialize ExpertC_Rank2 from HF MLP via top-2 SVD of an E×E map.
    Robust: handles transposed/variant layouts; always produces M:[E,E].
    Expected HF shapes (GPT-2): W1:[H,E], W2:[E,H], b1:[H], b2:[E]  (H=2E or 4E)
    """
    device = expert.w.device
    dtype_w = expert.w.dtype
    dtype_v = expert.v.dtype

    # Infer E from expert (safest source of truth)
    E = expert.w.size(1)        # w: [2, E]
    # Candidate matrices that *might* yield [E,E]
    candidates = []

    # Put all combos that could align to [E,E]
    # (matmul order matters; we test dimensions before using)
    combos = [
        ("W2@W1",      lambda: W2 @ W1),
        ("W2@W1.t()",  lambda: W2 @ W1.t()),
        ("W2.t()@W1",  lambda: W2.t() @ W1),
        ("W2.t()@W1.t()", lambda: W2.t() @ W1.t()),
        ("W1@W2",      lambda: W1 @ W2),
        ("W1@W2.t()",  lambda: W1 @ W2.t()),
        ("W1.t()@W2",  lambda: W1.t() @ W2),
        ("W1.t()@W2.t()", lambda: W1.t() @ W2.t()),
    ]

    for name, fn in combos:
        try:
            A = fn()
        except Exception:
            continue
        if A.dim() == 2 and A.size(0) == E and A.size(1) == E:
            candidates.append(A)

    if not candidates:
        raise RuntimeError(
            f"Could not form an E×E map from provided W1/W2. "
            f"Got W1:{tuple(W1.shape)} W2:{tuple(W2.shape)} while expert expects E={E}."
        )

    # Prefer first valid candidate (W2@W1 will hit on GPT-2), use CPU/float32 for SVD stability
    M = candidates[0].to(torch.float32).cpu()  # [E,E]

    # SVD
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)  # U:[E,E], S:[E], Vh:[E,E]

    # Top-2 components
    for i in range(2):
        u_i = U[:, i].to(device=device, dtype=dtype_v)        # [E]
        v_i = Vh[i, :].to(device=device, dtype=dtype_w)       # [E]
        expert.w[i].copy_(v_i)                                # input projection
        expert.v[i].copy_(u_i)                                # output shaping
        denom = v_i.norm().clamp_min(1e-8).item()
        expert.alpha[i].fill_( 2.0 * S[i].item() / denom )    # ~compensate GELU 0.5

    # Bias path (project b1 through W2)
    u_bias = (W2 @ b1).to(device=device, dtype=expert.biasvec.dtype) \
             if W2.size(1) == b1.size(0) else \
             (W2.t() @ b1).to(device=device, dtype=expert.biasvec.dtype)

    b_scalar = u_bias.mean().item() if u_bias.numel() == E else 0.0
    expert.b.copy_( torch.tensor([b_scalar, b_scalar], device=device, dtype=expert.b.dtype) )

    # biasvec = b2 + 0.5 * (W2 @ b1)/E   (fallback to b2 if mismatch)
    if b2.numel() == E and u_bias.numel() == E:
        expert.biasvec.copy_( b2.to(device=device, dtype=expert.biasvec.dtype) + 0.5 * (u_bias / E) )
    else:
        expert.biasvec.copy_( b2.to(device=device, dtype=expert.biasvec.dtype) )

def _maybe_init_all_perc_from_hf(model, config_name):
    """
    Initialize any perceptron expert with parameters (w[K,E], v[K,E], alpha[K], b[K], biasvec[E])
    from the pretrained HF MLP via top-K SVD of an inferred E×E map (usually W2@W1).
    Works for perc2, perc4, etc.
    """
    ref = GPT2LMHeadModel.from_pretrained(config_name).transformer
    ref_blocks = ref.h
    mdl_blocks = model.h

    for i, block in enumerate(mdl_blocks):
        layer_mlp = getattr(block, "mlp", None)
        if layer_mlp is None or not hasattr(layer_mlp, "mlps"):
            continue

        hf_block = ref_blocks[i]
        W1 = hf_block.mlp.c_fc.weight.data     # [H,E]
        b1 = hf_block.mlp.c_fc.bias.data       # [H]
        W2 = hf_block.mlp.c_proj.weight.data   # [E,H]
        b2 = hf_block.mlp.c_proj.bias.data     # [E]

        for m in layer_mlp.mlps:
            w = getattr(m, "w", None)
            v = getattr(m, "v", None)
            alpha = getattr(m, "alpha", None)
            b = getattr(m, "b", None)
            biasvec = getattr(m, "biasvec", None)

            # Only perceptron-style experts (perc2/perc4/…)
            if any(x is None for x in (w, v, alpha, b, biasvec)):
                continue
            if not (isinstance(w, torch.nn.Parameter) and isinstance(v, torch.nn.Parameter)):
                continue
            if w.dim() != 2 or v.dim() != 2:
                continue

            K, E = w.size(0), w.size(1)
            if K < 1 or v.size(0) != K or v.size(1) != E:
                continue

            # Try several ways to form an E×E map robustly
            candidates = []
            combos = [
                lambda: W2 @ W1, lambda: W2 @ W1.t(),
                lambda: W2.t() @ W1, lambda: W2.t() @ W1.t(),
                lambda: W1 @ W2, lambda: W1 @ W2.t(),
                lambda: W1.t() @ W2, lambda: W1.t() @ W2.t(),
            ]
            for fn in combos:
                try:
                    A = fn()
                except Exception:
                    continue
                if A.dim() == 2 and A.size(0) == E and A.size(1) == E:
                    candidates.append(A)
            if not candidates:
                continue

            M = candidates[0].to(torch.float32).cpu()  # [E,E]
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)

            # Top-K projections
            for r in range(min(K, S.numel())):
                u_r = U[:, r].to(device=v.device, dtype=v.dtype)
                v_r = Vh[r, :].to(device=w.device, dtype=w.dtype)
                with torch.no_grad():
                    w[r].copy_(v_r)  # input projection
                    v[r].copy_(u_r)  # output shaping
                    denom = v_r.norm().clamp_min(1e-8).item()
                    alpha[r].fill_( 2.0 * S[r].item() / denom )  # ~compensate GELU 0.5

            # Bias path & biasvec
            try:
                u_bias = (W2 @ b1) if W2.size(1) == b1.size(0) else (W2.t() @ b1)
                u_bias = u_bias.to(device=biasvec.device, dtype=biasvec.dtype)
                with torch.no_grad():
                    b.fill_(float(u_bias.mean().item() if u_bias.numel() == E else 0.0))
                    if b2.numel() == E and u_bias.numel() == E:
                        biasvec.copy_( b2.to(biasvec.device, biasvec.dtype) + 0.5 * (u_bias / E) )
                    else:
                        biasvec.copy_( b2.to(biasvec.device, b2.dtype) )
            except Exception:
                pass

    print("[init] Perceptron experts (any K) initialized from HF MLP via top-K SVD.")

def build_optimizer(model, args):
    wd = getattr(args, "weight_decay", 1e-2)
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight", "ln_", "ln1", "ln_1", "ln2", "ln_2"}

    mlp_w, mlp_nd = [], []
    gate_w, gate_nd = [], []
    embed_w, embed_nd = [], []
    backbone_w, backbone_nd = [], []

    # --- 1) Collect exact gate params by module type ---
    # Adjust the import/type to your actual class name if different
    from context_readers_model import GatingMLP  # or wherever it's defined
    gate_param_ids = set()
    for blk in getattr(model, "h", []):
        mlp = getattr(blk, "mlp", None)
        if mlp is None:
            continue
        gate = getattr(mlp, "gate", None)
        if isinstance(gate, GatingMLP):
            for p in gate.parameters(recurse=True):
                gate_param_ids.add(id(p))

    # --- 2) Bucket all params ---
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_no_decay = any(s in name for s in no_decay)
        is_embed = name.startswith("wte.") or name.startswith("lm_head.")
        is_gate = id(p) in gate_param_ids

        # Treat non-gate MLP experts as "mlp_*" so you can freeze/unfreeze them separately if you wish
        # Heuristic: parameters in expert stacks `.mlps` and not in gate
        is_mlp = ((".mlps" in name) or (".mlp." in name)) and not is_gate

        if is_embed:
            (embed_nd if is_no_decay else embed_w).append(p)
        elif is_gate:
            (gate_nd if is_no_decay else gate_w).append(p)
        elif is_mlp:
            (mlp_nd if is_no_decay else mlp_w).append(p)
        else:
            (backbone_nd if is_no_decay else backbone_w).append(p)

    groups = []
    if mlp_w:   groups.append({"params": mlp_w,   "lr": args.lr,      "weight_decay": wd,   "name": "mlp_w"})
    if mlp_nd:  groups.append({"params": mlp_nd,  "lr": args.lr,      "weight_decay": 0.0, "name": "mlp_nd"})
    if gate_w:  groups.append({"params": gate_w,  "lr": args.gate_lr, "weight_decay": wd,   "name": "gate_w"})
    if gate_nd: groups.append({"params": gate_nd, "lr": args.gate_lr, "weight_decay": 0.0, "name": "gate_nd"})
    if embed_w: groups.append({"params": embed_w, "lr": args.lr,      "weight_decay": wd,   "name": "embed_w"})
    if embed_nd:groups.append({"params": embed_nd,"lr": args.lr,      "weight_decay": 0.0, "name": "embed_nd"})
    if backbone_w:  groups.append({"params": backbone_w,  "lr": args.lr, "weight_decay": wd,   "name": "backbone_w"})
    if backbone_nd: groups.append({"params": backbone_nd, "lr": args.lr, "weight_decay": 0.0, "name": "backbone_nd"})

    # Sanity: warn if gate groups ended up empty
    if not gate_w and not gate_nd:
        print("[WARN] No gate parameters were found; routing may stay uniform. Check GatingMLP detection.")

    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    return optimizer

def _str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes","true","t","y","1"): return True
    if v in ("no","false","f","n","0"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
   
    parser = argparse.ArgumentParser(description="Multi-MLP Model Training")
    
    # Model configuration
    parser.add_argument("--config_name", type=str, default="gpt2",
                        help="HF config name (e.g., gpt2, gpt2-medium)")
    parser.add_argument("--num_mlps", type=int, default=4,  
                        help="Number of MLPs per layer")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gate_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=0, help="0 = use epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_pretrained", type=_str2bool, default=True)
    
    # Data parameters
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_length", type=int, default=256)
    
    # Routing parameters (set via command line)
    parser.add_argument("--soft_routing", type=_str2bool, default=True,
                        help="Use soft routing (True) or hard routing (False)")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Temperature for routing")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Number of experts to use")
    parser.add_argument("--compute_only_selected", type=_str2bool, default=False,
                        help="Compute only selected experts (for efficiency)")
    parser.add_argument("--force_uniform_gate", type=_str2bool, default=False,
                        help="force_uniform_gate  (True) or (False)")
    
    # Regularization
    parser.add_argument("--load_balance_coef", type=float, default=0.01,
                        help="Load balancing regularization coefficient")
    parser.add_argument("--diversity_coef", type=float, default=0.001,
                        help="Perceptron diversity regularization coefficient")
    
    # Logging and saving
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    
    # Environment
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lm_lr", type=float, default=None, help="LR for lm_head (defaults to head_lr)")
    parser.add_argument("--backbone_lr", type=float, default=None, help="LR for transformer backbone (defaults to 0.5*head_lr)")
    parser.add_argument("--head_lr", type=float, default=None,
                    help="LR for perceptron (CosineLinear) weights; defaults to --lr")

    parser.add_argument("--freeze_backbone_steps", type=int, default=0,
                    help="Number of optimizer steps to freeze the Transformer backbone (set its LR to 0).")

    parser.add_argument("--debug_no_step", action="store_true",
                    help="Run forward/backward but skip optimizer.step for sanity checks.")
    
    # Knowledge Distillation
    parser.add_argument("--use_kd", type=_str2bool, default=False,
                        help="Enable knowledge distillation from teacher model")
    parser.add_argument("--teacher_model", type=str, default="gpt2",
                        help="Teacher model name for knowledge distillation")
    parser.add_argument("--kd_weight", type=float, default=0.3,
                        help="Weight for knowledge distillation loss")
    parser.add_argument("--kd_temperature", type=float, default=4.0,
                        help="Temperature for knowledge distillation")
    
    parser.add_argument("--scheduler", type=str, default="linear", choices=["linear","cosine"])

    parser.add_argument("--ce_weight", type=float, default=None)

    # Heterogeneous experts per FFN (comma-separated)
    parser.add_argument("--expert_kinds", type=str, default="mlp2e,mlpe,perc2,perc4",
                    help='Comma-separated list per FFN layer, e.g. "mlp2e,mlpe,perc2,perc4". '
                         'You can also include "reglu1d" and "film".')

    # General perceptron init (works for perc2/perc4/…)
    parser.add_argument("--init_perc_from_hf", type=_str2bool, default=True,
                        help="Initialize any perceptron expert from pretrained MLP via top-K SVD (e.g., perc4).")

    # Back-compat alias; if explicitly set, it overrides the general flag
    parser.add_argument("--init_perc2_from_hf", type=_str2bool, default=None,
                        help="DEPRECATED: use --init_perc_from_hf.")


    # Mixed precision & speed knobs
    parser.add_argument("--amp", type=_str2bool, default=True,
                        help="Use mixed precision (autocast + GradScaler).")
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"],
                        help="AMP dtype: fp16 (faster on most GPUs) or bf16 (more stable on Ada/Hopper).")
    parser.add_argument("--compile", type=_str2bool, default=False,
                        help="Use torch.compile to optimize the model (PyTorch 2.x).")
    parser.add_argument("--allow_tf32", type=_str2bool, default=True,
                        help="Allow TF32 for matmul/cudnn on Ampere+ (fast fp32 path).")
    
    parser.add_argument(
        "--expert_gate_bias",
        type=str,
        default="",   # e.g. "mlp2e:-0.35,mlpe:0.0,perc2:+0.1"
        help="Comma list of kind:bias to add to gate logits per expert kind."
    )
    parser.add_argument(
        "--bias_warmup_steps",
        type=int,
        default=0,
        help="If >0, linearly anneal bias from its value to 0 across these steps."
    )

    args = parser.parse_args()

    # Materialize chosen AMP dtype into a torch dtype
    args.amp_torch_dtype = torch.float16 if args.amp_dtype.lower() == "fp16" else torch.bfloat16

    # head_lr falls back to --lr if unset
    if args.head_lr is None and hasattr(args, "lr"):
        args.head_lr = args.lr

    return args

class MultiMLPTrainingManager:
    """Simplified trainer without automatic phase management"""
    
    def __init__(self, args, model, tokenizer, device, checkpoint_dir=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.global_step = 0

        # ----- Mixed precision settings -----
        # CLI is optional; we fall back to sensible defaults if flags don't exist.
        amp_on = bool(getattr(args, "amp", True) and device.type == "cuda")
        amp_dtype_str = getattr(args, "amp_dtype", "fp16").lower()
        self.autocast_dtype = torch.float16 if amp_dtype_str == "fp16" else torch.bfloat16
        self.use_autocast = amp_on

        # GradScaler only for fp16 (bf16 doesn't need it)
        use_scaler = self.use_autocast and (self.autocast_dtype == torch.float16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        # ----- Fast math (optional but safe) -----
        allow_tf32 = bool(getattr(args, "allow_tf32", True))
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            try:
                torch.set_float32_matmul_precision("high")  # PyTorch 2.x
            except Exception:
                pass

        # Initialize base trainer
        self.trainer = MultiMLPTrainer(model, tokenizer, device)

        # Optional torch.compile (can speed up longer runs; warms up first steps)
        if bool(getattr(args, "compile", False)):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[compile] torch.compile enabled")
            except Exception as e:
                print(f"[compile] disabled: {e}")

        # Set routing configuration from command line args
        self._update_routing_config()

        self._setup_knowledge_distillation()

        # Optional: Initialize perceptron experts from HF MLP (perc2/perc4/etc.)
        init_flag = bool(getattr(args, "init_perc_from_hf", True))
        if getattr(args, "init_perc2_from_hf", None) is not None:
            # honor explicit deprecated flag if provided
            init_flag = bool(args.init_perc2_from_hf)
        if init_flag:
            try:
                _maybe_init_all_perc_from_hf(self.model, args.config_name)
            except Exception as e:
                print(f"[init] Skipped perceptron HF init due to error: {e}")

    def _setup_knowledge_distillation(self):
        """Setup knowledge distillation teacher model if enabled"""
        if self.args.use_kd:
            try:
                from transformers import GPT2LMHeadModel
                print(f"Loading teacher model: {self.args.teacher_model}")
                self.teacher_model = GPT2LMHeadModel.from_pretrained(self.args.teacher_model).to(self.device)
                self.teacher_model.eval()
                
                # Freeze teacher model parameters
                for param in self.teacher_model.parameters():
                    param.requires_grad = False
                    
                teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
                print(f"Teacher model loaded: {teacher_params:,} parameters")
                print(f"KD weight: {self.args.kd_weight}, temperature: {self.args.kd_temperature}")
                
            except Exception as e:
                print(f"Warning: Failed to load teacher model: {e}")
                print("Continuing without knowledge distillation")
                self.teacher_model = None
        else:
            self.teacher_model = None
            print("Knowledge distillation disabled")

    def compute_kd_loss(self, student_logits, teacher_logits, labels, temperature=4.0):
        """
        Compute knowledge distillation loss using KL divergence with proper shifting and masking
        
        Args:
            student_logits: [batch_size, seq_len, vocab_size]
            teacher_logits: [batch_size, seq_len, vocab_size] 
            labels: [batch_size, seq_len] with -100 for ignored positions
            temperature: Temperature for softmax
        """
        # Shift logits to match language modeling: predict next token
        student_shift = student_logits[..., :-1, :].contiguous()  # [B, S-1, V]
        teacher_shift = teacher_logits[..., :-1, :].contiguous()   # [B, S-1, V]
        labels_shift = labels[..., 1:].contiguous()                # [B, S-1]
        
        # Create mask for valid positions (ignore -100 labels)
        valid_mask = (labels_shift != -100)  # [B, S-1]
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        # Apply mask - only compute KD loss on valid positions
        student_valid = student_shift[valid_mask]  # [N, V] where N = number of valid positions
        teacher_valid = teacher_shift[valid_mask]  # [N, V]
        
        # Compute softmax with temperature
        student_soft = F.log_softmax(student_valid / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_valid / temperature, dim=-1)
        
        # KL divergence loss
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
        
        return kd_loss

    def _update_routing_config(self):
        """Set routing configuration from command line arguments"""
        config = {
            'soft_routing': self.args.soft_routing,
            'tau': self.args.tau,
            'top_k': self.args.top_k,
            'compute_only_selected': self.args.compute_only_selected,
            "force_uniform_gate": self.args.force_uniform_gate,
        }

        # Apply to all layers
        layer_configs = {}
        for i in range(self.model.config.n_layer):
            layer_configs[i] = config.copy()
        
        self.model.set_routing_mode(layer_configs)
        print(f"Routing configuration set:")
        print(f"  soft_routing: {config['soft_routing']}")
        print(f"  tau: {config['tau']}")
        print(f"  top_k: {config['top_k']}")
        print(f"  compute_only_selected: {config['compute_only_selected']}")

    @torch.no_grad()
    def teacher_forced_stats_from_batch(
        self,
        batch,
        tau: float = 0.6,
        num_mlps: int | None = None,
    ):
        import torch
        import torch.nn.functional as F

        was_training = self.model.training
        self.model.eval()
        device = self.device
        if num_mlps is None:
            num_mlps = int(getattr(self.args, "num_mlps", 2))

        saved_layer_cfg = getattr(self.model, "layer_configs", None)
        self.model.layer_configs = {}

        routing_cfg = {
            "soft_routing": True,
            "tau": float(tau),
            "top_k": int(getattr(self.args, "top_k", getattr(self.args, "num_mlps", 2))),
            "compute_only_selected": False,
        }

        try:
            x = batch["input_ids"].to(device)
            m = batch.get("attention_mask")
            m = m.to(device) if m is not None else None

            # derive labels from inputs for this diagnostic
            y = x.clone()
            if m is not None:
                y = y.masked_fill(m == 0, -100)

            out = self.model(
                input_ids=x,
                attention_mask=m,
                use_cache=False,
                output_attentions=False,
                return_dict=True,
                global_routing_config=routing_cfg,
            )
            logits = out.logits  # [B,S,V]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y[..., 1:].contiguous()
            valid = (shift_labels != -100)

            if not valid.any():
                print("[TF-ONE] tokens=0")
                return {"tokens": 0, "nll": float("nan"), "ppl": float("nan"),
                        "entropy": float("nan"), "p_max": float("nan"), "p_repeat_prev": float("nan")}

            log_probs = F.log_softmax(shift_logits[valid], dim=-1)  # [N,V]
            gold = shift_labels[valid].view(-1, 1)                  # [N,1]
            nll = -log_probs.gather(dim=-1, index=gold).mean().item()
            probs = log_probs.exp()
            ent = (-(probs * log_probs).sum(dim=-1)).mean().item()
            pmax = probs.max(dim=-1).values.mean().item()

            # previous-token probability (teacher-forced), aligned and masked
            prev_ids = y[..., :-1].contiguous()                 # [B,S-1]
            prev_valid = valid & (prev_ids != -100)             # [B,S-1]
            if prev_valid.any():
                prev_lp = F.log_softmax(shift_logits[prev_valid], dim=-1)
                prev_probs = prev_lp.exp()
                prev_ids_flat = prev_ids[prev_valid].view(-1, 1)
                prevrep = prev_probs.gather(dim=-1, index=prev_ids_flat).mean().item()
            else:
                prevrep = float("nan")

            n_tokens = valid.sum().item()
            ppl = float(torch.exp(torch.tensor(nll)).item())
            print(f"[TF-ONE] tokens={n_tokens}  nll={nll:.4f}  ppl={ppl:.2f}  "
                f"H={ent:.3f}  p_max={pmax:.3f}  p_repeat_prev={prevrep:.3f}")
            return {
                "tokens": n_tokens,
                "nll": nll,
                "ppl": ppl,
                "entropy": ent,
                "p_max": pmax,
                "p_repeat_prev": prevrep,
            }
        finally:
            self.model.layer_configs = saved_layer_cfg
            if was_training:
                self.model.train()

    def clamp_cosine_scales(self, min_log=-6.9, max_log=2.3):
        import math, torch
        with torch.no_grad():
            for m in self.model.modules():
                if hasattr(m, "log_s"):
                    m.log_s.clamp_(min_log, max_log)

    def evaluate(self, val_dataloader):
        return self.strict_eval_ppl(
            val_dataloader,
            tau=self.args.tau,
            num_mlps=self.args.num_mlps,
        )

    def _collect_gate_stats(self, model):
        std_vals, ent_vals = [], []
        for blk in getattr(model, "h", []):
            mlp = getattr(blk, "mlp", None)
            if mlp is None:
                continue
            if getattr(mlp, "_last_gate_std", None) is not None:
                std_vals.append(float(mlp._last_gate_std))
            if getattr(mlp, "_last_gate_entropy", None) is not None:
                ent_vals.append(float(mlp._last_gate_entropy))
        mean_std = sum(std_vals)/len(std_vals) if std_vals else 0.0
        mean_ent = sum(ent_vals)/len(ent_vals) if ent_vals else 0.0
        return mean_std, mean_ent

    @torch.no_grad()
    def strict_eval_ppl(
        self,
        dataloader,
        tau: float = 0.6,
        num_mlps: int | None = None
    ) -> float:
        self.model.eval()
        device = self.device

        with torch.amp.autocast(
                "cuda",
                enabled=bool(getattr(self, "use_autocast", False)),
                dtype=getattr(self, "autocast_dtype", torch.float16)):
            pass  # keep eval-side autocast off by default here

        # Use consistent routing with training
        saved_layer_cfg = getattr(self.model, "layer_configs", None)
        self.model.layer_configs = {}

        try:
            routing_cfg = {
                "soft_routing": True,
                "tau": float(tau),
                "top_k": int(getattr(self.args, "top_k", num_mlps or getattr(self.args, "num_mlps", 2))),
                "compute_only_selected": False,
                "force_uniform_gate": self.args.force_uniform_gate,  
            }

            total_nll = 0.0
            total_tokens = 0
            total_correct = 0  # <-- NEW

            for batch in dataloader:
                x = batch["input_ids"].to(device)
                m = batch.get("attention_mask")
                m = m.to(device) if m is not None else None
                
                # Create labels (same as training)
                y = x.clone()
                if m is not None:
                    y = y.masked_fill(m == 0, -100)

                # Forward pass with autocast
                with torch.amp.autocast("cuda", enabled=self.use_autocast, dtype=self.autocast_dtype):
                    out = self.model(
                        input_ids=x,
                        attention_mask=m,
                        labels=y,
                        return_dict=True,
                        use_cache=False,
                        output_attentions=False,
                        global_routing_config=routing_cfg,
                    )
                
                # Manual NLL calculation
                logits = out.logits[..., :-1, :].contiguous()  # [B, S-1, V]
                targets = y[..., 1:].contiguous()              # [B, S-1]
                valid_mask = (targets != -100)                 # [B, S-1]
                
                if valid_mask.any():
                    # Flatten valid positions
                    flat_logits  = logits[valid_mask]          # [N, V]
                    flat_targets = targets[valid_mask]         # [N]

                    # Log-probs for NLL
                    log_probs = F.log_softmax(flat_logits, dim=-1)
                    nll = -log_probs.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)

                    # Accuracy (argmax over logits)
                    preds = flat_logits.argmax(dim=-1)         # [N]
                    total_correct += (preds == flat_targets).sum().item()

                    # Accumulate NLL & count
                    total_nll += nll.sum().item()
                    total_tokens += flat_targets.numel()

            print(f"[DEBUG] Processed {len(dataloader)} batches")

            if total_tokens == 0:
                print("[EVAL] No valid tokens found")
                return float('inf')
                
            avg_nll = total_nll / total_tokens
            ppl = math.exp(avg_nll)
            acc = total_correct / total_tokens  # fraction in [0,1]

            std_eval, ent_eval = self._collect_gate_stats(self.model)
            print(f"train==eval? False")
            print(f"std eval (soft gate): {std_eval:.6f} | entropy: {ent_eval:.6f}")
            print(f"[EVAL] tokens={total_tokens} nll={avg_nll:.6f} ppl={ppl:.4f} acc={acc*100:.2f}%")
            return ppl
            
        finally:
            self.model.layer_configs = saved_layer_cfg

    def compute_regularization_loss(self):
        """Compute regularization losses for load balancing and diversity"""
        total_reg_loss = 0.0
        
        if self.args.load_balance_coef > 0 or self.args.diversity_coef > 0:
            layer_stats = self.model.get_layer_stats()
            
            for layer_idx, stats in layer_stats.items():
                # Load balancing: encourage uniform usage across perceptrons
                if self.args.load_balance_coef > 0:
                    usage_counts = torch.tensor(stats['usage_counts'], device=self.device, dtype=torch.float32)
                    if usage_counts.sum() > 0:
                        usage_probs = usage_counts / usage_counts.sum()
                        uniform_target = torch.ones_like(usage_probs) / len(usage_probs)
                        lb_loss = F.kl_div((usage_probs + 1e-8).log(), uniform_target, reduction='batchmean')
                        total_reg_loss += self.args.load_balance_coef * lb_loss
        
        return total_reg_loss
   
    def train_step(self, batch, optimizer):
        """Single training micro-step (accumulation-safe). No optimizer.step() here."""
        model = self.model
        device = self.device
        args = self.args
        scaler = getattr(self, "scaler", None)

        nb = torch.cuda.is_available() and getattr(self, "trainer", None) is None  # simple guard
        # You can just always use non_blocking=True when pin_memory=True in DataLoader
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
            # Make sure pads don't contribute to CE loss
            labels = labels.masked_fill(attention_mask == 0, -100)

        use_amp = bool(getattr(self, "use_autocast", False))
        amp_dtype = getattr(self, "autocast_dtype", torch.float16)

        with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            # Let HF compute CE internally (uses labels)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
            ce_loss = outputs.loss  # already averaged over non-ignored positions

            # Regularization term from your routing/entropy balancing etc.
            reg_loss = self.compute_regularization_loss()

            # ---- Knowledge Distillation (optional) ----
            kd_loss = torch.tensor(0.0, device=device)
            if getattr(args, "use_kd", False) and hasattr(self, "teacher_model") and (self.teacher_model is not None):
                T = float(getattr(args, "kd_temperature", 1.0))
                kd_w = float(getattr(args, "kd_weight", 0.0))

                # Teacher forward (eval + no grad). No labels needed.
                was_training = self.teacher_model.training
                self.teacher_model.eval()
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
                    teacher_logits = teacher_outputs.logits

                if was_training:
                    self.teacher_model.train()

                # Compute KD loss over non-ignored tokens (labels != -100)
                # Your compute_kd_loss(student_logits, teacher_logits, labels, temperature=...)
                kd_loss_raw = self.compute_kd_loss(
                    outputs.logits, teacher_logits, labels, temperature=T
                )
                kd_loss = kd_w * kd_loss_raw


            # ---- Mix CE + KD + Reg ----
            # If KD is on and user didn’t specify CE weight, default to a reasonable mix.
            ce_w = args.ce_weight if args.ce_weight is not None else \
                (0.5 if getattr(args, "use_kd", False) else 1.0)

            total_loss = ce_w * ce_loss + reg_loss + kd_loss

            # Normalize for gradient accumulation
            grad_accum = max(1, int(getattr(args, "grad_accum_steps", 1)))
            if grad_accum > 1:
                total_loss = total_loss / float(grad_accum)

        # ---- Backward (no step here) ----
        if scaler is not None and scaler.is_enabled():
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # Return *un*-divided values for logging
        return {
            "total_loss": float(total_loss.detach().item() * grad_accum),
            "reg_loss": float(reg_loss.detach().item()),
            "kd_loss": float(kd_loss.detach().item()),
        }

    def save_checkpoint(self, epoch, optimizer, scheduler=None, is_final=False):
        """Save training checkpoint"""
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": (self.scaler.state_dict() if hasattr(self, "scaler") and self.scaler is not None else None),
            "global_step": self.global_step,
            "epoch": epoch,
            "args": self.args,  # Save args for resuming
        }
        
        # Save layer statistics
        checkpoint['layer_stats'] = self.model.get_layer_stats()
        
        if is_final:
            filename = "final_model.pt"
        else:
            filename = f"checkpoint_step_{self.global_step}.pt"
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
        
        return save_path
    
    def load_checkpoint(self, checkpoint_path, optimizer, scheduler=None):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if scheduler is not None and checkpoint.get('scheduler') is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        if hasattr(self, 'scaler') and checkpoint.get('scaler') is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded: step {self.global_step}")
        return checkpoint
    
    def _get_transformer_layers(self):
        """
        Return a list/ModuleList of transformer blocks robustly.
        Supports model.h, model.transformer.h, DDP(model.module), or falls back
        to scanning modules for blocks that have an 'mlp' attribute.
        """
        mdl = getattr(self.model, "module", self.model)  # handle DDP/DP

        # Preferred: GPT-2 style
        if hasattr(mdl, "h"):
            return mdl.h
        tr = getattr(mdl, "transformer", None)
        if tr is not None and hasattr(tr, "h"):
            return tr.h

        # Fallback: collect any modules that look like blocks (have .mlp)
        candidates = []
        for m in mdl.modules():
            if hasattr(m, "mlp"):
                candidates.append(m)
        return candidates  # may be empty

    def print_layer_stats(self, step=None, end="\n"):
        if step is None:
            step = self.global_step

        layers = self._get_transformer_layers()
        if not layers:
            print(f"LayerStats step={int(step)}: <no layers found>", end=end, flush=True)
            return

        parts = []
        for i, block in enumerate(layers):
            mlp = getattr(block, "mlp", None)
            if mlp is None:
                continue

            # usage %
            uc = getattr(mlp, "usage_counts", None)
            if uc is None:
                use_str = "U:?"; up = None
            else:
                uc = uc.detach().float().cpu()
                u_sum = max(float(uc.sum().item()), 1.0)
                up = [100.0 * float(x) / u_sum for x in uc]
                use_str = "U[" + " ".join(f"M{j}:{p:.1f}%" for j, p in enumerate(up)) + "]"  # Changed P to M

            # selection %
            sc = getattr(mlp, "selection_counts", None)
            if sc is None or float(sc.sum().item()) == 0.0:
                sel_str = "S[—]"
            else:
                sc = sc.detach().float().cpu()
                s_sum = max(float(sc.sum().item()), 1.0)
                sp = [100.0 * float(x) / s_sum for x in sc]
                sel_str = "S[" + " ".join(f"M{j}:{p:.1f}%" for j, p in enumerate(sp)) + "]"  # Changed P to M

            parts.append(f"L{i:02d} {use_str} {sel_str}")

        line = f"LayerStats step={int(step)}: " + " | ".join(parts)
        print(line, end=end, flush=True)

    @torch.no_grad()
    def quick_preview_sample(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float | None = 0.9,
        top_k: int = 40,
    ) -> str:
        model, tok, device = self.model, self.tokenizer, self.device
        model.eval()

        # Use current training routing configuration for consistency
        current_routing_cfg = {
            "soft_routing": self.args.soft_routing,
            "tau": self.args.tau,
            "top_k": self.args.top_k,
            "compute_only_selected": self.args.compute_only_selected,
            "force_uniform_gate": self.args.force_uniform_gate,   # Phase A
        }

        # Encode prompt
        ctx = tok.encode(prompt, add_special_tokens=False)
        if not ctx:
            ctx = [tok.eos_token_id]
        input_ids = torch.tensor([ctx], device=device, dtype=torch.long)

        new_tokens = []
        for _ in range(max_new_tokens):
            attn = torch.ones_like(input_ids, device=device)
            
            # Use autocast if enabled
            with torch.amp.autocast("cuda", enabled=self.use_autocast, dtype=self.autocast_dtype):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                    global_routing_config=current_routing_cfg,  # Pass routing config
                )
            
            logits = out.logits[:, -1, :].float()

            # Sampling logic (temperature, top_k, top_p)
            if temperature <= 0.0:
                next_id = int(torch.argmax(logits, dim=-1))
            else:
                l = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    k = min(top_k, l.size(-1))
                    kth = torch.topk(l, k=k, dim=-1).values[..., -1, None]
                    l = torch.where(l < kth, torch.full_like(l, float("-inf")), l)
                
                # Top-p filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    probs = torch.softmax(l, dim=-1)
                    sp, si = torch.sort(probs, descending=True)
                    cdf = torch.cumsum(sp, dim=-1)
                    keep = (cdf <= top_p)
                    keep[..., 0] = True
                    mask = torch.full_like(l, float("-inf"))
                    mask.scatter_(1, si, torch.where(keep, torch.zeros_like(sp), torch.full_like(sp, float("-inf"))))
                    l = l + mask
                
                p = torch.softmax(l, dim=-1)
                next_id = int(torch.multinomial(p, 1)[0, 0])

            new_tokens.append(next_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

            if next_id == tok.eos_token_id:
                break

        return tok.decode(new_tokens, skip_special_tokens=True).strip()


def setup_environment():
    """Setup training environment"""
    # Determine environment
    RUN_MODE = "colab" if "COLAB_GPU" in os.environ else "local"
    
    if RUN_MODE == "colab":
        BASE_PATH = Path("/content/drive/MyDrive/Project1")
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except ImportError:
            pass
    else:
        BASE_PATH = Path("C:/Machine Learning/Project1")
    
    BASE_PATH.mkdir(parents=True, exist_ok=True)
    
    print(f"Running in {RUN_MODE} mode")
    print(f"Base path: {BASE_PATH}")
    
    return BASE_PATH, RUN_MODE

def create_collate_fn(tokenizer):
    """Create collate function for DataLoader; supports dict samples and TensorDataset rows."""
    def _collate(batch):
        if isinstance(batch[0], dict):
            input_ids = torch.stack([b['input_ids'] for b in batch])
            attention_mask = torch.stack([b['attention_mask'] for b in batch])
        else:
            # TensorDataset path (each item is a tuple of tensors)
            input_ids = torch.stack([b[0] for b in batch])
            attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    return _collate

def print_param_groups(optimizer):
    """Pretty-print optimizer parameter groups once (robust to missing lr/wd)."""
    print("\n=== Optimizer parameter groups ===")
    total = 0
    d_lr = float(optimizer.defaults.get("lr", 0.0))
    d_wd = float(optimizer.defaults.get("weight_decay", 0.0))
    for i, g in enumerate(optimizer.param_groups):
        n = sum(p.numel() for p in g.get("params", []))
        total += n
        name = str(g.get("name", f"group_{i}"))
        lr = float(g.get("lr", d_lr))
        wd = float(g.get("weight_decay", d_wd))
        print(f"{i:02d} | {name:14s} | lr={lr:.3e} | wd={wd:.3e} | n_params={n:,}")
    print(f"Total trainable params in optimizer: {total:,}\n")

def set_eval_seed(seed: int = 1234):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _probe_first_mlp_weights(model):
    """
    Prints quick stats for the first block's first MLP weights,
    without assuming any specific attribute names.
    """
    import torch
    try:
        mlp0 = model.h[0].mlp.mlps[0]
    except Exception as e:
        print(f"[MLP probe] skipped: {e}")
        return

    # Collect a few weight tensors if present
    weight_tensors = []
    for name, module in getattr(mlp0, 'named_modules', lambda: [])():
        pass  # in case SingleMLP doesn't expose submodules via named_modules

    # Fall back to named_parameters (works for both module styles)
    for name, p in mlp0.named_parameters():
        if p is not None and p.ndim >= 1:
            weight_tensors.append((name, p.detach()))

    if not weight_tensors:
        print("[MLP probe] no parameters found on first MLP; skipping.")
        return

    # Print up to two tensors to keep logs readable
    for name, w in weight_tensors[:2]:
        mean = w.mean().item()
        std  = w.std().item()
        mn   = w.min().item()
        mx   = w.max().item()
        l2   = w.norm().item()
        print(f"[MLP0.{name}] mean={mean:.2e} std={std:.2e} min={mn:.2e} max={mx:.2e} ||w||={l2:.2e}")
       

def _build_scheduler(optim, sched_name: str, warmup_steps: int, total_steps: int):
    if sched_name.lower() == "cosine":
        return get_cosine_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    return get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

def main():
    global training_interrupted
    args = get_args()

    # Parse expert kinds
    expert_kinds_cli = [s.strip().lower() for s in args.expert_kinds.split(",") if s.strip()]
    if not expert_kinds_cli:
        expert_kinds_cli = ["mlp2e", "mlpe", "perc2", "perc2"]
    args.num_mlps = len(expert_kinds_cli)  # <-- keep everyone on the same page

    # Set random seed
    torch.manual_seed(args.seed)
    set_eval_seed(1234)

    # Setup environment
    BASE_PATH, RUN_MODE = setup_environment()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fast math paths
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
        try:
            torch.set_float32_matmul_precision("high")  # or "medium"
        except Exception:
            pass

    # Setup checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = BASE_PATH / "multi_context_readers_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # Setup tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.config_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Creating Context-Readers model (pretrained={args.use_pretrained}) with experts: {expert_kinds_cli} ...")
    # Build model using your factory; then move the *model* to device
    model, config = MultiMLPGPT2Model.create_multi_lp_model(
        config_name=args.config_name,
        num_mlps=len(expert_kinds_cli),
        expert_kinds=expert_kinds_cli,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model created from pretrained:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Layers: {model.config.n_layer}")
    print(f"  MLPs per layer: {args.num_mlps}")
    
    # Load datasets
    if WikiTextDataset is not None:
        train_dataset = WikiTextDataset(
            data_dir=BASE_PATH / "wikitext-103",
            tokenizer=tokenizer,
            max_length=args.max_length,
            split="train",
            max_samples=args.max_samples
        )
        
        val_dataset = WikiTextDataset(
            data_dir=BASE_PATH / "wikitext-103",
            tokenizer=tokenizer,
            max_length=args.max_length,
            split="valid",
            max_samples=1000  # Smaller validation set
        )
    else:
        print("Warning: Using dummy datasets - WikiTextDataset not available")
        # Create dummy datasets for testing
        train_dataset = torch.utils.data.TensorDataset(
            torch.randint(0, tokenizer.vocab_size, (1000, args.max_length))
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.randint(0, tokenizer.vocab_size, (100, args.max_length))
        )
    
    loader_config = {
        'num_workers': 2 if RUN_MODE == "colab" else 0,
        'pin_memory': torch.cuda.is_available()
    }

    collate_fn = create_collate_fn(tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **loader_config
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_config
    )

    # Cache a deterministic validation batch (first batch, first 2 examples)
    fixed_valid_batch = next(iter(val_dataloader))
    for k in fixed_valid_batch:
        if torch.is_tensor(fixed_valid_batch[k]):
            # Keep it small & on CPU; .clone() to avoid later in-place changes
            fixed_valid_batch[k] = fixed_valid_batch[k][:2].clone()

    print(f"Training batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(val_dataloader)}")

    # Calculate total training steps
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        steps_per_epoch = max(1, len(train_dataloader) // max(1, args.grad_accum_steps))
        total_steps = max(1, steps_per_epoch * max(1, args.epochs))

    optimizer = build_optimizer(model, args)
    print_param_groups(optimizer)

    # Clamp warmup
    warmup_steps = min(args.warmup_steps, max(1, total_steps - 1))

    # Setup scheduler
    scheduler = _build_scheduler(optimizer, args.scheduler, warmup_steps, total_steps)

    print(f"Training schedule:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Routing mode: {'soft' if args.soft_routing else 'hard'}")
    print(f"  Tau: {args.tau}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Compute only selected: {args.compute_only_selected}")
    
    # Create trainer
    trainer = MultiMLPTrainingManager(args, model, tokenizer, device, checkpoint_dir)
    
    trainer.use_autocast = bool(args.amp and device.type == "cuda")
    trainer.autocast_dtype = args.amp_torch_dtype

    if not hasattr(trainer, 'scaler'):
        trainer.scaler = None

    # Identify backbone groups by the names you set in the builder
    backbone_idx = [i for i, g in enumerate(optimizer.param_groups)
                    if g.get("name") in ("backbone_w", "backbone_nd")]
    orig_backbone_lrs = [optimizer.param_groups[i]["lr"] for i in backbone_idx]

    # Freeze backbone only (leave embeddings & gate active)
    if args.freeze_backbone_steps > 0 and backbone_idx:
        for i in backbone_idx:
            optimizer.param_groups[i]["lr"] = 0.0
        trainer._backbone_unfrozen = False
        print(f"[optimizer] Freezing backbone (except embeddings & gate) for {args.freeze_backbone_steps} steps.")

    # Stash for later unfreeze
    trainer._backbone_group_idx = backbone_idx
    trainer._backbone_orig_lrs = orig_backbone_lrs
    trainer._backbone_unfrozen = (args.freeze_backbone_steps == 0)

    print("resid_pdrop =", getattr(model.config, "resid_pdrop", 0.1))

    # Set up target step for early stopping
    start_step = 0
    if args.max_steps and args.max_steps > 0:
        target_step = start_step + args.max_steps
    else:
        target_step = float("inf")
    trainer.target_step = target_step

    # Use the scaler already created in MultiMLPTrainingManager
    scaler = trainer.scaler

    # Load checkpoint if resuming
    if args.resume:
        print(f"[resume] Loading checkpoint: {args.resume}")

        # Load with safe path, then trusted fallback if needed
        try:
            ckpt = torch.load(args.resume, map_location=device)
        except Exception as e:
            print(f"[resume] Safe load failed: {e}")
            print("[resume] Retrying with weights_only=False (trusted checkpoint).")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        # Normalize possible formats
        def pick(d, *keys, default=None):
            for k in keys:
                if k in d:
                    return d[k]
            return default

        # If file itself is a raw model state_dict, wrap it
        if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()) and not any(
            k in ckpt for k in ("model", "model_state_dict", "state_dict")
        ):
            # Heuristic: if keys look like parameter names, treat as state_dict
            if any(k.endswith(".weight") or k.endswith(".bias") for k in ckpt.keys()):
                ckpt = {"model": ckpt}

        model_sd = pick(ckpt, "model", "model_state_dict", "state_dict")
        if model_sd is None:
            raise KeyError(f"[resume] No model weights in checkpoint. Keys: {list(ckpt.keys())[:10]}")

        # 1) Model
        missing, unexpected = model.load_state_dict(model_sd, strict=False)
        if missing or unexpected:
            print(f"[resume][model] missing={len(missing)} unexpected={len(unexpected)}")

        # 2) Optimizer (best-effort)
        opt_sd = pick(ckpt, "optimizer", "optimizer_state_dict")
        if opt_sd is not None:
            try:
                optimizer.load_state_dict(opt_sd)
                print("[resume] optimizer loaded")
            except Exception as e:
                print(f"[resume] optimizer fresh: {e}")
        else:
            print("[resume] optimizer fresh (no state found)")

        # 3) Scheduler - DON'T load old state, will recreate below
        # (Old scheduler state is incompatible when resuming with different max_steps)

        # 4) AMP scaler (optional)
        if "scaler" in ckpt and scaler is not None:
            try:
                if ckpt["scaler"] is not None:
                    scaler.load_state_dict(ckpt["scaler"])
                    print("[resume] scaler loaded")
            except Exception as e:
                print(f"[resume] scaler fresh: {e}")

        # 5) Step / epoch
        trainer.global_step = int(ckpt.get("global_step", 0))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"[resume] global_step={trainer.global_step} start_epoch={start_epoch}")

        # 6) Re-apply routing config from current args (not checkpoint)
        trainer._update_routing_config()

        # 7) Re-apply backbone freeze (LM head stays trainable)
        if hasattr(trainer, "_backbone_group_idx"):
            if args.freeze_backbone_steps > 0 and trainer.global_step < args.freeze_backbone_steps:
                for i in trainer._backbone_group_idx:
                    optimizer.param_groups[i]["lr"] = 0.0
                trainer._backbone_unfrozen = False
                print(f"[resume] backbone remains frozen (LM head active) until step {args.freeze_backbone_steps}")
            else:
                for j, i in enumerate(trainer._backbone_group_idx):
                    optimizer.param_groups[i]["lr"] = float(trainer._backbone_orig_lrs[j])
                trainer._backbone_unfrozen = True

        # 8) RECREATE scheduler with correct total steps for the new training run
        if args.max_steps > 0:
            new_total_steps = args.max_steps
            new_warmup = 0  # Already warmed up from previous training
        else:
            steps_per_epoch = max(1, len(train_dataloader) // max(1, args.grad_accum_steps))
            new_total_steps = steps_per_epoch * args.epochs
            new_warmup = min(args.warmup_steps, max(1, new_total_steps - 1))
        
        scheduler = _build_scheduler(optimizer, args.scheduler, new_warmup, new_total_steps)
        print(f"[resume] Recreated scheduler ({args.scheduler}) "
            f"with total={new_total_steps} warmup={new_warmup}")

        # 9) Update target step
        start_step = trainer.global_step
        if args.max_steps and args.max_steps > 0:
            target_step = start_step + args.max_steps
        else:
            target_step = float("inf")
        trainer.target_step = target_step
        
        print(f"Training until step {trainer.target_step} "
            f"(start={start_step}, +{args.max_steps}).")
    else:
        start_epoch = 0

    if args.resume and args.max_steps > 0:
        # When resuming with max_steps, use a large epoch count to ensure we don't exit early
        effective_epochs = 999  # Large number to avoid epoch-based exit
    else:
        effective_epochs = args.epochs

    #enable_tracing(max_depth=10)  # Adjust depth as needed
    
    # Check model health
    #check_model_health(model)

    kind2bias = parse_kind_bias(args.expert_gate_bias)  # dict or {}

    with torch.no_grad():
        for blk in model.h:  # GPT-2 blocks
            mlp = getattr(blk, "mlp", None)
            if mlp is None: continue
            # set base bias per kind
            bias_vec = torch.zeros(mlp.num_mlps, device=mlp.expert_bias.device)
            for i, kind in enumerate(getattr(mlp, "expert_kinds", [])):
                if kind in kind2bias:
                    bias_vec[i] = kind2bias[kind]
            mlp.expert_bias.copy_(bias_vec)

    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    '''
    print("\n[Sanity] Locked eval x2 (should be identical):")
    p1 = trainer.strict_eval_ppl(val_dataloader, tau=max(0.6, args.tau))
    p2 = trainer.strict_eval_ppl(val_dataloader, tau=max(0.6, args.tau))
    print(f"[Sanity] PPL1={p1:.4f}  PPL2={p2:.4f}  Δ={abs(p1-p2):.6f}\n")

    print("[Sanity] Teacher-forced ONE-BATCH (locked routing) – BEFORE")
    trainer.teacher_forced_stats_from_batch(fixed_valid_batch, tau=args.tau)
    '''

    try:
        step_count = 0
        accum_loss = 0.0
        
        for epoch in range(start_epoch, effective_epochs):
            if training_interrupted:
                break

            optimizer.zero_grad()
            
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                if training_interrupted:
                    break
                
                if trainer.global_step >= trainer.target_step:
                    print(f"Reached target step: {trainer.target_step}")
                    training_interrupted = True
                    break

                # Training step (only forward + backward, no optimizer step)
                step_results = trainer.train_step(batch, optimizer)
                accum_loss += step_results['total_loss']
                step_count += 1
                
                # Gradient accumulation - SINGLE STEP ONLY
                if step_count % args.grad_accum_steps == 0:
                    if not getattr(args, 'debug_no_step', False):
                        # Gradient clipping (unscale first for AMP)
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if scaler is not None and scaler.is_enabled():
                                scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        
                        # Optimizer step (ONLY ONCE)
                        if scaler is not None and scaler.is_enabled():
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        
                        # Scheduler step (ONLY ONCE)
                        if scheduler is not None:
                            scheduler.step()
                    
                    # Post-step maintenance
                    #trainer.clamp_cosine_scales(min_log=-1.6, max_log=1.6)
                    trainer.clamp_cosine_scales(min_log=-0.69, max_log=3.0)  # s ∈ [0.5, 20]

                    optimizer.zero_grad(set_to_none=True)
                    trainer.global_step += 1
                    
                    # Unfreeze backbone if needed
                    if (not trainer._backbone_unfrozen) and (trainer.global_step >= args.freeze_backbone_steps):
                        for j, i in enumerate(trainer._backbone_group_idx):
                            optimizer.param_groups[i]["lr"] = float(trainer._backbone_orig_lrs[j])
                        trainer._backbone_unfrozen = True
                        print(f"[optimizer] Unfroze backbone at step {trainer.global_step}.")
                    
                    # Calculate average loss for display
                    avg_loss = accum_loss / args.grad_accum_steps
                    accum_loss = 0.0
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'step': trainer.global_step
                    })

                    # Logging
                    if trainer.global_step > 0 and (trainer.global_step % args.log_every == 0):
                        print(" | ".join(
                            [f"Step {trainer.global_step}", f"Loss {avg_loss:.4f}"]
                            + ([f"Reg {step_results['reg_loss']:.6f}"] if step_results.get("reg_loss", 0) > 0 else [])
                            + ([f"KD {step_results['kd_loss']:.6f}"]   if step_results.get("kd_loss", 0)   > 0 else [])
                        ))

                        with torch.no_grad():
                            w = trainer.model.lm_head.weight
                            print("[LM HEAD] mean={:.4e} std={:.4e} min={:.4e} max={:.4e}".format(
                                w.mean().item(), w.std().item(), w.min().item(), w.max().item()
                            ))

                        for n, p in model.named_parameters():
                            if ".mlp." in n and p.grad is not None:
                                print(f"[GRAD] {n} ||g||={p.grad.data.norm().item():.3e}")
                                break

                    if trainer.global_step > 0 and (trainer.global_step % (args.log_every * 5) == 0):
                        trainer.print_layer_stats()

                    # Lightweight eval/preview
                    if args.eval_every > 0 and trainer.global_step > 0 and (trainer.global_step % args.eval_every == 0):
                        prev_mode = trainer.model.training

                        device = next(model.parameters()).device
                        E = model.config.n_embd
                        x = torch.ones(2, 3, E, device=device)

                        model.train()
                        y_train = model.h[0].mlp_dropout(x)          # should apply dropout noise

                        model.eval()
                        with torch.no_grad():
                            y_eval = model.h[0].mlp_dropout(x)       # should be identity

                        print("train==eval?", torch.allclose(y_train, y_eval))  # usually False
                        print("std train:", float(y_train.std().cpu()), "std eval:", float(y_eval.std().cpu()))

                        _probe_first_mlp_weights(model)

                        trainer.model.eval()

                        ctx_ids = batch["input_ids"][0][:32].detach().cpu()
                        prompt_text = trainer.tokenizer.decode(ctx_ids, skip_special_tokens=True)
                        
                        # Generate with consistent routing configuration
                        sample_text = trainer.quick_preview_sample(
                            prompt_text,
                            max_new_tokens=64,  # Reasonable length for preview
                            temperature=0.8,
                            top_p=0.9,
                            top_k=40
                        )

                        print(f"\nSample: '{prompt_text[:50]}...'")
                        print(f"Generated: '{sample_text}'")
                        if prev_mode:
                            trainer.model.train()

                        val_ppl = trainer.evaluate(val_dataloader)
                        #print(f"  Validation - Perplexity: {val_ppl:.2f}")

                    # Checkpointing
                    if args.save_every > 0 and trainer.global_step > 0 and (trainer.global_step % args.save_every == 0):
                        trainer.save_checkpoint(epoch, optimizer, scheduler)

                    # Early stop on max steps
                    if trainer.global_step >= trainer.target_step:
                        print(f"Reached target step: {trainer.target_step}")
                        training_interrupted = True
                        break

            progress_bar.close()
            
            # End of epoch evaluation
            if not training_interrupted and val_dataloader is not None:
                val_ppl = trainer.evaluate(val_dataloader)
                print(f"End of Epoch {epoch + 1} - Validation Perplexity: {val_ppl:.2f}")
        
        # Final checkpoint
        if not training_interrupted:
            trainer.save_checkpoint(args.epochs, optimizer, scheduler, is_final=True)
        
        # Final statistics
        trainer.print_layer_stats()
        
        total_time = time.time() - start_time
        print(f"\nTraining completed!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Total steps: {trainer.global_step}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final checkpoint if interrupted
        if training_interrupted:
            trainer.save_checkpoint(epoch, optimizer, scheduler)
        
        print(f"\nAll files saved to: {checkpoint_dir}")
        
        if device.type == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()