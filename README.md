# 🧠 New LLM Transformer Design – SwiftTransformer
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status: R&D](https://img.shields.io/badge/status-R%26D-orange.svg)]()
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)]()

**Author:** Songnian Qian  
**Status:** Research & Development  
**Goal:** Redesign transformer-based language models for *semantic alignment*, *efficiency*, and *scalable specialization*.

---

## 🌟 Overview

![SwiftTransformer Architecture](https://github.com/songnianqian/dynamic-transformer/blob/main/SwiftTransformer.png)

This repository presents a **SwiftTransformer Architecture** built around  
**Next-N-Token Prediction** with **Semantic Coherence Evaluation**.

The design introduces eight complementary innovations that together redefine how large language models learn and operate.

| Part | Title | Core Goal |
|------|--------|-----------|
| 1 | Next-N-Token Prediction with Semantic Coherence | Capture meaning beyond single-token accuracy |
| 2 | Reduced Token Embedding Dimension for Attention-FFN Paths | Lower compute without losing capacity |
| 3 | Multi-Path Routing | Route tokens through specialized sub-networks |
| 4 | Adaptive Depth Selection | Dynamically adjust computation per query |
| 5 | Attention Type Pool Selection | Use best attention type per context |
| 6 | Mixture-of-Experts FFN (Breaking O(E²) Bottleneck) | Replace dense FFN with sparse expert activation |
| 7 | Multiple Specialized LM Heads | Domain-specific vocabulary projection |
| 8 | Evaluation Metrics – Redefining Perplexity (sPPL⁺) | Measure *semantic* quality instead of token accuracy |

Together these mechanisms aim for **5–10 × inference speed-up**, **semantic robustness**, and **modular scalability** to 100 K + token contexts. Increase model size without adding computation costs.

---

## 🧩 1 · Next-N-Token Prediction with Semantic Coherence

Traditional transformers predict one token at a time using cross-entropy.  
Here, the model predicts *N tokens at once* (typically N = 3–7) and evaluates entire sequences for:

- **Semantic similarity**
- **Syntactic correctness**
- **Factual and logical coherence**

Training alternates between:

1. **Phase 1 – Conventional Convergence**  
   Standard next-token training for stability.
2. **Phase 2 – Sliding-Window Optimization**  
   Multi-token loss computed on overlapping windows.

This yields smoother, context-aware learning signals and reduces over-penalization of valid alternatives.

> **Status:** ✅ Concept validated & tested in inference.  
> **Next Step:** 🔧 Implement semantic loss for training phase – *to be finished*.

---

## ⚙️ 2 · Reduced Token Embedding Dimension for Attention-FFN Paths

GPT-2 → GPT-4 expanded embedding sizes (768 → 4096), which increases computation quadratically across all downstream layers.  
While this provides higher capacity, it also **amplifies cost in every attention and FFN operation** — the entire model becomes heavier without selective routing.

This design partitions embeddings into multiple **reduced-dimension expert paths** (for example 4096 → 1024), routed by a lightweight gating mechanism.

### 🔍 Core Design

1. **Input-Sequence–Based Routing**  
   Routing decisions are made from the entire sequence embedding, not per token.  
   A small routing network analyzes global context and determines which reduced-dimension path to activate.

2. **Multi-Query Gating (Attention-Style)**  
   Similar to multi-head attention, several learnable query vectors attend over the input sequence.  
   Their outputs form a context representation used to compute expert selection probabilities.

3. **Lightweight Attention-FFN Layers**  
   Each expert path contains a compact Attention + FFN stack with smaller hidden size and fewer parameters.  
   These are efficient enough for experimentation on standard GPUs while retaining expressive capacity.

### ⚙️ Insight

Large embedding sizes do not make sense when the model uses **multi-path or expert-based specialization** (introduced in the next part).  
Without routing, every token passes through the entire 4096-dimensional stack — resulting in massive, unnecessary computation.

### 💡 Benefits

- Maintains representational diversity with lower cost  
- Reduces O(E) computation in all layers  
- Enables specialization across input types without enlarging the full model

> **Status:** 🔬 Concept implemented and validated in prototype.  
> **Next Step:** 🧠 Test multi-query router with several lightweight Attention-FFN path variants – *in progress*.

---

## 🔀 3 · Multi-Path Routing

> **Repository:** [Dynamic-Multiple-Path-Transformer](https://github.com/songnianqian/-Dynamic-Multiple-Path-Transformer)  
> **Purpose:** Enable token- or sequence-level specialization across multiple parallel transformer paths.

### 🧠 Concept

Standard transformers process all tokens through the same stack of attention-FFN layers, regardless of content type or difficulty.  
The **Multi-Path Routing** design breaks this uniformity by introducing several independent *attention-FFN paths* that handle different token categories — such as **code**, **mathematics**, **creative writing**, or **conversational text**.  

Each path acts as a semi-independent “expert transformer,” while a shared backbone maintains general linguistic understanding.

### ⚙️ Architecture

1. **Gated Router**
   - Learns routing probabilities for each token or sequence.  
   - Implemented with a lightweight attention-style gate using multiple learnable queries.  
   - During training, soft routing mixes paths; during inference, hard routing selects the best path.

2. **Specialized Paths**
   - Each path has its own reduced-dimension Attention + FFN block (from Part 2).  
   - Paths can vary in hidden size, dropout, or activation style (GELU, ReLU, or linear-perceptron).  
   - Lightweight variants can run on modest GPUs.

3. **Shared Backbone**
   - A few initial layers remain shared to capture general syntax and semantics.  
   - Prevents over-fragmentation and ensures cross-domain stability.

4. **Routing Criteria**
   - Context-aware: based on sequence embedding, attention entropy, or early-layer activations.  
   - Optionally augmented by domain tags or dataset clustering.

### 📈 Implementation

- Complete working prototype in:  
  👉 [**songnianqian/-Dynamic-Multiple-Path-Transformer**](https://github.com/songnianqian/-Dynamic-Multiple-Path-Transformer)  
- Implemented in PyTorch, compatible with GPT-2 small backbone.  
- Supports both **soft** and **hard** routing.  
- Includes visualization tools for path-usage statistics and gating entropy.

### 🔬 Results & Observations

- **Quality:** Improves contextual consistency without hurting speed.  
- **Efficiency:** Each token activates only one attention-FFN path → significant compute reduction.  
- **Stability:** Keeping a few shared layers helps prevent path collapse.  
- **Scalability:** Architecture can expand to MoE-style routing or adaptive-depth control (see Parts 5–6).

### 🚧 Next Step

- Combine multi-path routing with **multi-LM-head specialization** (Part 7).  
- Explore **load-balance regularization** for stable multi-expert usage.  
- Add benchmark comparisons against Switch-Transformer and DeepSeek-V2 routing strategies.

> **Status:** ✅ Repository live and functional.  
> **Next Step:** 🔧 Extend to adaptive depth and LM-head integration – *in progress*.


---

## 🧱 4 · Adaptive Depth Selection

Not all queries require equal depth.  
If intermediate layers already yield high-confidence and coherent predictions, inference halts early.  
Simple tasks traverse fewer layers; complex reasoning activates more — saving compute without hurting quality.

> **Status:** 🔬 Concept implemented and validated in prototype.  
> **Next Step:** 🧠  *in progress*.
---

## 🧮 5 · Attention Type Pool Selection

Each layer dynamically selects from a pool of attention mechanisms:

| Type | Complexity | Use Case |
|------|-------------|----------|
| Full | O(n²) | Short sequences, global context |
| Sliding Window | O(n × w) | Local grammar dependencies |
| Sparse | O(n√n) | Mixed long/short context |
| Linear | O(n) | Very long sequences |
| Global-Local Hybrid | O(n) | Balanced context trade-off |

The model learns which attention type best fits each layer and sequence.

> **Status:** 🔬 Concept implemented and validated in prototype.  
> **Next Step:** 🧠  *in progress*.

---

## 🧠 6 · Mixture-of-Experts FFN – Breaking the O(E²) Bottleneck
> **Repository code:** `context_readers_model.py` + `context_readers_training.py`

### 🔹 The Problem
In GPT-style transformers, the **FFN (MLP)** block dominates both parameter count and compute.  
It scales as **≈ 4 E²** FLOPs, making it the slowest and largest component.  
However, *not every token* needs that full capacity — different contexts demand different computation.

### 🔹 The Solution — *SpeedyGate*
This design replaces the dense MLP with a **Mixture-of-Experts (MoE) FFN**, built from lightweight experts:

- **Experts:** `FiLM`, `ReGLU1D`, `percN` (rank-N perceptrons)  
- **Routing:** Gating network assigns tokens to experts  
  - **Soft routing** → smooth gradient training  
  - **Hard routing** → fast inference (compute only selected experts)

Implemented in [`MultiMLPLayer`](context_readers_model.py):contentReference[oaicite:0]{index=0} and trained via [`context_readers_training.py`](context_readers_training.py):contentReference[oaicite:1]{index=1}.

### 🔹 Why It Works
- ⚡ **Efficiency:** up to **3–4× faster inference**  
- 🧩 **Scalability:** add more experts → capacity grows, speed unchanged  
- 🌈 **Diversity:** millions of token-specific computation paths  
- 🧠 **Simplicity:** all experts share the same interface, easy to train

Without MLP, Experts FFN achieved same performance.

Layer statistics show light experts dominate — many layers select **FiLM** or **perc4** over 70 % of the time.  
This proves the heavy GPT-style 2E MLP is unnecessary for most tokens.

### 🔹 A New Scaling Path
Instead of endlessly enlarging embedding sizes (GPT-2 → GPT-3 → GPT-4),  
we can **scale horizontally** — add experts and smarter gating — to gain capacity  
*without increasing inference cost.*

> ✅ **Tested:** all linear experts perform equivalently; MLP no longer required.  
> 🔧 **Next Step:** extend SpeedyGate to adaptive-depth and multi-LM-head systems (see Part 7 & 8).

---

## 🧩 7 · Multiple Specialized LM Heads
> **Fast-K Multi-Header Language Models: Sparse Routing at the LM Head for Efficient Generation**  
> Repository coming soon – implementation integrated with SpeedyGate MoE (Part 6).

### 🔹 The Problem
Traditional language models rely on a **single large projection head** to map hidden states to a vocabulary of 50 000 + tokens.  
This creates a computational bottleneck — every token update multiplies by the entire vocabulary —  
and fails to capture that *the same word can mean different things in different contexts*.

### 🔹 The Solution — Multi-Header Architecture with Fast-K Inference
Replace the single LM head with **P parallel heads**.  
Each head can specialize in a different linguistic or semantic domain.

During **training**:
- Only the head assigning the *highest probability* to the gold token is updated.  
  *(Token-aware, sparse update — only one head learns per token.)*

During **inference**:
1. A **pilot head** proposes a Top-K shortlist (e.g., 50 tokens).  
2. Other heads score only these K candidates.  
3. The best *(head, token)* pair is selected.

This reduces computational complexity from  
**O(P × V)** → **O(V + (P – 1) × K)** per decoding step —  
massive savings when V = 50 000 and K = 50.

---

### 🧠 Architecture at a Glance

Pilot Head → Top-K shortlist
│
Head 1, Head 2, … Head P-1 → Score Top-K tokens
│
Select (best head, token) → Next token output

Each head maintains its own **reduced vocabulary header** and projection weights, enabling context-specific interpretation.

---

### 🔑 Key Innovations
**1. Multiple Specialized Heads**  
Each head focuses on a subset of language patterns — syntax, math, dialogue, or reasoning.  
Only the best-performing head learns from each example, yielding *self-organized specialization.*

**2. Fast-K Sparse Inference**  
Instead of evaluating all heads over all tokens, only shortlisted columns are scored.  
This yields near-constant latency even as more heads are added.

---

### 📊 Results That Matter
| Metric | GPT-2 Baseline | Fast-K Multi-Header |
|:-------|:---------------|:--------------------|
| Perplexity | 25.66 | **16.11** |
| Generation Speed | 1.0 × | **1.65 × faster** |
| Rare-Token Accuracy | baseline | **↑ improved top-n accuracy** |

✅ **65 % faster generation** with 2.9 % better overall quality  
✅ **Improved rare-token handling** across domains  
✅ **Semantic specialization:** same word routed to different heads depending on context  

Example (“make” routed dynamically):  
- “make decisions” → Head 3 *(create/produce)*  
- “make them for themselves” → Head 1 *(act for others)*  
- “make it a problem” → Head 3 *(transformative)*  
- “make the solution” → Head 2 *(causative)*  

---

### ⚙️ Implementation Notes
- **Reduced Vocabulary Header:** each head may use a smaller domain-specific sub-vocabulary (*work in progress*).  
- **Efficient slice-matmul path:** shortlist columns are contiguous for cache locality.  
- **Pilot selection:** entropy-based, utilization-balanced, or round-robin strategies all work.  
- **Quantization compatible:** works with 4-bit heads via slice de-quantization.  

---

### 🌍 Why This Matters
- **Efficiency:** lower computational cost → faster inference  
- **Quality:** better handling of rare and context-dependent words  
- **Scalability:** easily integrates with existing transformers  
- **Applicability:** ideal for latency-sensitive or edge deployments where output projection dominates  

---

### 🧩 The Bigger Picture
Like humans making hierarchical decisions (*screening resumes before interviews*),  
language models can first **filter possibilities**, then **refine** on a small candidate set.  
Fast-K follows this principle: quick broad screening, then targeted scoring.

---

### 🚀 What’s Next
- Finish reduced-vocabulary header implementation.  
- Integrate Fast-K heads with SpeedyGate MoE (Part 6).  
- Release training + evaluation code on GitHub and submit arXiv preprint.  
- Explore reward-based training as an alternative to gradient descent.

> **Status:** ✅ Core architecture complete and validated.  
> **Next Step:** 🔧 Implement reduced-vocabulary Fast-K headers – *in progress*.


---

## 📏 8 · Evaluation Metrics – Semantic Perplexity Plus (sPPL⁺)
> **Status:** 🧪 In Design – aims to measure true semantic quality beyond token-level accuracy.

### 🔹 The Problem
Traditional **Perplexity (PPL)** is a token-exact metric: it rewards models that assign high probability to the *exact* next token in a reference sequence.  
However, this fails to recognize semantically equivalent outputs:

| Reference | Prediction | PPL Judgment |
|------------|-------------|--------------|
| "The car stopped." | "The vehicle halted." | ❌ *Penalized* despite identical meaning |

Large-scale LLMs often generate many valid continuations.  
A metric that only checks surface tokens underestimates real performance and encourages over-fitting to literal text.

---

### 🔹 The Solution — Semantic Perplexity Plus (sPPL⁺)
**sPPL⁺** extends classical perplexity with semantic, syntactic, and factual coherence evaluation.  
It compares *multi-token windows* rather than individual tokens and weighs losses by meaning similarity.

#### 🧠 Core Components
1. **Window-Level Evaluation**  
   Compute probabilities over overlapping N-token windows (N ≈ 3–7) instead of single tokens.

2. **Semantic Weighting**  
   For each window, scale cross-entropy by embedding similarity between predicted and reference segments  
   (e.g., cosine similarity of encoder-based sentence embeddings).

3. **Multi-Aspect Integration**  
   Combine penalties for:
   - semantic divergence (meaning mismatch)  
   - syntactic error (grammar violation)  
   - factual contradiction (from NLI or QA checks)  
   - logical inconsistency (detected via reasoning probes)

4. **Composite Score**  
   Aggregate window losses to produce **sPPL⁺**, where lower = better contextual and semantic alignment.

---

### 🧮 Prototype Formula
Let  
- *pₜ* = model probability of token t  
- *E(w)* = embedding of window w  
- *sim* = cosine similarity(E_pred, E_ref)  

Then for window w of length N:
sPPL⁺ = exp( - (1/T) · Σ [ log pₜ · (1 + λ · sim(E_pred, E_ref)) ] )
λ controls semantic weighting strength (typ. 0.3–0.7).  
This formulation rewards semantically correct alternatives even if the exact token differs.

---

### 🔍 Planned Extensions
- **Dynamic window size** – adapt N by sequence entropy  
- **Cross-model evaluation** – use teacher embeddings to judge coherence  
- **Domain weighting** – stronger penalties in factual or code domains  
- **Integration with reward training** – use sPPL⁺ as reinforcement signal

---

### 🧪 Current Progress
- ✅ Concept defined and baseline scripts drafted  
- ⚙️ Prototype running on small GPT-2 variants using sentence-embedding similarity  
- 🚧 Needs calibration for long-context tasks (≥ 1 K tokens)  
- 🧩 Next step: integrate with Fast-K heads and SpeedyGate MoE for semantic-aware evaluation

---

### 🌍 Why It Matters
- **Semantic-Aware Evaluation:** distinguishes meaningful paraphrases from true errors  
- **Better Training Feedback:** encourages models to prefer coherent continuations  
- **Cross-Model Comparison:** enables fairer benchmarks between architectures with different tokenizations  
- **Bridge to Human Judgment:** aligns automatic scores with how humans perceive fluency and meaning.

---

> **Goal:** redefine what “good text generation” means — not just low perplexity,  
> but *semantic coherence, logical flow, and factual alignment.*

> **Next Step:** build open-source evaluation toolkit and release benchmark results comparing GPT-2, Mixtral, and SpeedyGate-based models.


---

## 🚀 Future Directions & Challenges

- Integrate multi-LM-head routing with adaptive attention pools  
- Reinforcement-style semantic reward fine-tuning  
- Publish open evaluation scripts for sPPL⁺  
- Explore genetic evolution of automata and expert paths  
- Study compute scaling and energy efficiency trade-offs  

---

## 📂 Repository Structure (Planned)
├── context_readers_model.py         # SpeedyGate MoE FFN module
├── context_readers_training.py      # Training loop for MoE + routing
├── multi_header_fastk.py            # Multi-LM-Head Fast-K implementation (In different project)
├── evaluation/                      # Scripts for sPPL⁺ and semantic scoring
├── checkpoints/                     # Sample or reference weights
├── README.md
└── LICENSE

---

## 📜 License

Released under the MIT License for academic and research use.  
Feel free to fork and experiment — citations are appreciated.

---

## 🙌 Acknowledgments

This project draws on insights from:
- Switch Transformer, GLaM, Mixtral, DeepSeek-V2  
- DeeBERT and ElasticBERT (adaptive depth)  
- XLNet and SpanBERT (multi-token objectives)  
- OpenAI GPT-2 baseline for proof-of-concept training  

---




















