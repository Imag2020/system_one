# 🧠 System 1: building a "JEV-like" decision model from scratch 

<img width="1102" height="619" alt="{7D74CF8C-BAAE-43F0-84CF-730EED867CC1}" src="https://github.com/user-attachments/assets/4358deaf-12bf-46f9-a3c0-a84b81439211" />

**Educational notebook — webinar handout**

We build, piece by piece, a model that receives a **state**, a **question** and a **variable-size list of choices**, and returns **in a single forward pass** a calibrated probability distribution over those choices. No generation, no autoregressive loop, no KV-cache: this is a fast "System 1", meant to serve as a decision module in larger architectures (robotics, ARC-AGI-3 agents, fast trading decisions…).

| Part | Content |
|---|---|
| 0 | Corrections and important clarifications |
| 1 | The idea: a decision = scoring choices in one pass |
| 2 | Setup and configuration |
| 3 | Anatomy of Qwen3-0.6B (and why not Qwen3.5) |
| 4 | Input format: question → state → choices → forced ending |
| 5 | Shared `position_ids`: invariance to order |
| 6 | The 4D attention mask: choices never see each other |
| 7 | Pooling: one vector per choice + one answer vector |
| 8 | No-training tests (shapes, invariance) |
| 9 | Data: schema, procedural generators, augmentation |
| 10 | Decision heads: dot, bilinear, self-attention + bilinear |
| 11 | Per-layer probing, truncation 20/28, LoRA |
| 12 | Training |
| 13 | Validation: accuracy, NLL, Brier, ECE, temperature, invariance, latency |
| 14 | Calibration and RL: what we know, what we propose |
| 15 | Saving, inference, integration (IERM, ARC-AGI-3, trading) |
| A | Appendix: moving to production code with Claude Code |

**Two execution modes**

- `tiny_debug=True` (automatic if no GPU): a **random** mini-Qwen3 (same code, same HF classes, 6 layers of width 128). The whole notebook runs on CPU in a few minutes: perfect for showing tensors, masks and tests live during the webinar. The metrics obviously carry no meaning.
- `tiny_debug=False`: the real `Qwen/Qwen3-0.6B` on GPU (A100/L4/4090…).

