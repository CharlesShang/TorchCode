---
title: TorchCode
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🔥 TorchCode

**Crack the PyTorch interview.**

Practice implementing operators and architectures from scratch — the exact skills top ML teams test for.

*An interactive coding platform, but for tensors. Self-hosted. Jupyter-based. Instant feedback.*

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com)
[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![GitHub stars](https://img.shields.io/github/stars/CharlesShang/TorchCode?style=social)](https://github.com/CharlesShang/TorchCode)
[![GitHub Container Registry](https://img.shields.io/badge/ghcr.io-TorchCode-blue?style=flat-square&logo=github)](https://ghcr.io/duoan/torchcode)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-TorchCode-blue?style=flat-square)](https://huggingface.co/spaces/duoan/TorchCode)
![Problems](https://img.shields.io/badge/problems-83-orange?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-not%20required-brightgreen?style=flat-square)

[![Star History Chart](https://api.star-history.com/svg?repos=CharlesShang/TorchCode&type=Date)](https://star-history.com/#CharlesShang/TorchCode&Date)

</div>

---

## 🎯 Why TorchCode?

Top companies (Meta, Google DeepMind, OpenAI, etc.) expect ML engineers to implement core operations **from memory on a whiteboard**. Reading papers isn't enough — you need to write `softmax`, `LayerNorm`, `MultiHeadAttention`, and full Transformer blocks code.

TorchCode gives you a **structured practice environment** with:

| | Feature | |
|---|---|---|
| 🧩 | **83 curated problems** | The most frequently asked PyTorch interview topics |
| ⚖️ | **Automated judge** | Correctness checks, gradient verification, and timing |
| 🎨 | **Instant feedback** | Colored pass/fail per test case, just like competitive programming |
| 💡 | **Hints when stuck** | Nudges without full spoilers |
| 📖 | **Reference solutions** | Study optimal implementations after your attempt |
| 📊 | **Progress tracking** | What you've solved, best times, and attempt counts |
| 🔄 | **One-click reset** | Toolbar button to reset any notebook back to its blank template — practice the same problem as many times as you want |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) | **Open in Colab** | Every notebook has an "Open in Colab" badge + toolbar button — run problems in Google Colab with zero setup |

No cloud. No signup. No GPU needed. Just `make run` — or try it instantly on Hugging Face.

---

## 🚀 Quick Start

### Option 0 — Try it online (zero install)

**[Launch on Hugging Face Spaces](https://huggingface.co/spaces/duoan/TorchCode)** — opens a full JupyterLab environment in your browser. Nothing to install.

Or open any problem directly in Google Colab — every notebook has an [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/01_relu.ipynb) badge.

### Option 0b — Use the judge in Colab

In Google Colab, install the judge from this repo so `check(...)` uses the same tasks as the notebooks:

```bash
!pip install -q --force-reinstall --no-deps git+https://github.com/CharlesShang/TorchCode.git@master
```

Then in a notebook cell:

```python
from torch_judge import check, status, hint, reset_progress
status()           # list all problems and your progress
check("relu")      # run tests for the "relu" task
hint("relu")       # show a hint
```

### Option 1 — Pull the pre-built image (fastest)

```bash
docker run -p 8888:8888 -e PORT=8888 ghcr.io/duoan/torchcode:latest
```

If the registry image is unavailable for your platform, use Option 2 instead. This is the common path on Apple Silicon / `arm64`.

### Option 2 — Build locally

```bash
make run
```

`make run` will try the prebuilt image first and automatically fall back to a local build when needed.

Open **<http://localhost:8888>** — that's it. Works with both Docker and Podman (auto-detected).

### Option 3 — Standalone Web UI (Next.js + FastAPI)

For a modern, standalone coding experience with an integrated IDE and dual-pane layout:

1. **Start Backend (FastAPI):**
   ```bash
   pip install -r api/requirements.txt
   python -m uvicorn api.main:app --port 8000 --reload
   ```
2. **Start Frontend (Next.js):**
   ```bash
   cd web
   npm install
   npm run dev
   ```
3. Open **<http://localhost:3000>** in your browser.

![TorchCode UI Preview](assets/ui_preview.png)

---

## 📋 Problem Set

> **Frequency**: 🔥 = very likely in interviews, ⭐ = commonly asked, 💡 = emerging / differentiator

### 🧱 Fundamentals — "Implement X from scratch"

The bread and butter of ML coding interviews. You'll be asked to write these without `torch.nn`.

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 1 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/01_relu.ipynb" target="_blank">ReLU</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/01_relu.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `relu(x)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Activation functions, element-wise ops |
| 2 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/02_softmax.ipynb" target="_blank">Softmax</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/02_softmax.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_softmax(x, dim)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Numerical stability, exp/log tricks |
| 16 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/16_cross_entropy.ipynb" target="_blank">Cross-Entropy Loss</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/16_cross_entropy.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `cross_entropy_loss(logits, targets)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Log-softmax, logsumexp trick |
| 42 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/42_label_smoothing.ipynb" target="_blank">Label Smoothing Loss</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/42_label_smoothing.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `label_smoothing_loss(logits, targets, ...)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Smoothed targets, stable logsumexp CE |
| 17 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/17_dropout.ipynb" target="_blank">Dropout</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/17_dropout.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MyDropout` (nn.Module) | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Train/eval mode, inverted scaling |
| 18 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/18_embedding.ipynb" target="_blank">Embedding</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/18_embedding.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MyEmbedding` (nn.Module) | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Lookup table, `weight[indices]` |
| 19 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/19_gelu.ipynb" target="_blank">GELU</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/19_gelu.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_gelu(x)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | ⭐ | Gaussian error linear unit, `torch.erf` |
| 20 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/20_weight_init.ipynb" target="_blank">Kaiming Init</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/20_weight_init.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `kaiming_init(weight)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | ⭐ | `std = sqrt(2/fan_in)`, variance scaling |
| 21 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/21_gradient_clipping.ipynb" target="_blank">Gradient Clipping</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/21_gradient_clipping.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `clip_grad_norm(params, max_norm)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | ⭐ | Norm-based clipping, direction preservation |
| 31 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/31_gradient_accumulation.ipynb" target="_blank">Gradient Accumulation</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/31_gradient_accumulation.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `accumulated_step(model, opt, ...)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 💡 | Micro-batching, loss scaling |
| 40 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/40_linear_regression.ipynb" target="_blank">Linear Regression</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/40_linear_regression.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `LinearRegression` (3 methods) | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Normal equation, GD from scratch, nn.Linear |
| 3 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/03_linear.ipynb" target="_blank">Linear Layer</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/03_linear.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `SimpleLinear` (nn.Module) | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | `y = xW^T + b`, Kaiming init, `nn.Parameter` |
| 4 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/04_layernorm.ipynb" target="_blank">LayerNorm</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/04_layernorm.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_layer_norm(x, γ, β)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Normalization, running stats, affine transform |
| 7 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/07_batchnorm.ipynb" target="_blank">BatchNorm</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/07_batchnorm.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_batch_norm(x, γ, β)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Batch vs layer statistics, train/eval behavior |
| 8 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/08_rmsnorm.ipynb" target="_blank">RMSNorm</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/08_rmsnorm.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `rms_norm(x, weight)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | LLaMA-style norm, simpler than LayerNorm |
| 15 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/15_mlp.ipynb" target="_blank">SwiGLU MLP</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/15_mlp.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `SwiGLUMLP` (nn.Module) | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Gated FFN, `SiLU(gate) * up`, LLaMA/Mistral-style |
| 22 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/22_conv2d.ipynb" target="_blank">Conv2d</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/22_conv2d.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_conv2d(x, weight, ...)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Convolution, unfold, stride/padding |
| 41 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/41_max_pool2d.ipynb" target="_blank">MaxPool2d</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/41_max_pool2d.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `my_max_pool2d(x, kernel_size, ...)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Pooling windows, padding with `-inf`, unfold |

### 🧠 Attention Mechanisms — The heart of modern ML interviews

If you're interviewing for any role touching LLMs or Transformers, expect at least one of these.

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 23 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/23_cross_attention.ipynb" target="_blank">Cross-Attention</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/23_cross_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MultiHeadCrossAttention` (nn.Module) | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Encoder-decoder, Q from decoder, K/V from encoder |
| 5 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/05_attention.ipynb" target="_blank">Scaled Dot-Product Attention</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/05_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `scaled_dot_product_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | `softmax(QK^T/√d_k)V`, the foundation of everything |
| 6 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/06_multihead_attention.ipynb" target="_blank">Multi-Head Attention</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/06_multihead_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MultiHeadAttention` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | Parallel heads, split/concat, projection matrices |
| 9 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/09_causal_attention.ipynb" target="_blank">Causal Self-Attention</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/09_causal_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `causal_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | Autoregressive masking with `-inf`, GPT-style |
| 10 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/10_gqa.ipynb" target="_blank">Grouped Query Attention</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/10_gqa.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `GroupQueryAttention` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | ⭐ | GQA (LLaMA 2), KV sharing across heads |
| 11 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/11_sliding_window.ipynb" target="_blank">Sliding Window Attention</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/11_sliding_window.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `sliding_window_attention(Q, K, V, w)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | ⭐ | Mistral-style local attention, O(n·w) complexity |
| 12 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/12_linear_attention.ipynb" target="_blank">Linear Attention</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/12_linear_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `linear_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Kernel trick, `φ(Q)(φ(K)^TV)`, O(n·d²) |
| 14 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/14_kv_cache.ipynb" target="_blank">KV Cache Attention</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/14_kv_cache.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `KVCacheAttention` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | Incremental decoding, cache K/V, prefill vs decode |
| 24 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/24_rope.ipynb" target="_blank">RoPE</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/24_rope.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `apply_rope(q, k)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | Rotary position embedding, relative position via rotation |
| 25 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/25_flash_attention.ipynb" target="_blank">Flash Attention</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/25_flash_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `flash_attention(Q, K, V, block_size)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Tiled attention, online softmax, memory-efficient |

### 🏗️ Architecture & Adaptation — Put it all together

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 26 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/26_lora.ipynb" target="_blank">LoRA</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/26_lora.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `LoRALinear` (nn.Module) | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Low-rank adaptation, frozen base + `BA` update |
| 27 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/27_vit_patch.ipynb" target="_blank">ViT Patch Embedding</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/27_vit_patch.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `PatchEmbedding` (nn.Module) | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 💡 | Image → patches → linear projection |
| 43 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/43_sinusoidal_positional_encoding.ipynb" target="_blank">Sinusoidal Positional Encoding</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/43_sinusoidal_positional_encoding.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `sinusoidal_positional_encoding(seq_len, dim)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Transformer positions, sin/cos frequencies |
| 13 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/13_gpt2_block.ipynb" target="_blank">GPT-2 Block</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/13_gpt2_block.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `GPT2Block` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | ⭐ | Pre-norm, causal MHA + MLP (4x, GELU), residual connections |
| 28 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/28_moe.ipynb" target="_blank">Mixture of Experts</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/28_moe.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MixtureOfExperts` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | ⭐ | Mixtral-style, top-k routing, expert MLPs |

### ⚙️ Training & Optimization

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 29 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/29_adam.ipynb" target="_blank">Adam Optimizer</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/29_adam.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MyAdam` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Momentum + RMSProp, bias correction |
| 44 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/44_sgd_momentum.ipynb" target="_blank">SGD with Momentum</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/44_sgd_momentum.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MySGD` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Momentum buffers, weight decay, optimizer state |
| 30 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/30_cosine_lr.ipynb" target="_blank">Cosine LR Scheduler</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/30_cosine_lr.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `cosine_lr_schedule(step, ...)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Linear warmup + cosine annealing |

### 🎯 Inference & Decoding

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 32 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/32_topk_sampling.ipynb" target="_blank">Top-k / Top-p Sampling</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/32_topk_sampling.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `sample_top_k_top_p(logits, ...)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Nucleus sampling, temperature scaling |
| 33 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/33_beam_search.ipynb" target="_blank">Beam Search</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/33_beam_search.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `beam_search(log_prob_fn, ...)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Hypothesis expansion, pruning, eos handling |
| 34 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/34_speculative_decoding.ipynb" target="_blank">Speculative Decoding</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/34_speculative_decoding.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `speculative_decode(target, draft, ...)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Accept/reject, draft model acceleration |

### 🔬 Advanced — Differentiators

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 35 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/35_bpe.ipynb" target="_blank">BPE Tokenizer</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/35_bpe.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `SimpleBPE` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Byte-pair encoding, merge rules, subword splits |
| 36 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/36_int8_quantization.ipynb" target="_blank">INT8 Quantization</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/36_int8_quantization.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `Int8Linear` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Per-channel quantize, scale/zero-point, buffer vs param |
| 45 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/45_clip_contrastive_loss.ipynb" target="_blank">CLIP Contrastive Loss</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/45_clip_contrastive_loss.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `clip_contrastive_loss(image, text, ...)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 💡 | InfoNCE, bidirectional retrieval, temperature |
| 37 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/37_dpo_loss.ipynb" target="_blank">DPO Loss</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/37_dpo_loss.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `dpo_loss(chosen, rejected, ...)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Direct preference optimization, alignment training |
| 38 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/38_grpo_loss.ipynb" target="_blank">GRPO Loss</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/38_grpo_loss.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `grpo_loss(logps, rewards, group_ids, eps)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Group relative policy optimization, RLAIF, within-group normalized advantages |
| 39 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/39_ppo_loss.ipynb" target="_blank">PPO Loss</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/39_ppo_loss.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `ppo_loss(new_logps, old_logps, advantages, clip_ratio)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | PPO clipped surrogate loss, policy gradient, trust region |

### 🖼️ MMDiT & Diffusion Transformers

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 46 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/46_patchify_latents.ipynb" target="_blank">Patchify / Unpatchify Latents</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/46_patchify_latents.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `PatchifyLatents` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Latent image tokens, patch flattening, inverse reshape |
| 47 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/47_timestep_embedding.ipynb" target="_blank">Diffusion Timestep Embedding</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/47_timestep_embedding.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `timestep_embedding(t, dim)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Sinusoidal diffusion timesteps, odd dimensions |
| 48 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/48_adaln_zero_modulation.ipynb" target="_blank">AdaLN-Zero Modulation</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/48_adaln_zero_modulation.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `AdaLNZero` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Adaptive LayerNorm, shift/scale/gate, zero init |
| 49 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/49_dit_block.ipynb" target="_blank">DiT Block</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/49_dit_block.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `DiTBlock` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | AdaLN-Zero, self-attention, gated residuals |
| 50 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/50_mmdit_joint_attention.ipynb" target="_blank">MMDiT Joint Attention</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/50_mmdit_joint_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MMDiTJointAttention` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | Separate modality projections, joint attention, split outputs |
| 51 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/51_classifier_free_guidance.ipynb" target="_blank">Classifier-Free Guidance</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/51_classifier_free_guidance.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `classifier_free_guidance(uncond, cond, scale)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Conditional sampling, guidance scale, rescale trick |
| 52 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/52_rectified_flow_loss.ipynb" target="_blank">Rectified Flow Loss</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/52_rectified_flow_loss.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `rectified_flow_loss(model, x0, x1, t)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Flow matching, velocity target, interpolation |
| 53 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/53_euler_flow_sampler.ipynb" target="_blank">Euler Flow Sampler</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/53_euler_flow_sampler.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `euler_flow_sample(model, x0, steps)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | ODE sampling, rectified flow integration |
| 54 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/54_logit_normal_timestep_sampling.ipynb" target="_blank">Logit-Normal Timestep Sampling</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/54_logit_normal_timestep_sampling.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `sample_logit_normal_timesteps(batch_size)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Rectified-flow timestep distribution, sigmoid normal |
| 55 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/55_rope_2d_image_tokens.ipynb" target="_blank">2D RoPE for Image Tokens</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/55_rope_2d_image_tokens.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `apply_2d_rope(x, H, W)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | ⭐ | 2D rotary embeddings, image patch positions |

### 🧬 Modern LLM Systems & Architecture

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 56 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/56_qk_norm_attention.ipynb" target="_blank">QK Norm Attention</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/56_qk_norm_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `qk_norm_attention(q, k, v)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Normalized queries/keys, stable attention logits |
| 57 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/57_rope_scaling.ipynb" target="_blank">RoPE Scaling</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/57_rope_scaling.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `apply_scaled_rope(x, scaling_factor)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Long context, linear position scaling, norm preservation |
| 58 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/58_multi_head_latent_attention.ipynb" target="_blank">Multi-Head Latent Attention</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/58_multi_head_latent_attention.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MultiHeadLatentAttention` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | MLA, compressed KV cache, latent up-projection |
| 59 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/59_paged_kv_cache.ipynb" target="_blank">Paged KV Cache</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/59_paged_kv_cache.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `PagedKVCache` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | vLLM-style blocks, non-contiguous KV storage, cache reuse |
| 60 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/60_prefix_cache.ipynb" target="_blank">Prefix Cache</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/60_prefix_cache.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `PrefixCache` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Prompt KV reuse, longest-prefix lookup |
| 61 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/61_continuous_batching_scheduler.ipynb" target="_blank">Continuous Batching Scheduler</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/61_continuous_batching_scheduler.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `ContinuousBatchingScheduler` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | LLM serving, prefill/decode scheduling, max batch size |
| 62 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/62_speculative_decoding_verification.ipynb" target="_blank">Speculative Decoding with Verification</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/62_speculative_decoding_verification.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `speculative_decode_verify(...)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 🔥 | Draft model proposals, target verification, greedy acceptance |
| 63 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/63_multi_token_prediction_loss.ipynb" target="_blank">Multi-Token Prediction Loss</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/63_multi_token_prediction_loss.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `multi_token_prediction_loss(logits, targets)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 💡 | Future-token auxiliary heads, stable CE, ignore index |
| 64 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/64_qlora_nf4_quantization.ipynb" target="_blank">QLoRA NF4 Quantization</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/64_qlora_nf4_quantization.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `NF4Quantizer` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | ⭐ | NF4 codebook, block quantization, dequantization |
| 65 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/65_fp8_quantization.ipynb" target="_blank">FP8 Quantization Basics</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/65_fp8_quantization.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `FP8Quantizer` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | FP8-style dynamic scaling, clipping, dequantization |
| 66 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/66_lora_merge_unmerge.ipynb" target="_blank">LoRA Merge / Unmerge</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/66_lora_merge_unmerge.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `MergeableLoRALinear` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Adapter deployment, low-rank update, reversible merge |
| 67 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/67_fused_rmsnorm_residual.ipynb" target="_blank">Fused RMSNorm + Residual</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/67_fused_rmsnorm_residual.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `fused_rmsnorm_residual(x, residual, weight)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | RMSNorm, residual stream, fused kernels |

### 🖼️ Image & Video Processing

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 68 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/68_rgb_to_grayscale.ipynb" target="_blank">RGB to Grayscale</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/68_rgb_to_grayscale.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `rgb_to_grayscale(x)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Channel weighting, broadcasting, image tensor shapes |
| 69 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/69_bilinear_resize.ipynb" target="_blank">Bilinear Resize NCHW</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/69_bilinear_resize.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `bilinear_resize(x, out_h, out_w)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Interpolation coordinates, gather/broadcast, align_corners |
| 70 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/70_sobel_edges.ipynb" target="_blank">Sobel Edge Magnitude</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/70_sobel_edges.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `sobel_edges(x)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Depthwise convolution, gradient magnitude, image edges |
| 71 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/71_box_iou.ipynb" target="_blank">Bounding Box IoU</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/71_box_iou.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `box_iou(boxes1, boxes2)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | Object detection geometry, pairwise broadcasting |
| 72 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/72_non_max_suppression.ipynb" target="_blank">Non-Max Suppression</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/72_non_max_suppression.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `non_max_suppression(boxes, scores, iou_threshold)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | Object detection post-processing, IoU filtering |
| 73 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/73_video_uniform_temporal_sample.ipynb" target="_blank">Video Uniform Temporal Sampling</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/73_video_uniform_temporal_sample.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `uniform_temporal_sample(video, num_frames)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Video frame sampling, temporal indices, batch support |
| 74 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/74_temporal_average_pool.ipynb" target="_blank">Temporal Average Pooling for Video</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/74_temporal_average_pool.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `temporal_avg_pool(video, kernel_size, stride)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Video temporal pooling, unfold, BCTHW layout |
| 75 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/75_video_tubelet_patchify.ipynb" target="_blank">Video Tubelet Patchify</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/75_video_tubelet_patchify.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `VideoTubeletPatcher` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | Video transformers, tubelet tokens, inverse reshape |

### 🧵 Python Concurrency

| # | Problem | What You'll Implement | Difficulty | Freq | Key Concepts |
|:---:|---------|----------------------|:----------:|:----:|--------------|
| 76 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/76_thread_safe_counter.ipynb" target="_blank">Thread-Safe Counter</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/76_thread_safe_counter.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `ThreadSafeCounter` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | 🔥 | threading.Lock, race conditions, atomic increments |
| 77 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/77_threaded_map_ordered.ipynb" target="_blank">Ordered Threaded Map</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/77_threaded_map_ordered.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `threaded_map(fn, items, max_workers)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | ThreadPoolExecutor, ordered results, exception propagation |
| 78 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/78_bounded_blocking_queue.ipynb" target="_blank">Bounded Blocking Queue</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/78_bounded_blocking_queue.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `BoundedBlockingQueue` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | Condition variables, producer/consumer, blocking put/get |
| 79 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/79_process_pool_map.ipynb" target="_blank">Process Pool Map</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/79_process_pool_map.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `process_pool_map(fn, items, max_workers)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | multiprocessing, CPU-bound work, ordered process results |
| 80 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/80_async_gather_limited.ipynb" target="_blank">Async Gather with Concurrency Limit</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/80_async_gather_limited.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `async_gather_limited(coros, limit)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | 🔥 | asyncio.Semaphore, ordered results, bounded concurrency |
| 81 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/81_async_timeout_gather.ipynb" target="_blank">Async Gather with Timeout Defaults</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/81_async_timeout_gather.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `gather_with_timeout(coros, timeout, default)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | asyncio.wait_for, timeout handling, ordered results |
| 82 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/82_async_retry.ipynb" target="_blank">Async Retry Helper</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/82_async_retry.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `async_retry(fn, retries, delay)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | ⭐ | retry loops, async sleeps, transient failures |
| 83 | <a href="https://github.com/CharlesShang/TorchCode/blob/master/templates/83_async_queue_pipeline.ipynb" target="_blank">Async Queue Pipeline</a> <a href="https://colab.research.google.com/github/CharlesShang/TorchCode/blob/master/templates/83_async_queue_pipeline.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="20"></a> | `async_queue_pipeline(items, worker_fn, num_workers)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | 💡 | asyncio.Queue, worker tasks, ordered result collection |

---

## ⚙️ How It Works

Each problem has **two** notebooks:

| File | Purpose |
|------|---------|
| `01_relu.ipynb` | ✏️ Blank template — write your code here |
| `01_relu_solution.ipynb` | 📖 Reference solution — check when stuck |

### Workflow

```text
1. Open a blank notebook           →  Read the problem description
2. Implement your solution         →  Use only basic PyTorch ops
3. Debug freely                    →  print(x.shape), check gradients, etc.
4. Run the judge cell              →  check("relu")
5. See instant colored feedback    →  ✅ pass / ❌ fail per test case
6. Stuck? Get a nudge              →  hint("relu")
7. Review the reference solution   →  01_relu_solution.ipynb
8. Click 🔄 Reset in the toolbar  →  Blank slate — practice again!
```

### In-Notebook API

```python
from torch_judge import check, hint, status

check("relu")               # Judge your implementation
hint("causal_attention")    # Get a hint without full spoiler
status()                    # Progress dashboard — solved / attempted / todo
```

---

## 📅 Suggested Study Plan

> **Total: ~32–42 hours spread across 7–8 weeks. Perfect for interview prep on a deadline.**

| Week | Focus | Problems | Time |
|:----:|-------|----------|:----:|
| **1** | 🧱 Foundations | ReLU → Softmax → CE Loss → Label Smoothing → Dropout → Embedding → GELU → Linear → LayerNorm → BatchNorm → RMSNorm → SwiGLU MLP → Conv2d → MaxPool2d | 3–4 hrs |
| **2** | 🧠 Attention Deep Dive | SDPA → MHA → Cross-Attn → Causal → GQA → KV Cache → Sliding Window → RoPE → Linear Attn → Flash Attn | 3–4 hrs |
| **3** | 🏗️ Architecture + Training | GPT-2 Block → LoRA → MoE → ViT Patch → Sinusoidal PE → SGD Momentum → Adam → Cosine LR → Grad Clip → Grad Accumulation → Kaiming Init | 4–5 hrs |
| **4** | 🎯 Inference + Advanced | Top-k/p Sampling → Beam Search → Speculative Decoding → CLIP Loss → BPE → INT8 Quant → DPO Loss → GRPO Loss → PPO Loss + speed run | 4–5 hrs |
| **5** | 🖼️ MMDiT / DiT | Patchify → Timestep Embedding → AdaLN-Zero → CFG → Rectified Flow → Euler Sampler → Logit-Normal Sampling → 2D RoPE → DiT Block → MMDiT Joint Attention | 5–7 hrs |
| **6** | 🧬 Modern LLM Systems | QK Norm → RoPE Scaling → MLA → Paged KV Cache → Prefix Cache → Continuous Batching → Speculative Verification → Multi-Token Prediction → NF4 → FP8 → LoRA Merge → Fused RMSNorm | 6–8 hrs |
| **7** | 🖼️ Image / Video Processing | RGB→Gray → Bilinear Resize → Sobel → IoU → NMS → Temporal Sampling → Temporal Pooling → Tubelet Patchify | 4–5 hrs |
| **8** | 🧵 Python Concurrency | Thread-safe Counter → Threaded Map → Blocking Queue → Process Pool → Async Gather Limit → Async Timeout → Async Retry → Async Queue Pipeline | 4–5 hrs |

---

## 🏛️ Architecture

```text
┌──────────────────────────────────────────┐
│           Docker / Podman Container      │
│                                          │
│  JupyterLab (:8888)                      │
│    ├── templates/  (reset on each run)   │
│    ├── solutions/  (reference impl)      │
│    ├── torch_judge/ (auto-grading)       │
│    ├── torchcode-labext (JLab plugin)    │
│    │     🔄 Reset — restore template     │
│    │     🔗 Colab — open in Colab        │
│    └── PyTorch (CPU), NumPy              │
│                                          │
│  Judge checks:                           │
│    ✓ Output correctness (allclose)       │
│    ✓ Gradient flow (autograd)            │
│    ✓ Shape consistency                   │
│    ✓ Edge cases & numerical stability    │
└──────────────────────────────────────────┘
```

Single container. Single port. No database. No frontend framework. No GPU.

## 🛠️ Commands

```bash
make run    # Build & start (http://localhost:8888)
make stop   # Stop the container
make clean  # Stop + remove volumes + reset all progress
```

## 🧩 Adding Your Own Problems

TorchCode uses auto-discovery — just drop a new file in `torch_judge/tasks/`:

```python
TASK = {
    "id": "my_task",
    "title": "My Custom Problem",
    "difficulty": "medium",
    "function_name": "my_function",
    "hint": "Think about broadcasting...",
    "tests": [ ... ],
}
```

No registration needed. The judge picks it up automatically.

---

## 📦 Publishing `torch-judge` to PyPI (maintainers)

The notebooks install `torch-judge` directly from this GitHub repo so Colab always sees the current task set. For a stable public release, publish the judge package to PyPI so users can also `pip install torch-judge`.

### Automatic (GitHub Action)

Pushing to `master` after changing the package version triggers [`.github/workflows/pypi-publish.yml`](.github/workflows/pypi-publish.yml), which builds and uploads to PyPI. No git tag is required.

1. **Bump version** in `torch_judge/_version.py` (e.g. `__version__ = "0.1.1"`).
2. **Configure PyPI Trusted Publisher** (one-time):
   - PyPI → Your project **torch-judge** → **Publishing** → **Add a new pending publisher**
   - Owner: `CharlesShang`, Repository: `TorchCode`, Workflow: `pypi-publish.yml`, Environment: (leave empty)
   - Run the workflow once (push a version bump to `master` or **Actions → Publish torch-judge to PyPI → Run workflow**); PyPI will then link the publisher.
3. **Release**: commit the version bump and `git push origin master`.

Alternatively, use an API token: add repository secret `PYPI_API_TOKEN` (value = `pypi-...` from PyPI) and set `TWINE_USERNAME=__token__` and `TWINE_PASSWORD` from that secret in the workflow if you prefer not to use Trusted Publishing.

### Manual

```bash
pip install build twine
python -m build
twine upload dist/*
```

Version is in `torch_judge/_version.py`; bump it before each release.

---

## ❓ FAQ

<details>
<summary><b>Do I need a GPU?</b></summary>
<br>
No. Everything runs on CPU. The problems test correctness and understanding, not throughput.
</details>

<details>
<summary><b>Can I keep my solutions between runs?</b></summary>
<br>
Blank templates reset on every <code>make run</code> so you practice from scratch. Save your work under a different filename if you want to keep it. You can also click the <b>🔄 Reset</b> button in the notebook toolbar at any time to restore the blank template without restarting.
</details>

<details>
<summary><b>Can I use Google Colab instead?</b></summary>
<br>
Yes! Every notebook has an <b>Open in Colab</b> badge at the top. Click it to open the problem directly in Google Colab — no Docker or local setup needed. You can also use the <b>Colab</b> toolbar button inside JupyterLab.
</details>

<details>
<summary><b>How are solutions graded?</b></summary>
<br>
The judge runs your function against multiple test cases using <code>torch.allclose</code> for numerical correctness, verifies gradients flow properly via autograd, and checks edge cases specific to each operation.
</details>

<details>
<summary><b>Who is this for?</b></summary>
<br>
Anyone preparing for ML/AI engineering interviews at top tech companies, or anyone who wants to deeply understand how PyTorch operations work under the hood.
</details>

---

## 🤝 Contributors

Thanks to everyone who has contributed to TorchCode.

<!-- readme: contributors -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/duoan">
                    <img src="https://avatars.githubusercontent.com/u/2378740?v=4" width="100;" alt="duoan"/>
                    <br />
                    <sub><b>duoan</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Ando233">
                    <img src="https://avatars.githubusercontent.com/u/74404658?v=4" width="100;" alt="Ando233"/>
                    <br />
                    <sub><b>Ando233</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/abhijitmjj">
                    <img src="https://avatars.githubusercontent.com/u/22732909?v=4" width="100;" alt="abhijitmjj"/>
                    <br />
                    <sub><b>abhijitmjj</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/HareshKarnan">
                    <img src="https://avatars.githubusercontent.com/u/5285984?v=4" width="100;" alt="HareshKarnan"/>
                    <br />
                    <sub><b>HareshKarnan</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/ThierryHJ">
                    <img src="https://avatars.githubusercontent.com/u/51846529?v=4" width="100;" alt="ThierryHJ"/>
                    <br />
                    <sub><b>ThierryHJ</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: contributors -end -->

Auto-generated from the [GitHub contributors graph](https://github.com/CharlesShang/TorchCode/graphs/contributors) with avatars and GitHub usernames.

---

<div align="center">

**Built for engineers who want to deeply understand what they build.**

If this helped your interview prep, consider giving it a ⭐

---

### ☕ Buy Me a Coffee

<a href="https://buymeacoffee.com/duoan" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

<img src="./bmc_qr.png" alt="BMC QR Code" width="150" height="150">

*Scan to support*

</div>
