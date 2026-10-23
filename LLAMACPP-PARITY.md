# NeuralFree and llama.cpp: Architecture Parity

This page maps every llama.cpp architecture to its status in NeuralFree's GGUF engine (`com.eonmux.neuralfree.gguf`). It also describes the memory, speculative-decoding, LoRA, encoder-decoder, diffusion and audio features the engine implements, using llama.cpp's terms.

**How the status was established.** NeuralFree's architecture set is generated from `ArchRegistry` and the `ArchTable*` classes in `gguf/arch`. It names the same 164 entries as llama.cpp's `LLM_ARCH_NAMES`, including the `(unknown)` sentinel. Each architecture was implemented by tracing llama.cpp's loader (`llama-model.cpp`) and its graph builder (`src/models/*.cpp`). The engine was verified by:
- compiling the Java sources;
- compiling every GPU kernel with the kernel compilers: clang (OpenCL C 1.2), nvcc (PTX), and glslc followed by spirv-val (Vulkan);
- tracing the code against llama.cpp.

No model was run to produce this page, and no row claims hardware validation.

## Support classes

`SupportClass` (`gguf/arch/SupportClass.java`) decides what a model can do:

| Class | Meaning | Count |
|---|---|---|
| GEN | Autoregressive text generation | 136 |
| EMBED | Pooled or per-token embeddings | 12 |
| CLS | Classification or ranking scores (decision heads) | 2 |
| DIFF | Masked-diffusion generation | 4 |
| DRAFT | Speculative drafter bound to a target model | 3 |
| AUDIO | Audio codec or speech token model | 3 |
| ENC_DEC | Encoder-decoder generation | 1 |
| NA | Rejected at load with the reason shown in the matrix | 3 |

Every non-NA architecture has both a CPU forward path (`GGUFModel`) and a GPU forward path (`GGUFModelGPU`, on OpenCL, CUDA or Vulkan). Both are built from the same `ModelSpec`. `CpuFactoryCatalog` and `GpuFactoryCatalog` register a block factory for every `MixerKind` and `FfnKind`.

## Architecture matrix

Legend:
- **#**: llama.cpp `llm_arch` order.
- **Cache kind**: the model's memory, from `MemoryKind` and the layer specs.
  - `KV`: attention KV cache.
  - `SWA`: sliding-window layers, stored as ring caches where llama.cpp builds an ISWA cache.
  - `MLA`: multi-head latent attention cache, latent by default.
  - `recurrent`: per-sequence recurrent state; the mixer is named in the notes.
  - `hybrid`: recurrent layers and attention layers together.
  - `DSA indexer` / `MSA index` / `compressed KV`: the DeepSeek sparse-attention, MiniMax sparse-attention and DeepSeek-V4 side caches.
  - `none`: a cache-less forward pass (encoders, classifiers, diffusion, the WavTokenizer decoder).
- **MTP / NextN**:
  - `MTP`: the architecture has a NextN/MTP draft graph, usable with `--mtp` (the target's own blocks) or with an MTP draft file.
  - `chained heads`: several NextN blocks draft one token each.
  - `NextN tensors, no MTP graph`: the blocks are loaded and validated, but llama.cpp defines no MTP draft graph for the architecture, so drafting with them is refused.
- **Experts**: the architecture's tensor table includes routed-expert tensors.

| # | Architecture | Class | Cache kind | MTP / NextN | Notes |
|---|---|---|---|---|---|
| 0 | `clip` | NA | none | - | Rejected: architecture 'clip' is a vision/audio projector without a text model (llama.cpp loads no hparams for clip); load the language model GGUF instead. |
| 1 | `llama` | GEN | KV | - | Dense FFN without experts, Mixtral-style MoE with them. |
| 2 | `llama4` | GEN | KV + chunked SWA | - | Experts. Chunked attention (8192) on 3 of 4 layers, NoPE + temperature on the 4th; MoE every interleave step with a shared expert; weightless QK norm except on 128-expert files. |
| 3 | `deci` | GEN | KV | - | Per-layer attention-free and linear-replacement blocks are resolved from n_head / n_head_kv. |
| 4 | `falcon` | GEN | KV | - | Attention and MLP read the same normalized input and are summed. |
| 5 | `grok` | GEN | KV | - | Experts. GELU experts; attn_output_norm and the block output norm wrap both branches. |
| 6 | `gpt2` | GEN | KV | - |  |
| 7 | `gptj` | NA | none | - | Rejected: architecture 'gptj' has no model implementation in llama.cpp and is not supported. |
| 8 | `gptneox` | GEN | KV | - | Parallel residual decided per file (use_parallel_residual). |
| 9 | `mpt` | GEN | KV | - |  |
| 10 | `baichuan` | GEN | KV | - |  |
| 11 | `starcoder` | GEN | KV | - |  |
| 12 | `refact` | GEN | KV | - | Experts. |
| 13 | `bert` | EMBED | none | - |  |
| 14 | `modern-bert` | EMBED | none | - | Symmetric sliding-window mask on the cache-less encoder. GEGLU-erf with the gate packed in ffn_up. |
| 15 | `nomic-bert` | EMBED | none | - | BERT-family graph. |
| 16 | `nomic-bert-moe` | EMBED | none | - | BERT-family graph. Experts. |
| 17 | `neo-bert` | EMBED | none | - | SwiGLU with the gate packed in ffn_up. |
| 18 | `jina-bert-v2` | EMBED | none | - | BERT-family graph. |
| 19 | `jina-bert-v3` | EMBED | none | - | BERT-family graph. |
| 20 | `eurobert` | EMBED | none | - |  |
| 21 | `bloom` | GEN | KV | - |  |
| 22 | `stablelm` | GEN | KV | - | Without ffn_norm (StableLM 2 12B) the MLP reads the attention norm output, a parallel residual decided by tensor presence. |
| 23 | `qwen` | GEN | KV | - |  |
| 24 | `qwen2` | GEN | KV | - |  |
| 25 | `qwen2moe` | GEN | KV | - | Experts. Shared expert gated by sigmoid(ffn_gate_inp_shexp). |
| 26 | `qwen2vl` | GEN | KV | - | M-RoPE (text positions). |
| 27 | `sensenova_u1` | GEN | KV | - | llama.cpp reports MROPE, but the graph rotates text with NEOX over head/2 with text positions [t, 0, 0, 0]. |
| 28 | `qwen3` | GEN | KV | - |  |
| 29 | `flint` | GEN | KV + SWA | - | Experts. Explicit SWA and RoPE layer arrays, attention sinks, biased routing. |
| 30 | `qwen3moe` | GEN | KV | - | Experts. |
| 31 | `qwen3next` | GEN | hybrid: recurrent + KV | MTP | Gated DeltaNet mixer. Experts. post_attention_norm is the pre-FFN norm; attention output is sigmoid-gated by the Q half. |
| 32 | `qwen3vl` | GEN | KV | - | Interleaved M-RoPE (text positions). |
| 33 | `qwen3vlmoe` | GEN | KV | - | Experts. Interleaved M-RoPE over the four position sections. |
| 34 | `qwen35` | GEN | hybrid: recurrent + KV | MTP | Gated DeltaNet mixer. Interleaved M-RoPE (text positions). |
| 35 | `qwen35moe` | GEN | hybrid: recurrent + KV | MTP | Gated DeltaNet mixer. Interleaved M-RoPE (text positions). Experts. |
| 36 | `clef` | CLS | none | NextN tensors, no MTP graph | Gated DeltaNet mixer. Interleaved M-RoPE (text positions). qwen35 backbone run without memory, joint decision head. |
| 37 | `vllm-decision2` | CLS | KV | - | Runs the qwen3 graph. qwen3 backbone with a per-option decision head. |
| 38 | `qwen4exp` | GEN | hybrid: recurrent + KV + indexer | MTP | Gated DeltaNet mixer. Interleaved M-RoPE (text positions). Experts. Hyper-connection mixers replace the residual adds and the block norms; the QSA indexer of the attention layers needs the indexed hybrid memory. |
| 39 | `phi2` | GEN | KV | - | Attention and MLP read the same normalized input and are summed. |
| 40 | `phi3` | GEN | KV | - |  |
| 41 | `phimoe` | GEN | KV | - | Experts. phi3 graph: RMS norms with bias tensors, LongRoPE factors. |
| 42 | `plamo` | GEN | KV | - | The MLP reuses the attention norm output; both branches are summed. |
| 43 | `plamo2` | GEN | hybrid: recurrent + KV | - | PLaMo-2 Mamba mixer. Sandwich norms around both branches; ffn_up holds the fused SwiGLU gate and up halves. |
| 44 | `plamo3` | GEN | KV + SWA (if set) | - | Sliding window when the file sets one. |
| 45 | `codeshell` | GEN | KV | - |  |
| 46 | `orion` | GEN | KV | - |  |
| 47 | `internlm2` | GEN | KV | - |  |
| 48 | `minicpm` | GEN | KV | - | Experts. Runs the Granite graph: scaled residual branches and divided logits. |
| 49 | `minicpm3` | GEN | MLA | - | Non-absorbed MLA with a dense FFN. |
| 50 | `gemma` | GEN | KV | - |  |
| 51 | `gemma2` | GEN | KV + SWA | - |  |
| 52 | `gemma3` | GEN | KV + SWA | - |  |
| 53 | `gemma3n` | GEN | KV + SWA | - |  |
| 54 | `gemma4` | GEN | KV + SWA | - | Experts. GELU FFN and experts, sandwich norms, per-layer embeddings, KV sharing. |
| 55 | `gemma4-assistant` | DRAFT | reads the target's KV | is a drafter | Gemma 4 assistant drafter: Q-only attention over the target's KV caches (writes no cache of its own), GELU FFN, sandwich norms. |
| 56 | `gemma-embedding` | EMBED | none | - | Symmetric sliding-window mask on the cache-less encoder. Always bidirectional; gemma sandwich norms. |
| 57 | `gemma-embedding2` | EMBED | none | - | Symmetric sliding-window mask on the cache-less encoder. Always bidirectional; gemma sandwich norms. |
| 58 | `starcoder2` | GEN | KV | - |  |
| 59 | `mamba` | GEN | recurrent | - | Mamba-1 mixer. |
| 60 | `mamba2` | GEN | recurrent | - | Mamba-2 mixer. |
| 61 | `maple` | GEN | KV + SWA | - | Experts. RoPE only on SWA layers; clamped routed-expert SwiGLU. |
| 62 | `jamba` | GEN | hybrid: recurrent + KV | - | Mamba-1 mixer. Experts. |
| 63 | `falcon-h1` | GEN | hybrid: recurrent + KV | - | Mamba-2 mixer. Every block runs the Mamba-2 and attention mixers in parallel and keeps both memories. |
| 64 | `xverse` | GEN | KV | - |  |
| 65 | `command-r` | GEN | KV | - | The MLP reuses the attention norm output; both branches are summed. |
| 66 | `cohere2` | GEN | KV + SWA | - |  |
| 67 | `cohere2moe` | GEN | KV + SWA | MTP | Experts. Attention and MoE read the same norm output (parallel residual). The block norm is LayerNorm when the file has no RMS epsilon (stored as 0), RMS otherwise. |
| 68 | `dbrx` | GEN | KV | - | Experts. LayerNorm without bias; attn_output_norm is the pre-FFN norm. |
| 69 | `olmo` | GEN | KV | - | Parameter-free LayerNorm: no norm tensors. |
| 70 | `olmo2` | GEN | KV + SWA | - | No pre-norms: norms apply to the branch outputs only. |
| 71 | `olmoe` | GEN | KV | - | Experts. Full-width QK norm; unnormalized softmax routing. |
| 72 | `muse-glimmer` | GEN | KV + SWA | - |  |
| 73 | `muse-spark` | GEN | KV + SWA | - | Experts. Gated attention, learned residual alpha/beta mixing, latent MoE; RoPE on SWA layers. |
| 74 | `openelm` | GEN | KV | - |  |
| 75 | `arctic` | GEN | KV | - | Experts. Dense FFN residual in parallel with the expert branch (own norm ffn_norm_exps). |
| 76 | `deepseek` | GEN | KV | - | Experts. Dense lead, routed plus shared experts. |
| 77 | `deepseek2` | GEN | MLA | MTP | Experts. MLA (absorbed when k_b/v_b are split), YaRN, grouped and biased routing. |
| 78 | `deepseek2-ocr` | GEN | KV | - | Experts. deepseek2 graph with plain multi-head attention. |
| 79 | `deepseek32` | GEN | MLA + DSA indexer | MTP | Experts. MLA plus the lightning indexer selecting top-k keys. |
| 80 | `deepseek4` | GEN | compressed KV + indexer + SWA | MTP | Experts. Compressed KV caches with indexer, hyper-connections, hash routing on the first layers and sqrt(softplus) scoring. |
| 81 | `chatglm` | GEN | KV | - |  |
| 82 | `glm4` | GEN | KV | NextN tensors, no MTP graph | NORM, or MROPE when the file carries M-RoPE sections. |
| 83 | `glm4moe` | GEN | KV | MTP | Experts. MROPE when the file carries rope sections, NEOX otherwise. post_attention_norm is the pre-FFN norm. |
| 84 | `glm-dsa` | GEN | MLA + DSA indexer | MTP | Experts. MLA plus a DSA indexer on full-indexer layers; shared layers reuse the top-k. |
| 85 | `bitnet` | GEN | KV | - | The FFN is a gated SiLU MLP. |
| 86 | `t5` | ENC_DEC | encoder + self/cross KV | - | GELU-gated FFN when ffn_gate exists, ReLU otherwise. |
| 87 | `t5encoder` | EMBED | none | - | Runs the t5 graph. Encoder-only T5: pooled embeddings. |
| 88 | `jais` | GEN | KV | - |  |
| 89 | `jais2` | GEN | KV | - |  |
| 90 | `nemotron` | GEN | KV | - |  |
| 91 | `nemotron_h` | GEN | hybrid: recurrent + KV | NextN tensors, no MTP graph | Mamba-2 mixer. Experts. Single-kind layers; the FFN layers run an ungated squared-ReLU MLP, routed experts use sigmoid gating with a selection bias. |
| 92 | `nemotron_h_moe` | GEN | hybrid: recurrent + KV | MTP | Mamba-2 mixer. Experts. |
| 93 | `exaone` | GEN | KV | - |  |
| 94 | `exaone4` | GEN | KV + SWA | NextN tensors, no MTP graph | No pre-norms: norms apply to the branch outputs only. |
| 95 | `exaone-moe` | GEN | KV + SWA | NextN tensors, no MTP graph | Experts. RoPE only on local (SWA) layers; dense lead and NextN blocks are dense. |
| 96 | `rwkv6` | GEN | recurrent | - | RWKV-6 mixer. LayerNorm throughout; the channel mix is an ungated squared-ReLU MLP behind a receptance gate. |
| 97 | `rwkv6qwen2` | GEN | recurrent | - | RWKV-6 (Qwen2) mixer. |
| 98 | `rwkv7` | GEN | recurrent | - | RWKV-7 mixer. |
| 99 | `arwkv7` | GEN | recurrent | - | RWKV-7 mixer. |
| 100 | `granite` | GEN | KV | - | Experts. Scaled residual branches and divided logits. |
| 101 | `granitemoe` | GEN | KV | - | Experts. |
| 102 | `granitehybrid` | GEN | hybrid: recurrent + KV | - | Mamba-2 mixer. Experts. Residual branches are scaled by residual_scale and logits divided by logit_scale. |
| 103 | `graniteswitch` | GEN | KV | - |  |
| 104 | `granite_swa` | GEN | KV + SWA | - | Experts. granite-style scaled residuals and divided logits; sinks; RoPE per rope_pattern. |
| 105 | `chameleon` | GEN | KV | - |  |
| 106 | `wavtokenizer-dec` | AUDIO | none | - | Decodes WavTokenizer codes to spectral features (no iSTFT waveform synthesis); group norms in the posnet, LayerNorm and GELU in the ConvNeXt blocks. |
| 107 | `plm` | GEN | MLA | - | MLA without Q LoRA; ungated squared-ReLU FFN. |
| 108 | `bailingmoe` | GEN | KV | - | Experts. Softmax routing, weight normalization from the file. |
| 109 | `bailingmoe2` | GEN | KV | NextN tensors, no MTP graph | Experts. Grouped, biased routing (gating named by the file); partial NEOX rotation. |
| 110 | `bailingmoe3` | GEN | hybrid: recurrent + MLA | MTP | KDA mixer. Experts. M-RoPE when the file has rope sections, NORM otherwise. |
| 111 | `dots1` | GEN | KV | - | Experts. Dense lead, biased routed plus shared experts (gating named by the file). |
| 112 | `dots3note` | GEN | MLA + DSA indexer + SWA | NextN tensors, no MTP graph | Experts. MLA with separate sliding-window ranks, interleaved SWA, DSA indexer. |
| 113 | `arcee` | GEN | KV | - |  |
| 114 | `afmoe` | GEN | KV + SWA | - | Experts. Sandwich norms, gated attention, NoPE every 4th layer. |
| 115 | `laguna` | GEN | KV + SWA | - | Experts. Gated attention; YaRN on full layers, plain RoPE on SWA layers. |
| 116 | `ernie4_5` | GEN | KV | - |  |
| 117 | `ernie4_5-moe` | GEN | KV | - | Experts. MoE on layers past the dense lead at the configured step. |
| 118 | `hunyuan-moe` | GEN | KV | - | Experts. QK norms; shared expert on every layer. |
| 119 | `hunyuan-dense` | GEN | KV | - | Reuses the hunyuan_vl loader and graph; always NEOX, M-RoPE sections are ignored. |
| 120 | `hunyuan_vl` | GEN | KV | - | NEOX, or MROPE when the file carries M-RoPE sections. |
| 121 | `hy_v3` | GEN | KV | MTP (chained heads) | Experts. A block is MoE when its ffn_gate_inp is present; sigmoid routing by default. |
| 122 | `hy_v4` | GEN | MLA + DSA indexer | - | Experts. MLA with sinks and output gate, hyper-connections, optional DSA indexer. |
| 123 | `smollm3` | GEN | KV | - |  |
| 124 | `gpt-oss` | GEN | KV + SWA | - | Experts. Attention sinks; post_attention_norm is the pre-FFN norm; softmax over the selected router logits; clamped OpenAI SwiGLU experts. |
| 125 | `lfm2` | GEN | hybrid: recurrent + KV + SWA (if set) | - | Short-convolution mixer. Experts. Sliding window when the file sets one. |
| 126 | `lfm2moe` | GEN | hybrid: recurrent + KV | - | Short-convolution mixer. Experts. |
| 127 | `dream` | DIFF | none | - | Masked diffusion (Dream timestep schedule). |
| 128 | `smallthinker` | GEN | KV + SWA | - | Experts. ReLU experts; the router reads the block input before the attention norm. |
| 129 | `llada` | DIFF | none | - | Masked diffusion (LLaDA block schedule). |
| 130 | `llada-moe` | DIFF | none | - | Experts. Masked diffusion; softmax routing. |
| 131 | `seed_oss` | GEN | KV | - | post_attention_norm is the pre-FFN norm; there is no ffn_norm. |
| 132 | `grovemoe` | GEN | KV | - | Experts. Routed experts plus scaled chunk experts per expert group. |
| 133 | `apertus` | GEN | KV | - |  |
| 134 | `minimax-01` | GEN | hybrid: recurrent + KV | - | Lightning attention mixer. Experts. Both residual branches add the normed block input scaled by residual_scale. |
| 135 | `hrm_text` | GEN | KV | - | Weightless RMS norms around each sub-block. |
| 136 | `minimax-m2` | GEN | KV | - | Experts. Full-width QK norm; biased routing (gating named by the file). |
| 137 | `minimax-m3` | GEN | KV + MSA index | - | Experts. Block-sparse MiniMax attention; clamped OpenAI SwiGLU experts. |
| 138 | `cogvlm` | GEN | KV | - | The vis_* expert weights serve image tokens only. |
| 139 | `rnd1` | DIFF | none | - | Experts. Masked diffusion; softmax routing. |
| 140 | `pangu-embedded` | GEN | KV | - |  |
| 141 | `mistral3` | GEN | KV | - | Experts. |
| 142 | `eagle3` | DRAFT | KV | - | EAGLE-3 drafter: fuses three target-layer hidden states, one decoder layer with its own cache, optional d2t vocabulary map. |
| 143 | `dflash` | DRAFT | KV | - | Experts. Block-diffusion drafter (DFlash, DFlash2, DSpark, LiLiCorr); target-layer features injected into its own KV cache. |
| 144 | `mistral4` | GEN | MLA | NextN tensors, no MTP graph | Experts. Loads and runs as deepseek2. |
| 145 | `paddleocr` | GEN | KV | - | M-RoPE (text positions). Reuses the ernie4_5 loader with its own MROPE graph. |
| 146 | `mimo2` | GEN | KV + SWA | MTP (chained heads) | Experts. A block is MoE when its ffn_gate_inp is present; sigmoid routing. |
| 147 | `step35` | GEN | KV + SWA | MTP (chained heads) | Experts. Half-rotary full layers, optional head-wise attention gate, clamped SwiGLU; a block is MoE when its ffn_gate_inp is present. |
| 148 | `spark2_5` | GEN | KV + SWA | - |  |
| 149 | `llama-embed` | EMBED | none | - | llama hparams and tensors with the cache-less embedding graph. |
| 150 | `maincoder` | GEN | KV | - |  |
| 151 | `maion-coder` | GEN | KV + SWA | - | Experts. Sandwich norms; latent MoE every interleave step; RoPE on SWA layers. |
| 152 | `kimi-linear` | GEN | hybrid: recurrent + MLA | - | KDA mixer. Experts. |
| 153 | `kimi-k3` | GEN | hybrid: recurrent + MLA | - | KDA mixer. Experts. Cross-layer attention residuals (attn_res_block_size) and the SiTU GLU (situ_beta, situ_linear_beta) are file-resolved hparams. |
| 154 | `glm5-next` | GEN | hybrid: recurrent + MLA + indexer | MTP | KDA mixer. Experts. KDA + k-pool DSA hybrid with mHC residual streams. |
| 155 | `talkie` | GEN | KV | - | Weightless norms, embedding skip scaled by layer_output_scale, multiplied logits. |
| 156 | `mellum` | GEN | KV + SWA | MTP | Experts. Softmax routing by default; SWA layers use plain RoPE. |
| 157 | `nanbeige` | GEN | KV | - |  |
| 158 | `qwen3tts` | AUDIO | KV | - | Interleaved M-RoPE (text positions). qwen3vl hparams, tensors and graph; generates codec tokens. |
| 159 | `pockettts` | AUDIO | KV | - | LayerNorm, ungated GELU FFN; returns backbone hidden states (input of the PocketTTS flow network). |
| 160 | `k2-horizon` | GEN | KV | - | Experts. Block norms are RMS norms over attention.groupnorm_groups channel groups (1 = plain RMS); MoE from the dense lead, value experts (MoVA) on MoE layers; sigmoid routing by default. |
| 161 | `beam` | GEN | KV + SWA | - | Experts. Sandwich norms, biased per-head attention gate, NoPE full layers with power temperature, depth-scaled residual adds. |
| 162 | `kolibri1` | GEN | KV + SWA | - | Experts. Sandwich norms; biased routing with one shared expert; RoPE on SWA layers. |
| 163 | `(unknown)` | NA | none | - | Rejected: unknown architecture: general.architecture does not name a llama.cpp architecture. |

## KV cache types

`KvDType` (`gguf/memory/KvDType.java`) holds the nine cache types of llama.cpp's `-ctk` / `-ctv`. They are selected with `--kv-type`, `--kv-type-k` and `--kv-type-v`, or with `neuralfree.kv.type`, `neuralfree.kv.typeK` and `neuralfree.kv.typeV`. The default is `f16`.

| Type | Storage | Block |
|---|---|---|
| `f32` | 4 bytes per value | - |
| `f16` | 2 bytes per value | - |
| `bf16` | 2 bytes per value | - |
| `q8_0` | 34 bytes per 32 values | 32 |
| `q4_0` | 18 bytes per 32 values | 32 |
| `q4_1` | 20 bytes per 32 values | 32 |
| `iq4_nl` | 18 bytes per 32 values | 32 |
| `q5_0` | 22 bytes per 32 values | 32 |
| `q5_1` | 24 bytes per 32 values | 32 |

Rules the memory planner (`MemoryPlanner`) enforces:
- **Whole blocks.** A block type needs every head to be whole 32-value blocks. For an MLA latent row, the aliased `kv_lora` V prefix must be whole blocks too. Otherwise the configuration is refused, as llama.cpp's flash-attention check refuses it.
- **Mixed K and V types.** K and V types may differ. MLA models and DeepSeek-V4 store K and V in one row, so they need equal types: "model does not support different K (...) and V (...) cache types".
- **Hadamard rotation.** Quantized K/V rows are Hadamard-rotated before they are stored (`RotationPlan`, llama.cpp `llama-kv-cache.cpp`). `--no-kv-rotation` or `neuralfree.kv.rotation=false` turns rotation off.
  - Only block types are rotated by `neuralfree.kv.rotation`, and only when every cached layer shares one head width that is a multiple of 64. F32, F16 and BF16 rows are not rotated, except by the forced rotation below.
  - MLA latent caches and DSA attention caches are never rotated.
  - deepseek32, deepseek4, glm-dsa and dots3note force the rotation, whatever the cache type and `neuralfree.kv.rotation`, when the K head width equals the indexer head size. It applies to the q/k of their DSA indexer caches, and to the K rows of a main cache that is not an MLA latent or DSA cache.
- **Indexer caches.** Indexer caches use the K type.
- **Draft contexts.** Draft contexts use F16 rows (`MemoryConfig.DRAFT_KV_TYPE`), as llama.cpp's speculative defaults do. This covers separate draft models, NextN/MTP heads, eagle3 and dflash.

## Cache features

- **Sliding-window rings.** Causal STANDARD and CHUNKED sliding-window layers get a ring of `min(n_ctx, roundUp(n_swa + ubatch, 256))` rows (`KvLayout.ringRows`). This applies to the memory kinds llama.cpp builds with an ISWA cache: KV, HYBRID, DSA_ISWA and the DeepSeek-V4 raw cache. DSA, HYBRID_IDX, MSA and ENC_DEC keep full-size caches and mask the window. `--swa-full` or `neuralfree.swa.full=true` gives full-size caches.
- **Shared KV.** Layers that reuse another layer's cache read that layer's rows and own nothing (gemma3n, gemma4). The gemma4-assistant drafter reads the target's caches.
- **Prefix reuse.** Each generation keeps the longest common prefix of the stored tokens and the prompt, up to the last prompt token, and evaluates only the rest. The rest must be droppable. Full caches can always drop it. Rings can drop it until they wrap, and recurrent state only when nothing is dropped. Otherwise the memory is reset; encoder-decoder models always reset. This is on by default; turn it off with `--no-prefix-reuse` or `neuralfree.prefix.reuse=false`.
- **Context shift.** Off by default; turn it on with `--context-shift` or `neuralfree.context.shift=true`. When the context is full, the loop keeps the leading BOS, discards half of the remaining positions and re-rotates the kept keys (llama.cpp `seq_rm` + `seq_add`). Without shift, generation stops with `CONTEXT_FULL`. The shift is refused, and nothing moves, for:
  - recurrent state;
  - sparse-attention side caches (DSA indexer, MSA index keys, DeepSeek-V4 compressed rows);
  - layers that read a target model's cache;
  - several positions per token (M-RoPE, `n_pos_per_embd > 1`, sensenova_u1);
  - step35 (per-layer RoPE dims);
  - a ring that already overwrote kept positions the next query attends.
- **Rollback.** Dropping the last positions, as speculative verification does, is refused for:
  - recurrent state, which rolls back through snapshots instead;
  - DeepSeek-V4 compressor state beyond its rollback snapshots (`CompressorRollback`): more than `n_rs_seq` positions, a second rollback before the next step, or more positions than the last step stored;
  - a ring that no longer holds the window of the next query.
- **DeepSeek-V4 rollback snapshots.** Each compressor (CSA, HCA, indexer) keeps `1 + n_rs_seq` F32 planes of kv and score, as llama.cpp's `llama_dsv4_comp_state` does.
  - Each step writes plane d, the state before the step's last d tokens (`LLM_DSV4_StateSnapshot`). A rollback of d positions copies plane d back into the live plane and drops only the raw KV rows.
  - Each forward call is split into ubatch steps so that its trailing `1 + n_rs_seq` tokens stay in one step (`MemoryPlan.stepTokens`, llama.cpp `split_equal` with `n_keep_tail`), on CPU and GPU alike.
  - `n_rs_seq` (`MemoryConfig.rollbackSnapshots`) is the draft max for `--mtp`, an MTP draft file, eagle3, gemma4-assistant and dflash drafters, and 0 for a separate draft model or no drafter (llama.cpp `need_n_rs_seq`). It is capped at the ubatch and counted in the context fit and VRAM budget.
- **Recurrent-state snapshots.** Memory with recurrent state can reserve a shadow copy of every state (`neuralfree.state.snapshots`). The default is on when speculative decoding is configured, as llama.cpp sizes its recurrent rollback rows from the speculative settings. Snapshots are refused for DeepSeek-V4 compressed caches.
- **MLA cache form.** `--mla-cache auto|latent|decompressed` (`neuralfree.mla.cache`); `auto` selects latent for MLA architectures. On the GPU, a decompressed cache over absorbed `attn_k_b`/`attn_v_b` weights has no kernel and is refused: "use neuralfree.mla.cache=latent (or auto)".

## Speculative decoding

`LLMModelFactory.openSpeculative` and `LLMModelFactory.speculate` bind exactly one drafter to a target. `SpeculativeDecoder` verifies the drafts with the target's sampler. The drafter is chosen from the draft file, as llama.cpp chooses the speculative type of a draft GGUF, unless `--spec-type` (`neuralfree.draft.type`) forces one:

| Mode | How to enable | Drafter |
|---|---|---|
| Draft model | `--draft <gguf>` with any GEN model | `ModelDrafter`: a separate model on the target's compute backend (on the CPU for a CPU target). Needs memory on both models and a compatible vocabulary. |
| MTP, in-file | `--mtp` | `NextnDraftModel` over the target's own NextN blocks. A GPU target opened without MTP places them on the device when the drafter attaches. |
| MTP draft file | `--draft <gguf>` whose last block holds `nextn.eh_proj` and that is an MTP draft of the target's model | `NextnDraftModel` over the file's NextN blocks, embedding and head. The file may be a full model or the MTP blocks alone (`convert_hf_to_gguf.py --mtp`). Its trunk is never run, uploaded or reserved. |
| eagle3 | `--draft <eagle3 gguf>` | `Eagle3DraftModel`: three target-layer taps fused by `fc`, one decoder layer with its own cache, optional `d2t` vocabulary map. |
| gemma4-assistant | `--draft <assistant gguf>` | `Gemma4AssistantDraftModel`: Q-only attention over the target's last two layers' KV caches. The target's `n_embd` and vocabulary must match the assistant's backbone width and head. |
| dflash | `--draft <dflash gguf>` | `DflashDrafter`: block-diffusion drafting (DFlash, DFlash2, DSpark, LiLiCorr). The draft length is clamped to its block. |

Draft parameters are `--draft-max` (default 3), `--draft-min` (default 0) and `--draft-p-min` (default 0), or the `neuralfree.draft.*` properties.
- **Chained MTP heads.** hy_v3, mimo2 and step35 with several NextN blocks chain them, at most one token per head.
- **Single-block MTP graphs.** Every other MTP graph runs a single block, as llama.cpp asserts `n_layer_nextn == 1`.
- **Refused targets.** Speculation is refused for EMBED, CLS, NA, DIFF, ENC_DEC and DRAFT targets, and for models without memory.
- **One drafter at a time.** A loaded model's forward backend (`ForwardBackend`) has one hidden-state tap (`HiddenTapSlot`), so a target serves one drafter at a time.
- **DeepSeek-V4 targets.** A round drafts at most the target's compressor rollback snapshots (`n_rs_seq`). A separate draft model gives the target no snapshots, so on DeepSeek-V4 it does not speculate; llama.cpp falls back to state checkpoints there, which NeuralFree does not implement.

**Draft type** (`--spec-type`, llama.cpp's speculative type):
- **`auto`** (the default) detects the drafter from the file:
  - dflash, eagle3 and gemma4-assistant files draft as themselves;
  - a file with NextN blocks drafts with them when it is an MTP draft of the target's model, meaning the same architecture, trunk depth and layer schedule, `n_embd` and vocabulary;
  - a full model with NextN blocks that is not such a draft runs as a separate draft model, and the choice is logged;
  - a file that holds only MTP blocks of another model is refused.
- **`draft-mtp`** (alias `mtp`) runs the file's NextN blocks and refuses any file that is not an MTP draft of the target's model. Without `--draft`, it uses the target's own blocks, like `--mtp`.
- **`draft-simple`** (aliases `simple`, `model`) runs the file as a separate draft model. It refuses drafter architectures and files that hold only MTP blocks.
- `draft-eagle3` and `draft-dflash` are refused as values; `auto` detects those drafters from the file's architecture.

llama.cpp checks only the MTP input width of a draft file (see [GAPS.md](GAPS.md)).

## LoRA adapters

`--lora FNAME[:SCALE][,...]` and `--lora-scaled FNAME:SCALE[,...]` (or `--lora-scaled FNAME SCALE`), or `neuralfree.lora`, attach llama.cpp LoRA adapter GGUFs (`general.type = adapter`, `adapter.type = lora`). They attach on both the CPU and the GPU path.
- **Applying deltas.** Every `lora_a` / `lora_b` pair becomes a delta `s * B(A x)` on its base tensor, as `build_lora_mm` computes it. Here `s = scale * alpha / rank`, or `scale` when the adapter has no alpha. This covers projections, expert tensors and token embeddings. Several adapters on one tensor add up in load order.
- **Load checks.** The adapter's architecture must equal the base model's, and the shapes are checked against the base tensors.
- **Unservable tensors.** The whole attachment fails, naming the adapter and tensor, when:
  - no bound block applies deltas to a pair's base tensor;
  - a block refuses deltas (a PRISM Hadamard-rotated input, the absorbed MLA `attn_kv_b`).
- **Activated LoRA.** Activated LoRA (`adapter.alora.invocation_tokens`) is refused.
- **Switched LoRA.** The graniteswitch architecture's switched LoRA is a model feature, routed by its own router.

## Encoder-decoder

`t5` (ENC_DEC) runs its encoder once over the source. The decoder then generates with self-attention caches plus cross-attention K/V of the encoder rows. T5 relative-position buckets are supported. The FFN is GELU-gated when the file has `ffn_gate` and ReLU otherwise. `t5encoder` (EMBED) runs the encoder alone for embeddings. With `--context-shift`, only the decoder self-attention cache shifts. The cross-attention caches hold the encoder output and stay.

## Masked diffusion

`dream`, `llada`, `llada-moe` and `rnd1` (DIFF) generate with `DiffusionGenerator` (llama.cpp `diffusion-cli`).

| Setting | Default | Notes |
|---|---|---|
| Steps | 128 | |
| Algorithm | confidence | origin, entropy, margin, random or confidence |
| Schedule | timestep (epsilon 0.001) | Dream; or the block schedule (`--block-length`), LLaDA |
| Algorithm temperature | | |
| Classifier-free guidance scale | 0 (off) | |
| Gumbel noise | | optional |

A diffusion model has no cache and no autoregressive loop, so speculation and context shift do not apply.

## Audio

| Architecture | Output |
|---|---|
| `wavtokenizer-dec` | Decodes WavTokenizer codes (`--audio-codes`, or parsed from the prompt) to spectral features. Waveform synthesis by iSTFT is not part of the engine. |
| `pockettts` | Returns its backbone hidden states, the input of the PocketTTS flow network (`PocketTtsSession`). |
| `qwen3tts` | Generates codec tokens from the raw prompt. |

`--audio-out <file>` writes features or hidden rows as float32 little-endian.

## Embeddings, classification and reranking

EMBED models return pooled (`NONE`, `MEAN`, `CLS`, `LAST`) or per-token embeddings, with BERT token types (`--token-types`). A model with `RANK` pooling scores a query against a document: `--mode rerank --document <text>`, or the model's `tokenizer.chat_template.rerank` prompt. Decision models (clef, vllm-decision2) classify with a per-token decision order (`--decision-order`).

## Out of scope

Vision and audio projectors (`mmproj`, the `clip` architecture) are rejected. The text part of a vision-language model (qwen2vl, qwen3vl, cogvlm, paddleocr, hunyuan_vl and others) runs on text input only. See [GAPS.md](GAPS.md).
