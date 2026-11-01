# CONTINUE.md — handoff for NeuralFree doc-sync + parity work

Hand this file to a fresh agent/harness session. It contains everything needed to resume.
Repo: `/home/aeslampanah/Desktop/Projects/NeuralFree` (Java, Gradle, branch `main`).
Reference sources for parity work: `/home/aeslampanah/Desktop/Projects/{vllm,llama.cpp,sglang}`.
Full reusable plan lives at `~/.commandcode/plans/neuralfree-doc-sync-and-parity.md`.

## Mission

1. Keep all documentation (`README.md`, `USAGE.md`, `LLM-INFERENCE.md`, `TRAINING.md`, `GAPS.md`,
   `design/parity-work-order.md`) exactly in sync with the code.
2. Close capability gaps vs vLLM / llama.cpp / SGLang, following the priority order in
   `design/parity-work-order.md`. Every new implementation must reuse existing house patterns
   (`Cpu*`/`Gpu*` twins, refusal-throwing `ForwardBackend` contracts, `PROP_*` constants,
   arch tables, kernel inventory, plugin SPI).

## Status: DONE (committed on main)

Recent batch (2026-10-29, commits `7d438e0`..`dc15cd3`) — the committed HF/speculative
breakage the old "GPU flake envelope" was hiding, plus the parity fixture repairs:

- **The parity fixtures now match the converter contract** (`dc15cd3`):
  `HfGgufParityTest` 16 → 4 failures. `--outtype auto` defaults to MOSTLY_F16
  (base.py:181-196) and converts rank-2 matrices — the F32 goldens now say file_type 1
  with F16 matrices; the fixture vocab's chkhsh resolves to `mpt` (a miss raises in
  llama.cpp, base.py:2132-2140) and stores the byte-decoded piece; `LlamaModel.permute`
  falls back to `n_head_kv` and swaps the halves of each head (llama.py:176-179);
  phi3 carries a fused `mlp.gate_up_proj` → FFN_UP (PHI3 has no FFN_GATE); gemma3's
  post-block norm is `post_attention_norm`; qwen3moe writes `expert_used_count`
  (base.py:1694) and gemma3 writes no `vocab_size` for a gpt2 vocab; the
  tensor-info duplicate check asserted the wrong direction (assertNotNull on the first
  insertion — unreachable until the KV fixes).

- **The HF registry was empty at HEAD** — `HfArchRegistry.mergeProviders` recorded the
  built-in rows in its collision map but never merged them into the result, so every
  HF resolve failed since the plugin SPI commit (`6996f8e`): `710b45d`.- **`HfNameTables` failed class init on legal data** — llama.cpp's `MODEL_TENSORS`
  lists `V_PROJ_NORM`/`V_QF_PROJ_*` without `TENSOR_NAMES` entries (and lists
  `V_QF_PROJ_NORM` twice); the loader treated both as defects: `710b45d`.
- **`NF_HF_DROPPED_PREFIXES` vanished when empty** — `set()` mirrors gguf-writer's
  empty-array skip; the bookkeeping key writes through `setTyped`: `710b45d`.
- **DIRECT-mode regions were broken twice** — `readAligned` handed back the aligned
  window's capacity (a 2048-byte read at offset 88 returned 4008 bytes) and read past
  EOF on any file whose last tensor's sector window crossed the end: `c922933`.
- **`SpeculativeDecoder`'s verify round never advanced `accepted`** — every second
  confirmation threw `confirm(1) after 0 confirmed draft tokens`; the draft length no
  longer shrinks with the token budget (llama.cpp drafts `n_draft` regardless, the
  stop-in-verify replay handles overshoot); `build()` now throws IAE for a null mode
  and refuses synthetic settings outside synthetic mode: `f23ec0f`.
- `RejectionVerifierTest` rewritten to the landed contracts (whole-batch verify
  scripts, an overflow-aware enumeration walk, the block mixture law) and two
  `SpeculativeDecoderTest` MATCH traces updated to the budget contract: `f23ec0f`.
- `SuffixDrafter`'s first draft of a generation read `token(-1)` off its -1 sentinel:
  `f23ec0f`.
- Earlier in the same session: the dist draw moved to the chain's `Mt19937`
  (`nextCanonicalDouble`'s signed-word fold produced negative uniforms), the DRY
  breakers keep their whitespace, `ForwardBatch` sets its set-once flags after the
  validation, `RegexToGbnf`'s whole-string wrap keeps its closing quote, the qkv
  fixture names its rope: `7d438e0`, `a7dfc0c`, `eebedec`.

- P1 (B1) — paged multi-sequence CPU batch engine: `98cfb90`.
- P2 core (B2) — transport-neutral batch engine (`gguf/engine/batch`): `18bc40c`.
- P2 grammar — `GrammarStage` (llama.cpp grammar sampling) + the batch `ConstraintSpec` route: `5c6fb42`,
  NUL-piece mask fix `6e6024c`.
- P2 server — OpenAI-compatible HTTP (`/v1/models`, `/v1/completions`, `/v1/chat/completions` + SSE,
  `/v1/embeddings`, `/v1/rerank`, llama.cpp `/health` + `/props`) on `EngineService`'s single pump thread: `3d6831f`,
  embeddings/rerank + lazy engine `9e57212`. Docs `efad67f`.
- P3 (B3) — prefix cache (`PrefixCache` adopt-in-place over retired paged sequences, LRU eviction,
  `neuralfree.prefix.reuse` gate, SWA refusal) + the `PagedSequences.remove` cursor fix it exposed: `b109d83`.
- P4 — logprobs (`TokenScore`, `SequenceRun` post-chain capture, `GenerationResult.logprobs()`,
  host-logits guard, server `logprobs`/`top_logprobs`/`n_probs`): `b544a91`.
- P5 — `min_tokens`/`bad_words` sampler stages (`9dc8718`), structural-tag parser (`74f0b78`),
  the P5 session after `efad67f`, and the straggler session after `4b14deb`:
  - imatrix collector (`cpu/quant/ImatrixCollector` over `CpuMatVec.observeGlobally`) + `nftool imatrix`
    (`ImatrixCommand`); `ImatrixFile.writeGguf` now overwrites as its javadoc and llama.cpp promise: `aa27b99`.
  - `CvectorGenerator.generate` — the prefill loop over one loaded CPU engine (llama.cpp get_hidden_layers:
    BOS-policy tokenization, space-token padding, beginPair/positive/setSide/negative/endPair, tracked layers
    0..nLayer-2) + the `nftool cvector` route: `78961e2`.
  - Streamed logprobs — the `ReasoningFilter.Sink` capture overload, per-token pieces in `SequenceRun`, the
    OpenAI `logprobs.content` shape per SSE chunk: `18f539a`.
  - Session files — `HeapModelMemory.saveSession`/`loadSession` (`memory/cpu/SessionFile`, llama.cpp
    llama_state_* scope: tokens, position mirrors, owning layers' K/V rows, live recurrent state;
    plan-identity-checked, reset-then-restore): `be6513b`.
  - `logit_bias` — additive biases in `BiasStage` at the chain's first stage (llama.cpp
    `llama_sampler_init_logit_bias` position, the -100 ban threshold), `Sampler.create` map overloads,
    `GenerationRequest.logitBias`, the server mapping with [-100, 100] validation; the refusal left: `90eb014`.
  - The Anthropic Messages API — `POST /v1/messages` (+ `/v1/messages/count_tokens`, `AnthropicResource`): the
    body shape, stop reasons, usage counts, the full event sequence on streams: `05cd921`.
  - Speculative logprobs — `SpeculativeDecoder.build()` rejected EVERY configuration (`(rates == null) ==
    (length == null)` throws for the plain MATCH case); fixed, and MATCH + plain decode steps capture the
    per-token distributions into the result and the sink: `f704642` region (`SpeculativeDecoder`).
  - The adopt-cut page walk — `PagedSequences.removeGroup` ran ascending with back removals, leaving freed
    page ids in the list that the pool re-handed out as live pages of the same sequence; the next eviction
    unref'd a live page and every later submission failed. The walk runs from the back; an emptied list zeros
    the slot counter; the regression pins six adopt-cut rounds over a wide context: `c208338`.
  - The launcher usage names all four implemented subcommands (`fe41829`).
  - Committed-broken suites repaired: `GgufEncoderTest` (`c4443eb`), `CvectorGeneratorTest` ×7 (`78961e2`),
    `ExtVecDotTest` (`7decab5`), `SequenceRunTest` ×3 (`a94827c`), `SpeculativeDecoderTest` ×8 — the builder
    validation gated every decoder build (`f70f642`) — plus its two stale trace scripts, and the order tests'
    `PreSplitChain` twin (`da29ee1`).
  - Real implementation bugs the suites exposed: `Sampler.Candidates.siftDown` dropped the replaced root
    (top-k kept the wrong set; `f4a4821`), `ThinkMachine`'s prelude flipped to literal content on the first
    non-tag text (`ac9d50a`), `GgufTensorSink`'s `ATOMIC_MOVE` silently replaced existing outputs on
    rename(2) platforms (`e98d42c`), the adopt-cut walk above, and the SpeculativeDecoder builder gate.
- Docs synced to the landed code (nftool table, control vectors, imatrix, session files, streamed logprobs,
  structural tags, the logit_bias mapping, the Anthropic Messages section, GAPS entries closed with the
  remaining named): `4b14deb` and `ef2a8ad`.

## Full-suite baseline (corrected 2026-10-29 — the old GPU-flake envelope was a misclassification)

The earlier claim ("581 failures inside a 606–746 GPU-flake envelope, non-GPU delta
empty") was wrong: diffing failure SETS against earlier baselines only proved the sets
were stable, not that they were GPU flakes. Ground truth after the fixes below:

- A filtered run over the GPU-infra suites (`/tmp/opencode/skip-gpu.init.gradle`
  excludes `tensor/gpu/**`, `gguf/gpu/**`, `acceleration/backend/cuda/**`,
  `GpuAttentionMixerTest`, `GpuMsaMixerTest`, `RecurrentVisionParityTest$GpuParity`,
  heap 2g) completes in ~15 min and yields the real picture.
- ~264 deterministic non-GPU failures remain across ~98 classes (see the list below);
  ~90 of those are genuine committed breakage (HF parity/hooks/torch, grammar, memory,
  json, sampling order, lora, quant), the rest are GPU-parameterized rows (host rows
  included, so not all are environmental) plus known GPU-by-assumption flakes
  (`autograd/GradientCheckUtilTest`).
- GPU-infra failures on this box: `cuDevicePrimaryCtxRetain CUDA_ERROR_OUT_OF_MEMORY`
  and `OpenCL -34 clCreateCommandQueue` — environmental, ignore them; judge classes
  individually, not by count.
- Full unfiltered runs can HANG (a GC/reference storm in the test worker, 33+ min
  without progress). If a run stalls, jstack the worker; prefer the filtered run.
- Snapshots: `/tmp/opencode/nf-nogpu-results2` (after this session's fixes).

## Known deterministic failures still open (the remaining committed breakage)

Biggest clusters, each needing its own focused session (failure messages recorded in
`/tmp/opencode/nf-nogpu-results2`):

- Big clusters, each needing its own focused session (failure messages recorded in
  `/tmp/opencode/nf-nogpu-results3`, the post-parity-fix snapshot):

- `HfRegistryCoverageTest` ×2 + `HfDynamicArchCoverageTest` ×1 — the coverage tests iterate
  every table row with a shared stub `config.json`; the granite-hybrid family's resolver
  needs `layer_types`/`attn_layer_indices`/`attn_layer_period`, deepseek2's OCR row needs
  `model_type: deepseek_v2`, and the families branch on contradictory keys — the registry
  test needs the per-row minimal-config catalog that `HfDynamicArchCoverageTest`
  (`minimalConfigs`) already carries, merged in. Attempted and reverted this session: a
  shared stub cannot satisfy all families at once.
- `HfQuantParityTest` — FIXED this session (0 failures): qzeros pack 8 nibbles per word,
  AWQ scales cover each output channel in vllm's reverse AWQ slot order, gpt-oss blocks
  need 128 codes for [1,2,2,16], fp_quant scales are per-group [n, k/32], the bitsandbytes
  quant state is plain JSON, HQQ packs one byte per slot pair, and the rewrite serves codes
  plus the output multiplier (CT global scale as its reciprocal; fp8's fold is
  F8_E4M3_B128) — verified against vllm's awq_triton and nvfp4_emulation references.
- `HfGgufParityTest` ×4 — the bert fixture needs the standard bert WPM special ids
  ([PAD]=0, [UNK]=100, [CLS]=101, [SEP]=102, [MASK]=103, vocab ≥ 104) that llama.cpp's WPM
  defaults assume (SpecialTokenRules.applyTypeDefaults); the mamba2 fixture's `ssm_a`
  reaches the view as `[4]` where the converter unsqueezes to `[1, 4, 1]`
  (mamba.py:188-196 `reshape((*shape, 1))`) and `ssm_conv1d` skips its squeeze — the
  `mamba2Tensor` hook implements both, so the wiring/matching is what needs investigation.
- The HF hooks families — FIXED and green (298 tests, 0 failures as of `a21ee75`): all
  20 classes. The recurring shapes, all fixed and recorded: hook tests driving
  post-converter names where the hook expects the checkpoint name (dbrx/llama4/grok);
  expert-collector index pre-checks vs leftover-at-extra semantics (arctic, bailing, afmoe);
  per-variant GGUF naming and template quirks (PLaMo2's suffix-less post norms, mamba2's
  suffix-less SSM_A/SSM_D, ssm_dt.bias routing, gemma3n's `_scale` tail rule); fixture
  configs missing keys the resolvers read (`mamba_n_groups`, `num_local_experts`,
  D1Omni's `intermediate_size` which lfm2.py:178 reads directly); float-vs-delta assertion
  overloads. Production bugs found and fixed: the SSM hooks passed runtime gguf fragments
  into matches() (MODEL_TENSOR ids required) so no mamba squeeze/unsqueeze/negExp ever ran;
  DenseHooksA's normal CDF used the upper erfc tail (diverging every icdf to infinity);
  HfTensorHook.DEFAULT lacked the base filter's unconditional language_model. strip
  (base.py:591-592) so SenseNova's prefixed tensors failed to map; the sensenova q/k concat
  left its consumed spatial half unrecorded; the ASR bos/eos ids resolved through the vocab
  instead of the added_tokens_decoder map key (qwen3vl.py:393); the granite rope pattern was
  width-sensitive on bool(0) (granite.py:116-119); the LFM2 MoE key/value patch inverted the
  reference (`value if layer_type == "full_attention" else 0`, lfm2.py:290-293) —
  `perLayerKeepKv` now mirrors it; the d1OmniFixup re-application (the harness context
  builder re-runs the entry fixup) needed an idempotence guard (`intermediate_size`
  already a list → return, lfm2.py:174-183 loads once); the nemotron-h/puzzle tensor path
  recomputed its layer lists through the GRANITE reader (`layer_types`) while the reference
  resolves them through the Nemotron pattern (`hybrid_override_pattern`/`layers_block_type`,
  nemotron.py:481-490) — `graniteHybridTensor`/`nemotronHTensor` now take the caller's
  lists, with the folded MTP block registered as an attention layer (nemotron.py:472-474)
  and Puzzle forced `moe=true` (nemotron.py:741); the gpt-oss mxfp4 split called
  `interleaveRows` (literal dim-0 = experts) instead of the flat-row de-interleave the
  contract names — new `RowTransforms.interleaveFlatRows` maps the [E·2F, K] rows.
  The mxfp4 packed-vs-F32 accounting question is RESOLVED on the harness side: a stored
  source of a declared block format reports packed rows (17 bytes per 32-wide MXFP4 row,
  what a file-backed source reports) and the mxfp4 test asserts byte-level de-interleave.
- The HF sources families — FIXED and green (all 14 `Hf*SourcesTest` classes, commits
  `a21ee75`..`88e3df9` and the two follow-ups): every fixture now matches the converter
  contract (verified per case against the reference sources). Production fixes found on
  the way: the gemma3n meta writes the raw `sliding_window` (gemma.py:239-251); the
  OpenELM entry carries the `n_layers` fixup and unwrapped head arrays (openelm.py:68-72);
  the grok expert buffer scopes the layer check to the expert tensors (grok.py:77-79);
  the granite MoE SWA keeps the fused gate/up stacks whole (granite.py:150-172) while the
  GraniteMoeModel contract splits (granite.py:201-227); the qwen3vl-moe Omni entry takes
  the transposed-experts tensor path (qwen3vl.py:355); the nemotron meta writes the plain
  `layer_norm_eps` (nemotron.py:379-380); the LFM2 loader keeps the metadata head dims on
  a mamba-first layer 0 (HparamsLoader.readHeadDims); the plamo2 vocab gets an empty
  normal-token set. Fixture lessons: `SyntheticHf.binaryFile` writes binary tokenizer
  files (the sentencepiece `tokenizer.model` of arctic/dots1/ernie/internlm3/plamo2);
  the LFM2 per-expert names go through the collector (the merged w1/w2/w3 stacks are
  emitted, never written); the arch_tensors row (not the fixture prefix) decides
  mappability — several granite/nemotron tensors only exist in the `model.layers.` form.
- `TorchCheckpointTest` ×9 + `RestrictedUnpicklerTest` ×2 — torch checkpoint
  zip/pickle acceptance cases.
- `NgramDrafterTest` ×6 — the SuffixTree/SuffixDrafter contract: the drafter tests
  expect a draft from a shorter productive context when the longest match dead-ends
  (`suffixDrafterDraftsFromTheLocalTree` expects `[5,1,2,3]`), while the SuffixTree
  unit tests pin dead-end-means-empty (`suffixTreeMatchesAndDraftsTheBestChain`);
  plus `NgramCacheFile` record validation (no UncheckedIOException on truncated
  records) and `NgramCorpus` eviction counts. The `SuffixDrafter` first-call crash is
  FIXED; the algorithm contract needs re-derivation against SGLang
  (`vllm/v1/spec_decode` ngrampropose + SGLang suffix cache) before touching more.
- `GgufCompatConverterTest` ×6 (byte counts 0), `ModelSourcesTest` ×4,
  `MetadataOverridesTest` ×5 (override parsing), `LoraRegistryTest` ×5,
  `SamplerOrderTest` ×4, `GbnfParserTest` ×3 + `JsonSchemaToGbnfTest` ×3,
  `JsonStreamWriterTest` ×3 + `JsonLimitsTest` ×1, `ModelSerializationRoundTripTest` ×3,
  `TransformerTextGenerationIntegrationTest` ×3, `SentencePieceModelTest` ×2,
  `RequantPolicyTest` ×3, `F8E5M2Test` ×3, `Mxfp6Test` ×2, `FlashAttnTilingPagedTest` ×2,
  `MemoryPlannerPagedTest` ×2, `TiledGemmPlanTest` ×2, `HfRegistryCoverageTest` ×2,
  `SidecarDirectoryTest` ×1 (`NoSuchFileException t0.bin`), and the singles across
  `LayerSecondOrderTest` (host rows), `VisionGradCheckTest` (host rows fail:
  `GradientCheckResult{passed=false}`), `EndToEndGradientTest`,
  `TensorOpsGradCheckTest`, `CpuRopeTest`, `CpuMatVecTest`, `MoeRoutingTest`,
  `SwitchedLoraDeltaTest`, `CpuDsv4Test`, `SequenceStateReuseTest`,
  `SingleSequenceMemoryTest`, `LRSchedulerTest`, `HadamardTransformsTest`,
  `Fp4SchemeTest`, `PeftAdapterSourceTest`, `GgufKeyTableCoverageTest`,
  `HfDynamicArchCoverageTest`, `HfStRowSourceTest`, `HparamsLoaderTest`,
  `JinjaFiltersTest`, `LlmKernelInventoryTest`, `RopeOverridesTest`,
  `InputTransformLoaderTest`, `BlockDecodersTest`, `ExtBlockDecodersTest`,
  `AlgorithmsTest`, `ReplayBufferTest`, `GpuTensorOpsParityTest` (NaN-vs-NaN compare),
  `LLMDemoSpecTypeTest`, `DownloadCommandTest`, `ArchRegistryTest`,
  `GGUFFileParserHeaderTest`, `GGUFHeaderReaderExtensionTypesTest`,
  `GgufWriterTest` ×2, `GgufWriterSplitTest` ×2, `ExtTypeFastPathTest` ×1,
  `GradientClippingTest$Device` ×8 / `LayerSecondOrderTest` ×12 API rows (GPU-open
  failures), `SamplingDefaultsTest`, `SamplerConfigModelDefaultsTest`,
  `HfDenseClassicHooksETest` ×2, `HfMoeClassicHooksDTest` ×1, `FlashAttnTilingPagedTest`.
  Repaired this session (all green): `RejectionVerifierTest` ×20, the two rewritten
  `SpeculativeDecoderTest` MATCH traces, `FileTensorRegionsTest` ×4, `HfTensorRegionsTest`,
  `HfModelSourceTest` ×18, `DraftSourceRoutingTest` ×2, `HfNameTablesTest`,
  `HfTensorNameMapTest`, `TensorNamesAliasConsistencyTest`, `PluginsTest`.

## Remaining roadmap (P6+, from design/parity-work-order.md)

- **P5 stragglers closed this pair of sessions**: `logit_bias` (through `BiasStage`), speculative
  MATCH/decode-step logprobs, the Anthropic `/v1/messages` mapper, the paged adopt-cut crash.
- **P5/P6 stragglers still open** (each needs its own focused session): the GPU `EventTap` driver for
  cvector collection (needs reliable GPU runs); trigger-gated structural tags and the modern xgrammar
  `format` object; session-file coverage for the paged page lists, DSV4 compressed caches and the sparse
  side caches; tool-call output parsing (the `ToolParserProvider` consumer, per-model call formats);
  the lossless verify modes' per-token scores; the `privatize` shared-source reference that the trim's
  own copy leaves held (a benign leak: the shared original keeps one reference until its other holder
  drops it — measured by `swaTrimOfASharedPageEmitsOneCopy`).
- **~176 deterministic test failures** (7,397 tests; closed this session:
  `HfQuantParityTest` 8 → 0 (`35d8277`), the MoE-classic-A/B + Mistral3 hooks 14 → 0
  (`120ac64`, `96ca289`, `77719fb`), `HfSsmHooksTest` 8 → 0, `HfDenseHooksATest` 6 → 0,
  `HfMoeClassicHooksBTest` 6 → 0 (`0a340f1`, `77719fb`), `HfMoeSwaHooksATest` 5 → 0 and
  `HfDenseHooksBTest` 5 → 0 (`a62a3fa`)) — down from 581 at session start;
  ~90 of the remainder are GPU-infra rows and the rest are the clusters listed above. The HF
  parity cluster's method is proven: read the failing message, verify the expected value
  against the llama.cpp/vLLM/SGLang reference, fix production or rewrite the fixture to the
  reference contract. Beware: the Gradle daemon's file-state cache can serve stale compiled
  classes after tool-driven edits (`compileTestJava UP-TO-DATE` while the class lacks the
  new code) — `--stop` the daemon or diff the class file before trusting a run.
- **P6+** (multi-day epics): GPU paged path / device page pools / paged kernel dispatch (MS-3),
  FP8 KV cache, int8/FP8 GEMM, AWQ/GPTQ deconvs, tool-call/reasoning parser consumers,
  metrics/benchmarks, multimodal epic.

## Load-bearing invariants (do not break)

- Paged step lifecycle order: `prepare(layout)` → host copies → `beginPagedStep` → trunk →
  `endPagedStep` → `commit(step)`; commit AFTER endPagedStep.
- `HeapModelMemory.beginPagedStep` must drain `pendingReports` (SWA trim/shift/copy
  `RowReport`s) into the `PagedSequences.kvPos` / `GroupPositions` mirrors, or attention
  reads stale positions.
- `pageShift` is log₂(pageSize): pageSize 16 → shift 4. Never pass raw page size.
- `PagedStep.pageTable(g)` is the full `[seq * maxPages + k]` table; mixers must pass the
  per-sequence slice.
- `CpuLayerExecutor.forward` routes contiguous steps through the original `run()` logic
  (`n == 1 && hasMemory() ? decode : prefill`); routing contiguous through `batch()` drops
  `hasMemory()` and breaks a test contract.
- Fresh sequences must start at position 0 — the engine throws otherwise. Context-overflow
  tests must fill nCtx (e.g. push 64 tokens into nCtx 64), not start a fresh seq at pos 63.
- `UBatch` rejects duplicate output rows — in multi-seq batches mark only the final row of
  each sequence as output.
- `PagedSequences.remove` keeps the append cursor on the first dead slot of a tail cut (the
  cache adopts-and-cuts; a stale cursor writes K/V where attention cannot read it).
- The prefix cache adopts the CACHED sequence's own slot (never `copy` into the same seq id —
  `copy` refuses `src == dst`); a cached token array must equal the STORED tokens (the final
  armed-but-unfed token is not in the pages).
- `ForwardBackend.decode(token, pos)` stores the row AND appends the token to the sequence
  record; a test that decodes twice at the same position overwrites the row but appends once.
- NEVER revert the user's pre-existing P1 groundwork (`HeapModelMemory`/`PagedSequences`
  wiring, `CpuAttentionCore.attendRows`, `GroupPositions.set()`).

## House style / gotchas

- Tests open `GGUFModel` directly on a tiny synthetic llama GGUF (`SyntheticModels.llama()`,
  house style), not wrapper harnesses. Core tests are JUnit 4; server tests are Jupiter
  (message-last assertions).
- Docs are written in a strict factual register: every claim names its class/property, gaps
  are stated as refusals with the throwing class in parentheses.
- Core (`src/main`) is Java 11; `server/` is Java 21 and may use Quarkus types — the core must not.
- An unrelated TradingVisor live-trial process runs on this machine — leave it alone.
- Build/test: `./gradlew test` (~20 min full suite). Console: `--console=plain`. Scope
  `--tests` filters to `:test` (they otherwise hit `:server:test` too, which then fails on
  "no tests found").
