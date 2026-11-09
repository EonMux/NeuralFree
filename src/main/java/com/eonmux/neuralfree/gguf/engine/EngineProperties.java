package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.chat.tools.ToolStrict;
import com.eonmux.neuralfree.gguf.cpu.numa.NumaStrategy;
import com.eonmux.neuralfree.gguf.engine.EngineConfig.BackendMode;
import com.eonmux.neuralfree.gguf.engine.EngineConfig.DraftType;
import com.eonmux.neuralfree.gguf.engine.EngineConfig.SplitMode;
import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.gguf.rpc.RpcServerSpec;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Properties;

import static com.eonmux.neuralfree.gguf.engine.EngineConfig.GPU_LAYERS_ALL;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.GPU_LAYERS_AUTO;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.N_CPU_LAYERS_UNSET;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_ACT_QUANT;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_BACKEND;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_BEAM_ALWAYS_CONSTRAIN;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_CONTROL_VECTOR;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_CONTROL_VECTOR_LAYER_RANGE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_CPU_MOE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_CPU_MOE_ALIAS;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_CPU_OFFLOAD_MB;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_CPU_OFFLOAD_PARAMS;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DEVICE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_EXPERT_PARALLEL;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_MAX;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_MIN;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_P_MIN;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_CPU_MOE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_GPU_DEVICES;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_N_CPU_MOE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_N_GPU_LAYERS;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_TENSOR_OVERRIDES;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_TYPE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_FIT;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_GPU_DEVICES;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_GPU_VRAM_LIMIT;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_KV_OFFLOAD;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_LORA;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_LOGITS_PROCESSORS;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_MAIN_GPU;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_MODEL_SAMPLING_DEFAULTS;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_MOE_CACHE_MB;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_MOE_CACHE_POOL;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_MTP;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_N_CPU_FFN;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_N_CPU_MOE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_N_CPU_MOE_ALIAS;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_N_GPU_LAYERS;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_NUMA;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_OVERRIDE_TENSOR;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_POOLING;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_RAM_CACHE_MB;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_REASONING_PARSER;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_RPC;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_SPLIT_MODE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_STREAM_PARAMS;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_TENSOR_OVERRIDES;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_TENSOR_SPLIT;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_TOOL_PARSER;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_TOOL_STRICT;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_VRAM_DEPTH;

/**
 * Reads an {@link EngineConfig} from {@code neuralfree.*} properties ({@link EngineConfig#fromProperties}): the
 * parsing of each {@link EngineConfig} key, and the typed property readers the engine's other property-driven
 * configurations ({@link SamplerConfig}) share. Every reader treats a missing or blank value as absent and refuses
 * a value it cannot parse by naming the property.
 *
 * <p>Properties read by {@code read(Properties)} ({@link EngineConfig#fromProperties}; absent keys keep their
 * defaults):
 * <ul>
 *   <li>{@code neuralfree.backend}: auto | opencl | cuda | vulkan | hip | metal | cpu, or an accelerator API served by
 *       an installed backend provider (sycl, xpu, cann, hexagon, openvino, webgpu, musa, zdnn, zendnn);
 *       default auto.</li>
 *   <li>{@code neuralfree.device}: device-name substring the backend selection must match, default none.</li>
 *   <li>{@code neuralfree.batch.invariant}: {@code true} runs the batch-invariant kernels, vLLM's
 *       {@code VLLM_BATCH_INVARIANT}: a row's logits no longer depend on the other rows batched with it (see
 *       {@link com.eonmux.neuralfree.gguf.BatchInvariance}). The environment variable
 *       {@code NEURALFREE_BATCH_INVARIANT} is read when the property is unset; default false.</li>
 *   <li>{@code neuralfree.split.mode}: none (default) | layer | row | tensor | context; llama.cpp
 *       {@code --split-mode}. {@code layer} spreads the model over several GPUs, each taking a contiguous span of
 *       layers; {@code row} splits every trunk layer's dense feed-forward over the GPUs, each holding a slice of the
 *       hidden dimension and a device-to-device all-reduce summing the partials; {@code tensor} is vLLM's tensor
 *       parallelism, sharding the attention (column-parallel QKV with the heads divided over the ranks, per-rank KV
 *       caches, a row-parallel output all-reduced) and the dense feed-forward together; {@code context} is vLLM's
 *       decode context parallelism, dividing the context positions between the ranks while every rank keeps every
 *       head and merges the partial softmaxes with their log-sum-exp. The GPUs are
 *       {@code neuralfree.gpu.devices}, the layer shares {@code neuralfree.tensor.split} and the GPU holding the
 *       embedding and the output head {@code neuralfree.main.gpu}.</li>
 *   <li>{@code neuralfree.gpu.devices}: the GPUs a split runs over (llama.cpp {@code --device}).</li>
 *   <li>{@code neuralfree.rpc}: the remote compute servers a layer split adds as devices, as
 *       {@code host:port[@key]} entries (llama.cpp {@code --rpc}); an entry without an inline key reads
 *       {@code NEURALFREE_RPC_KEY}. Each server runs {@code nftool rpc-server} and contributes its own GPU as a
 *       layer target, so trunk layers can run on another machine. Every connection is authenticated with the
 *       shared key. Requires {@code neuralfree.split.mode=layer}.</li>
 *   <li>{@code neuralfree.expert.parallel}: whether a tensor split places whole experts on each rank instead of an
 *       intermediate-width slice of every expert (vLLM {@code --enable-expert-parallel}).</li>
 *   <li>The {@link EplbConfig} keys: {@code neuralfree.eplb} off (default) | on, {@code neuralfree.eplb.profile},
 *       {@code neuralfree.eplb.save} and {@code neuralfree.eplb.policy}, the expert-parallel load balancer that packs
 *       the experts over the ranks by their recorded routed-row load (vLLM EPLB) instead of contiguous blocks.</li>
 *   <li>The {@link MemoryConfig} keys, including {@code neuralfree.context.shift}.</li>
 *   <li>The {@link SamplerConfig} keys, including {@code neuralfree.seed}.</li>
 *   <li>{@code neuralfree.ram.cache.mb}: host cache for streamed weights in MiB; -1 (default) = automatic, sized
 *       against the maximum heap, the physical RAM and the model size, and off for a model larger than RAM
 *       ({@code WeightDataProvider.ramCacheBytes}); 0 = no cache.</li>
 *   <li>{@code neuralfree.vram.depth}: layers kept resident in VRAM when streaming; 0 (default) = from the VRAM
 *       budget.</li>
 *   <li>{@code neuralfree.numa}: the CPU pool's NUMA placement (llama.cpp {@code --numa}, {@link NumaStrategy}):
 *       {@code distribute}, {@code isolate}, {@code numactl} or {@code disabled} (default). The pool reads the
 *       property when it starts; the strategy places each worker thread on its node's CPUs.</li>
 *   <li>{@code neuralfree.cpu.offload.mb}: the MiB of device-resident trunk weights a GPU load instead keeps in
 *       page-locked host memory that the device reads in place (vLLM's UVA offloader,
 *       {@code cpu_offload_gb}); 0 (default) = off. The tensors are chosen in weight-layer order up to the
 *       budget, or by {@code neuralfree.cpu.offload.params} when that names parameter segments. The weights stay
 *       read once per pass over the interconnect, costing no VRAM and no upload. Needs a backend with host-mapped
 *       buffers (CUDA mapped host memory, Vulkan host-visible memory) on one device.</li>
 *   <li>{@code neuralfree.cpu.offload.params}: the parameter name segments of the UVA offload (vLLM's
 *       {@code cpu_offload_params}), comma-separated; for example {@code experts} keeps the routed expert tensors
 *       in mapped host memory and the rest of every layer on the device. A segment matches whole dotted name
 *       parts ({@code ffn_gate_exps} matches {@code blk.3.ffn_gate_exps.weight}, {@code expert} does not).
 *       Empty (default) offloads tensors of any kind up to the budget.</li>
 *   <li>{@code neuralfree.stream.params}: the parameter name segments of a streamed layer that go through the
 *       streaming slots (vLLM's prefetch {@code offload_params}), comma-separated; the other tensors of those
 *       layers stay resident. For example {@code experts} streams only routed expert stacks of the layers the
 *       placement streams, keeping their attention and dense feed-forward on the device. Empty (default) streams
 *       every tensor of a streamed layer.</li>
 *   <li>{@code neuralfree.gpu.vram.limit}: the device memory in MiB a GPU load may allocate on each of its devices
 *       (weights, caches, scratch, streaming slots, adapters, the MoE expert cache and the reserved drafter memory
 *       together), below what the device reports free; 0 (default) = no limit.</li>
 *   <li>{@code neuralfree.n.gpu.layers}: how many layers a GPU load keeps on the device (llama.cpp
 *       {@code -ngl}): {@code auto} (default; as many as the VRAM holds), {@code all}, or a count that covers the
 *       output head, so {@code N} puts {@code N - 1} trunk layers on the device and the rest compute on the host,
 *       each side with its KV on its own device ({@code com.eonmux.neuralfree.gguf.hybrid}).</li>
 *   <li>{@code neuralfree.override.tensor}: tensor placement overrides as
 *       {@code <pattern>=<cpu | gpu | opencl | cuda | vulkan | hip | metal>} entries (llama.cpp {@code -ot}); the
 *       first pattern that matches a tensor name decides where it is kept. A host match keeps a routed expert tensor
 *       in RAM while its layer stays on the device and computes the experts there
 *       ({@code com.eonmux.neuralfree.gguf.ops.moe}); any other host match keeps the tensor's whole layer on the
 *       host.</li>
 *   <li>{@code neuralfree.cpu.moe} (llama.cpp {@code --cpu-moe}) and {@code neuralfree.n.cpu.moe} (llama.cpp
 *       {@code --n-cpu-moe}): keep the expert weights of every, or of the first N, mixture-of-experts layers on
 *       the host, where the layer's experts compute while the rest of the layer runs on the device.</li>
 *   <li>{@code neuralfree.n-cpu-ffn}: the dense FFN weights of the first N decoder layers in RAM, on the host
 *       (llama.cpp {@code --n-cpu-ffn}); unset by default. The weights shrink the VRAM fit; a dense FFN that then
 *       has to leave the device fails its load.</li>
 *   <li>{@code neuralfree.moe.cache.mb}: the MoE expert-cache budget in MiB (llama.cpp {@code --moe-cache-mib});
 *       0 (default) leaves the budget to {@code neuralfree.moe.cache}, whose VRAM-left budget it overrides.</li>
 *   <li>{@code neuralfree.fit}: keep the experts in RAM when the dense weights and the context fit the VRAM
 *       without them (Avalon {@code --fit}), instead of streaming; default false.</li>
 *   <li>{@code neuralfree.cpu-moe}, {@code neuralfree.n-cpu-moe} and {@code neuralfree.override-tensor}:
 *       aliases of {@code neuralfree.cpu.moe}, {@code neuralfree.n.cpu.moe} and {@code neuralfree.override.tensor}
 *       with llama.cpp's flag-shaped keys, taking the same targets; the dotted key wins when both are set.</li>
 *   <li>The {@link MoeCacheConfig} keys: {@code neuralfree.moe.cache} off (default) | on | auto | MiB,
 *       {@code neuralfree.moe.cache.ranking}, {@code neuralfree.moe.cache.ranking.save},
 *       {@code neuralfree.moe.cache.prefetch}, {@code neuralfree.moe.cache.hosted.rows} and
 *       {@code neuralfree.moe.cache.tail.skip}, the GPU MoE expert cache
 *       of host-resident routed experts, the expert ranking it preloads from, its prefill expert prefetch and the
 *       approximate tail-skip rule (off by default; a rank drops uncached experts every token routes only at that
 *       top-k position or lower).
 *       {@code on}, {@code auto} or a MiB value hosts the routed expert stacks in RAM as {@code neuralfree.cpu.moe}
 *       does and serves their routed experts from the device cache; {@code neuralfree.moe.cache.mb} sets the same
 *       budget and wins when both give one.</li>
 *   <li>{@code neuralfree.draft}: path of a draft GGUF for speculative decoding, default none: a separate draft model,
 *       an eagle, eagle3, gemma4-assistant or dflash drafter, or a GGUF whose last block holds NextN / MTP tensors
 *       (with or without its trunk), which drafts with those blocks only ({@link LLMModelFactory}).</li>
 *   <li>{@code neuralfree.draft.type}: the drafter types, a comma-separated list as llama.cpp's {@code --spec-type}
 *       ({@link DraftType}, {@link DraftTypes}). At most one draft-file type: auto (default; detected from the
 *       file), draft-mtp (alias mtp; its NextN / MTP blocks, or the model's own without a draft file) or
 *       draft-simple (aliases simple, model; a separate draft model). Any of the lookup types, which draft without a
 *       draft file: ngram-simple (alias prompt-lookup; the target's own token history), ngram-map-k (aliases
 *       ngram-map, map-k), ngram-map-k4v (alias map-k4v), ngram-mod (alias mod), ngram-cache (the
 *       {@link com.eonmux.neuralfree.gguf.engine.draft.NgramCache} chain) and suffix (suffix decoding). Several types
 *       chain ({@code DrafterChain}) in llama.cpp's order, the lookup types first and the draft file last: each
 *       round the first drafter with a proposal drafts. A draft file needs a draft-file type in the list.</li>
 *   <li>The {@link SpeculativeOptions} keys: the n-gram sizes of each lookup type, the {@code ngram-cache} files,
 *       the suffix decoding limits and the verification mode ({@code neuralfree.draft.verify}).</li>
 *   <li>The {@link LookaheadOptions} keys ({@code neuralfree.lookahead}, {@code .window}, {@code .ngram.size} and
 *       {@code .max.ngrams}): lookahead (Jacobi) decoding instead of speculative decoding, over a paged memory of
 *       {@code window + maxNgrams + 1} sequences with unified KV.</li>
 *   <li>{@code neuralfree.draft.max} (default 3), {@code neuralfree.draft.min} (default 0),
 *       {@code neuralfree.draft.p.min} (default 0): llama.cpp {@code common_params_speculative_draft}. The draft
 *       max bounds every drafter's round; a lookup drafter also stops at its own draft size.</li>
 *   <li>The draft model's own placement, applied to the load of a separate draft model
 *       ({@code neuralfree.draft.*}; every one needs {@code neuralfree.draft}): {@code neuralfree.draft.gpu.devices}
 *       (llama.cpp {@code -devd}) a comma-separated GPU selector list like {@code neuralfree.gpu.devices}, or
 *       {@code none} alone to keep the draft on the host; unset (default) keeps the draft beside the target's
 *       weights on the target's backend. {@code neuralfree.draft.n.gpu.layers} (llama.cpp {@code -ngld}): a count
 *       that covers the output head, {@code auto} (default) or {@code all}, like {@code neuralfree.n.gpu.layers}
 *       but for the draft's load. {@code neuralfree.draft.override.tensor} (llama.cpp {@code -otd}),
 *       {@code neuralfree.draft.cpu.moe} (llama.cpp {@code -cmoed}) and {@code neuralfree.draft.n.cpu.moe}
 *       (llama.cpp {@code -ncmoed}): the draft load's own overrides and expert placement, replacing the target's.</li>
 *   <li>{@code neuralfree.mtp}: self-speculation with the model's own NextN / MTP layers, default false. A GPU
 *       load places the NextN / MTP blocks on the device and in its context fit only when it is set (or
 *       {@code neuralfree.nextn.memory} is), as llama.cpp loads them only with {@code load_mtp}.</li>
 *   <li>{@code neuralfree.lora}: LoRA adapters as {@code path[:scale][,path[:scale]...]} (llama.cpp {@code --lora} /
 *       {@code --lora-scaled}), scale 1 when omitted, default none; see {@link LoraSpec#parseList}.</li>
 *   <li>The {@link LoraConfig} keys of the per-request adapter registry ({@code neuralfree.lora.enable},
 *       {@code neuralfree.lora.max}, {@code neuralfree.lora.max.rank}, {@code neuralfree.lora.cache}).</li>
 *   <li>{@code neuralfree.control-vector}: control vectors as {@code path[:scale][,path[:scale]...]} (llama.cpp
 *       {@code --control-vector} / {@code --control-vector-scaled}), scale 1 when omitted, default none; see
 *       {@link ControlVectorSpec#parseList}.</li>
 *   <li>{@code neuralfree.control-vector-layer-range}: {@code START:END}, the 1-based decoder layers the control
 *       vectors apply to, both inclusive (llama.cpp {@code --control-vector-layer-range START END}); 0 or a negative
 *       value keeps the llama.cpp default of that side ({@code 1} for the start, the model's layer count for the
 *       end).</li>
 *   <li>The {@link ModelSourceOptions} keys ({@code neuralfree.hf.*}, {@code neuralfree.quant.online},
 *       {@code neuralfree.lora.base.config}): how a checkpoint that is not GGUF is turned into its GGUF view.</li>
 *   <li>{@code neuralfree.act.quant}: weight-only (default) | checkpoint, how a checkpoint's activation quantization
 *       is applied ({@link ActivationQuant}).</li>
 *   <li>{@code neuralfree.sampling.model.defaults}: whether a model's sampling defaults
 *       ({@link EngineModel#modelSamplingDefaults()}) fill the sampler settings and token budget a request leaves
 *       unset, default true.</li>
 *   <li>{@code neuralfree.reasoning.parser}: the reasoning parser (vLLM {@code --reasoning-parser} names,
 *       {@link com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers}), default auto: the chat template's.</li>
 *   <li>{@code neuralfree.tool.parser}: the tool-call parser (vLLM {@code --tool-call-parser} names,
 *       {@link com.eonmux.neuralfree.gguf.chat.tools.ToolParserRegistry}), default auto: detected from the chat
 *       template; none leaves tool-call syntax in the content. The response-template parsers {@code hf} and
 *       {@code beam} (both properties) read the model's {@code response_template} metadata and pair only with
 *       the same parser, {@code auto} or {@code none}.</li>
 *   <li>{@code neuralfree.tool.strict}: the tool strict floor (vLLM {@code --tool-strict-level},
 *       {@code auto | function | parameter}, {@link com.eonmux.neuralfree.gguf.chat.tools.ToolStrict}),
 *       default auto: a tool's argument schema is enforced when the tool declares {@code strict}, and a
 *       tool-call grammar applies under {@code tool_choice: auto} only for a strict tool.</li>
 *   <li>{@code neuralfree.pooling}: the pooling type override (llama.cpp {@code --pooling}, vLLM
 *       {@code --override-pooler-config} {@code pooling_type}): {@code none | mean | cls | last | rank}, vLLM's
 *       {@code ALL} naming per-token pooling. Unset (default) takes the GGUF's {@code <arch>.pooling_type};
 *       {@code rank} refuses a model that stores no classifier head.</li>
 *   <li>{@code neuralfree.logits.processors}: the fully-qualified class names of custom logits processor
 *       factories to load beside the plugin providers (vLLM's {@code --logits-processors}), comma-separated.</li>
 * </ul>
 */
final class EngineProperties {

    private EngineProperties() {
    }

    /**
     * Configuration from a property set; absent keys keep the {@link EngineConfig#defaults()} values.
     *
     * @throws IllegalArgumentException naming the property when a value cannot be parsed or is out of range
     */
    static EngineConfig read(Properties p) {
        Objects.requireNonNull(p, "properties");
        String batchInvariant = stringProperty(p, BatchInvariance.PROPERTY);
        if (batchInvariant != null) {
            /* The kernels read the process-wide property; refuse a malformed value here instead of at the first
             * projection. */
            BatchInvariance.parse(batchInvariant, "-D" + BatchInvariance.PROPERTY);
        }
        EngineConfig d = EngineConfig.defaults();
        int[] cvRange = controlVectorLayerRangeProperty(p);
        EngineConfig.Builder b = EngineConfig.builder()
                .memory(MemoryConfig.fromProperties(p))
                .samplerDefaults(SamplerConfig.fromProperties(p))
                .ramCacheMb(longProperty(p, PROP_RAM_CACHE_MB, d.ramCacheMb()))
                .vramDepth(intProperty(p, PROP_VRAM_DEPTH, d.vramDepth()))
                .numa(numaProperty(p))
                .cpuOffloadMb(longProperty(p, PROP_CPU_OFFLOAD_MB, d.cpuOffloadMb()))
                .cpuOffloadParams(listProperty(p, PROP_CPU_OFFLOAD_PARAMS))
                .streamParams(listProperty(p, PROP_STREAM_PARAMS))
                .gpuVramLimitMb(longProperty(p, PROP_GPU_VRAM_LIMIT, d.gpuVramLimitMb()))
                .gpuLayers(gpuLayersProperty(p, PROP_N_GPU_LAYERS))
                .kvOffload(booleanProperty(p, PROP_KV_OFFLOAD, d.kvOffload()))
                .tensorOverrides(tensorOverridesProperty(p))
                .cpuMoe(booleanPropertyAlias(p, PROP_CPU_MOE, PROP_CPU_MOE_ALIAS, d.cpuMoe()))
                .nCpuMoe(nCpuLayersProperty(p, PROP_N_CPU_MOE, PROP_N_CPU_MOE_ALIAS))
                .draftGpuLayers(gpuLayersProperty(p, PROP_DRAFT_N_GPU_LAYERS))
                .draftTensorOverrides(tensorOverridesProperty(p, PROP_DRAFT_TENSOR_OVERRIDES))
                .draftCpuMoe(booleanProperty(p, PROP_DRAFT_CPU_MOE, d.draftCpuMoe()))
                .draftNCpuMoe(nCpuLayersProperty(p, PROP_DRAFT_N_CPU_MOE, null))
                .draftDevices(draftDevicesProperty(p))
                .moeCache(MoeCacheConfig.fromProperties(p))
                .moeCachePool(booleanProperty(p, PROP_MOE_CACHE_POOL, d.moeCachePool()))
                .splitMode(splitModeProperty(p))
                .expertParallel(booleanProperty(p, PROP_EXPERT_PARALLEL, d.expertParallel()))
                .prefillContextParallelSize(intProperty(p, PROP_PREFILL_CONTEXT_PARALLEL,
                        d.prefillContextParallelSize()))
                .eplb(EplbConfig.fromProperties(p))
                .gpuDevices(deviceListProperty(p))
                .rpcServers(rpcServersProperty(p))
                .tensorSplit(tensorSplitProperty(p))
                .mainGpu(intProperty(p, PROP_MAIN_GPU, d.mainGpu()))
                .draftModelPath(stringProperty(p, PROP_DRAFT))
                .draftTypes(draftTypeProperty(p))
                .speculativeOptions(SpeculativeOptions.fromProperties(p))
                .lookaheadOptions(LookaheadOptions.fromProperties(p))
                .draftMax(intProperty(p, PROP_DRAFT_MAX, d.draftMax()))
                .draftMin(intProperty(p, PROP_DRAFT_MIN, d.draftMin()))
                .draftPMin(floatProperty(p, PROP_DRAFT_P_MIN, d.draftPMin()))
                .mtp(booleanProperty(p, PROP_MTP, d.mtp()))
                .deviceFilter(stringProperty(p, PROP_DEVICE))
                .loras(loraProperty(p))
                .loraConfig(LoraConfig.fromProperties(p))
                .controlVectors(controlVectorProperty(p))
                .controlVectorLayerRange(cvRange[0], cvRange[1])
                .sourceOptions(ModelSourceOptions.fromProperties(p))
                .activationQuant(activationQuantProperty(p))
                .modelSamplingDefaults(booleanProperty(p, PROP_MODEL_SAMPLING_DEFAULTS, d.modelSamplingDefaults()))
                .reasoningParser(stringProperty(p, PROP_REASONING_PARSER))
                .toolParser(stringProperty(p, PROP_TOOL_PARSER))
                .toolStrict(strictProperty(p))
                .beamAlwaysConstrain(booleanProperty(p, PROP_BEAM_ALWAYS_CONSTRAIN, d.beamAlwaysConstrain()))
                .nCpuFfn(nCpuLayersProperty(p, PROP_N_CPU_FFN, null))
                .moeCacheMb(moeCacheMbProperty(p))
                .pooling(poolingProperty(p))
                .logitsProcessors(listProperty(p, PROP_LOGITS_PROCESSORS))
                .fit(booleanProperty(p, PROP_FIT, d.fit()));
        String backend = stringProperty(p, PROP_BACKEND);
        if (backend != null) {
            b.backend(BackendMode.parse(backend));
        }
        return b.build();
    }

    /** The trimmed value of {@code key}, or null when it is unset or blank. */
    static String stringProperty(Properties p, String key) {
        String v = p.getProperty(key);
        if (v == null) {
            return null;
        }
        String s = v.trim();
        return s.isEmpty() ? null : s;
    }

    /**
     * The comma-separated entries of {@code key}, trimmed; empty when it is unset or blank.
     *
     * @throws IllegalArgumentException naming the property when an entry is empty
     */
    static List<String> listProperty(Properties p, String key) {
        String v = stringProperty(p, key);
        if (v == null) {
            return Collections.emptyList();
        }
        List<String> out = new ArrayList<>();
        for (String part : v.split(",", -1)) {
            String entry = part.trim();
            if (entry.isEmpty()) {
                throw new IllegalArgumentException(key + " has an empty entry in '" + v + "'");
            }
            out.add(entry);
        }
        return out;
    }

    /**
     * The integer value of {@code key}, or {@code dflt} when it is unset.
     *
     * @throws IllegalArgumentException naming the property when the value is not an integer
     */
    static int intProperty(Properties p, String key, int dflt) {
        String v = stringProperty(p, key);
        if (v == null) {
            return dflt;
        }
        try {
            return Integer.parseInt(v);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + "=" + v + " is not an integer", e);
        }
    }

    /**
     * The long value of {@code key}, or {@code dflt} when it is unset.
     *
     * @throws IllegalArgumentException naming the property when the value is not an integer
     */
    static long longProperty(Properties p, String key, long dflt) {
        String v = stringProperty(p, key);
        if (v == null) {
            return dflt;
        }
        try {
            return Long.parseLong(v);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + "=" + v + " is not an integer", e);
        }
    }

    /**
     * The float value of {@code key}, or {@code dflt} when it is unset.
     *
     * @throws IllegalArgumentException naming the property when the value is not a number
     */
    static float floatProperty(Properties p, String key, float dflt) {
        String v = stringProperty(p, key);
        if (v == null) {
            return dflt;
        }
        try {
            return Float.parseFloat(v);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + "=" + v + " is not a number", e);
        }
    }

    /**
     * The boolean value of {@code key} ({@code true} or {@code false}, ignoring case), or {@code dflt} when it is
     * unset.
     *
     * @throws IllegalArgumentException naming the property for any other value
     */
    static boolean booleanProperty(Properties p, String key, boolean dflt) {
        String v = stringProperty(p, key);
        if (v == null) {
            return dflt;
        }
        String s = v.toLowerCase(Locale.ROOT);
        if (s.equals("true")) {
            return true;
        }
        if (s.equals("false")) {
            return false;
        }
        throw new IllegalArgumentException(key + "=" + v + " is not a boolean; use true or false");
    }

    /** A boolean property with an alias key; the primary key wins when both are set. */
    static boolean booleanPropertyAlias(Properties p, String key, String alias, boolean dflt) {
        return stringProperty(p, key) != null ? booleanProperty(p, key, dflt) : booleanProperty(p, alias, dflt);
    }

    /** The configured tool strict level, refusing an unknown name by the property it came from. */
    private static ToolStrict strictProperty(Properties p) {
        String v = stringProperty(p, PROP_TOOL_STRICT);
        if (v == null) {
            return ToolStrict.AUTO;
        }
        try {
            return ToolStrict.fromName(v);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(PROP_TOOL_STRICT + "=" + v + " is not a tool strict level;"
                    + " use auto, function or parameter");
        }
    }

    /** The pooling override ({@link PoolingKind} names), or null when the property is unset. */
    private static PoolingKind poolingProperty(Properties p) {
        String v = stringProperty(p, PROP_POOLING);
        if (v == null) {
            return null;
        }
        try {
            return PoolingKind.parse(v);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(PROP_POOLING + "=" + v + ": " + e.getMessage(), e);
        }
    }

    /** The CPU pool's NUMA strategy ({@code --numa}); {@link NumaStrategy#DISABLED} when the key is unset. */
    private static NumaStrategy numaProperty(Properties p) {
        String v = stringProperty(p, PROP_NUMA);
        if (v == null) {
            return NumaStrategy.DISABLED;
        }
        return NumaStrategy.parse(v);
    }

    private static SplitMode splitModeProperty(Properties p) {
        String v = stringProperty(p, PROP_SPLIT_MODE);
        if (v == null) {
            return SplitMode.NONE;
        }
        try {
            return SplitMode.parse(v);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(e.getMessage(), e);
        }
    }

    /** Comma-separated GPU selectors ({@code 0}, {@code 1}, or a name substring); blank entries are refused. */
    private static List<String> deviceListProperty(Properties p) {
        String v = stringProperty(p, PROP_GPU_DEVICES);
        if (v == null) {
            return new ArrayList<>();
        }
        List<String> out = new ArrayList<>();
        for (String entry : v.split(",")) {
            String s = entry.trim();
            if (s.isEmpty()) {
                throw new IllegalArgumentException(PROP_GPU_DEVICES + "=" + v + " has an empty device entry");
            }
            out.add(s);
        }
        return out;
    }

    /**
     * Comma-separated remote compute servers ({@code host:port[@key]}) the layer split adds as devices; a malformed
     * entry or one without a key is refused by {@link RpcServerSpec#parse(String)}.
     */
    private static List<RpcServerSpec> rpcServersProperty(Properties p) {
        String v = stringProperty(p, PROP_RPC);
        if (v == null) {
            return new ArrayList<>();
        }
        List<RpcServerSpec> out = new ArrayList<>();
        for (String entry : v.split(",", -1)) {
            String s = entry.trim();
            if (s.isEmpty()) {
                throw new IllegalArgumentException(PROP_RPC + "=" + RpcServerSpec.masked(v)
                        + " has an empty server entry");
            }
            out.add(RpcServerSpec.parse(s));
        }
        return out;
    }

    /**
     * Comma-separated GPU selectors of the draft model's own load ({@code 0}, {@code 1}, or a name substring), or
     * the lone {@link EngineConfig#DRAFT_NO_DEVICE}; blank entries are refused, and {@code none} beside devices is
     * refused by the {@link EngineConfig.Builder}.
     */
    private static List<String> draftDevicesProperty(Properties p) {
        String v = stringProperty(p, PROP_DRAFT_GPU_DEVICES);
        if (v == null) {
            return new ArrayList<>();
        }
        List<String> out = new ArrayList<>();
        for (String entry : v.split(",")) {
            String s = entry.trim();
            if (s.isEmpty()) {
                throw new IllegalArgumentException(PROP_DRAFT_GPU_DEVICES + "=" + v + " has an empty device entry");
            }
            out.add(s);
        }
        return out;
    }

    /** Comma-separated positive layer shares ({@code 3,1} takes three quarters of the layers on the first GPU). */
    private static double[] tensorSplitProperty(Properties p) {
        String v = stringProperty(p, PROP_TENSOR_SPLIT);
        if (v == null) {
            return new double[0];
        }
        String[] parts = v.split(",");
        double[] out = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                out[i] = Double.parseDouble(parts[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(PROP_TENSOR_SPLIT + "=" + v + ": '" + parts[i]
                        + "' is not a number", e);
            }
            if (!(out[i] > 0.0) || !Double.isFinite(out[i])) {
                throw new IllegalArgumentException(PROP_TENSOR_SPLIT + "=" + v + ": every share must be a finite"
                        + " value above 0, got " + parts[i].trim());
            }
        }
        return out;
    }

    private static List<DraftType> draftTypeProperty(Properties p) {
        String v = stringProperty(p, PROP_DRAFT_TYPE);
        if (v == null) {
            return DraftTypes.DEFAULT;
        }
        try {
            return DraftTypes.parse(v);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(PROP_DRAFT_TYPE + "=" + v + ": " + e.getMessage(), e);
        }
    }

    private static ActivationQuant activationQuantProperty(Properties p) {
        String v = stringProperty(p, PROP_ACT_QUANT);
        return v == null ? ActivationQuant.WEIGHT_ONLY : ActivationQuant.parse(v);
    }

    /**
     * The {@code -ngl} value of a property: {@code auto} or {@code all}, or a layer count.
     *
     * @throws IllegalArgumentException naming the property for any other value
     */
    private static int gpuLayersProperty(Properties p, String key) {
        String v = stringProperty(p, key);
        if (v == null) {
            return GPU_LAYERS_AUTO;
        }
        String s = v.toLowerCase(Locale.ROOT);
        if (s.equals("auto")) {
            return GPU_LAYERS_AUTO;
        }
        if (s.equals("all")) {
            return GPU_LAYERS_ALL;
        }
        try {
            int n = Integer.parseInt(s);
            if (n >= 0) {
                return n;
            }
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(key + "=" + v + " is invalid; use auto, all or a layer count >= 0", e);
        }
        throw new IllegalArgumentException(key + "=" + v + " is invalid; use auto, all or a layer count >= 0");
    }

    /**
     * The {@code -ot} entries: comma-separated {@code pattern=target} under {@link EngineConfig#PROP_TENSOR_OVERRIDES},
     * plus the same entries under the {@link EngineConfig#PROP_OVERRIDE_TENSOR} alias (llama.cpp's flag-shaped key;
     * the same targets are accepted there).
     *
     * @throws IllegalArgumentException naming the entry when one cannot be parsed
     */
    private static List<TensorOverride> tensorOverridesProperty(Properties p) {
        List<TensorOverride> out = new ArrayList<>(tensorOverridesProperty(p, PROP_TENSOR_OVERRIDES));
        out.addAll(tensorOverridesProperty(p, PROP_OVERRIDE_TENSOR));
        return out;
    }

    /** The {@code -ot} entries under one property key: comma-separated {@code pattern=target}. */
    private static List<TensorOverride> tensorOverridesProperty(Properties p, String key) {
        String v = stringProperty(p, key);
        if (v == null) {
            return Collections.emptyList();
        }
        List<TensorOverride> out = new ArrayList<>();
        for (String entry : v.split(",")) {
            String e = entry.trim();
            if (e.isEmpty()) {
                continue;
            }
            int eq = e.indexOf('=');
            if (eq <= 0 || eq == e.length() - 1) {
                throw new IllegalArgumentException(key + " entry '" + e + "' is invalid; use"
                        + " <tensor name pattern>=<" + String.join(" | ", TensorOverride.targets()) + ">");
            }
            try {
                out.add(TensorOverride.parse(e.substring(0, eq), e.substring(eq + 1)));
            } catch (IllegalArgumentException ex) {
                throw new IllegalArgumentException(key + " entry '" + e + "': " + ex.getMessage(), ex);
            }
        }
        return out;
    }

    private static List<LoraSpec> loraProperty(Properties p) {
        try {
            return LoraSpec.parseList(stringProperty(p, PROP_LORA));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(PROP_LORA + "=" + p.getProperty(PROP_LORA) + ": " + e.getMessage(), e);
        }
    }

    private static List<ControlVectorSpec> controlVectorProperty(Properties p) {
        try {
            return ControlVectorSpec.parseList(stringProperty(p, PROP_CONTROL_VECTOR));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(PROP_CONTROL_VECTOR + "=" + p.getProperty(PROP_CONTROL_VECTOR)
                    + ": " + e.getMessage(), e);
        }
    }

    /**
     * The {@code START:END} layer range, both parts optional ({@code :END}, {@code START:}, a lone number as the
     * start); each part {@code <= 0} or missing keeps the 0 default that selects llama.cpp's range end default at
     * load.
     *
     * @throws IllegalArgumentException when the value holds more than one {@code ':'} or a part is not an integer
     */
    private static int[] controlVectorLayerRangeProperty(Properties p) {
        String v = stringProperty(p, PROP_CONTROL_VECTOR_LAYER_RANGE);
        if (v == null) {
            return new int[]{0, 0};
        }
        String[] parts = v.split(":", -1);
        if (parts.length > 2) {
            throw new IllegalArgumentException(PROP_CONTROL_VECTOR_LAYER_RANGE + "=" + v
                    + " is not a START:END layer range");
        }
        int[] out = new int[2];
        for (int i = 0; i < parts.length; i++) {
            String part = parts[i].trim();
            if (part.isEmpty()) {
                continue;
            }
            try {
                out[i] = Integer.parseInt(part);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(PROP_CONTROL_VECTOR_LAYER_RANGE + "=" + v + " holds the"
                        + " non-integer layer " + part, e);
            }
        }
        return out;
    }

    /** An {@code --n-cpu-moe} / {@code --n-cpu-ffn} layer count: unset when absent, else non-negative. */
    private static int nCpuLayersProperty(Properties p, String key, String alias) {
        String name = key;
        String v = stringProperty(p, key);
        if (v == null && alias != null) {
            v = stringProperty(p, alias);
            if (v != null) {
                name = alias;
            }
        }
        if (v == null) {
            return N_CPU_LAYERS_UNSET;
        }
        int n = intProperty(p, name, N_CPU_LAYERS_UNSET);
        if (n < 0) {
            throw new IllegalArgumentException(name + " must be >= 0, got " + n);
        }
        return n;
    }

    /** The {@code --moe-cache-mib} expert-cache budget in MiB: 0 (default) or a non-negative count. */
    private static long moeCacheMbProperty(Properties p) {
        String v = stringProperty(p, PROP_MOE_CACHE_MB);
        if (v == null) {
            return 0L;
        }
        long mb = longProperty(p, PROP_MOE_CACHE_MB, 0L);
        if (mb < 0) {
            throw new IllegalArgumentException(PROP_MOE_CACHE_MB + " must be >= 0, got " + mb);
        }
        return mb;
    }
}
