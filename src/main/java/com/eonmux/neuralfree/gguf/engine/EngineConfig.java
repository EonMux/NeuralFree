package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.Config;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.chat.tools.ToolStrict;
import com.eonmux.neuralfree.gguf.cpu.numa.NumaStrategy;
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

/**
 * Everything a caller chooses when opening a model: backend and device, memory, sampling defaults, weight
 * streaming, speculative decoding, LoRA adapters, control vectors, how checkpoints that are not GGUF are opened and
 * how checkpoint activation quantization is applied. Immutable; built with {@link #builder()} or read from system
 * properties with {@link #fromSystemProperties()}.
 *
 * <p>The {@code neuralfree.*} properties {@link #fromProperties(Properties)} reads (absent keys keep their
 * defaults) are documented beside the reader that parses them, {@code EngineProperties}; a value it cannot
 * parse is refused with an {@link IllegalArgumentException} naming the property.
 */
public final class EngineConfig {

    /**
     * The compute backend: {@code auto}, {@code cpu}, one of the built-in APIs ({@code opencl}, {@code cuda},
     * {@code vulkan}, {@code hip}, {@code metal}) or an accelerator API served by an installed backend provider
     * ({@code sycl}, {@code xpu}, {@code cann}, {@code hexagon}, {@code openvino}, {@code webgpu}, {@code musa},
     * {@code zdnn}, {@code zendnn}). See {@link BackendMode#parse(String)}.
     */
    public static final String PROP_BACKEND = "neuralfree.backend";
    public static final String PROP_DEVICE = "neuralfree.device";
    /** llama.cpp {@code --split-mode} ({@code -sm}): how a model spreads over several GPUs. */
    public static final String PROP_SPLIT_MODE = "neuralfree.split.mode";
    /**
     * vLLM {@code --prefill-context-parallel-size}: the rank count a context split splits every prefill's query
     * rows over. 1 (the default) keeps every rank running every prefill row against its own key shard; the rank
     * count skips the rows below the rank's context block — their attention launches, their projections and their
     * stores — and fills their merge slots with empty partitions. Needs {@link #PROP_SPLIT_MODE}=context and the
     * context split's rank count.
     */
    public static final String PROP_PREFILL_CONTEXT_PARALLEL = "neuralfree.prefill_context_parallel_size";
    /**
     * llama.cpp {@code --tensor-split} ({@code -ts}): the share of the model each GPU takes, one value per device.
     * A layer split sizes the layer spans; a row split sizes the feed-forward hidden ranges in proportion (rounded
     * down to whole device blocks, the last rank taking the remainder). A tensor split divides evenly and refuses
     * explicit shares.
     */
    public static final String PROP_TENSOR_SPLIT = "neuralfree.tensor.split";
    /**
     * vLLM {@code --enable-expert-parallel}: with {@link SplitMode#TENSOR}, every rank holds whole experts (its own
     * contiguous expert range) instead of a slice of every expert's intermediate width. The routing stays replicated,
     * a rank computes only its experts' pairs (the others contribute zero) and the group all-reduce sums the ranks'
     * partial outputs. Requires the model's expert count to divide into the ranks, and no expert offload.
     */
    public static final String PROP_EXPERT_PARALLEL = "neuralfree.expert.parallel";
    /**
     * llama.cpp {@code --device} ({@code -dev}): the GPUs a split runs over, as an index into the enumeration or a
     * device-name substring per entry. When set, the selectors choose from the local devices and the
     * {@link #PROP_RPC} servers ({@code rpc<i>}) together, so a remote server can also be left out.
     */
    public static final String PROP_GPU_DEVICES = "neuralfree.gpu.devices";
    /**
     * llama.cpp {@code --rpc}: the remote compute servers a layer split adds as devices, comma-separated as
     * {@code [tls://]host:port[@key]}; an entry without an inline key reads it from {@value
     * com.eonmux.neuralfree.gguf.rpc.RpcServerSpec#KEY_ENV}, and the {@code tls://} scheme runs the transport over
     * TLS (the client's trust store in {@code neuralfree.rpc.tls.truststore}). Remote devices are appended after the
     * local ones and take a contiguous span of layers like any other device, so inference offloads to another machine
     * (an {@code nftool rpc-server} there); a server serving several devices contributes one device per entry of its
     * {@code -d} list. When {@link #PROP_GPU_DEVICES} is set, its selectors choose from the local devices and the
     * remote servers ({@code rpc<i>}) together, so a remote can also be left out. Requires {@link
     * #PROP_SPLIT_MODE} {@code layer}.
     */
    public static final String PROP_RPC = "neuralfree.rpc";
    /** llama.cpp {@code --main-gpu} ({@code -mg}): the GPU that holds the embedding and the output head. */
    public static final String PROP_MAIN_GPU = "neuralfree.main.gpu";
    public static final String PROP_RAM_CACHE_MB = "neuralfree.ram.cache.mb";
    public static final String PROP_VRAM_DEPTH = "neuralfree.vram.depth";
    /** The MiB of trunk weights kept in host-mapped memory the device reads in place (UVA zero-copy); 0 is off. */
    public static final String PROP_CPU_OFFLOAD_MB = "neuralfree.cpu.offload.mb";
    /** The parameter name segments of the UVA offload (vLLM {@code cpu_offload_params}); empty offloads any. */
    public static final String PROP_CPU_OFFLOAD_PARAMS = "neuralfree.cpu.offload.params";
    /** The parameter name segments of a streamed layer that stream (vLLM prefetch {@code offload_params}). */
    public static final String PROP_STREAM_PARAMS = "neuralfree.stream.params";
    /** The MiB of device memory a GPU load may allocate on each of its devices; 0 for no limit. */
    public static final String PROP_GPU_VRAM_LIMIT = "neuralfree.gpu.vram.limit";
    public static final String PROP_N_GPU_LAYERS = "neuralfree.n.gpu.layers";
    /**
     * llama.cpp {@code --numa}: the NUMA placement of the CPU pool's threads, {@code distribute}, {@code isolate}
     * or {@code numactl} ({@link NumaStrategy}); {@code disabled} (default) leaves them to the OS. The pool reads
     * {@link NumaStrategy#PROPERTY} when it starts, so the property is what places threads; this key validates and
     * reports it.
     */
    public static final String PROP_NUMA = NumaStrategy.PROPERTY;
    /**
     * Whether a GPU load keeps the layers' KV caches on the device (llama.cpp {@code offload_kqv}): {@code false}
     * ({@code -nkvo}) runs every layer's projections and FFN on the device with its attention on the host over a
     * host-resident cache. Default true.
     */
    public static final String PROP_KV_OFFLOAD = "neuralfree.kv.offload";
    public static final String PROP_TENSOR_OVERRIDES = "neuralfree.override.tensor";
    public static final String PROP_CPU_MOE = "neuralfree.cpu.moe";
    public static final String PROP_N_CPU_MOE = "neuralfree.n.cpu.moe";
    public static final String PROP_DRAFT = "neuralfree.draft";
    public static final String PROP_DRAFT_TYPE = "neuralfree.draft.type";
    public static final String PROP_DRAFT_MAX = "neuralfree.draft.max";
    public static final String PROP_DRAFT_MIN = "neuralfree.draft.min";
    public static final String PROP_DRAFT_P_MIN = "neuralfree.draft.p.min";
    public static final String PROP_MTP = "neuralfree.mtp";
    public static final String PROP_LORA = "neuralfree.lora";
    public static final String PROP_CONTROL_VECTOR = "neuralfree.control-vector";
    public static final String PROP_CONTROL_VECTOR_LAYER_RANGE = "neuralfree.control-vector-layer-range";
    public static final String PROP_ACT_QUANT = ActivationQuant.PROPERTY;
    public static final String PROP_MODEL_SAMPLING_DEFAULTS = "neuralfree.sampling.model.defaults";
    /** The reasoning parser name ({@code ReasoningParsers}); {@code auto} or unset follows the chat template. */
    public static final String PROP_REASONING_PARSER = "neuralfree.reasoning.parser";
    /**
     * The tool-call parser name ({@code ToolParserRegistry}); {@code auto} or unset detects it from the chat
     * template, {@code none} leaves tool-call syntax in the content.
     */
    public static final String PROP_TOOL_PARSER = "neuralfree.tool.parser";
    /** The tool strict level ({@code ToolStrict}); {@code auto} or unset follows each tool's strict flag. */
    public static final String PROP_TOOL_STRICT = "neuralfree.tool.strict";
    /** {@code beam} holds every output to its template grammar ({@code VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT}); default true. */
    public static final String PROP_BEAM_ALWAYS_CONSTRAIN = "neuralfree.beam.always.constrain";
    /** llama.cpp-flag-shaped alias of {@link #PROP_CPU_MOE}; the dotted key wins when both are set. */
    public static final String PROP_CPU_MOE_ALIAS = "neuralfree.cpu-moe";
    /** llama.cpp-flag-shaped alias of {@link #PROP_N_CPU_MOE}; the dotted key wins when both are set. */
    public static final String PROP_N_CPU_MOE_ALIAS = "neuralfree.n-cpu-moe";
    /** {@code --n-cpu-ffn N}: the dense FFN weights of the first N decoder layers in RAM, on the host. */
    public static final String PROP_N_CPU_FFN = "neuralfree.n-cpu-ffn";
    /**
     * Alias of {@link #PROP_TENSOR_OVERRIDES} with llama.cpp's flag-shaped key, taking the same targets; the dotted
     * key wins when both are set.
     */
    public static final String PROP_OVERRIDE_TENSOR = "neuralfree.override-tensor";
    /** {@code --moe-cache-mib}: the MoE expert-cache budget in MiB, 0 (default) for the configured default. */
    public static final String PROP_MOE_CACHE_MB = "neuralfree.moe.cache.mb";
    /**
     * {@code --moe-cache-pool}: pool the expert cache over the GPUs of a layer split. The routed hosted MoE layers
     * are divided among the GPUs in layer order, each GPU taking a share of the experts' bytes proportional to the
     * VRAM it has left, and each layer's hosted MoE block runs on the GPU whose cache holds its experts.
     */
    public static final String PROP_MOE_CACHE_POOL = "neuralfree.moe.cache.pool";
    /** {@code --fit}: keep the experts in RAM when the dense weights and the context fit the VRAM without them. */
    public static final String PROP_FIT = "neuralfree.fit";
    /**
     * The pooling type override (llama.cpp {@code --pooling}, vLLM {@code --override-pooler-config}
     * {@code pooling_type}): {@code none | mean | cls | last | rank}, vLLM's {@code ALL} naming per-token pooling.
     * Unset (default) takes the GGUF's {@code <arch>.pooling_type}; {@code rank} refuses a model that stores no
     * classifier head.
     */
    public static final String PROP_POOLING = "neuralfree.pooling";

    /**
     * The custom logits processor factory classes to load beside the plugin providers (vLLM's
     * {@code --logits-processors}); each is a fully-qualified class name of a
     * {@link com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessorFactory} with a public no-argument
     * constructor, comma-separated.
     */
    public static final String PROP_LOGITS_PROCESSORS = "neuralfree.logits.processors";

    /** llama.cpp {@code --spec-draft-ngl} ({@code -ngld}): how many layers the draft model's GPU load keeps. */
    public static final String PROP_DRAFT_N_GPU_LAYERS = "neuralfree.draft.n.gpu.layers";
    /** llama.cpp {@code --spec-draft-override-tensor} ({@code -otd}): the draft model's tensor placement overrides. */
    public static final String PROP_DRAFT_TENSOR_OVERRIDES = "neuralfree.draft.override.tensor";
    /** llama.cpp {@code --spec-draft-cpu-moe} ({@code -cmoed}): every MoE layer of the draft model hosts its experts. */
    public static final String PROP_DRAFT_CPU_MOE = "neuralfree.draft.cpu.moe";
    /** llama.cpp {@code --spec-draft-n-cpu-moe} ({@code -ncmoed}): the first N draft-model MoE layers host theirs. */
    public static final String PROP_DRAFT_N_CPU_MOE = "neuralfree.draft.n.cpu.moe";
    /** llama.cpp {@code --spec-draft-device} ({@code -devd}): the GPUs the draft model's own load runs over. */
    public static final String PROP_DRAFT_GPU_DEVICES = "neuralfree.draft.gpu.devices";
    /** The lone {@link #PROP_DRAFT_GPU_DEVICES} entry that keeps the draft model off every GPU (llama.cpp: no offload). */
    public static final String DRAFT_NO_DEVICE = "none";

    /** {@link #nCpuMoe()} and {@link #nCpuFfn()} value meaning "not set". */
    public static final int N_CPU_LAYERS_UNSET = 0;

    /** {@link #ramCacheMb()} value meaning "automatic: heap, physical RAM and model size decide
     *  ({@code WeightDataProvider.ramCacheBytes}); off for a model larger than RAM". */
    public static final long RAM_CACHE_AUTO = -1L;
    /** {@link #gpuVramLimitMb()} value meaning "no limit: the free VRAM the device reports decides". */
    public static final long GPU_VRAM_UNLIMITED = 0L;
    /** {@link #vramDepth()} value meaning "derive from the VRAM budget". */
    public static final int VRAM_DEPTH_AUTO = 0;
    /** {@link #gpuLayers()} value meaning "as many layers as fit the VRAM" (llama.cpp {@code -ngl auto}). */
    public static final int GPU_LAYERS_AUTO = -1;
    /** {@link #gpuLayers()} value meaning "every layer" (llama.cpp {@code -ngl all}). */
    public static final int GPU_LAYERS_ALL = -2;

    /** Compute backend choice. */
    public enum BackendMode {
        /** The fastest GPU backend that can run the LLM kernels, else the CPU. */
        AUTO,
        OPENCL,
        CUDA,
        VULKAN,
        HIP,
        /** Apple GPUs through Metal. */
        METAL,
        /** SYCL/Level Zero, through an installed backend provider. */
        SYCL,
        /** Intel XPU, through an installed backend provider. */
        XPU,
        /** Huawei CANN, through an installed backend provider. */
        CANN,
        /** Qualcomm Hexagon, through an installed backend provider. */
        HEXAGON,
        /** Intel OpenVINO, through an installed backend provider. */
        OPENVINO,
        /** WebGPU, through an installed backend provider. */
        WEBGPU,
        /** Moore Threads MUSA, through an installed backend provider. */
        MUSA,
        /** IBM Z zDNN, through an installed backend provider. */
        ZDNN,
        /** AMD ZenDNN, through an installed backend provider. */
        ZENDNN,
        /** The CPU engine only. */
        CPU;

        /** Whether a GPU backend is requested or allowed. */
        public boolean usesGpu() {
            return this != CPU;
        }

        /**
         * The acceleration mode used for GPU backend selection.
         *
         * @throws IllegalStateException for {@link #CPU}
         */
        public Config.ACCELERATION_MODE toAccelerationMode() {
            if (this == CPU) {
                throw new IllegalStateException("backend mode CPU has no GPU acceleration mode");
            }
            return Config.ACCELERATION_MODE.valueOf(name());
        }

        /**
         * The mode a user names, case-insensitively; accelerator APIs accept their runtime spellings
         * ({@code level-zero}, {@code web-gpu}, {@code z-dnn}, {@code zen-dnn}) through
         * {@link ComputeBackend.BackendType#fromApiName(String)}.
         *
         * @throws IllegalArgumentException for an unknown name, and for {@code tpu}, whose platform
         *                                  runs through JAX/torch_xla and is not reachable from this engine
         */
        public static BackendMode parse(String value) {
            String s = Objects.requireNonNull(value, "backend mode").trim().toUpperCase(Locale.ROOT);
            if ("AUTO".equals(s)) {
                return AUTO;
            }
            if ("CPU".equals(s)) {
                return CPU;
            }
            if ("TPU".equals(s)) {
                throw new IllegalArgumentException(PROP_BACKEND + "=" + value + " is not supported: vLLM's TPU"
                        + " platform runs through JAX/torch_xla, which a Java inference engine cannot drive; use auto, "
                        + ComputeBackend.BackendType.acceptedApiNames() + " or cpu");
            }
            try {
                return valueOf(ComputeBackend.BackendType.fromApiName(value).name());
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(PROP_BACKEND + "=" + value + " is invalid; use auto, "
                        + ComputeBackend.BackendType.acceptedApiNames() + " or cpu");
            }
        }

        /**
         * The lowercase names {@link #parse} accepts, in declaration order ({@code auto}, {@code opencl},
         * {@code cuda}, {@code vulkan}, {@code hip}, {@code metal}, the accelerator APIs, {@code cpu});
         * unmodifiable. Usage lines and target lists derive from it, so they stay complete when a backend is added.
         */
        public static List<String> names() {
            List<String> names = new ArrayList<>(values().length);
            for (BackendMode m : values()) {
                names.add(m.name().toLowerCase(Locale.ROOT));
            }
            return Collections.unmodifiableList(names);
        }

        /** The names joined {@code " | "}, as a usage line prints them. */
        public static String namesBar() {
            return String.join(" | ", names());
        }
    }

    /**
     * How a model spreads over several GPUs (llama.cpp {@code --split-mode}, {@code include/llama.h}
     * {@code LLAMA_SPLIT_MODE}).
     */
    public enum SplitMode {
        /** One GPU runs the whole model (the default). */
        NONE("none"),
        /**
         * Each GPU takes a contiguous span of layers and their KV caches (llama.cpp {@code LLAMA_SPLIT_MODE_LAYER},
         * vLLM pipeline parallelism): the activation hand-off between two GPUs is staged through the host, and the
         * embedding and the output head run on the main GPU ({@link #mainGpu()}).
         */
        LAYER("layer"),
        /**
         * Each GPU holds a slice of every split layer's feed-forward projections and a device-to-device all-reduce
         * sums the partials (llama.cpp {@code LLAMA_SPLIT_MODE_ROW}, vLLM's MLP tensor parallelism with
         * {@code tensor_parallel_size} the device count): the split covers the trunk's dense FFN, and the
         * attention, norms, embedding and output head stay whole on the main GPU. {@code neuralfree.tensor.split}
         * sizes the hidden ranges ({@code --tensor-split}), rounded to whole device blocks.
         */
        ROW("row"),
        /**
         * vLLM's tensor parallelism over the attention and the dense feed-forward: every rank holds a slice of the
         * query, key, value and output projections (the query heads divided over the ranks, the KV heads likewise,
         * so each rank keeps its own KV cache), a row-parallel output projection all-reduced into the main device,
         * and its slice of the dense FFN. Contrary to {@link #ROW}, the attention is part of the split.
         */
        TENSOR("tensor"),
        /**
         * vLLM's decode context parallelism ({@code decode_context_parallel_size}): every rank keeps every head and
         * the ranks divide the context positions between them, each attending over its own block of the model's KV
         * cache. The ranks' partial softmaxes merge through their log-sum-exp ({@code LLM_AttnCpMerge}), so the
         * layer output is the single-device attention. Unlike {@link #TENSOR} the split is over the context, not
         * the heads; a paged multi-sequence batch is refused.
         */
        CONTEXT("context");

        private final String llamaName;

        SplitMode(String llamaName) {
            this.llamaName = llamaName;
        }

        /** llama.cpp's name of the mode. */
        public String llamaName() {
            return llamaName;
        }

        /**
         * The mode of a llama.cpp name, ignoring case.
         *
         * @throws IllegalArgumentException for an unknown name
         */
        public static SplitMode parse(String value) {
            String s = Objects.requireNonNull(value, "split mode").trim().toLowerCase(Locale.ROOT);
            if (s.equals("layer")) {
                return LAYER;
            }
            if (s.equals("row")) {
                return ROW;
            }
            if (s.equals("tensor")) {
                return TENSOR;
            }
            if (s.equals("context")) {
                return CONTEXT;
            }
            if (s.equals("none") || s.isEmpty()) {
                return NONE;
            }
            throw new IllegalArgumentException(PROP_SPLIT_MODE + "=" + value + " is invalid; use none, layer, row,"
                    + " tensor or context");
        }
    }

    /**
     * A speculative drafter type: llama.cpp's speculative type ({@code --spec-type}, common/speculative.cpp:75-85)
     * or vLLM's {@code suffix}. The draft-file types ({@code auto}, {@code draft-mtp}, {@code draft-simple}) say how
     * the draft file drafts; llama.cpp detects that from the file only when no type is given
     * (common/arg.cpp:566-572), and an explicit type overrides the detection. The lookup types draft without a
     * draft file and are declared in llama.cpp's ask order ({@link DraftTypes#ordered}).
     */
    public enum DraftType {
        /**
         * Detected from the draft file as llama.cpp detects it (common/speculative.cpp:2398-2433): dflash, eagle3 and
         * gemma4-assistant files draft as their architecture; a file whose last block holds {@code nextn.eh_proj}
         * drafts with its NextN / MTP blocks when it is an MTP draft of the target's model, and otherwise, when it
         * holds a model of its own, as a separate draft model; any other file is a separate draft model.
         */
        AUTO("auto"),
        /**
         * {@code draft-mtp}: the draft file's NextN / MTP blocks over the target's hidden states, refused unless the
         * file is an MTP draft of the target's model; without a draft file, the target's own NextN / MTP blocks
         * (common/speculative.cpp:2691-2700), as {@link EngineConfig#mtp()}.
         */
        MTP("draft-mtp"),
        /** {@code draft-simple}: the draft file as a separate draft model, NextN / MTP blocks or not. */
        MODEL("draft-simple"),
        /**
         * {@code ngram-simple}: prompt-lookup drafting from the target's own token history with no draft model
         * (vLLM {@code ngram_proposer.py}, llama.cpp {@code common_ngram_simple_draft},
         * {@code NgramDrafter}), aliased {@code prompt-lookup}.
         */
        NGRAM("ngram-simple"),
        /**
         * {@code ngram-map-k}: the n-gram map of the target's history, keys only (common/ngram-map.cpp), aliased
         * {@code ngram-map} and {@code map-k}.
         */
        NGRAM_MAP_K("ngram-map-k"),
        /**
         * {@code ngram-map-k4v}: the n-gram map with the 4 most frequent value m-grams per key, aliased
         * {@code map-k4v}.
         */
        NGRAM_MAP_K4V("ngram-map-k4v"),
        /** {@code ngram-mod}: the hashed n-gram index ({@code common_ngram_mod}), aliased {@code mod}. */
        NGRAM_MOD("ngram-mod"),
        /**
         * {@code ngram-cache}: drafting over the {@link com.eonmux.neuralfree.gguf.engine.draft.NgramCache} chain
         * (context, dynamic and static parts, with cache files in the {@code common_ngram_cache_save} layout,
         * {@link com.eonmux.neuralfree.gguf.engine.draft.NgramCacheFile}) (common/ngram-cache.cpp), as llama.cpp's
         * {@code ngram-cache} type.
         */
        NGRAM_CACHE("ngram-cache"),
        /**
         * {@code suffix}: suffix decoding over a suffix tree (vLLM {@code suffix_decoding.py},
         * {@code SuffixDrafter}), no draft model.
         */
        SUFFIX("suffix");

        private final String llamaName;

        DraftType(String llamaName) {
            this.llamaName = llamaName;
        }

        /** llama.cpp's name of the type ({@code auto} for automatic detection). */
        public String llamaName() {
            return llamaName;
        }

        /**
         * The type of a llama.cpp name ({@code auto}, {@code draft-mtp}, {@code draft-simple}, {@code ngram-simple},
         * {@code ngram-map-k}, {@code ngram-map-k4v}, {@code ngram-mod}, {@code ngram-cache}), vLLM's
         * {@code suffix}, or an alias ({@code mtp}; {@code simple}, {@code model}; {@code prompt-lookup};
         * {@code ngram-map}, {@code map-k}; {@code map-k4v}; {@code mod}), ignoring case.
         *
         * @throws IllegalArgumentException for any other value, naming the valid ones
         */
        public static DraftType parse(String value) {
            String s = Objects.requireNonNull(value, "draft type").trim().toLowerCase(Locale.ROOT);
            switch (s) {
                case "auto": return AUTO;
                case "draft-mtp": case "mtp": return MTP;
                case "draft-simple": case "simple": case "model": return MODEL;
                case "ngram-simple": case "prompt-lookup": return NGRAM;
                case "ngram-map-k": case "ngram-map": case "map-k": return NGRAM_MAP_K;
                case "ngram-map-k4v": case "map-k4v": return NGRAM_MAP_K4V;
                case "ngram-mod": case "mod": return NGRAM_MOD;
                case "ngram-cache": return NGRAM_CACHE;
                case "suffix": return SUFFIX;
                case "draft-eagle3":
                case "draft-dflash":
                    throw new IllegalArgumentException("draft type " + value + " is detected from the draft file's"
                            + " architecture; use auto");
                default:
                    throw new IllegalArgumentException("draft type " + value + " is invalid; use auto, draft-mtp,"
                            + " draft-simple, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache or"
                            + " suffix");
            }
        }

        /** Whether the type drafts from the target's own tokens or a suffix text, needing no draft file. */
        public boolean lookupWithoutDraftModel() {
            return this != AUTO && this != MTP && this != MODEL;
        }
    }

    private final BackendMode backend;
    private final String deviceFilter;
    private final SplitMode splitMode;
    private final List<String> gpuDevices;
    /** The remote compute servers the layer split adds as devices ({@link #PROP_RPC}); unmodifiable. */
    private final List<RpcServerSpec> rpcServers;
    private final double[] tensorSplit;
    private final int mainGpu;
    /** Whether a tensor split places whole experts on each rank ({@link #PROP_EXPERT_PARALLEL}). */
    private final boolean expertParallel;
    /** The prefill query-split rank count ({@link #PROP_PREFILL_CONTEXT_PARALLEL}); 1 runs whole prefills. */
    private final int prefillContextParallelSize;
    /** The expert-parallel load balancer ({@link EplbConfig}); off by default. */
    private final EplbConfig eplb;
    private final MemoryConfig memory;
    private final SamplerConfig samplerDefaults;
    private final long ramCacheMb;
    private final int vramDepth;
    private final NumaStrategy numa;
    /** Trunk weight bytes the load keeps in host-mapped memory (UVA); 0 is off. */
    private final long cpuOffloadMb;
    /** The parameter name segments of the host-mapped offload; empty offloads any tensor up to the budget. */
    private final List<String> cpuOffloadParams;
    /** The parameter name segments of a streamed layer that stream; empty streams every tensor. */
    private final List<String> streamParams;
    private final long gpuVramLimitMb;
    private final int gpuLayers;
    /** Whether the layers' KV caches stay on the device ({@link #PROP_KV_OFFLOAD}). */
    private final boolean kvOffload;
    private final List<TensorOverride> tensorOverrides;
    private final boolean cpuMoe;
    private final int nCpuMoe;
    private final MoeCacheConfig moeCache;
    private final String draftModelPath;
    private final List<DraftType> draftTypes;
    private final SpeculativeOptions speculativeOptions;
    private final LookaheadOptions lookaheadOptions;
    private final int draftMax;
    private final int draftMin;
    private final float draftPMin;
    private final boolean mtp;
    /** {@link Builder#mtp(boolean)} as set, without the self-speculation {@link DraftType#MTP} implies. */
    private final boolean mtpSet;
    private final List<LoraSpec> loras;
    private final LoraConfig loraConfig;
    private final List<ControlVectorSpec> controlVectors;
    private final int controlVectorLayerStart;
    private final int controlVectorLayerEnd;
    private final ModelSourceOptions sourceOptions;
    private final ActivationQuant activationQuant;
    private final boolean modelSamplingDefaults;
    private final String reasoningParser;
    private final String toolParser;
    private final ToolStrict toolStrict;
    private final boolean beamAlwaysConstrain;
    /** The dense FFN weights of the first N decoder layers stay in RAM, or {@link #N_CPU_LAYERS_UNSET}. */
    private final int nCpuFfn;
    /** The MoE expert-cache budget in MiB; 0 leaves it to the {@link MoeCacheConfig} default. */
    private final long moeCacheMb;
    /** Whether the expert cache is pooled over the GPUs of a layer split ({@link #PROP_MOE_CACHE_POOL}). */
    private final boolean moeCachePool;
    /** Whether the load keeps the experts in RAM when the dense weights alone fit the usable VRAM. */
    private final boolean fit;
    /** The pooling type override ({@link #PROP_POOLING}), or null for the GGUF's pooling. */
    private final PoolingKind pooling;
    /** The custom logits processor factory classes ({@link #PROP_LOGITS_PROCESSORS}); unmodifiable, empty for none. */
    private final List<String> logitsProcessors;
    /** How many layers the draft model's GPU load keeps on its device ({@link #PROP_DRAFT_N_GPU_LAYERS}). */
    private final int draftGpuLayers;
    /** The draft model's tensor placement overrides in match order (unmodifiable, empty for none). */
    private final List<TensorOverride> draftTensorOverrides;
    /** Whether every MoE layer of the draft model keeps its expert weights on the host. */
    private final boolean draftCpuMoe;
    /** How many of the first MoE layers of the draft model keep their expert weights on the host. */
    private final int draftNCpuMoe;
    /** The GPUs the draft model's own load runs over, or the lone {@link #DRAFT_NO_DEVICE}; empty follows the target. */
    private final List<String> draftDevices;

    private EngineConfig(Builder b, MemoryConfig memory) {
        this.backend = b.backend;
        this.deviceFilter = b.deviceFilter;
        this.splitMode = b.splitMode;
        this.gpuDevices = Collections.unmodifiableList(new ArrayList<>(b.gpuDevices));
        this.rpcServers = Collections.unmodifiableList(new ArrayList<>(b.rpcServers));
        this.tensorSplit = b.tensorSplit.clone();
        this.mainGpu = b.mainGpu;
        this.expertParallel = b.expertParallel;
        this.prefillContextParallelSize = b.prefillContextParallelSize;
        this.eplb = b.eplb;
        this.memory = memory;
        this.samplerDefaults = b.samplerDefaults;
        this.ramCacheMb = b.ramCacheMb;
        this.vramDepth = b.vramDepth;
        this.numa = b.numa;
        this.cpuOffloadMb = b.cpuOffloadMb;
        this.cpuOffloadParams = Collections.unmodifiableList(new ArrayList<>(b.cpuOffloadParams));
        this.streamParams = Collections.unmodifiableList(new ArrayList<>(b.streamParams));
        this.gpuVramLimitMb = b.gpuVramLimitMb;
        this.gpuLayers = b.gpuLayers;
        this.kvOffload = b.kvOffload;
        this.tensorOverrides = Collections.unmodifiableList(new ArrayList<>(b.tensorOverrides));
        this.cpuMoe = b.cpuMoe;
        this.nCpuMoe = b.nCpuMoe;
        this.moeCache = b.moeCache;
        this.draftModelPath = b.draftModelPath;
        this.draftTypes = b.draftTypes;
        this.speculativeOptions = b.speculativeOptions;
        this.lookaheadOptions = b.lookaheadOptions;
        this.draftMax = b.draftMax;
        this.draftMin = b.draftMin;
        this.draftPMin = b.draftPMin;
        this.mtp = b.mtp || EngineConfigRules.selfMtp(b);
        this.mtpSet = b.mtp;
        this.loras = Collections.unmodifiableList(new ArrayList<>(b.loras));
        this.loraConfig = b.loraConfig;
        this.controlVectors = Collections.unmodifiableList(new ArrayList<>(b.controlVectors));
        this.controlVectorLayerStart = b.controlVectorLayerStart;
        this.controlVectorLayerEnd = b.controlVectorLayerEnd;
        this.sourceOptions = b.sourceOptions;
        this.activationQuant = b.activationQuant;
        this.modelSamplingDefaults = b.modelSamplingDefaults;
        this.reasoningParser = b.reasoningParser;
        this.toolParser = b.toolParser;
        this.toolStrict = b.toolStrict;
        this.beamAlwaysConstrain = b.beamAlwaysConstrain;
        this.nCpuFfn = b.nCpuFfn;
        this.moeCacheMb = b.moeCacheMb;
        this.moeCachePool = b.moeCachePool;
        this.fit = b.fit;
        this.pooling = b.pooling;
        this.logitsProcessors = Collections.unmodifiableList(new ArrayList<>(b.logitsProcessors));
        this.draftGpuLayers = b.draftGpuLayers;
        this.draftTensorOverrides = Collections.unmodifiableList(new ArrayList<>(b.draftTensorOverrides));
        this.draftCpuMoe = b.draftCpuMoe;
        this.draftNCpuMoe = b.draftNCpuMoe;
        this.draftDevices = Collections.unmodifiableList(new ArrayList<>(b.draftDevices));
    }

    /**
     * Defaults: AUTO backend, no device filter, default memory and sampling, automatic streaming, no drafting, no
     * LoRA adapters, no control vectors, default source options, weight-only activations and the model's sampling
     * defaults applied.
     */
    public static EngineConfig defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Configuration from {@link System#getProperties()}. */
    public static EngineConfig fromSystemProperties() {
        return fromProperties(System.getProperties());
    }

    /**
     * Configuration from a property set; absent keys keep their defaults.
     *
     * @throws IllegalArgumentException naming the property when a value cannot be parsed or is out of range
     */
    public static EngineConfig fromProperties(Properties p) {
        return EngineProperties.read(p);
    }

    public BackendMode backend() { return backend; }
    /** Device-name substring for GPU backend selection, or null for none; unused by the CPU backend. */
    public String deviceFilter() { return deviceFilter; }
    /** How the model spreads over several GPUs ({@link SplitMode#NONE} runs one). */
    public SplitMode splitMode() { return splitMode; }
    /** The GPUs a split runs over, as selected ({@link #gpuDevices()} entries or all of one API); unmodifiable. */
    public List<String> gpuDevices() { return gpuDevices; }
    /**
     * The remote compute servers the layer split adds as devices, in configuration order after the local devices
     * ({@link #PROP_RPC}); unmodifiable, empty for none.
     */
    public List<RpcServerSpec> rpcServers() { return rpcServers; }
    /** The layer share of each GPU ({@link #PROP_TENSOR_SPLIT}); empty for equal shares. */
    public double[] tensorSplit() { return tensorSplit.clone(); }
    /**
     * Index into {@link #gpuDevices()} of the GPU that holds the embedding and the output head of a layer split; with
     * {@link SplitMode#NONE} the GPU the one-device load runs on ({@link #picksMainGpu()}).
     */
    public int mainGpu() { return mainGpu; }

    /**
     * True when a one-device load runs on the GPU {@code neuralfree.main.gpu} picks from the device list, as llama.cpp
     * keeps only {@code devices[main_gpu]} under {@code --split-mode none} (src/llama.cpp,
     * {@code llama_model_load_from_splits}): a single listed device index ({@link #isDeviceIndex}) in either split
     * mode, since a layer split over one device runs on it; and without a layer split, entry {@link #mainGpu()} of a
     * {@link #gpuDevices()} list of two or more, or, without a list, device {@link #mainGpu()} of the GPUs a split
     * would select when it is above 0 ({@code GpuDevices#openMainGpu}). False for a layer split over two or more
     * listed or selected devices, a single listed device-name substring (the {@link #deviceFilter()} picks it across
     * the APIs) and the default main GPU 0 without a list (the backend's ranking picks it).
     */
    public boolean picksMainGpu() {
        if (gpuDevices.size() == 1) {
            return isDeviceIndex(gpuDevices.get(0));
        }
        if (split()) {
            return false;
        }
        return gpuDevices.size() > 1 || mainGpu > 0;
    }

    /**
     * True when a {@code neuralfree.gpu.devices} entry is a device index (all digits) rather than a device-name
     * substring: the device's 0-based position inside its API's list, the index {@code GpuDevices} matches and
     * {@code --list-devices} prints.
     */
    public static boolean isDeviceIndex(String selector) {
        String s = Objects.requireNonNull(selector, "selector").trim();
        if (s.isEmpty()) {
            return false;
        }
        for (int i = 0; i < s.length(); i++) {
            if (!Character.isDigit(s.charAt(i))) {
                return false;
            }
        }
        return true;
    }
    /** True when a load may spread the model over several GPUs ({@link SplitMode#LAYER}); a load on a single-GPU
     * host then runs that one device, as {@code llama.cpp} does. */
    public boolean split() { return splitMode == SplitMode.LAYER; }
    /** True when the load splits every trunk layer's dense feed-forward over the ranks ({@link SplitMode#ROW}); a
     * load on a single-GPU host runs that one device, as llama.cpp does and as vLLM does for
     * {@code tensor_parallel_size=1}. */
    public boolean rowSplit() { return splitMode == SplitMode.ROW; }
    /** True when the load shards the attention and the dense feed-forward over the ranks
     * ({@link SplitMode#TENSOR}); a load on a single-GPU host runs that one device, as vLLM does for
     * {@code tensor_parallel_size=1}. */
    public boolean tensorMode() { return splitMode == SplitMode.TENSOR; }
    /**
     * True when the load divides the context positions between the ranks ({@link SplitMode#CONTEXT}); a load on a
     * single-GPU host runs that one device, as vLLM does for {@code decode_context_parallel_size=1}.
     */
    public boolean contextMode() { return splitMode == SplitMode.CONTEXT; }
    /**
     * The rank count every prefill's query rows split over ({@link #PROP_PREFILL_CONTEXT_PARALLEL}); 1 keeps every
     * rank running the whole prefill against its own key shard.
     */
    public int prefillContextParallelSize() { return prefillContextParallelSize; }
    /** True when the prefill query split is on ({@link #prefillContextParallelSize()} over 1). */
    public boolean prefillContextParallel() { return prefillContextParallelSize > 1; }
    /**
     * True when a tensor split places whole experts on each rank ({@link #PROP_EXPERT_PARALLEL}); false keeps every
     * rank holding an intermediate-width slice of every expert. Only meaningful with {@link #tensorMode()}.
     */
    public boolean expertParallel() { return expertParallel; }
    /** The expert-parallel load balancer ({@link EplbConfig}); off by default. */
    public EplbConfig eplb() { return eplb; }
    /** True for either tensor-parallel mode ({@link SplitMode#ROW} or {@link SplitMode#TENSOR}). */
    public boolean tensorParallel() { return splitMode == SplitMode.ROW || splitMode == SplitMode.TENSOR; }
    public MemoryConfig memory() { return memory; }
    /** Sampling used when a request does not bring its own. */
    public SamplerConfig samplerDefaults() { return samplerDefaults; }
    /** The default sampling seed ({@link SamplerConfig#RANDOM_SEED} for random). */
    public long seed() { return samplerDefaults.seed(); }
    /** Whether a full context shifts instead of stopping ({@link MemoryConfig#contextShift()}). */
    public boolean contextShift() { return memory.contextShift(); }
    /** Host cache for streamed weights in MiB, or {@link #RAM_CACHE_AUTO}. */
    public long ramCacheMb() { return ramCacheMb; }
    /** Layers kept resident in VRAM when streaming, or {@link #VRAM_DEPTH_AUTO}. */
    public int vramDepth() { return vramDepth; }
    /** The CPU pool's NUMA placement ({@link #PROP_NUMA}); {@link NumaStrategy#DISABLED} by default. */
    public NumaStrategy numaStrategy() { return numa; }
    /** MiB of trunk weights kept in host-mapped memory the device reads in place; 0 is off. */
    public long cpuOffloadMb() { return cpuOffloadMb; }
    /** {@link #cpuOffloadMb()} in bytes, 0 for off. */
    public long cpuOffloadBytes() { return Math.multiplyExact(cpuOffloadMb, 1024L * 1024L); }
    /** The parameter name segments of the host-mapped offload; empty offloads any tensor up to the budget. */
    public List<String> cpuOffloadParams() { return cpuOffloadParams; }
    /** The parameter name segments of a streamed layer that stream; empty streams every tensor (unmodifiable). */
    public List<String> streamParams() { return streamParams; }
    /**
     * MiB of device memory a GPU load may allocate on each of its devices, or {@link #GPU_VRAM_UNLIMITED}; the
     * placement, context and streaming fits see at most this much ({@code VRAMBudget#limitedBytes}).
     */
    public long gpuVramLimitMb() { return gpuVramLimitMb; }
    /** {@link #gpuVramLimitMb()} in bytes, 0 for no limit. */
    public long gpuVramLimitBytes() { return Math.multiplyExact(gpuVramLimitMb, 1024L * 1024L); }
    /**
     * How many layers a GPU load keeps on the device (llama.cpp {@code -ngl}): a count that covers the output
     * head, so {@code N} trunk layers on the device is {@code N + 1}, and a count above the model's layers plus
     * the output head is clamped to that full offload at load, as llama.cpp clamps {@code -ngl 99};
     * {@link #GPU_LAYERS_AUTO} (default) fits as many as the VRAM holds, {@link #GPU_LAYERS_ALL} keeps every
     * layer on the device.
     */
    public int gpuLayers() { return gpuLayers; }
    /**
     * Whether a GPU load keeps the KV caches on the device (llama.cpp {@code offload_kqv}, default true). With
     * {@code false} the layers' projections and FFN run on the device and their attention on the host over a
     * host-resident cache ({@link #PROP_KV_OFFLOAD}).
     */
    public boolean kvOffload() { return kvOffload; }
    /**
     * Tensor buffer overrides in match order (llama.cpp {@code -ot}); the first entry whose pattern matches a
     * tensor decides where its tensors are kept (unmodifiable, empty for none).
     */
    public List<TensorOverride> tensorOverrides() { return tensorOverrides; }
    /** Whether every mixture-of-experts layer keeps its expert weights on the host (llama.cpp {@code --cpu-moe}). */
    public boolean cpuMoe() { return cpuMoe; }
    /**
     * How many of the first MoE layers keep their expert weights on the host (llama.cpp
     * {@code --n-cpu-moe}); {@code <= 0} keeps none.
     */
    public int nCpuMoe() { return nCpuMoe; }
    /** The GPU MoE expert cache configuration ({@link MoeCacheConfig#off()} by default). */
    public MoeCacheConfig moeCache() { return moeCache; }
    /** Draft model path, or null. */
    public String draftModelPath() { return draftModelPath; }
    /**
     * The configured drafter types in llama.cpp's ask order ({@link DraftTypes#ordered}): the lookup types, then
     * the draft-file type when one is given; {@code [auto]} by default. Unmodifiable.
     */
    public List<DraftType> draftTypes() { return draftTypes; }
    /**
     * How the draft file drafts for its target: the draft-file member of {@link #draftTypes()}
     * ({@code auto}, {@code draft-mtp} or {@code draft-simple}), {@link DraftType#AUTO} when it has none.
     */
    public DraftType draftType() { return DraftTypes.fileType(draftTypes); }
    /** The lookup members of {@link #draftTypes()} (drafting without a draft file), in ask order; unmodifiable. */
    public List<DraftType> lookupTypes() { return DraftTypes.lookups(draftTypes); }
    /** The n-gram sizes, lookup cache files, suffix limits and verification of speculative decoding. */
    public SpeculativeOptions speculativeOptions() { return speculativeOptions; }
    /** The lookahead (Jacobi) decoding settings ({@link LookaheadOptions}, off by default). */
    public LookaheadOptions lookaheadOptions() { return lookaheadOptions; }
    /** Whether speculative decoding is configured: a draft model, NextN / MTP self-speculation, or a lookup type
     * that drafts from the target's own tokens or an external corpus without a draft file. */
    public boolean speculative() { return draftModelPath != null || mtp || !lookupTypes().isEmpty(); }
    /** Maximum tokens drafted per step. */
    public int draftMax() { return draftMax; }
    /** Minimum draft length worth verifying; shorter drafts are dropped. */
    public int draftMin() { return draftMin; }
    /** Minimum drafter probability of a greedy draft token; drafting stops below it. */
    public float draftPMin() { return draftPMin; }
    /**
     * Whether the model's own NextN / MTP layers draft for it: set explicitly, or {@link DraftType#MTP} without a
     * draft file.
     */
    public boolean mtp() { return mtp; }
    /** The LoRA adapters applied to the model, in load order (unmodifiable, empty for none). */
    public List<LoraSpec> loras() { return loras; }
    /** The settings of the model's per-request adapter registry ({@link LoraConfig#defaults()} unless set). */
    public LoraConfig loraConfig() { return loraConfig; }
    /** The control vectors applied to the model, in load order (unmodifiable, empty for none). */
    public List<ControlVectorSpec> controlVectors() { return controlVectors; }
    /**
     * The 1-based decoder layer the control-vector adds start at ({@code <= 0}: llama.cpp's default, layer 1,
     * {@code common/common.cpp:1484}).
     */
    public int controlVectorLayerStart() { return controlVectorLayerStart; }
    /**
     * The 1-based decoder layer the control-vector adds end at, inclusive ({@code <= 0}: llama.cpp's default, the
     * model's layer count, {@code common/common.cpp:1485}).
     */
    public int controlVectorLayerEnd() { return controlVectorLayerEnd; }
    /** How a checkpoint that is not GGUF is turned into its GGUF view (models, drafts and adapters alike). */
    public ModelSourceOptions sourceOptions() { return sourceOptions; }
    /** How a checkpoint's activation quantization is applied ({@link ActivationQuant#WEIGHT_ONLY} by default). */
    public ActivationQuant activationQuant() { return activationQuant; }
    /**
     * Whether generation applies the model's sampling defaults ({@link EngineModel#modelSamplingDefaults()}) to the
     * sampler settings and token budget a request leaves unset (default true; llama.cpp applies
     * {@code general.sampling.*}, vLLM its {@code generation_config}).
     */
    public boolean modelSamplingDefaults() { return modelSamplingDefaults; }
    /**
     * The reasoning parser that splits generated text into reasoning and content
     * ({@link com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers}), or {@code auto} for the chat
     * template's (vLLM {@code --reasoning-parser}).
     */
    public String reasoningParser() { return reasoningParser; }
    /**
     * The tool-call parser that reads calls out of generated content
     * ({@link com.eonmux.neuralfree.gguf.chat.tools.ToolParserRegistry}): {@code auto} detects it from the chat
     * template, {@code none} turns parsing off (vLLM {@code --tool-call-parser}).
     */
    public String toolParser() { return toolParser; }
    /**
     * The floor of the tool-call grammars ({@link ToolStrict}): how far the calls are constrained in the model's own
     * syntax and the argument schemas enforced (vLLM {@code --tool-strict-level}).
     */
    public ToolStrict toolStrict() { return toolStrict; }
    /** Whether {@code beam} constrains every output to its template's grammar ({@link #PROP_BEAM_ALWAYS_CONSTRAIN}). */
    public boolean beamAlwaysConstrain() { return beamAlwaysConstrain; }
    /**
     * Decoder layers whose dense FFN weights stay in RAM on the host (llama.cpp {@code --n-cpu-ffn}), or
     * {@link #N_CPU_LAYERS_UNSET}.
     */
    public int nCpuFfn() { return nCpuFfn; }
    /** The MoE expert-cache budget in MiB (llama.cpp {@code --moe-cache-mib}); 0 leaves it to {@link #moeCache()}. */
    public long moeCacheMb() { return moeCacheMb; }

    /**
     * Whether the expert cache is pooled over the GPUs of a layer split ({@link #PROP_MOE_CACHE_POOL}): each GPU
     * caches the routed layers assigned to it by its free VRAM, and each hosted MoE block runs on the GPU that caches
     * it.
     */
    public boolean moeCachePool() { return moeCachePool; }
    /**
     * Whether the load keeps the experts in RAM when the dense weights and the context fit the VRAM without them
     * (Avalon {@code --fit}), instead of streaming or refusing.
     */
    public boolean fit() { return fit; }

    /**
     * The pooling type override (llama.cpp {@code --pooling}, vLLM {@code --override-pooler-config}
     * {@code pooling_type}), or null for the GGUF's {@code <arch>.pooling_type}. {@code rank} refuses a model
     * without a classifier head when the model loads.
     */
    public PoolingKind pooling() { return pooling; }

    /**
     * The custom logits processor factory classes loaded beside the plugin providers (vLLM's
     * {@code --logits-processors}), as fully-qualified class names; unmodifiable and empty for none.
     */
    public List<String> logitsProcessors() { return logitsProcessors; }

    /**
     * How many layers the draft model's GPU load keeps on its device (llama.cpp {@code -ngld}): a count that
     * covers the output head like {@link #gpuLayers()}, {@link #GPU_LAYERS_AUTO} (default) fits as many as the
     * VRAM holds, {@link #GPU_LAYERS_ALL} keeps every layer on the device.
     */
    public int draftGpuLayers() { return draftGpuLayers; }
    /**
     * The tensor buffer overrides of the draft model's own load in match order (llama.cpp {@code -otd}); the first
     * entry whose pattern matches a tensor decides where its tensors are kept (unmodifiable, empty for none).
     */
    public List<TensorOverride> draftTensorOverrides() { return draftTensorOverrides; }
    /** Whether every MoE layer of the draft model keeps its expert weights on the host (llama.cpp {@code -cmoed}). */
    public boolean draftCpuMoe() { return draftCpuMoe; }
    /**
     * How many of the first MoE layers of the draft model keep their expert weights on the host (llama.cpp
     * {@code -ncmoed}); {@code <= 0} keeps none.
     */
    public int draftNCpuMoe() { return draftNCpuMoe; }
    /**
     * The GPUs the draft model's own load runs over (llama.cpp {@code -devd}), as {@link #PROP_DRAFT_GPU_DEVICES}
     * selectors, or the lone {@link #DRAFT_NO_DEVICE} entry that keeps the draft on the host; empty (default)
     * keeps the draft beside the target's weights on the target's backend (unmodifiable).
     */
    public List<String> draftDevices() { return draftDevices; }

    /**
     * True when a draft model placement setting is configured ({@link #draftGpuLayers()},
     * {@link #draftTensorOverrides()}, {@link #draftCpuMoe()}, {@link #draftNCpuMoe()}, {@link #draftDevices()}):
     * all of them apply to the load of a separate draft model only, so without {@link #draftModelPath()} they
     * are refused ({@link Builder#build()}).
     */
    public boolean hasDraftPlacement() {
        return draftGpuLayers != GPU_LAYERS_AUTO || !draftTensorOverrides.isEmpty() || draftCpuMoe
                || draftNCpuMoe > N_CPU_LAYERS_UNSET || !draftDevices.isEmpty();
    }

    /**
     * The sampler of the legacy {@code LLMModel.generate} parameters, the one mapping every orchestrator uses: the
     * configured {@link #samplerDefaults()} with the given temperature, top-p, min-p, top-k and repetition penalty
     * in place of theirs, so typical-p, the presence and frequency penalties, greedy decoding and the seed stay as
     * configured. Values pass through unchanged, as llama.cpp's sampler parameters do, except
     * {@code repeatWindow == 0}, which penalizes the whole context ({@link SamplerConfig#PENALTY_WHOLE_CONTEXT},
     * llama.cpp {@code penalty_last_n = -1}); {@code -1} also means the whole context and {@code n > 0} the last
     * {@code n} tokens. A repetition penalty of 1 disables it. The given values are explicit
     * ({@link SamplerConfig#explicit()}), so a model's sampling defaults fill only the other settings.
     *
     * @throws IllegalArgumentException for a value {@link SamplerConfig.Builder#build()} rejects: a non-finite
     *                                  temperature or penalty, top-p or min-p outside [0, 1], a penalty <= 0 or a
     *                                  window below -1
     */
    public SamplerConfig legacySampler(float temperature, float topP, float repetitionPenalty, int repeatWindow,
                                       float minP, int topK) {
        return samplerDefaults.toBuilder()
                .temperature(temperature)
                .topP(topP)
                .minP(minP)
                .topK(topK)
                .repeatPenalty(repetitionPenalty)
                .penaltyLastN(repeatWindow == 0 ? SamplerConfig.PENALTY_WHOLE_CONTEXT : repeatWindow)
                .build();
    }

    public Builder toBuilder() {
        return builder().backend(backend).deviceFilter(deviceFilter).splitMode(splitMode).gpuDevices(gpuDevices)
                .rpcServers(rpcServers)
                .tensorSplit(tensorSplit).mainGpu(mainGpu).memory(memory).samplerDefaults(samplerDefaults)
                .ramCacheMb(ramCacheMb).vramDepth(vramDepth).numa(numa)
                .cpuOffloadMb(cpuOffloadMb).cpuOffloadParams(cpuOffloadParams).streamParams(streamParams)
                .gpuVramLimitMb(gpuVramLimitMb).gpuLayers(gpuLayers)
                .kvOffload(kvOffload).tensorOverrides(tensorOverrides)
                .cpuMoe(cpuMoe).nCpuMoe(nCpuMoe).moeCache(moeCache).eplb(eplb).draftModelPath(draftModelPath)
                .draftTypes(draftTypes).speculativeOptions(speculativeOptions).lookaheadOptions(lookaheadOptions)
                .draftMax(draftMax).draftMin(draftMin).draftPMin(draftPMin).mtp(mtpSet).loras(loras)
                .loraConfig(loraConfig)
                .controlVectors(controlVectors).controlVectorLayerRange(controlVectorLayerStart,
                        controlVectorLayerEnd)
                .sourceOptions(sourceOptions).activationQuant(activationQuant)
                .modelSamplingDefaults(modelSamplingDefaults).reasoningParser(reasoningParser)
                .toolParser(toolParser).toolStrict(toolStrict).beamAlwaysConstrain(beamAlwaysConstrain)
                .nCpuFfn(nCpuFfn).moeCacheMb(moeCacheMb).moeCachePool(moeCachePool).fit(fit).pooling(pooling)
                .draftGpuLayers(draftGpuLayers).draftTensorOverrides(draftTensorOverrides)
                .draftCpuMoe(draftCpuMoe).draftNCpuMoe(draftNCpuMoe).draftDevices(draftDevices)
                .logitsProcessors(logitsProcessors);
    }

    @Override
    public String toString() {
        return EngineConfigText.print(this);
    }

    /** The parser name that follows the chat template. */
    public static final String PARSER_AUTO = "auto";
    /** The tool-parser name that turns tool-call parsing off. */
    public static final String PARSER_NONE = "none";

    public static final class Builder {
        BackendMode backend = BackendMode.AUTO;
        String deviceFilter;
        SplitMode splitMode = SplitMode.NONE;
        boolean expertParallel;
        int prefillContextParallelSize = 1;
        EplbConfig eplb = EplbConfig.off();
        List<String> gpuDevices = new ArrayList<>();
        final List<RpcServerSpec> rpcServers = new ArrayList<>();
        double[] tensorSplit = new double[0];
        int mainGpu;
        MemoryConfig memory = MemoryConfig.defaults();
        SamplerConfig samplerDefaults = SamplerConfig.defaults();
        long ramCacheMb = RAM_CACHE_AUTO;
        int vramDepth = VRAM_DEPTH_AUTO;
        NumaStrategy numa = NumaStrategy.DISABLED;
        long cpuOffloadMb;
        final List<String> cpuOffloadParams = new ArrayList<>();
        final List<String> streamParams = new ArrayList<>();
        long gpuVramLimitMb = GPU_VRAM_UNLIMITED;
        int gpuLayers = GPU_LAYERS_AUTO;
        boolean kvOffload = true;
        final List<TensorOverride> tensorOverrides = new ArrayList<>();
        boolean cpuMoe;
        int nCpuMoe = N_CPU_LAYERS_UNSET;
        MoeCacheConfig moeCache = MoeCacheConfig.off();
        String draftModelPath;
        List<DraftType> draftTypes = DraftTypes.DEFAULT;
        SpeculativeOptions speculativeOptions = SpeculativeOptions.defaults();
        LookaheadOptions lookaheadOptions = LookaheadOptions.disabled();
        int draftMax = 3;
        int draftMin;
        float draftPMin;
        boolean mtp;
        final List<LoraSpec> loras = new ArrayList<>();
        LoraConfig loraConfig = LoraConfig.defaults();
        final List<ControlVectorSpec> controlVectors = new ArrayList<>();
        int controlVectorLayerStart;
        int controlVectorLayerEnd;
        ModelSourceOptions sourceOptions = ModelSourceOptions.defaults();
        ActivationQuant activationQuant = ActivationQuant.WEIGHT_ONLY;
        boolean modelSamplingDefaults = true;
        String reasoningParser = PARSER_AUTO;
        String toolParser = PARSER_AUTO;
        ToolStrict toolStrict = ToolStrict.AUTO;
        boolean beamAlwaysConstrain = true;
        int nCpuFfn = N_CPU_LAYERS_UNSET;
        long moeCacheMb;
        boolean moeCachePool;
        boolean fit;
        PoolingKind pooling;
        final List<String> logitsProcessors = new ArrayList<>();
        int draftGpuLayers = GPU_LAYERS_AUTO;
        final List<TensorOverride> draftTensorOverrides = new ArrayList<>();
        boolean draftCpuMoe;
        int draftNCpuMoe = N_CPU_LAYERS_UNSET;
        final List<String> draftDevices = new ArrayList<>();

        private Builder() {
        }

        public Builder backend(BackendMode v) { this.backend = Objects.requireNonNull(v, "backend"); return this; }

        /** Device-name substring; null or blank removes the filter. */
        public Builder deviceFilter(String v) {
            this.deviceFilter = v == null || v.trim().isEmpty() ? null : v.trim();
            return this;
        }

        /** How the model spreads over several GPUs ({@link SplitMode}). */
        public Builder splitMode(SplitMode v) { this.splitMode = Objects.requireNonNull(v, "splitMode"); return this; }

        /** Whether a tensor split places whole experts on each rank ({@link #PROP_EXPERT_PARALLEL}). */
        public Builder expertParallel(boolean v) { this.expertParallel = v; return this; }

        /** The prefill query-split rank count ({@link #PROP_PREFILL_CONTEXT_PARALLEL}); 1 runs whole prefills. */
        public Builder prefillContextParallelSize(int v) {
            this.prefillContextParallelSize = v;
            return this;
        }

        /** The expert-parallel load balancer ({@link EplbConfig}); null keeps the default (off). */
        public Builder eplb(EplbConfig v) {
            this.eplb = v == null ? EplbConfig.off() : v;
            return this;
        }

        /** Replaces the GPU selector list of a split ({@code --device} entries in order). */
        public Builder gpuDevices(List<String> v) {
            Objects.requireNonNull(v, "gpuDevices");
            List<String> copy = new ArrayList<>(v.size());
            for (String s : v) {
                copy.add(Objects.requireNonNull(s, "gpu device"));
            }
            gpuDevices.clear();
            gpuDevices.addAll(copy);
            return this;
        }

        /** Appends one GPU selector to the split. */
        public Builder addGpuDevice(String selector) {
            gpuDevices.add(Objects.requireNonNull(selector, "selector"));
            return this;
        }

        /** Replaces the remote compute servers the layer split adds as devices ({@link #PROP_RPC}). */
        public Builder rpcServers(List<RpcServerSpec> v) {
            Objects.requireNonNull(v, "rpcServers");
            List<RpcServerSpec> copy = new ArrayList<>(v.size());
            for (RpcServerSpec spec : v) {
                copy.add(Objects.requireNonNull(spec, "rpc server"));
            }
            rpcServers.clear();
            rpcServers.addAll(copy);
            return this;
        }

        /** Appends one remote compute server to the layer split. */
        public Builder addRpcServer(RpcServerSpec spec) {
            rpcServers.add(Objects.requireNonNull(spec, "rpc server"));
            return this;
        }

        /**
         * Appends one remote compute server written as a {@code neuralfree.rpc} entry.
         *
         * @throws IllegalArgumentException when the entry is malformed or carries no key
         */
        public Builder addRpcServer(String entry) {
            return addRpcServer(RpcServerSpec.parse(entry));
        }

        /** Replaces the per-GPU layer shares; every value must be a finite number above 0. */
        public Builder tensorSplit(double[] v) {
            Objects.requireNonNull(v, "tensorSplit");
            for (double d : v) {
                if (!(d > 0.0) || !Double.isFinite(d)) {
                    throw new IllegalArgumentException(PROP_TENSOR_SPLIT + ": every share must be a finite value"
                            + " above 0, got " + d);
                }
            }
            tensorSplit = v.clone();
            return this;
        }

        /** Index into the GPU list of the GPU that holds the embedding and the output head. */
        public Builder mainGpu(int v) {
            if (v < 0) {
                throw new IllegalArgumentException(PROP_MAIN_GPU + " must be >= 0, got " + v);
            }
            mainGpu = v;
            return this;
        }

        public Builder memory(MemoryConfig v) { this.memory = Objects.requireNonNull(v, "memory"); return this; }
        public Builder samplerDefaults(SamplerConfig v) { this.samplerDefaults = Objects.requireNonNull(v, "samplerDefaults"); return this; }

        /** Sets the seed of the default sampler. */
        public Builder seed(long v) { this.samplerDefaults = samplerDefaults.withSeed(v); return this; }

        /** Sets {@link MemoryConfig#contextShift()}. */
        public Builder contextShift(boolean v) { this.memory = memory.toBuilder().contextShift(v).build(); return this; }

        public Builder ramCacheMb(long v) { this.ramCacheMb = v; return this; }
        public Builder vramDepth(int v) { this.vramDepth = v; return this; }

        /**
         * The CPU pool's NUMA placement ({@link #PROP_NUMA}); {@link NumaStrategy#DISABLED} when null.
         * The pool reads the property when it starts, so a programmatic setting also belongs in the property.
         */
        public Builder numa(NumaStrategy v) {
            this.numa = v == null ? NumaStrategy.DISABLED : v;
            return this;
        }

        /** The MiB of trunk weights kept in host-mapped memory (UVA zero-copy); 0 turns the offload off. */
        public Builder cpuOffloadMb(long v) { this.cpuOffloadMb = v; return this; }

        /**
         * The parameter name segments of the host-mapped offload (vLLM {@code cpu_offload_params}), replacing
         * the list; empty offloads any tensor up to the budget.
         *
         * @throws IllegalArgumentException naming {@link #PROP_CPU_OFFLOAD_PARAMS} for a blank segment or an
         *                                  empty dotted part
         */
        public Builder cpuOffloadParams(List<String> v) {
            EngineConfigRules.replaceSegments(cpuOffloadParams, v, PROP_CPU_OFFLOAD_PARAMS);
            return this;
        }

        /**
         * The parameter name segments of a streamed layer that stream (vLLM prefetch {@code offload_params}),
         * replacing the list; empty streams every tensor of a streamed layer.
         *
         * @throws IllegalArgumentException naming {@link #PROP_STREAM_PARAMS} for a blank segment or an empty
         *                                  dotted part
         */
        public Builder streamParams(List<String> v) {
            EngineConfigRules.replaceSegments(streamParams, v, PROP_STREAM_PARAMS);
            return this;
        }

        /** MiB of device memory the load may allocate on each device; {@link #GPU_VRAM_UNLIMITED} for none. */
        public Builder gpuVramLimitMb(long v) { this.gpuVramLimitMb = v; return this; }

        /** The {@code -ngl} value: a layer count, {@link #GPU_LAYERS_AUTO} or {@link #GPU_LAYERS_ALL}. */
        public Builder gpuLayers(int v) { this.gpuLayers = v; return this; }

        /** Whether a GPU load keeps the KV caches on the device ({@link #PROP_KV_OFFLOAD}); default true. */
        public Builder kvOffload(boolean v) {
            this.kvOffload = v;
            return this;
        }

        /** Replaces the tensor buffer override list ({@code -ot} entries in match order). */
        public Builder tensorOverrides(List<TensorOverride> v) {
            Objects.requireNonNull(v, "tensorOverrides");
            List<TensorOverride> copy = new ArrayList<>(v.size());
            for (TensorOverride o : v) {
                copy.add(Objects.requireNonNull(o, "tensor override"));
            }
            tensorOverrides.clear();
            tensorOverrides.addAll(copy);
            return this;
        }

        /** Appends one {@code -ot} entry. */
        public Builder addTensorOverride(String pattern, String target) {
            tensorOverrides.add(TensorOverride.parse(pattern, target));
            return this;
        }

        /** Whether every MoE layer keeps its expert weights on the host ({@code --cpu-moe}). */
        public Builder cpuMoe(boolean v) { this.cpuMoe = v; return this; }

        /**
         * How many of the first MoE layers keep their expert weights on the host ({@code --n-cpu-moe}); a
         * negative value clears it ({@link #N_CPU_LAYERS_UNSET}).
         */
        public Builder nCpuMoe(int v) { this.nCpuMoe = Math.max(v, N_CPU_LAYERS_UNSET); return this; }

        /** The GPU MoE expert cache configuration; null keeps the default (off). */
        public Builder moeCache(MoeCacheConfig v) {
            this.moeCache = v == null ? MoeCacheConfig.off() : v;
            return this;
        }


        /** Draft model path; null or blank removes it. */
        public Builder draftModelPath(String v) {
            this.draftModelPath = v == null || v.trim().isEmpty() ? null : v.trim();
            return this;
        }

        /** One drafter type ({@link DraftType}), replacing the type list. */
        public Builder draftType(DraftType v) {
            this.draftTypes = Collections.singletonList(Objects.requireNonNull(v, "draftType"));
            return this;
        }

        /**
         * The drafter types (llama.cpp {@code --spec-type a,b,c}), replacing the list; kept in llama.cpp's ask
         * order ({@link DraftTypes#ordered}).
         *
         * @throws IllegalArgumentException naming {@link #PROP_DRAFT_TYPE} for an empty list or two draft-file
         *                                  types
         */
        public Builder draftTypes(List<DraftType> v) {
            try {
                this.draftTypes = DraftTypes.ordered(v);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(PROP_DRAFT_TYPE + ": " + e.getMessage(), e);
            }
            return this;
        }

        /** The drafter and verifier settings ({@link SpeculativeOptions}). */
        public Builder speculativeOptions(SpeculativeOptions v) {
            this.speculativeOptions = Objects.requireNonNull(v, "speculativeOptions");
            return this;
        }

        /** The lookahead (Jacobi) decoding settings ({@link LookaheadOptions}); null restores off. */
        public Builder lookaheadOptions(LookaheadOptions v) {
            this.lookaheadOptions = v == null ? LookaheadOptions.disabled() : v;
            return this;
        }

        public Builder draftMax(int v) { this.draftMax = v; return this; }
        public Builder draftMin(int v) { this.draftMin = v; return this; }
        public Builder draftPMin(float v) { this.draftPMin = v; return this; }
        public Builder mtp(boolean v) { this.mtp = v; return this; }

        /** The per-request adapter registry settings; null restores {@link LoraConfig#defaults()}. */
        public Builder loraConfig(LoraConfig v) {
            this.loraConfig = v == null ? LoraConfig.defaults() : v;
            return this;
        }

        /** Replaces the LoRA adapter list ({@code --lora} / {@code --lora-scaled} entries in order). */
        public Builder loras(List<LoraSpec> v) {
            Objects.requireNonNull(v, "loras");
            List<LoraSpec> copy = new ArrayList<>(v.size());
            for (LoraSpec s : v) {
                copy.add(Objects.requireNonNull(s, "lora entry"));
            }
            loras.clear();
            loras.addAll(copy);
            return this;
        }

        /** Appends one LoRA adapter at {@code scale}. */
        public Builder addLora(String path, float scale) {
            loras.add(new LoraSpec(path, scale));
            return this;
        }

        /** Replaces the control-vector list ({@code --control-vector} / {@code --control-vector-scaled} entries). */
        public Builder controlVectors(List<ControlVectorSpec> v) {
            Objects.requireNonNull(v, "controlVectors");
            List<ControlVectorSpec> copy = new ArrayList<>(v.size());
            for (ControlVectorSpec s : v) {
                copy.add(Objects.requireNonNull(s, "control vector entry"));
            }
            controlVectors.clear();
            controlVectors.addAll(copy);
            return this;
        }

        /** Appends one control vector at {@code scale}. */
        public Builder addControlVector(String path, float scale) {
            controlVectors.add(new ControlVectorSpec(path, scale));
            return this;
        }

        /**
         * The 1-based decoder layer range the control vectors apply to, both inclusive; a value {@code <= 0} keeps
         * llama.cpp's default of that side (llama.cpp {@code --control-vector-layer-range START END}).
         */
        public Builder controlVectorLayerRange(int start, int end) {
            this.controlVectorLayerStart = start;
            this.controlVectorLayerEnd = end;
            return this;
        }

        /** How a checkpoint that is not GGUF is turned into its GGUF view. */
        public Builder sourceOptions(ModelSourceOptions v) {
            this.sourceOptions = Objects.requireNonNull(v, "sourceOptions");
            return this;
        }

        /** How a checkpoint's activation quantization is applied. */
        public Builder activationQuant(ActivationQuant v) {
            this.activationQuant = Objects.requireNonNull(v, "activationQuant");
            return this;
        }

        /** Whether generation applies the model's sampling defaults to the settings a request leaves unset. */
        public Builder modelSamplingDefaults(boolean v) { this.modelSamplingDefaults = v; return this; }

        /** The reasoning parser name; null or blank is {@code auto}. */
        public Builder reasoningParser(String v) { this.reasoningParser = EngineConfigRules.parserName(v); return this; }

        /** The tool-call parser name; null or blank is {@code auto}, {@code none} turns parsing off. */
        public Builder toolParser(String v) { this.toolParser = EngineConfigRules.parserName(v); return this; }

        /** The tool strict floor ({@code auto | function | parameter}); null or blank is {@code auto}. */
        public Builder toolStrict(ToolStrict v) { this.toolStrict = v == null ? ToolStrict.AUTO : v; return this; }

        /** The tool strict floor by name, as {@link #toolStrict(ToolStrict)} validates it. */
        public Builder toolStrict(String v) { return toolStrict(ToolStrict.fromName(v)); }

        /** Whether {@code beam} constrains every output to its response template's grammar; default true. */
        public Builder beamAlwaysConstrain(boolean v) { this.beamAlwaysConstrain = v; return this; }

        /**
         * Decoder layers whose dense FFN weights stay in RAM ({@code --n-cpu-ffn}); a negative value clears it
         * ({@link #N_CPU_LAYERS_UNSET}).
         */
        public Builder nCpuFfn(int v) { this.nCpuFfn = Math.max(v, N_CPU_LAYERS_UNSET); return this; }

        /** The MoE expert-cache budget in MiB ({@code --moe-cache-mib}); 0 leaves it to {@link #moeCache(MoeCacheConfig)}. */
        public Builder moeCacheMb(long v) { this.moeCacheMb = v; return this; }

        /** Whether the expert cache is pooled over the GPUs of a layer split ({@link #PROP_MOE_CACHE_POOL}). */
        public Builder moeCachePool(boolean v) { this.moeCachePool = v; return this; }

        /** Whether the load keeps the experts in RAM when the dense weights fit the VRAM without them. */
        public Builder fit(boolean v) { this.fit = v; return this; }

        /**
         * The pooling type override (llama.cpp {@code --pooling}); null clears it, so the GGUF's
         * {@code <arch>.pooling_type} runs.
         */
        public Builder pooling(PoolingKind v) { this.pooling = v; return this; }

        /**
         * Replaces the custom logits processor factory classes ({@link #PROP_LOGITS_PROCESSORS}); every entry must
         * be a non-blank, distinct fully-qualified class name.
         */
        public Builder logitsProcessors(List<String> v) {
            Objects.requireNonNull(v, "logitsProcessors");
            logitsProcessors.clear();
            for (String name : v) {
                String s = name == null ? null : name.trim();
                if (s == null || s.isEmpty()) {
                    throw new IllegalArgumentException(PROP_LOGITS_PROCESSORS + " holds a blank class name");
                }
                if (logitsProcessors.contains(s)) {
                    throw new IllegalArgumentException(PROP_LOGITS_PROCESSORS + " repeats " + s);
                }
                logitsProcessors.add(s);
            }
            return this;
        }

        /**
         * The {@code -ngld} value of the draft model's own load: a layer count that covers the output head,
         * {@link #GPU_LAYERS_AUTO} or {@link #GPU_LAYERS_ALL}.
         */
        public Builder draftGpuLayers(int v) { this.draftGpuLayers = v; return this; }

        /** Replaces the draft model's tensor buffer override list ({@code -otd} entries in match order). */
        public Builder draftTensorOverrides(List<TensorOverride> v) {
            Objects.requireNonNull(v, "draftTensorOverrides");
            List<TensorOverride> copy = new ArrayList<>(v.size());
            for (TensorOverride o : v) {
                copy.add(Objects.requireNonNull(o, "draft tensor override"));
            }
            draftTensorOverrides.clear();
            draftTensorOverrides.addAll(copy);
            return this;
        }

        /** Appends one {@code -otd} entry. */
        public Builder addDraftTensorOverride(String pattern, String target) {
            draftTensorOverrides.add(TensorOverride.parse(pattern, target));
            return this;
        }

        /** Whether every MoE layer of the draft model keeps its expert weights on the host ({@code -cmoed}). */
        public Builder draftCpuMoe(boolean v) { this.draftCpuMoe = v; return this; }

        /**
         * How many of the first MoE layers of the draft model keep their expert weights on the host
         * ({@code -ncmoed}); a negative value clears it ({@link #N_CPU_LAYERS_UNSET}).
         */
        public Builder draftNCpuMoe(int v) { this.draftNCpuMoe = Math.max(v, N_CPU_LAYERS_UNSET); return this; }

        /**
         * Replaces the GPU selector list of the draft model's own load ({@code -devd} entries), or the lone
         * {@link #DRAFT_NO_DEVICE} entry that keeps the draft on the host.
         */
        public Builder draftDevices(List<String> v) {
            Objects.requireNonNull(v, "draftDevices");
            List<String> copy = new ArrayList<>(v.size());
            for (String s : v) {
                copy.add(Objects.requireNonNull(s, "draft device"));
            }
            draftDevices.clear();
            draftDevices.addAll(copy);
            return this;
        }

        /** Appends one GPU selector, or the lone {@link #DRAFT_NO_DEVICE}, to the draft model's own load. */
        public Builder addDraftDevice(String selector) {
            draftDevices.add(Objects.requireNonNull(selector, "draft device selector"));
            return this;
        }

        /**
         * Builds the configuration. The memory settings receive the speculative defaults of the draft model and MTP
         * settings ({@link MemoryConfig#forSpeculation}) and the compressor rollback snapshots its drafter needs
         * ({@link MemoryConfig#forRollback}, llama.cpp's {@code need_n_rs_seq}, common/common.h:401-407): the draft
         * max for the model's own NextN / MTP blocks and for a draft file under {@link DraftType#MTP} or
         * {@link DraftType#AUTO}, 0 under {@link DraftType#MODEL} or without a drafter. A file under auto may still
         * draft as a separate draft model; {@link LLMModelFactory} resolves and pins that value from the file before
         * loading. Memory values set explicitly are kept.
         *
         * @throws IllegalArgumentException naming the setting when a value is out of range
         */
        public EngineConfig build() {
            return new EngineConfig(this, EngineConfigRules.validated(this));
        }
    }
}
