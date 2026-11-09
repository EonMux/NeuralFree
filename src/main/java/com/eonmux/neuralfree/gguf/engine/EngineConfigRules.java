package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.chat.response.TemplateParsing;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.moe.MoeCacheRanking;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static com.eonmux.neuralfree.gguf.engine.EngineConfig.DRAFT_NO_DEVICE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.GPU_LAYERS_AUTO;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.N_CPU_LAYERS_UNSET;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PARSER_AUTO;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PARSER_NONE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.RAM_CACHE_AUTO;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_BACKEND;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_CPU_MOE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_CPU_OFFLOAD_MB;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_CPU_OFFLOAD_PARAMS;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_CPU_MOE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_GPU_DEVICES;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_MAX;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_MIN;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_N_CPU_MOE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_N_GPU_LAYERS;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_P_MIN;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_TENSOR_OVERRIDES;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_DRAFT_TYPE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_EXPERT_PARALLEL;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_GPU_DEVICES;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_GPU_VRAM_LIMIT;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_MAIN_GPU;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_MOE_CACHE_MB;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_MOE_CACHE_POOL;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_MTP;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_N_CPU_FFN;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_N_CPU_MOE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_RAM_CACHE_MB;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_REASONING_PARSER;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_RPC;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_SPLIT_MODE;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_STREAM_PARAMS;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_TENSOR_OVERRIDES;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_TOOL_PARSER;
import static com.eonmux.neuralfree.gguf.engine.EngineConfig.PROP_VRAM_DEPTH;

/**
 * The build-time rules of {@link EngineConfig}: the cross-setting refusals {@link EngineConfig.Builder#build()}
 * applies (llama.cpp's common/arg.cpp checks), and the memory settings the drafting configuration derives — the
 * speculative defaults of the draft model and MTP settings ({@link MemoryConfig#forSpeculation}) and the compressor
 * rollback snapshots the drafter needs ({@link MemoryConfig#forRollback}). Also the parameter-segment and
 * parser-name rules the builder's setters share.
 */
final class EngineConfigRules {

    private EngineConfigRules() {
    }

    /**
     * Validates {@code b} and derives the memory settings of the built configuration: the speculative defaults of
     * the draft model and MTP settings ({@link MemoryConfig#forSpeculation}) and the compressor rollback snapshots
     * its drafter needs ({@link MemoryConfig#forRollback}, llama.cpp's {@code need_n_rs_seq}, common/common.h:401-407):
     * the draft max for the model's own NextN / MTP blocks and for a draft file under {@code draft-mtp} or {@code
     * auto}, 0 under {@code draft-simple} or without a drafter. A file under auto may still draft as a separate
     * draft model; {@link LLMModelFactory} resolves and pins that value from the file before loading. Memory values
     * set explicitly are kept.
     *
     * @throws IllegalArgumentException naming the setting when a value is out of range
     */
    static MemoryConfig validated(EngineConfig.Builder b) {
        if (!b.rpcServers.isEmpty() && b.splitMode != EngineConfig.SplitMode.LAYER) {
            throw new IllegalArgumentException(PROP_RPC + " adds remote devices to a layer split; set "
                    + PROP_SPLIT_MODE + "=" + EngineConfig.SplitMode.LAYER.llamaName() + ", got "
                    + b.splitMode.llamaName());
        }
        if (!b.rpcServers.isEmpty() && !b.backend.usesGpu()) {
            throw new IllegalArgumentException(PROP_RPC + " adds remote GPU devices; it cannot run with "
                    + PROP_BACKEND + "=cpu");
        }
        if (b.ramCacheMb < RAM_CACHE_AUTO) {
            throw new IllegalArgumentException(PROP_RAM_CACHE_MB + " must be >= 0, or -1 for automatic, got " + b.ramCacheMb);
        }
        if (b.gpuVramLimitMb < 0) {
            throw new IllegalArgumentException(PROP_GPU_VRAM_LIMIT + " must be >= 0 MiB (0 = no limit), got "
                    + b.gpuVramLimitMb);
        }
        if (b.vramDepth < 0) {
            throw new IllegalArgumentException(PROP_VRAM_DEPTH + " must be >= 0 (0 = automatic), got " + b.vramDepth);
        }
        if (b.cpuOffloadMb < 0) {
            throw new IllegalArgumentException(PROP_CPU_OFFLOAD_MB + " must be >= 0 (0 = off), got "
                    + b.cpuOffloadMb);
        }
        if (b.cpuOffloadMb > Long.MAX_VALUE / (1024L * 1024L)) {
            throw new IllegalArgumentException(PROP_CPU_OFFLOAD_MB + "=" + b.cpuOffloadMb + " does not fit a byte"
                    + " count");
        }
        if (b.cpuOffloadMb == 0 && !b.cpuOffloadParams.isEmpty()) {
            throw new IllegalArgumentException(PROP_CPU_OFFLOAD_PARAMS + " names " + b.cpuOffloadParams
                    + " but " + PROP_CPU_OFFLOAD_MB + " is 0; set a host-mapped offload budget in MiB");
        }
        if (b.nCpuMoe < N_CPU_LAYERS_UNSET) {
            throw new IllegalArgumentException(PROP_N_CPU_MOE + " must be >= 0, got " + b.nCpuMoe);
        }
        if (b.mainGpu < 0) {
            throw new IllegalArgumentException(PROP_MAIN_GPU + " must be >= 0, got " + b.mainGpu);
        }
        if (b.splitMode != EngineConfig.SplitMode.LAYER && !b.gpuDevices.isEmpty() && b.mainGpu >= b.gpuDevices.size()) {
            throw new IllegalArgumentException(PROP_MAIN_GPU + "=" + b.mainGpu + " is beyond the "
                    + b.gpuDevices.size() + " device(s) of " + PROP_GPU_DEVICES + " " + b.gpuDevices
                    + " (indices 0.." + (b.gpuDevices.size() - 1) + ")");
        }
        if ((b.splitMode == EngineConfig.SplitMode.LAYER || b.splitMode == EngineConfig.SplitMode.ROW
                || b.splitMode == EngineConfig.SplitMode.TENSOR || b.splitMode == EngineConfig.SplitMode.CONTEXT)
                && b.backend == EngineConfig.BackendMode.CPU) {
            throw new IllegalArgumentException(PROP_SPLIT_MODE + "=" + b.splitMode.llamaName()
                    + " spreads the model over GPUs, but " + PROP_BACKEND + "=cpu runs none");
        }
        if (b.expertParallel && b.splitMode != EngineConfig.SplitMode.TENSOR) {
            throw new IllegalArgumentException(PROP_EXPERT_PARALLEL + "=true places whole experts per rank of a"
                    + " tensor split, but " + PROP_SPLIT_MODE + "=" + b.splitMode.llamaName());
        }
        if (b.prefillContextParallelSize < 1) {
            throw new IllegalArgumentException(PROP_PREFILL_CONTEXT_PARALLEL + " must be >= 1, got "
                    + b.prefillContextParallelSize);
        }
        if (b.prefillContextParallelSize > 1 && b.splitMode != EngineConfig.SplitMode.CONTEXT) {
            throw new IllegalArgumentException(PROP_PREFILL_CONTEXT_PARALLEL + "="
                    + b.prefillContextParallelSize + " splits every prefill's query rows over the ranks of a"
                    + " context split, but " + PROP_SPLIT_MODE + "=" + b.splitMode.llamaName() + "; set "
                    + PROP_SPLIT_MODE + "=context or keep " + PROP_PREFILL_CONTEXT_PARALLEL + "=1");
        }
        if (b.expertParallel && (b.cpuMoe || b.nCpuMoe != N_CPU_LAYERS_UNSET || !b.tensorOverrides.isEmpty()
                || b.moeCache.enabled())) {
            throw new IllegalArgumentException(PROP_EXPERT_PARALLEL + "=true needs every routed expert resident"
                    + " on its rank, but the expert offload settings (" + PROP_CPU_MOE + ", "
                    + PROP_N_CPU_MOE + ", " + PROP_TENSOR_OVERRIDES + ", " + PROP_MOE_CACHE_MB
                    + ") keep experts in host RAM");
        }
        if (b.eplb.enabled() && !b.expertParallel) {
            throw new IllegalArgumentException(EplbConfig.PROP_EPLB + "=on balances the whole experts of expert"
                    + " parallelism, but " + PROP_EXPERT_PARALLEL + " is off");
        }
        if (b.eplb.enabled() && b.eplb.profile() == null && b.eplb.save() == null && !b.eplb.rebalance()) {
            throw new IllegalArgumentException(EplbConfig.PROP_EPLB + "=on needs " + EplbConfig.PROP_EPLB_PROFILE
                    + " (a recorded load to balance from), " + EplbConfig.PROP_EPLB_SAVE
                    + " (to record this run's load) or " + EplbConfig.PROP_EPLB_REBALANCE
                    + "=on (to rearrange from the live load)");
        }
        if (b.eplb.policy() == EplbConfig.Policy.DEFAULT && b.eplb.numNodes() != 1) {
            throw new IllegalArgumentException(EplbConfig.PROP_EPLB_NUM_NODES + " groups the experts of the"
                    + " hierarchical policy; give " + EplbConfig.PROP_EPLB_POLICY + "=hierarchical");
        }
        if (b.eplb.save() != null && b.moeCache.ranking().savePath() != null) {
            throw new IllegalArgumentException(EplbConfig.PROP_EPLB_SAVE + " and " + MoeCacheRanking.PROP_SAVE
                    + " both write a routed-row profile, but the forward path records into one; keep one");
        }
        if (b.nCpuFfn < N_CPU_LAYERS_UNSET) {
            throw new IllegalArgumentException(PROP_N_CPU_FFN + " must be >= 0, got " + b.nCpuFfn);
        }
        if (b.moeCacheMb < 0) {
            throw new IllegalArgumentException(PROP_MOE_CACHE_MB + " must be >= 0, got " + b.moeCacheMb);
        }
        if (b.moeCachePool && !(b.moeCache.enabled() || b.moeCacheMb > 0)) {
            throw new IllegalArgumentException(PROP_MOE_CACHE_POOL + "=true pools the expert cache over the"
                    + " GPUs of a layer split, but neither " + MoeCacheConfig.PROP_MOE_CACHE + " nor "
                    + PROP_MOE_CACHE_MB + " enables the cache");
        }
        if (b.moeCachePool && b.splitMode != EngineConfig.SplitMode.LAYER) {
            throw new IllegalArgumentException(PROP_MOE_CACHE_POOL + "=true divides the routed MoE layers over"
                    + " the GPUs of a " + PROP_SPLIT_MODE + "=layer load, but " + PROP_SPLIT_MODE + "="
                    + b.splitMode.llamaName() + "; list two or more --device GPUs or set " + PROP_SPLIT_MODE
                    + "=layer");
        }
        if (b.draftMax <= 0) {
            throw new IllegalArgumentException(PROP_DRAFT_MAX + " must be positive, got " + b.draftMax);
        }
        if (b.draftMin < 0 || b.draftMin > b.draftMax) {
            throw new IllegalArgumentException(PROP_DRAFT_MIN + " must be in [0, " + PROP_DRAFT_MAX + "=" + b.draftMax
                    + "], got " + b.draftMin);
        }
        if (!(b.draftPMin >= 0f && b.draftPMin <= 1f)) {
            throw new IllegalArgumentException(PROP_DRAFT_P_MIN + " must be in [0, 1], got " + b.draftPMin);
        }
        if (b.speculativeOptions.heterogeneousVocab()) {
            if (b.draftModelPath == null) {
                throw new IllegalArgumentException(SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB + " applies to a"
                        + " separate draft model, but " + PROP_DRAFT + " names none");
            }
            if (b.speculativeOptions.probabilisticDraft()) {
                throw new IllegalArgumentException(SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB + " supports greedy"
                        + " drafting only; " + SpeculativeOptions.PROP_DRAFT_SAMPLE_METHOD + "=probabilistic"
                        + " needs a shared vocabulary");
            }
            if (DraftTypes.fileType(b.draftTypes) == EngineConfig.DraftType.MTP) {
                throw new IllegalArgumentException(SpeculativeOptions.PROP_HETEROGENEOUS_VOCAB + " needs a"
                        + " separate draft model (draft-simple), but " + PROP_DRAFT_TYPE + "="
                        + DraftTypes.names(b.draftTypes) + " drafts with the target's NextN / MTP blocks");
            }
        }
        checkDrafters(b);
        if (b.draftNCpuMoe < N_CPU_LAYERS_UNSET) {
            throw new IllegalArgumentException(PROP_DRAFT_N_CPU_MOE + " must be >= 0, got " + b.draftNCpuMoe);
        }
        checkDraftDevices(b);
        if (draftPlacementSet(b) && b.draftModelPath == null) {
            throw new IllegalArgumentException(draftPlacementRefusal(b));
        }
        if (!b.reasoningParser.equals(PARSER_AUTO)
                && com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers.forName(b.reasoningParser) == null) {
            throw new IllegalArgumentException(PROP_REASONING_PARSER + "=" + b.reasoningParser
                    + " is not a reasoning parser; use auto or one of "
                    + com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers.names());
        }
        if (!b.toolParser.equals(PARSER_AUTO) && !b.toolParser.equals(PARSER_NONE)
                && com.eonmux.neuralfree.gguf.chat.tools.ToolParserRegistry.forName(b.toolParser) == null) {
            throw new IllegalArgumentException(PROP_TOOL_PARSER + "=" + b.toolParser
                    + " is not a tool-call parser; use auto, none or one of "
                    + com.eonmux.neuralfree.gguf.chat.tools.ToolParserRegistry.names());
        }
        checkTemplateParsers(b.reasoningParser, b.toolParser);
        checkLookahead(b);
        boolean ownHead = b.mtp || selfMtp(b);
        EngineConfig.DraftType fileType = DraftTypes.fileType(b.draftTypes);
        int rollback = ownHead || b.draftModelPath != null && fileType != EngineConfig.DraftType.MODEL
                ? b.draftMax : 0;
        return b.memory.forSpeculation(b.draftModelPath != null || ownHead, ownHead).forRollback(rollback);
    }

    /**
     * Refuses drafter settings nothing would read: a draft file beside a type list without a draft-file type,
     * {@code draft-simple} with the MTP head and no draft file, and lookup cache files without {@code ngram-cache}.
     */
    private static void checkDrafters(EngineConfig.Builder b) {
        EngineConfig.DraftType fileType = DraftTypes.fileType(b.draftTypes);
        boolean fileListed = !b.draftTypes.get(b.draftTypes.size() - 1).lookupWithoutDraftModel();
        if (b.draftModelPath != null && !fileListed) {
            throw new IllegalArgumentException(PROP_DRAFT + " is set, but " + PROP_DRAFT_TYPE + "="
                    + DraftTypes.names(b.draftTypes) + " has no draft-file type to run it; add auto, draft-simple"
                    + " or draft-mtp to the list");
        }
        if (fileType == EngineConfig.DraftType.MODEL && b.draftModelPath == null && b.mtp) {
            throw new IllegalArgumentException(PROP_DRAFT_TYPE + "=" + fileType.llamaName() + " runs a separate"
                    + " draft model (" + PROP_DRAFT + "), but none is set and " + PROP_MTP + " drafts with the"
                    + " model's own NextN / MTP blocks");
        }
        if (b.speculativeOptions.hasLookupCacheFiles() && !b.draftTypes.contains(EngineConfig.DraftType.NGRAM_CACHE)) {
            throw new IllegalArgumentException(SpeculativeOptions.PROP_LOOKUP_CACHE_STATIC + " / "
                    + SpeculativeOptions.PROP_LOOKUP_CACHE_DYNAMIC + " are read by the ngram-cache drafter, but "
                    + PROP_DRAFT_TYPE + "=" + DraftTypes.names(b.draftTypes) + " does not include ngram-cache");
        }
    }

    /**
     * Refuses lookahead settings the memory or the drafters cannot serve: lookahead decoding replaces
     * speculative decoding, runs over paged memory with {@code W + G + 1} sequences that share the context
     * ({@code neuralfree.kv.unified}) and manages the stored positions itself.
     */
    private static void checkLookahead(EngineConfig.Builder b) {
        if (!b.lookaheadOptions.enabled()) {
            return;
        }
        if (b.draftModelPath != null || b.mtp || selfMtp(b) || !DraftTypes.lookups(b.draftTypes).isEmpty()) {
            throw new IllegalArgumentException(LookaheadOptions.PROP_ENABLED + " and speculative decoding"
                    + " exclude each other: lookahead decoding drafts and verifies with the target's own"
                    + " Jacobi guesses, so clear " + PROP_DRAFT + ", " + PROP_MTP + " and " + PROP_DRAFT_TYPE
                    + " (or turn " + LookaheadOptions.PROP_ENABLED + " off)");
        }
        if (!b.memory.paged()) {
            throw new IllegalArgumentException(LookaheadOptions.PROP_ENABLED + " needs paged memory: set "
                    + MemoryConfig.PROP_PARALLEL + "=" + b.lookaheadOptions.sequenceCount() + " (or "
                    + MemoryConfig.PROP_KV_PAGED + "=true)");
        }
        if (!b.memory.kvUnified()) {
            throw new IllegalArgumentException(LookaheadOptions.PROP_ENABLED + " runs " + b.lookaheadOptions
                    .sequenceCount() + " sequences over one context: set " + MemoryConfig.PROP_KV_UNIFIED
                    + "=true (each sequence spans the whole context)");
        }
        if (b.memory.parallel() < b.lookaheadOptions.sequenceCount()) {
            throw new IllegalArgumentException(LookaheadOptions.PROP_ENABLED + " with window "
                    + b.lookaheadOptions.window() + " and max n-grams " + b.lookaheadOptions.maxNgrams() + " needs "
                    + MemoryConfig.PROP_PARALLEL + "=" + b.lookaheadOptions.sequenceCount() + ", got "
                    + b.memory.parallel());
        }
        if (b.memory.contextShift()) {
            throw new IllegalArgumentException(LookaheadOptions.PROP_ENABLED + " manages the stored positions of"
                    + " its step and cannot shift the context; clear " + MemoryConfig.PROP_CONTEXT_SHIFT);
        }
    }

    /** {@link EngineConfig#DRAFT_NO_DEVICE} keeps the draft model off every GPU and cannot be listed beside devices. */
    private static void checkDraftDevices(EngineConfig.Builder b) {
        for (String d : b.draftDevices) {
            if (!d.equalsIgnoreCase(DRAFT_NO_DEVICE)) {
                continue;
            }
            if (b.draftDevices.size() != 1) {
                throw new IllegalArgumentException(PROP_DRAFT_GPU_DEVICES + " " + b.draftDevices + ": "
                        + DRAFT_NO_DEVICE + " keeps the draft model off every GPU and cannot be listed beside"
                        + " devices");
            }
        }
    }

    /** The draft placement settings configured against the builder's fields ({@link EngineConfig#hasDraftPlacement()}). */
    private static boolean draftPlacementSet(EngineConfig.Builder b) {
        return b.draftGpuLayers != GPU_LAYERS_AUTO || !b.draftTensorOverrides.isEmpty() || b.draftCpuMoe
                || b.draftNCpuMoe > N_CPU_LAYERS_UNSET || !b.draftDevices.isEmpty();
    }

    /** Names the draft placement settings configured without a draft model to apply them to. */
    private static String draftPlacementRefusal(EngineConfig.Builder b) {
        List<String> set = new ArrayList<>();
        if (b.draftGpuLayers != GPU_LAYERS_AUTO) {
            set.add(PROP_DRAFT_N_GPU_LAYERS);
        }
        if (!b.draftTensorOverrides.isEmpty()) {
            set.add(PROP_DRAFT_TENSOR_OVERRIDES);
        }
        if (b.draftCpuMoe) {
            set.add(PROP_DRAFT_CPU_MOE);
        }
        if (b.draftNCpuMoe > N_CPU_LAYERS_UNSET) {
            set.add(PROP_DRAFT_N_CPU_MOE);
        }
        if (!b.draftDevices.isEmpty()) {
            set.add(PROP_DRAFT_GPU_DEVICES);
        }
        return String.join(", ", set) + " (llama.cpp -ngld, -otd, -cmoed, -ncmoed, -devd) place the load of a"
                + " separate draft model, but no draft model is set (" + PROP_DRAFT + ")";
    }

    /**
     * A response-template parser ({@code hf}, {@code beam}) parses the reasoning and the tool calls alike, so beside it
     * the other parser is unset ({@code auto}, {@code none}) or the same one, as vLLM's parser manager requires.
     *
     * @throws IllegalArgumentException naming both parsers
     */
    private static void checkTemplateParsers(String reasoningParser, String toolParser) {
        boolean reasoningSet = !isUnset(reasoningParser);
        boolean toolSet = !isUnset(toolParser);
        boolean template = TemplateParsing.names(reasoningParser)
                || TemplateParsing.names(toolParser);
        if (template && reasoningSet && toolSet && !reasoningParser.equalsIgnoreCase(toolParser)) {
            throw new IllegalArgumentException(PROP_REASONING_PARSER + "=" + reasoningParser + " and "
                    + PROP_TOOL_PARSER + "=" + toolParser + " cannot be combined: a response-template parser"
                    + " (hf, beam) parses the reasoning and the tool calls alike");
        }
    }

    private static boolean isUnset(String parser) {
        return parser.equalsIgnoreCase(PARSER_AUTO) || parser.equalsIgnoreCase(PARSER_NONE)
                || parser.equalsIgnoreCase("identity");
    }

    /** {@link EngineConfig.DraftType#MTP} without a draft file drafts with the model's own NextN / MTP blocks. */
    static boolean selfMtp(EngineConfig.Builder b) {
        return DraftTypes.fileType(b.draftTypes) == EngineConfig.DraftType.MTP && b.draftModelPath == null;
    }

    /** The parser name of a setter value: null or blank is {@code auto}. */
    static String parserName(String v) {
        return v == null || v.trim().isEmpty() ? PARSER_AUTO : v.trim();
    }

    /** Replaces a parameter-segment list, refusing a blank segment or an empty dotted part by property name. */
    static void replaceSegments(List<String> target, List<String> v, String property) {
        Objects.requireNonNull(v, property);
        List<String> copy = new ArrayList<>(v.size());
        for (String segment : v) {
            copy.add(checkSegment(segment, property));
        }
        target.clear();
        target.addAll(copy);
    }

    /** One parameter name segment: non-blank, and no empty part between its dots. */
    private static String checkSegment(String segment, String property) {
        if (segment == null || segment.trim().isEmpty()) {
            throw new IllegalArgumentException(property + " has a blank parameter segment");
        }
        String s = segment.trim();
        for (String part : s.split("\\.", -1)) {
            if (part.isEmpty()) {
                throw new IllegalArgumentException(property + " segment '" + s + "' has an empty dotted part;"
                        + " use whole name parts, for example experts or ffn_gate_exps");
            }
        }
        return s;
    }
}
