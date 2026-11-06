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
