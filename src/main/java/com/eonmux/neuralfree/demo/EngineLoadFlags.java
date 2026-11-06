package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.cpu.numa.NumaStrategy;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.EplbConfig;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.memory.KvCacheTypes;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.utils.json.Json;

import java.math.BigInteger;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * The model-load flags {@link LLMDemo} and the batched tools ({@code nftool batched-bench}, {@link BatchedDemo},
 * {@link ParallelDemo}) share: the backend and device choice ({@code --backend}, {@code --cpu}/{@code --gpu},
 * {@code --device}), the multi-GPU layer split ({@code --split-mode}, {@code --tensor-split}, {@code --main-gpu};
 * {@link SplitFlags}), the weight placement ({@code -ngl}, {@code -ot}, {@code --cpu-moe} and the other MoE and
 * streaming flags), the CPU thread pool ({@code -t}), its NUMA placement ({@code --numa}), the tensor load mode and
 * RAM pinning ({@code -lm}, {@code --mlock}), flash attention ({@code -fa}) and the KV cache types
 * ({@code --kv-type}, {@code -ctk}, {@code -ctv}). Flag names follow llama.cpp ({@code common/arg.cpp}). The
 * expert-parallel placement ({@code --enable-expert-parallel}) and the expert-parallel load balancer
 * ({@code --enable-eplb}, its {@code --eplb-*} paths and policy and vLLM's {@code --eplb-config} JSON) belong to
 * the tensor split, and vLLM's context-parallel launch flags ({@code -dcp}, {@code -pcp},
 * {@code --dcp-kv-cache-interleave-size}) to the context split. Enabling the balancer turns the runtime
 * rearrangement on, as vLLM's EPLB does. The flags
 * write the {@link EngineConfig} and {@link MemoryConfig} builders they are given, so a caller's other flags and the
 * {@code -Dneuralfree.*} property base ({@link #fromSystemProperties()}, the defaults LLMDemo starts from) combine
 * with them; every invalid value is refused naming its flag.
 *
 * <p>{@code -fa} takes llama.cpp's {@code on|off|auto} spellings: attention runs the fused flash-attention kernels on
 * every backend (the GPU {@code LLM_FlashDecode_*} / {@code LLM_FlashAttn_*} kernels and their CPU reference), so
 * {@code on} and {@code auto} select what the engine runs and {@code off} is refused by name.
 *
 * <p>One instance parses one command line; it is not thread-safe.
 */
public final class EngineLoadFlags {

    /** An invalid load flag or value; the message names the flag. */
    public static final class FlagException extends Exception {

        private static final long serialVersionUID = 1L;

        FlagException(String message) {
            super(message);
        }
    }

    private final EngineConfig.Builder engine;
    private final MemoryConfig.Builder memory;
    private final SplitFlags split = new SplitFlags();
    /** The MoE expert cache the flags build on: the base's one, then each cache flag's change. */
    private MoeCacheConfig moeCache;
    /** The expert-parallel load balancer the flags build on: the base's one, then each EPLB flag's change. */
    private EplbConfig eplb;
    /** The {@code --eplb-config} JSON's schedule and redundancy, applied when the flags build. */
    private EplbConfigJson eplbConfigJson;
    private String[] args = new String[0];
    private int i;

    /**
     * Load flags writing {@code engine} and {@code memory}; {@code moeCache} is the expert cache the
     * {@code --moe-cache} flags refine (the base configuration's).
     */
    public EngineLoadFlags(EngineConfig.Builder engine, MemoryConfig.Builder memory, MoeCacheConfig moeCache) {
        this(engine, memory, moeCache, EplbConfig.off());
    }

    /**
     * Load flags writing {@code engine} and {@code memory}; {@code moeCache} and {@code eplb} are the expert cache
     * and load balancer the {@code --moe-cache} and {@code --eplb-*} flags refine (the base configuration's).
     */
    public EngineLoadFlags(EngineConfig.Builder engine, MemoryConfig.Builder memory, MoeCacheConfig moeCache,
                           EplbConfig eplb) {
        this.engine = Objects.requireNonNull(engine, "engine");
        this.memory = Objects.requireNonNull(memory, "memory");
        this.moeCache = Objects.requireNonNull(moeCache, "moeCache");
        this.eplb = Objects.requireNonNull(eplb, "eplb");
    }

    /** Load flags over {@code base}: its engine and memory settings until a flag changes them. */
    public static EngineLoadFlags over(EngineConfig base) {
        return new EngineLoadFlags(base.toBuilder(), base.memory().toBuilder(), base.moeCache(), base.eplb());
    }

    /**
     * Load flags over the {@code -Dneuralfree.*} property configuration ({@link EngineConfig#fromSystemProperties()}),
     * the base LLMDemo parses its flags on: the best GPU backend ({@code neuralfree.backend}, default auto) with the
     * CPU engine as the fallback, the automatic layer placement and F16 KV caches unless a property says otherwise.
     *
     * @throws FlagException when a property holds an invalid value
     */
    public static EngineLoadFlags fromSystemProperties() throws FlagException {
        try {
            return over(EngineConfig.fromSystemProperties());
        } catch (IllegalArgumentException e) {
            throw new FlagException(e.getMessage());
        }
    }

    /**
     * Parses the load flag at {@code args[index]} with its values.
     *
     * @return the index of the flag's last argument, or -1 when {@code args[index]} is not a load flag
     * @throws FlagException for a missing or invalid value, naming the flag
     */
    public int parse(String[] args, int index) throws FlagException {
        this.args = args;
        this.i = index;
        String a = args[index];
        if (backend(a) || placement(a) || kvTypes(a) || split.parse(a, this::value)) {
            return i;
        }
        return -1;
    }

    /**
     * The configuration the flags give: the engine builder with the memory builder's settings and the split flags,
     * which are checked together against the built configuration (a refusal spanning several flags sees all of
     * them).
     *
     * @throws FlagException when the combined settings are invalid, naming the flag or the setting
     */
    public EngineConfig build() throws FlagException {
        applyEplbConfig();
        try {
            return split.build(engine.memory(memory.build()));
        } catch (IllegalArgumentException e) {
            throw new FlagException(e.getMessage());
        }
    }

    /** Applies the {@code --eplb-config} JSON's redundancy and schedule onto the balancer the flags built. */
    private void applyEplbConfig() throws FlagException {
        if (eplbConfigJson == null || eplbConfigJson.isEmpty()) {
            return;
        }
        if (!eplb.enabled()) {
            throw new FlagException("--eplb-config configures the expert-load balancer, which is off; give"
                    + " --enable-eplb (or -Dneuralfree.eplb=on)");
        }
        EplbConfig next = eplb;
        if (eplbConfigJson.numRedundant() != null) {
            next = next.withNumRedundant(eplbConfigJson.numRedundant());
        }
        if (eplbConfigJson.windowSize() != null || eplbConfigJson.stepInterval() != null) {
            next = next.withSchedule(eplbConfigJson.stepInterval() != null
                    ? eplbConfigJson.stepInterval() : next.stepInterval(),
                    eplbConfigJson.windowSize() != null ? eplbConfigJson.windowSize() : next.windowSize());
        }
        setEplb(next);
    }

    /**
     * The shares of one {@code --tensor-split}/{@code -ts} value, read and validated exactly as
     * {@link SplitFlags} reads them: proportions separated by {@code ,}, {@code /} or {@code ;}, every one finite
     * and above 0.
     *
     * @throws FlagException naming {@code flag} for a malformed, zero or non-finite share
     */
    public static double[] tensorSplitShares(String flag, String value) throws FlagException {
        SplitFlags split = new SplitFlags();
        split.parse(flag, f -> value);
        return split.tensorSplitShares();
    }

    /**
     * One {@code --split-mode}/{@code -sm} value, read and validated exactly as {@link SplitFlags} reads it, in
     * llama.cpp's canonical spelling.
     *
     * @throws FlagException naming {@code flag} for an unknown mode
     */
    public static EngineConfig.SplitMode splitMode(String flag, String value) throws FlagException {
        SplitFlags split = new SplitFlags();
        split.parse(flag, f -> value);
        return split.splitModeValue();
    }

    /**
     * One {@code --tensor-split}/{@code -ts} value in the comma-separated form the property reader takes: the shares
     * are validated exactly as {@link SplitFlags} reads them, then written back as trimmed comma-separated values.
     *
     * @throws FlagException naming {@code flag} for a malformed, zero or non-finite share
     */
    public static String tensorSplit(String flag, String value) throws FlagException {
        tensorSplitShares(flag, value);
        String[] parts = value.trim().split("[,/;]+", -1);
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < parts.length; i++) {
            if (i > 0) {
                out.append(',');
            }
            out.append(parts[i].trim());
        }
        return out.toString();
    }

    /**
     * One {@code --main-gpu}/{@code -mg} value, read and validated exactly as {@link SplitFlags} reads it.
     *
     * @throws FlagException naming {@code flag} for a value that is not a device index {@code >= 0}
     */
    public static int mainGpu(String flag, String value) throws FlagException {
        SplitFlags split = new SplitFlags();
        split.parse(flag, f -> value);
        return split.mainGpuValue();
    }

    /**
     * One {@code --eplb-profile}/{@code --eplb-save} value: a path, or null for {@code off}/{@code none} (no file).
     *
     * @throws FlagException naming {@code flag} for a malformed path
     */
    public static Path eplbPath(String flag, String value) throws FlagException {
        String v = value.trim();
        if (v.equalsIgnoreCase("off") || v.equalsIgnoreCase("none")) {
            return null;
        }
        try {
            return Paths.get(v);
        } catch (InvalidPathException e) {
            throw new FlagException(flag + " '" + v + "' is not a path: " + e.getMessage());
        }
    }

    /**
     * One {@code --eplb-policy} value, read and validated exactly as the CLI reads it.
     *
     * @throws FlagException naming {@code flag} for an unsupported policy
     */
    public static EplbConfig.Policy eplbPolicy(String flag, String value) throws FlagException {
        try {
            return EplbConfig.Policy.parse(value);
        } catch (IllegalArgumentException e) {
            throw new FlagException(flag + ": " + e.getMessage());
        }
    }

    /**
     * One {@code --eplb-config} value, vLLM's EPLBConfig JSON object, read and validated exactly as the CLI reads
     * it. See {@link EplbConfigJson#parse}.
     *
     * @throws FlagException naming {@code flag} for a malformed object or an unsupported key or value
     */
    public static EplbConfigJson eplbConfig(String flag, String value) throws FlagException {
        return EplbConfigJson.parse(flag, value);
    }

    /**
     * One {@code --device}/{@code -dev} value, read and validated exactly as {@link SplitFlags} reads it (a
     * comma-separated list of indices or name substrings, or {@code none} alone).
     *
     * @return null for llama-bench's {@code auto} (the engine default), {@code none} for no offload, else the
     *         comma-joined selectors
     * @throws FlagException naming {@code flag} for a malformed or mixed device list
     */
    public static String deviceSelection(String flag, String value) throws FlagException {
        String v = value.trim();
        if (v.equalsIgnoreCase("auto")) {
            return null;
        }
        SplitFlags split = new SplitFlags();
        split.parse(flag, f -> v);
        if (split.noDevice()) {
            return "none";
        }
        return String.join(",", split.deviceSelectors());
    }

    /**
     * The configuration a batched tool opens its model with: {@code load}'s backend, placement and KV types over a
     * paged memory ({@code neuralfree.kv.paged}) of {@code sequences} sequences sharing {@code nCtx} rows, stepped
     * {@code ubatch} tokens at a time.
     *
     * @throws IllegalArgumentException when the memory settings are out of range
     */
    public static EngineConfig pagedConfig(EngineConfig load, int nCtx, int ubatch, int sequences) {
        MemoryConfig memory = load.memory().toBuilder()
                .nCtx(nCtx)
                .ubatch(ubatch)
                .parallel(sequences)
                .kvPaged(true)
                .build();
        return load.toBuilder().memory(memory).build();
    }

    /**
     * What a loaded model runs on: the GPU API name ({@code opencl}, {@code cuda}, {@code vulkan}, {@code hip},
     * {@code metal}) for a model on a GPU backend, else {@code cpu}, which names the reason when the model fell back
     * from a GPU load ({@link EngineModel#gpuFallbackReason()}) or notes the GPU failure when {@code requested} asked
     * for a GPU but the model was opened without the factory's fallback report.
     */
    public static String backendLabel(EngineModel model, EngineConfig requested) {
        if (model instanceof GGUFModelGPU) {
            return ((GGUFModelGPU) model).backend().backendType().name().toLowerCase(Locale.ROOT);
        }
        String fallback = model.gpuFallbackReason();
        if (fallback != null) {
            return "cpu (" + fallback + ")";
        }
        return requested.backend().usesGpu() ? "cpu (the GPU load failed; the reason is logged)" : "cpu";
    }

    // ------------------------------------------------------------------ flag groups

    private boolean backend(String a) throws FlagException {
        switch (a) {
            case "--cpu": engine.backend(EngineConfig.BackendMode.CPU); return true;
            case "--gpu": engine.backend(EngineConfig.BackendMode.AUTO); return true;
            case "-t": case "--threads": {
                int n = intValue(a, 1, Integer.MAX_VALUE);
                System.setProperty(CpuParallel.THREADS_PROPERTY, Integer.toString(n));
                return true;
            }
            case "-lm": case "--load-mode": loadMode(a); return true;
            case "--mlock": mlock(a); return true;
            case "--backend":
                try {
                    engine.backend(EngineConfig.BackendMode.parse(value(a)));
                } catch (IllegalArgumentException e) {
                    throw new FlagException(a + ": " + e.getMessage());
                }
                return true;
            case "-fa": case "--flash-attn": flashAttn(a); return true;
            default: return false;
        }
    }

    private boolean placement(String a) throws FlagException {
        switch (a) {
            case "--ram-cache-mb":
                engine.ramCacheMb(longValue(a, EngineConfig.RAM_CACHE_AUTO, Long.MAX_VALUE));
                return true;
            case "--vram-depth": engine.vramDepth(intValue(a, 0, Integer.MAX_VALUE)); return true;
            case "--gpu-layers": case "-ngl": case "--n-gpu-layers": engine.gpuLayers(gpuLayers(a)); return true;
            case "--override-tensor": case "-ot": addTensorOverrides(a); return true;
            case "--cpu-moe": case "-cmoe": engine.cpuMoe(true); return true;
            case "--n-cpu-moe": case "-ncmoe": engine.nCpuMoe(intValue(a, 0, Integer.MAX_VALUE)); return true;
            case "--n-cpu-ffn": case "-ncffn": engine.nCpuFfn(intValue(a, 0, Integer.MAX_VALUE)); return true;
            case "--no-kv-offload": case "-nkvo": engine.kvOffload(false); return true;
            case "--kv-offload": case "-kvo": engine.kvOffload(true); return true;
            case "--moe-cache-mib": engine.moeCacheMb(longValue(a, 0, Long.MAX_VALUE)); return true;
            case "--cpu-offload-mb":
                engine.cpuOffloadMb(longValue(a, 0, Long.MAX_VALUE));
                return true;
            case "--cpu-offload-params": setOffloadParams(a, true); return true;
            case "--stream-params": setOffloadParams(a, false); return true;
            case "--moe-cache": setMoeCache(a, false); return true;
            case "--moe-cache-ranking": setMoeCache(a, true); return true;
            case "--moe-cache-tail-skip": setTailSkip(a); return true;
            case "--moe-cache-pool": engine.moeCachePool(true); return true;
            case "--enable-expert-parallel": engine.expertParallel(true); return true;
            case "--enable-eplb":
                // vLLM's EPLB rearranges the experts every step-interval steps whenever it runs: the flag turns
                // the runtime rearrangement on, and a -D launch keeps its own setting
                setEplb(eplb.withEnabled(true).withRebalance(true));
                return true;
            case "--no-eplb": setEplb(eplb.withEnabled(false)); return true;
            case "--eplb-profile": setEplb(eplb.withProfile(eplbPath(a))); return true;
            case "--eplb-save": setEplb(eplb.withSave(eplbPath(a))); return true;
            case "--eplb-policy": setEplb(eplb.withPolicy(eplbPolicy(a))); return true;
            case "--eplb-config": setEplbConfig(eplbConfig(a, value(a))); return true;
            case "--fit": engine.fit(true); return true;
            case "--numa": numa(a); return true;
            default: return false;
        }
    }

    private boolean kvTypes(String a) throws FlagException {
        switch (a) {
            case "--kv-type": {
                KvCacheTypes t = kvCacheTypes(a);
                memory.typeK(t.typeK()).typeV(t.typeV());
                return true;
            }
            case "--kv-type-k": case "--cache-type-k": case "-ctk": memory.typeK(kvType(a)); return true;
            case "--kv-type-v": case "--cache-type-v": case "-ctv": memory.typeV(kvType(a)); return true;
            default: return false;
        }
    }

    // ------------------------------------------------------------------ values

    /**
     * {@code -fa on|off|auto} in llama.cpp's spellings ({@code common_arg_utils::is_truthy}, {@code is_falsey},
     * {@code is_autoy}): on and auto are what the engine runs, off is refused.
     */
    private void flashAttn(String flag) throws FlagException {
        String v = value(flag).trim().toLowerCase(Locale.ROOT);
        switch (v) {
            case "on": case "enabled": case "true": case "1":
            case "auto": case "-1":
                return;
            case "off": case "disabled": case "false": case "0":
                throw new FlagException(flag + " " + v + " is not supported: attention always runs the fused"
                        + " flash-attention kernels, the engine has no unfused attention path");
            default:
                throw new FlagException(flag + " '" + v + "' is invalid; use on, off or auto");
        }
    }

    /**
     * {@code -lm}/{@code --load-mode}: llama.cpp's load modes, applied to the process-wide tensor read property
     * before the model opens. {@code auto} leaves the configured mode alone.
     */
    private void loadMode(String flag) throws FlagException {
        String v = value(flag).trim().toLowerCase(Locale.ROOT);
        TensorReadMode mode;
        switch (v) {
            case "auto": return;
            case "none": mode = TensorReadMode.READ; break;
            case "mmap": mode = TensorReadMode.MMAP; break;
            case "mlock": mode = TensorReadMode.MLOCK; break;
            case "mmap+mlock": mode = TensorReadMode.MMAP_MLOCK; break;
            case "dio": mode = TensorReadMode.DIRECT; break;
            default:
                throw new FlagException(flag + " needs auto, none, mmap, mlock, mmap+mlock or dio, not '" + v + "'");
        }
        System.setProperty(TensorReadMode.PROP_READ_MODE, mode.flagName());
    }

    /**
     * {@code --mlock}: pin the model's tensor pages in RAM, on top of the configured load mode (mapped pages under
     * the default mmap, the pages of the read buffers with {@code --no-mmap}). Refused for the modes without
     * resident tensor bytes.
     */
    private void mlock(String flag) throws FlagException {
        TensorReadMode current;
        try {
            current = TensorReadMode.fromProperties(System.getProperties());
            current = current.withMlock(true);
        } catch (IllegalArgumentException e) {
            throw new FlagException(flag + ": " + e.getMessage());
        }
        System.setProperty(TensorReadMode.PROP_READ_MODE, current.flagName());
    }

    /** {@code --numa <distribute|isolate|numactl>}: the CPU pool's thread placement on a multi-node host. */
    private void numa(String flag) throws FlagException {
        NumaStrategy strategy;
        try {
            strategy = NumaStrategy.parse(value(flag));
        } catch (IllegalArgumentException e) {
            throw new FlagException(flag + ": " + e.getMessage());
        }
        engine.numa(strategy);
        System.setProperty(NumaStrategy.PROPERTY, strategy.label());
    }

    /** The {@code -ngl} value: {@code auto} or llama.cpp's {@code -1} auto marker, {@code all}, or a count. */
    private int gpuLayers(String flag) throws FlagException {
        String v = value(flag).trim().toLowerCase(Locale.ROOT);
        if (v.equals("auto") || v.equals("-1")) {
            return EngineConfig.GPU_LAYERS_AUTO;
        }
        if (v.equals("all")) {
            return EngineConfig.GPU_LAYERS_ALL;
        }
        try {
            int n = Integer.parseInt(v);
            if (n >= 0) {
                return n;
            }
        } catch (NumberFormatException e) {
            // reported below
        }
        throw new FlagException(flag + " is auto, all or a layer count >= 0, got '" + v + "'");
    }

    /**
     * {@code --moe-cache off|on|auto|<MiB>} (Avalon's spelling), which keeps the ranking and prefetch settings,
     * or {@code --moe-cache-ranking <path|off>}, which keeps the rest.
     */
    private void setMoeCache(String flag, boolean ranking) throws FlagException {
        String v = value(flag);
        try {
            if (ranking) {
                moeCache = moeCache.withRanking(moeCache.ranking().withPreload(v));
            } else {
                moeCache = MoeCacheConfig.parse(v).withRanking(moeCache.ranking())
                        .withPrefetchZones(moeCache.prefetchZones()).withHostedMaxRows(moeCache.hostedMaxRows())
                        .withTailSkipRank(moeCache.tailSkipRank());
            }
        } catch (IllegalArgumentException e) {
            throw new FlagException(flag + ": " + e.getMessage());
        }
        engine.moeCache(moeCache);
    }

    /**
     * {@code --moe-cache-tail-skip off|0|<rank>} (Avalon {@code GGML_CUDA_MOE_TAIL_SKIP}): drop the cached layers'
     * misses that every token routes only at a low rank, which changes the output; the other cache settings are
     * kept.
     */
    private void setTailSkip(String flag) throws FlagException {
        try {
            moeCache = moeCache.withTailSkipRank(MoeCacheConfig.parseTailSkipRank(value(flag)));
        } catch (IllegalArgumentException e) {
            throw new FlagException(flag + ": " + e.getMessage());
        }
        engine.moeCache(moeCache);
    }

    /** Applies an expert-load-balancer change to the engine builder, keeping the config JSON's keys for build. */
    private void setEplbConfig(EplbConfigJson parsed) throws FlagException {
        if (parsed.policy() != null) {
            setEplb(eplb.withPolicy(parsed.policy()));
        }
        eplbConfigJson = parsed;
    }

    /** Applies an expert-load-balancer change to the engine builder. */
    private void setEplb(EplbConfig next) {
        eplb = next;
        engine.eplb(next);
    }

    /** {@code --eplb-profile}/{@code --eplb-save}: a path, or {@code off} for none. */
    private Path eplbPath(String flag) throws FlagException {
        return eplbPath(flag, value(flag));
    }

    /** {@code --eplb-policy}'s policy name. */
    private EplbConfig.Policy eplbPolicy(String flag) throws FlagException {
        return eplbPolicy(flag, value(flag));
    }

    /**
     * The comma-separated parameter segments of {@code --cpu-offload-params} (the host-mapped offload) or
     * {@code --stream-params} (the selective streaming set); errors carry the flag.
     */
    private void setOffloadParams(String flag, boolean offload) throws FlagException {
        List<String> segments = new ArrayList<>();
        for (String entry : value(flag).split(",", -1)) {
            String e = entry.trim();
            if (e.isEmpty()) {
                throw new FlagException(flag + " has an empty parameter segment");
            }
            segments.add(e);
        }
        try {
            if (offload) {
                engine.cpuOffloadParams(segments);
            } else {
                engine.streamParams(segments);
            }
        } catch (IllegalArgumentException e) {
            throw new FlagException(e.getMessage());
        }
    }

    /** Comma-separated {@code -ot} entries of {@code pattern=target}; errors carry the flag. */
    private void addTensorOverrides(String flag) throws FlagException {
        String v = value(flag);
        for (String entry : v.split(",")) {
            String e = entry.trim();
            if (e.isEmpty()) {
                continue;
            }
            int eq = e.indexOf('=');
            if (eq <= 0 || eq == e.length() - 1) {
                throw new FlagException(flag + " entry '" + e + "' is <tensor name pattern>=<cpu | gpu>");
            }
            try {
                engine.addTensorOverride(e.substring(0, eq), e.substring(eq + 1));
            } catch (IllegalArgumentException ex) {
                throw new FlagException(flag + " entry '" + e + "': " + ex.getMessage());
            }
        }
    }

    private KvDType kvType(String flag) throws FlagException {
        try {
            return KvDType.parse(value(flag), flag);
        } catch (IllegalArgumentException e) {
            throw new FlagException(e.getMessage());
        }
    }

    /** A cache type name that may set both sides (a TurboQuant preset), for {@code --kv-type}. */
    private KvCacheTypes kvCacheTypes(String flag) throws FlagException {
        try {
            return KvCacheTypes.parse(value(flag), flag);
        } catch (IllegalArgumentException e) {
            throw new FlagException(e.getMessage());
        }
    }

    private String value(String flag) throws FlagException {
        if (i + 1 >= args.length) {
            throw new FlagException(flag + " needs a value");
        }
        return args[++i];
    }

    private int intValue(String flag, int min, int max) throws FlagException {
        String v = value(flag);
        try {
            int n = Integer.parseInt(v.trim());
            if (n < min || n > max) {
                throw new FlagException(flag + " " + v + " is outside [" + min + ", " + max + "]");
            }
            return n;
        } catch (NumberFormatException e) {
            throw new FlagException(flag + " needs an integer, got '" + v + "'");
        }
    }

    private long longValue(String flag, long min, long max) throws FlagException {
        String v = value(flag);
        try {
            long n = Long.parseLong(v.trim());
            if (n < min || n > max) {
                throw new FlagException(flag + " " + v + " is outside [" + min + ", " + max + "]");
            }
            return n;
        } catch (NumberFormatException e) {
            throw new FlagException(flag + " needs an integer, got '" + v + "'");
        }
    }

    // ------------------------------------------------------------------ eplb config json

    /**
     * The vLLM {@code --eplb-config} JSON object (vLLM's EPLBConfig) the launch flags of the CLI and the server
     * take: the keys this engine runs become {@link EplbConfig} settings, the always-on settings are accepted, and
     * the keys this engine does not run refuse by name.
     *
     * <ul>
     *   <li>{@code policy} maps to the placement policy ({@code neuralfree.eplb.policy}).</li>
     *   <li>{@code num_redundant_experts} maps to the redundant expert count
     *       ({@code neuralfree.eplb.num-redundant-experts}).</li>
     *   <li>{@code window_size} and {@code step_interval} map to the rearrangement's load window and boundary
     *       interval ({@code neuralfree.eplb.window-size}, {@code neuralfree.eplb.step-interval}).</li>
     *   <li>{@code log_balancedness} and {@code log_balancedness_interval} are accepted: the balance figures are
     *       logged at every rearrangement boundary, as vLLM's interval of 1 does.</li>
     *   <li>{@code use_async=false}, {@code enable_migration_batching=false} and {@code communicator=null} are
     *       accepted; {@code true} and any communicator backend refuse by name (the expert rows migrate inside
     *       this process, between steps, unbatched).</li>
     * </ul>
     *
     * <p>A JSON {@code null} value leaves its key unset. Immutable.</p>
     */
    public static final class EplbConfigJson {

        private final EplbConfig.Policy policy;
        private final Integer numRedundant;
        private final Integer windowSize;
        private final Integer stepInterval;
        private final boolean given;

        private EplbConfigJson(EplbConfig.Policy policy, Integer numRedundant, Integer windowSize,
                               Integer stepInterval, boolean given) {
            this.policy = policy;
            this.numRedundant = numRedundant;
            this.windowSize = windowSize;
            this.stepInterval = stepInterval;
            this.given = given;
        }

        /**
         * Reads vLLM's EPLBConfig object: every key is checked, the ones this engine runs are kept and the others
         * refuse by name.
         *
         * @throws FlagException naming {@code flag} for a malformed object, an unknown key, a key this engine
         *                       does not run or a value out of range
         */
        public static EplbConfigJson parse(String flag, String json) throws FlagException {
            Object parsed;
            try {
                parsed = Json.parse(json, Json.Limits.UNTRUSTED);
            } catch (RuntimeException e) {
                throw new FlagException(flag + " is not a JSON object: " + e.getMessage());
            }
            if (!(parsed instanceof Map)) {
                throw new FlagException(flag + " is a JSON object, got "
                        + (parsed == null ? "null" : parsed.getClass().getSimpleName()));
            }
            EplbConfig.Policy policy = null;
            Integer numRedundant = null;
            Integer windowSize = null;
            Integer stepInterval = null;
            boolean given = false;
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) parsed).entrySet()) {
                Object rawValue = entry.getValue();
                if (rawValue == null) {
                    continue;
                }
                given = true;
                String key = String.valueOf(entry.getKey());
                switch (key) {
                    case "policy":
                        if (!(rawValue instanceof String)) {
                            throw new FlagException(flag + " policy is a string, got " + jsonText(rawValue));
                        }
                        try {
                            policy = EplbConfig.Policy.parse((String) rawValue);
                        } catch (IllegalArgumentException e) {
                            throw new FlagException(flag + ": " + e.getMessage());
                        }
                        break;
                    case "num_redundant_experts":
                        numRedundant = wholeNumber(flag, key, rawValue, 0);
                        break;
                    case "window_size":
                        windowSize = wholeNumber(flag, key, rawValue, 1);
                        break;
                    case "step_interval":
                        stepInterval = wholeNumber(flag, key, rawValue, 1);
                        break;
                    case "log_balancedness":
                        // accepted: the balance figures are logged at every rearrangement boundary
                        bool(flag, key, rawValue);
                        break;
                    case "log_balancedness_interval":
                        // accepted: the boundary log runs at every boundary, vLLM's default interval
                        wholeNumber(flag, key, rawValue, 1);
                        break;
                    case "use_async":
                        if (bool(flag, key, rawValue)) {
                            throw new FlagException(flag + " use_async=true is not supported: the rearrangement"
                                    + " migrates the expert weights between steps, not while one runs; give"
                                    + " use_async=false or drop the key");
                        }
                        break;
                    case "enable_migration_batching":
                        if (bool(flag, key, rawValue)) {
                            throw new FlagException(flag + " enable_migration_batching=true is not supported:"
                                    + " the migrations of a rearrangement are not batched; give"
                                    + " enable_migration_batching=false or drop the key");
                        }
                        break;
                    case "communicator":
                        throw new FlagException(flag + " communicator " + jsonText(rawValue) + " is not supported:"
                                + " the expert rows migrate inside this process; give communicator=null or drop"
                                + " the key");
                    default:
                        throw new FlagException(flag + " key '" + key + "' is not supported; this engine takes"
                                + " policy, num_redundant_experts, window_size, step_interval, log_balancedness,"
                                + " log_balancedness_interval, use_async=false, enable_migration_batching=false"
                                + " and communicator=null");
                }
            }
            return new EplbConfigJson(policy, numRedundant, windowSize, stepInterval, given);
        }

        /** The placement policy, or null when the JSON does not name one. */
        public EplbConfig.Policy policy() {
            return policy;
        }

        /** The redundant expert count, or null when the JSON does not set one. */
        public Integer numRedundant() {
            return numRedundant;
        }

        /** The rearrangement's load window in steps, or null when the JSON does not set one. */
        public Integer windowSize() {
            return windowSize;
        }

        /** The steps between rearrangements, or null when the JSON does not set one. */
        public Integer stepInterval() {
            return stepInterval;
        }

        /** True when the JSON carries no key, so it changes nothing. */
        public boolean isEmpty() {
            return !given;
        }

        /** One whole-number JSON value, an integral string coerced as vLLM's reader does, at least {@code min}. */
        private static int wholeNumber(String flag, String key, Object value, int min) throws FlagException {
            long n = wholeOf(flag, key, value);
            if (n < min) {
                throw new FlagException(flag + " " + key + " must be >= " + min + ", got " + n);
            }
            if (n > Integer.MAX_VALUE) {
                throw new FlagException(flag + " " + key + " is out of range, got " + n);
            }
            return (int) n;
        }

        /** The whole number a JSON value holds, or the refusal naming the key. */
        private static long wholeOf(String flag, String key, Object value) throws FlagException {
            if (value instanceof Long) {
                return (Long) value;
            }
            if (value instanceof BigInteger) {
                BigInteger big = (BigInteger) value;
                if (big.bitLength() < Long.SIZE) {
                    return big.longValue();
                }
                throw new FlagException(flag + " " + key + " is out of range, got " + big);
            }
            if (value instanceof Double) {
                double d = (Double) value;
                if (Double.isFinite(d) && d == Math.rint(d) && Math.abs(d) <= 9.007199254740992E15) {
                    return (long) d;
                }
            }
            if (value instanceof String) {
                try {
                    return Long.parseLong(((String) value).trim());
                } catch (NumberFormatException e) {
                    // reported below
                }
            }
            throw new FlagException(flag + " " + key + " is a whole number, got " + jsonText(value));
        }

        /** One boolean JSON value; anything else refuses naming the key. */
        private static boolean bool(String flag, String key, Object value) throws FlagException {
            if (value instanceof Boolean) {
                return (Boolean) value;
            }
            throw new FlagException(flag + " " + key + " is true or false, got " + jsonText(value));
        }

        /** The JSON-ish spelling of a refused value: strings quoted, the rest as their Java text. */
        private static String jsonText(Object value) {
            return value instanceof String ? "'" + value + "'" : String.valueOf(value);
        }
    }

    // ------------------------------------------------------------------ usage

    /** The usage lines of the backend, device, layer-split, placement, thread and flash-attention flags. */
    public static List<String> backendUsage() {
        List<String> u = new ArrayList<>(Arrays.asList(
                "  --gpu | --cpu                GPU (best backend, default) or the CPU engine",
                "  --backend <type>             " + EngineConfig.BackendMode.namesBar()));
        u.addAll(SplitFlags.usage());
        u.addAll(Arrays.asList(
                "  --threads, -t <n>            CPU thread pool size (default: the machine's)",
                "  --numa <type>                CPU thread placement on a multi-node host: distribute | isolate |",
                "                               numactl (default: none; no effect with a single NUMA node)",
                "  -lm, --load-mode <m>         Tensor reads: auto | none | mmap | mlock | mmap+mlock | dio",
                "                               (default mmap; mlock pins the tensors in RAM, dio reads directly)",
                "  --mlock                      Pin the model's tensor pages in RAM (mmap+mlock, or mlock after",
                "                               --no-mmap)",
                "  -fa, --flash-attn <on|auto>  Flash attention: always on (the fused kernels); off is refused",
                "  --ram-cache-mb <n>           Host cache of streamed layers in MB (-1 = automatic, off for a"
                        + " model larger than RAM)",
                "  --vram-depth <n>             Streamed-layer slots in VRAM (0 = automatic)",
                "  --cpu-offload-mb <n>         Trunk weights kept in host-mapped memory the device reads in",
                "                               place (UVA zero-copy, vLLM cpu_offload_gb); 0 = off",
                "  --cpu-offload-params <seg,..>  Parameter segments the host-mapped offload takes, in",
                "                               weight-layer order up to the budget (empty: any tensor)",
                "  --stream-params <seg,..>     Parameter segments of a streamed layer that stream; the",
                "                               rest stay resident (empty: every tensor of a streamed layer)",
                "  --gpu-layers, -ngl <n|auto|all>",
                "                               Layers kept on the device, the rest computing on the host with"
                        + " their KV there (alias --n-gpu-layers; auto or -1 = as many as the VRAM holds;",
                "                               a count covers the output head, like llama.cpp; layers 0..N",
                "                               compute at RAM bandwidth)",
                "  --override-tensor, -ot <p=t>  Tensor placement overrides pattern=cpu|gpu, comma separated (first"
                        + " match wins; a cpu match keeps a routed expert tensor in RAM beside its device",
                "                               layer, and any other tensor's whole layer on the host)",
                "  --cpu-moe, -cmoe             MoE expert weights stay on the host, their experts computed there"
                        + " while the layers run on the device",
                "  --n-cpu-moe, -ncmoe <n>      The first <n> MoE layers keep their expert weights on the host",
                "  --n-cpu-ffn, -ncffn <n>      The dense FFN weights of the first <n> layers in RAM, on the host",
                "  --no-kv-offload, -nkvo       Keep every layer's KV cache in host RAM: its projections and FFN",
                "                               run on the device, its attention on the host over that cache",
                "  --kv-offload, -kvo           Keep the KV caches on the device (the default)",
                "  --moe-cache <off|on|auto|n>  Host the routed experts and keep the hot ones in a VRAM cache",
                "                               (auto/on: the VRAM left after loading; n: a MiB budget)",
                "  --moe-cache-mib <n>          The expert cache budget in MiB; enables the cache on its own",
                "  --moe-cache-ranking <p|off>  The moe-cache-ranking.bin the cache preloads, hottest first"
                        + " (off: none)",
                "  --moe-cache-tail-skip <n|off>  Drop uncached experts every token routes only at rank n or lower"
                        + " (1-9; changes the output; off by default)",
                "  --moe-cache-pool             Pool the expert cache over the GPUs of a layer split: each GPU caches",
                "                               the routed layers its free VRAM earns, and each hosted MoE block",
                "                               runs on the GPU that caches it",
                "  --enable-expert-parallel     Tensor split (--split-mode tensor): each rank holds whole experts"
                        + " instead of a slice of every expert (vLLM --enable-expert-parallel)",
                "  --enable-eplb                Pack the experts of expert parallelism over the ranks by their"
                        + " recorded routed-row load and rearrange them from the live load while the model"
                        + " runs (vLLM --enable-eplb)",
                "  --eplb-profile <path|off>    The expert-load profile read at load (a moe-cache-ranking.bin)",
                "  --eplb-save <path|off>       Write the routed rows counted while the model runs to this file",
                "  --eplb-policy <p>            Expert placement policy (default: vLLM's weighted packing)",
                "  --eplb-config <json>         vLLM's EPLBConfig object: policy, num_redundant_experts,"
                        + " window_size and step_interval map to their settings; log_balancedness is accepted"
                        + " (the balance figures log at every boundary) and use_async=false is; the async,"
                        + " batched-migration and communicator settings refuse by name",
                "  --fit                        Keep the experts in RAM when the dense weights and the context",
                "                               fit the VRAM without them"));
        return Collections.unmodifiableList(u);
    }

    /** The usage lines of the KV cache type flags. */
    public static List<String> kvTypeUsage() {
        return Collections.unmodifiableList(Arrays.asList(
                "  --kv-type <t>                KV cache type for K and V (default f16): f32 | f16 | bf16 | q8_0 |",
                "                               q4_0 | q4_1 | iq4_nl | q5_0 | q5_1 | fp8 (= fp8_e4m3) |",
                "                               fp8_e5m2 | fp8_inc | fp8_per_token_head | int8_per_token_head |",
                "                               int4_per_token_head | nvfp4 | nvfp4_4over6 | fp8_ds_mla |",
                "                               nvfp4_ds_mla | turboquant_k8v4 | turboquant_4bit_nc |",
                "                               turboquant_k3v4_nc | turboquant_3bit_nc (vLLM --kv-cache-dtype",
                "                               names; an FP8 or NVFP4 cache divides by the checkpoint",
                "                               blk.N.attn_k/attn_v.kv_scale on store and attention multiplies it",
                "                               back; a per-token-head cache computes absmax/qmax per (token, KV",
                "                               head) at store time, and then K and V must both be per-token-head",
                "                               or neither; the DeepSeek MLA latent rows only fill an MLA latent",
                "                               cache)",
                "  --kv-type-k / --kv-type-v <t> KV cache type of K / V only (MLA and DeepSeek-V4 need K = V);",
                "                               aliases -ctk / -ctv, --cache-type-k / --cache-type-v"));
    }

    /**
     * The load-flag section of a tool's usage text: a heading, then the {@link #backendUsage()} and
     * {@link #kvTypeUsage()} lines, newline-terminated.
     */
    public static String usageSection(String heading) {
        StringBuilder sb = new StringBuilder(heading).append('\n');
        for (String line : backendUsage()) {
            sb.append(line).append('\n');
        }
        for (String line : kvTypeUsage()) {
            sb.append(line).append('\n');
        }
        return sb.toString();
    }
}
