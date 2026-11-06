package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.demo.EngineLoadFlags;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EplbConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

/**
 * The multi-GPU launch flags of {@link ServerLaunchArgs}: llama-server's split flags and vLLM's parallel flags,
 * collected while the arguments are read and resolved together into the split, expert-parallel and EPLB properties
 * once all are known. Every value is read through the CLI's own parsers ({@link EngineLoadFlags}), so the server and
 * the demo refuse the same values.
 *
 * <ul>
 *   <li>{@code -sm}/{@code --split-mode}, {@code -ts}/{@code --tensor-split} and {@code -mg}/{@code --main-gpu} set
 *       {@code neuralfree.split.mode}, {@code neuralfree.tensor.split} and {@code neuralfree.main.gpu}.</li>
 *   <li>vLLM's {@code -tp}/{@code --tensor-parallel-size N} is a tensor split over {@code N} devices and
 *       {@code -pp}/{@code --pipeline-parallel-size N} a layer split over {@code N} devices: the first {@code N}
 *       device indices unless {@code --device} already lists exactly {@code N} GPUs. A size of 1 is vLLM's default
 *       and changes nothing.</li>
 *   <li>vLLM's {@code -dcp}/{@code --decode-context-parallel-size N} is the context split
 *       ({@code neuralfree.split.mode=context}) over {@code N} devices. vLLM nests its dcp ranks inside the
 *       tensor (or pipeline) ranks and widens its world with pcp; this engine runs one split mode, so a context
 *       size beside a tensor or pipeline size refuses by name, and a prefill size beside a decode size must name
 *       the same rank count (the prefill query split runs over the context split's own ranks).
 *       {@code -pcp}/{@code --prefill-context-parallel-size N} above 1 is that prefill query split: the batch
 *       engine runs the paged memory, whose DualChunkSwap prefill cuts every segment of a batched step over the
 *       context split's translated page tables, so the flag maps onto {@code neuralfree.prefill_context_parallel_size}.
 *       {@code --dcp-kv-cache-interleave-size} names the page-granular
 *       interleave of the paged context split (one page per rank in turn, {@code CpPageShard}), so it needs the
 *       context split over a paged memory and refuses every value but the page size
 *       ({@code neuralfree.kv.page.size}).</li>
 *   <li>{@code --enable-expert-parallel}/{@code -ep} sets {@code neuralfree.expert.parallel}, and
 *       {@code --enable-eplb}/{@code --no-eplb} with {@code --eplb-profile}, {@code --eplb-save} and
 *       {@code --eplb-policy} sets the expert-load balancer ({@link EplbConfig}). Enabling the balancer turns the
 *       runtime rearrangement on, as vLLM's EPLB rearranges every {@code step_interval} steps whenever it runs.
 *       vLLM's {@code --eplb-config} JSON maps its {@code policy}, {@code num_redundant_experts},
 *       {@code window_size} and {@code step_interval} keys and accepts the always-on logging and synchronous
 *       migration settings; the keys this engine does not run are refused by name.</li>
 * </ul>
 *
 * <p>The engine runs one split mode, so a tensor and a pipeline size above 1 together, a rank flag contradicting
 * {@code --split-mode}, a tensor split under tensor mode, a share count that differs from the device count,
 * expert parallelism without a tensor split, and EPLB without expert parallelism or a profile/save path are refused
 * by name, as the CLI refuses them. A parse accumulator, used by one parse only.
 */
final class ParallelLaunchFlags {

    /** The flags taking a value. */
    static final Set<String> VALUED = Set.of("-sm", "--split-mode", "-ts", "--tensor-split", "-mg", "--main-gpu",
            "-tp", "--tensor-parallel-size", "-pp", "--pipeline-parallel-size",
            "-dcp", "--decode-context-parallel-size", "-pcp", "--prefill-context-parallel-size",
            "--dcp-kv-cache-interleave-size",
            "--eplb-profile", "--eplb-save", "--eplb-policy", "--eplb-config");

    /** The switches. */
    static final Set<String> SWITCHES = Set.of("-ep", "--enable-expert-parallel", "--enable-eplb", "--no-eplb");

    private String splitFlag;
    private EngineConfig.SplitMode splitMode;
    private String tensorSplitFlag;
    private String tensorSplitText;
    private String mainGpuFlag;
    private Integer mainGpu;
    private String tensorParallelFlag;
    private Integer tensorParallel;
    private String pipelineParallelFlag;
    private Integer pipelineParallel;
    private String decodeContextParallelFlag;
    private Integer decodeContextParallel;
    private String prefillContextParallelFlag;
    private Integer prefillContextParallel;
    private Integer kvInterleaveSize;
    private boolean expertParallel;
    private Boolean eplbEnabled;
    private boolean eplbProfileGiven;
    private Path eplbProfile;
    private boolean eplbSaveGiven;
    private Path eplbSave;
    private String eplbPolicy;
    private EngineLoadFlags.EplbConfigJson eplbConfigJson;

    /** Records the value of one {@link #VALUED} flag, refused when it is not a usable value. */
    void value(String flag, String value) {
        switch (flag) {
            case "-sm", "--split-mode" -> {
                splitFlag = "--split-mode";
                splitMode = mode(flag, value);
            }
            case "-ts", "--tensor-split" -> {
                tensorSplitFlag = "--tensor-split";
                tensorSplitText = tensorSplit(flag, value);
            }
            case "-mg", "--main-gpu" -> {
                mainGpuFlag = "--main-gpu";
                mainGpu = mainGpu(flag, value);
            }
            case "-tp", "--tensor-parallel-size" -> {
                tensorParallelFlag = "--tensor-parallel-size";
                tensorParallel = count(flag, value);
            }
            case "-pp", "--pipeline-parallel-size" -> {
                pipelineParallelFlag = "--pipeline-parallel-size";
                pipelineParallel = count(flag, value);
            }
            case "-dcp", "--decode-context-parallel-size" -> {
                decodeContextParallelFlag = "--decode-context-parallel-size";
                decodeContextParallel = count(flag, value);
            }
            case "-pcp", "--prefill-context-parallel-size" -> {
                prefillContextParallelFlag = "--prefill-context-parallel-size";
                prefillContextParallel = count(flag, value);
            }
            case "--dcp-kv-cache-interleave-size" -> kvInterleaveSize = count(flag, value);
            case "--eplb-profile" -> {
                eplbProfileGiven = true;
                eplbProfile = eplbPath(flag, value);
            }
            case "--eplb-save" -> {
                eplbSaveGiven = true;
                eplbSave = eplbPath(flag, value);
            }
            case "--eplb-policy" -> eplbPolicy = eplbPolicy(flag, value);
            case "--eplb-config" -> eplbConfig(flag, value);
            default -> throw new IllegalArgumentException("not a parallel flag: " + flag);
        }
    }

    /** Records one {@link #SWITCHES} flag. */
    void toggle(String flag) {
        switch (flag) {
            case "-ep", "--enable-expert-parallel" -> expertParallel = true;
            case "--enable-eplb" -> eplbEnabled = true;
            case "--no-eplb" -> {
                eplbEnabled = false;
                // disabling drops the paths and policy, as EplbConfig.withEnabled(false) does
                eplbProfileGiven = false;
                eplbProfile = null;
                eplbSaveGiven = false;
                eplbSave = null;
                eplbPolicy = null;
            }
            default -> throw new IllegalArgumentException("not a parallel switch: " + flag);
        }
    }

    /**
     * Puts the resolved properties into {@code properties}; nothing when no parallel flag was given beyond what
     * {@code --device} and {@code --rpc} already wrote.
     *
     * @throws IllegalArgumentException when the flags contradict themselves or cannot map onto the engine's split
     */
    void resolveInto(Map<String, String> properties) {
        EngineConfig.SplitMode implied = impliedMode();
        if (splitMode != null && implied != null && splitMode != implied) {
            throw new IllegalArgumentException(rankFlag() + " " + requestedRanks() + " runs a "
                    + implied.llamaName() + " split, but " + splitFlag + " " + splitMode.llamaName()
                    + " was given; the engine runs one split mode");
        }
        EngineConfig.SplitMode chosen = splitMode != null ? splitMode : implied;
        String devices = effective(properties, EngineConfig.PROP_GPU_DEVICES);
        boolean rpc = effective(properties, EngineConfig.PROP_RPC) != null;
        int ranks = requestedRanks();
        if (ranks > 0) {
            if (rpc) {
                throw new IllegalArgumentException(rankFlag() + " " + ranks + " maps vLLM's rank count onto local"
                        + " GPUs, but --rpc adds remote layer targets it cannot name; list the split with"
                        + " --device and --split-mode layer instead");
            }
            if ("cpu".equalsIgnoreCase(effective(properties, EngineConfig.PROP_BACKEND))) {
                throw new IllegalArgumentException(rankFlag() + " " + ranks + " spreads the model over GPUs, but"
                        + " --device none runs the CPU engine");
            }
            if (devices != null) {
                int listed = deviceCount(devices);
                if (listed != ranks) {
                    throw new IllegalArgumentException(rankFlag() + " " + ranks + " name(s) " + ranks + " rank(s),"
                            + " but --device lists " + listed + " device(s); give one device per rank");
                }
            } else {
                devices = defaultDevices(ranks);
                properties.put(EngineConfig.PROP_GPU_DEVICES, devices);
            }
        }

        EngineConfig.SplitMode mode = chosen;
        if (mode == null) {
            String property = effective(properties, EngineConfig.PROP_SPLIT_MODE);
            if (property != null) {
                mode = mode(EngineConfig.PROP_SPLIT_MODE, property);
            }
        }
        if (mode == null && devices != null && deviceCount(devices) > 1) {
            // llama.cpp's default: a device list of two or more splits by layer
            mode = EngineConfig.SplitMode.LAYER;
        }
        boolean modeFromFlag = splitMode != null || implied != null;
        if (modeFromFlag && mode != null && mode != EngineConfig.SplitMode.NONE
                && "cpu".equalsIgnoreCase(effective(properties, EngineConfig.PROP_BACKEND))) {
            throw new IllegalArgumentException((splitFlag != null ? splitFlag : rankFlag())
                    + " spreads the model over GPUs, but --device none runs the CPU engine");
        }
        if (rpc && mode != null && mode != EngineConfig.SplitMode.LAYER && modeFromFlag) {
            throw new IllegalArgumentException((splitFlag != null ? splitFlag : rankFlag())
                    + " cannot run with --rpc: the remote servers add devices to a layer split");
        }
        if (chosen != null) {
            properties.put(EngineConfig.PROP_SPLIT_MODE, chosen.llamaName());
        }
        if (size(prefillContextParallel) > 1) {
            // the batch engine runs the paged memory, whose DualChunkSwap prefill cuts every segment of a batched
            // step over the context split's translated page tables, so the flag maps onto the engine's property
            properties.put(EngineConfig.PROP_PREFILL_CONTEXT_PARALLEL, Integer.toString(prefillContextParallel));
        }
        if (kvInterleaveSize != null) {
            checkKvInterleave(properties, mode);
        }

        checkTensorSplit(properties, mode, devices);
        checkMainGpu(properties, devices);
        boolean expert = checkExpertParallel(properties, mode);
        checkEplb(properties, expert);
    }

    /**
     * The mode vLLM's rank flags imply, or null for none (a size of 1 changes nothing). A context size beside a
     * tensor or pipeline size refuses: vLLM nests its dcp ranks inside the tensor ranks and widens its world with
     * pcp beside them, while this engine runs one split mode. A prefill size beside a decode size must name the
     * same rank count, as the engine's prefill query split runs over the context split's own ranks.
     */
    private EngineConfig.SplitMode impliedMode() {
        int tensor = size(tensorParallel);
        int pipeline = size(pipelineParallel);
        int decode = size(decodeContextParallel);
        int prefill = size(prefillContextParallel);
        if (tensor > 1 && pipeline > 1) {
            throw new IllegalArgumentException(tensorParallelFlag + " " + tensorParallel + " and "
                    + pipelineParallelFlag + " " + pipelineParallel + " together are not supported: the engine"
                    + " runs one split mode, not a tensor and a pipeline dimension; give one of them");
        }
        if (decode > 1 && (tensor > 1 || pipeline > 1)) {
            throw new IllegalArgumentException(decodeContextParallelFlag + " " + decodeContextParallel + " beside "
                    + (tensor > 1 ? tensorParallelFlag + " " + tensorParallel
                            : pipelineParallelFlag + " " + pipelineParallel) + " is not supported: vLLM nests the"
                    + " dcp ranks inside the tensor ranks, but the engine runs one split mode; give one of them");
        }
        if (prefill > 1 && (tensor > 1 || pipeline > 1)) {
            throw new IllegalArgumentException(prefillContextParallelFlag + " " + prefillContextParallel
                    + " splits every prefill's query rows over the ranks of a context split, and the engine runs"
                    + " one split mode, not " + (tensor > 1 ? tensorParallelFlag + " " + tensorParallel
                            : pipelineParallelFlag + " " + pipelineParallel) + " beside it; give one of them");
        }
        if (decode > 1 && prefill > 1 && decode != prefill) {
            throw new IllegalArgumentException(prefillContextParallelFlag + " " + prefillContextParallel
                    + " splits every prefill's query rows over the ranks of the context split "
                    + decodeContextParallelFlag + " " + decode + " runs, so it must equal " + decode);
        }
        if (decode > 1 || prefill > 1) {
            return EngineConfig.SplitMode.CONTEXT;
        }
        if (tensor > 1) {
            return EngineConfig.SplitMode.TENSOR;
        }
        if (pipeline > 1) {
            return EngineConfig.SplitMode.LAYER;
        }
        return null;
    }

    /** The rank count of the flag that implies a split, or 0 when neither does. */
    private int requestedRanks() {
        int decode = size(decodeContextParallel);
        int prefill = size(prefillContextParallel);
        if (decode > 1) {
            return decode;
        }
        if (prefill > 1) {
            return prefill;
        }
        if (tensorParallel != null && tensorParallel > 1) {
            return tensorParallel;
        }
        if (pipelineParallel != null && pipelineParallel > 1) {
            return pipelineParallel;
        }
        return 0;
    }

    /** The flag naming the requested rank count. */
    private String rankFlag() {
        int decode = size(decodeContextParallel);
        int prefill = size(prefillContextParallel);
        if (decode > 1) {
            return decodeContextParallelFlag;
        }
        if (prefill > 1) {
            return prefillContextParallelFlag;
        }
        return tensorParallel != null && tensorParallel > 1 ? tensorParallelFlag : pipelineParallelFlag;
    }

    /**
     * Validates {@code --dcp-kv-cache-interleave-size}: the context split it names runs over a paged memory, and
     * its interleave is the page-granular one ({@code CpPageShard}, one page per rank in turn), so the value must
     * equal the page size the pools run — vLLM's block-level interleave. Accepted, the launch configures nothing
     * further: the paged context split always interleaves whole pages.
     */
    private void checkKvInterleave(Map<String, String> properties, EngineConfig.SplitMode mode) {
        if (mode != EngineConfig.SplitMode.CONTEXT) {
            throw new IllegalArgumentException("--dcp-kv-cache-interleave-size " + kvInterleaveSize
                    + " configures the KV interleave of a decode-context-parallel split, but the launch runs a "
                    + (mode == null ? "single-GPU" : mode.llamaName()) + " split; give"
                    + " --split-mode context, --decode-context-parallel-size N or"
                    + " --prefill-context-parallel-size N");
        }
        int parallel = size(parseWhole(effective(properties, MemoryConfig.PROP_PARALLEL)));
        if (parallel < 2 && !isOn(effective(properties, MemoryConfig.PROP_KV_PAGED))) {
            throw new IllegalArgumentException("--dcp-kv-cache-interleave-size " + kvInterleaveSize
                    + " names the page-granular interleave of the paged KV pools, but the memory is contiguous:"
                    + " give -np N above 1 or -D" + MemoryConfig.PROP_KV_PAGED + "=true");
        }
        int pageSize = pageSize(properties);
        if (kvInterleaveSize != pageSize) {
            throw new IllegalArgumentException("--dcp-kv-cache-interleave-size " + kvInterleaveSize
                    + " does not match the page size " + pageSize + " the paged KV pools run: the context split"
                    + " interleaves whole pages, one page per rank in turn; give the value " + pageSize
                    + ", or set -D" + MemoryConfig.PROP_KV_PAGE_SIZE + "=" + kvInterleaveSize);
        }
    }

    /** The effective page size of the paged KV pools, or the builder default when none is set. */
    private static int pageSize(Map<String, String> properties) {
        Integer page = parseWhole(effective(properties, MemoryConfig.PROP_KV_PAGE_SIZE));
        return page == null ? MemoryConfig.DEFAULT_PAGE_SIZE : page;
    }

    /** The whole number {@code value} holds, or null for none or for a value that is not a whole number. */
    private static Integer parseWhole(String value) {
        if (value == null) {
            return null;
        }
        try {
            return Integer.valueOf(value.trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /** The size behind an optional vLLM rank flag: 1 when unset, as a size of 1 is vLLM's default. */
    private static int size(Integer v) {
        return v == null ? 1 : v;
    }

    /** Validates {@code --tensor-split} against the effective split mode and device count, then writes it. */
    private void checkTensorSplit(Map<String, String> properties, EngineConfig.SplitMode mode, String devices) {
        if (tensorSplitText == null) {
            return;
        }
        if (mode == EngineConfig.SplitMode.TENSOR) {
            throw new IllegalArgumentException(tensorSplitFlag + " gives per-device shares, but --split-mode"
                    + " tensor divides every attention head and feed-forward neuron evenly over the ranks; use"
                    + " --split-mode row for proportional feed-forward shards");
        }
        if (mode == null || mode == EngineConfig.SplitMode.NONE) {
            throw new IllegalArgumentException(tensorSplitFlag + " sizes the layer spans of --split-mode layer;"
                    + " split mode none runs every layer on one GPU");
        }
        int total = 0;
        if (devices != null) {
            total = deviceCount(devices);
            if (mode == EngineConfig.SplitMode.LAYER) {
                total += rpcCount(properties);
            }
        }
        int shares = tensorSplitText.split(",", -1).length;
        if (total > 0 && shares != total) {
            throw new IllegalArgumentException(tensorSplitFlag + " gives " + shares + " share(s) for the " + total
                    + " device(s); give one share per device");
        }
        properties.put(EngineConfig.PROP_TENSOR_SPLIT, tensorSplitText);
    }

    /** Validates {@code --main-gpu} against the device list, then writes it. */
    private void checkMainGpu(Map<String, String> properties, String devices) {
        if (mainGpu == null) {
            return;
        }
        int listed = devices == null ? 0 : deviceCount(devices);
        if (listed > 0 && mainGpu >= listed) {
            throw new IllegalArgumentException(mainGpuFlag + " " + mainGpu + " is beyond the " + listed
                    + " GPU(s) of --device (indices 0.." + (listed - 1) + ")");
        }
        properties.put(EngineConfig.PROP_MAIN_GPU, Integer.toString(mainGpu));
    }

    /**
     * Validates and writes expert parallelism.
     *
     * @return whether expert parallelism is on, by the flag or by a pre-set property
     */
    private boolean checkExpertParallel(Map<String, String> properties, EngineConfig.SplitMode mode) {
        boolean expert = expertParallel || isOn(effective(properties, EngineConfig.PROP_EXPERT_PARALLEL));
        if (!expertParallel) {
            return expert;
        }
        if (mode != EngineConfig.SplitMode.TENSOR) {
            throw new IllegalArgumentException("--enable-expert-parallel places whole experts on each rank of a"
                    + " tensor split, but the launch runs --split-mode "
                    + (mode == null ? "none" : mode.llamaName()) + "; give --split-mode tensor or"
                    + " --tensor-parallel-size N");
        }
        properties.put(EngineConfig.PROP_EXPERT_PARALLEL, "true");
        return true;
    }

    /** Validates and writes the expert-load balancer flags. */
    private void checkEplb(Map<String, String> properties, boolean expert) {
        boolean touched = eplbEnabled != null || eplbProfileGiven || eplbSaveGiven || eplbPolicy != null
                || (eplbConfigJson != null && !eplbConfigJson.isEmpty());
        if (!touched) {
            return;
        }
        if (eplbEnabled != null && !eplbEnabled) {
            if (eplbProfileGiven || eplbSaveGiven || eplbPolicy != null
                    || (eplbConfigJson != null && !eplbConfigJson.isEmpty())) {
                throw new IllegalArgumentException("the EPLB profile, save path, policy and --eplb-config"
                        + " configure the load balancer, which --no-eplb turns off; give --enable-eplb");
            }
            properties.put(EplbConfig.PROP_EPLB, "off");
            // blank masks a pre-set property, so the launch really turns the balancer off; the redundant count
            // and the rearrangement need it on, so they mask too
            properties.put(EplbConfig.PROP_EPLB_PROFILE, "");
            properties.put(EplbConfig.PROP_EPLB_SAVE, "");
            properties.put(EplbConfig.PROP_EPLB_POLICY, "");
            properties.put(EplbConfig.PROP_EPLB_NUM_REDUNDANT, "");
            properties.put(EplbConfig.PROP_EPLB_REBALANCE, "");
            return;
        }
        if (eplbEnabled != null) {
            properties.put(EplbConfig.PROP_EPLB, "on");
            // vLLM's EPLB rearranges the experts every step_interval steps whenever it runs, so the flag turns
            // the runtime rearrangement on; a -D launch without the flag keeps its own setting
            properties.put(EplbConfig.PROP_EPLB_REBALANCE, "on");
        }
        if (eplbProfileGiven) {
            properties.put(EplbConfig.PROP_EPLB_PROFILE, eplbProfile == null ? "" : eplbProfile.toString());
        }
        if (eplbSaveGiven) {
            properties.put(EplbConfig.PROP_EPLB_SAVE, eplbSave == null ? "" : eplbSave.toString());
        }
        if (eplbPolicy != null) {
            properties.put(EplbConfig.PROP_EPLB_POLICY, eplbPolicy);
        }
        if (eplbConfigJson != null && !eplbConfigJson.isEmpty()) {
            if (eplbConfigJson.numRedundant() != null) {
                properties.put(EplbConfig.PROP_EPLB_NUM_REDUNDANT, eplbConfigJson.numRedundant().toString());
            }
            if (eplbConfigJson.windowSize() != null) {
                properties.put(EplbConfig.PROP_EPLB_WINDOW_SIZE, eplbConfigJson.windowSize().toString());
            }
            if (eplbConfigJson.stepInterval() != null) {
                properties.put(EplbConfig.PROP_EPLB_STEP_INTERVAL, eplbConfigJson.stepInterval().toString());
            }
        }
        if (eplbEnabled == null && !isOn(effective(properties, EplbConfig.PROP_EPLB))) {
            throw new IllegalArgumentException("the EPLB profile, save path, policy and --eplb-config configure"
                    + " the load balancer; give --enable-eplb (or -Dneuralfree.eplb=on)");
        }
        if (!isOn(effective(properties, EplbConfig.PROP_EPLB))) {
            return;
        }
        if (!expert) {
            throw new IllegalArgumentException("--enable-eplb packs the whole experts of expert parallelism over"
                    + " the ranks, but --enable-expert-parallel is off; give --enable-expert-parallel (or"
                    + " neuralfree.expert.parallel=true)");
        }
        if (effective(properties, EplbConfig.PROP_EPLB_PROFILE) == null
                && effective(properties, EplbConfig.PROP_EPLB_SAVE) == null
                && !isOn(effective(properties, EplbConfig.PROP_EPLB_REBALANCE))) {
            throw new IllegalArgumentException("the expert-load balancer is on without anything to balance from:"
                    + " give --eplb-profile (a moe-cache-ranking.bin), --eplb-save (to record this run's load)"
                    + " or -D" + EplbConfig.PROP_EPLB_REBALANCE + "=on (to rearrange from the live load)");
        }
    }

    // ------------------------------------------------------------------ values

    private static EngineConfig.SplitMode mode(String flag, String value) {
        try {
            return EngineLoadFlags.splitMode(flag, value);
        } catch (EngineLoadFlags.FlagException e) {
            throw new IllegalArgumentException(e.getMessage(), e);
        }
    }

    private static String tensorSplit(String flag, String value) {
        try {
            return EngineLoadFlags.tensorSplit(flag, value);
        } catch (EngineLoadFlags.FlagException e) {
            throw new IllegalArgumentException(e.getMessage(), e);
        }
    }

    private static int mainGpu(String flag, String value) {
        try {
            return EngineLoadFlags.mainGpu(flag, value);
        } catch (EngineLoadFlags.FlagException e) {
            throw new IllegalArgumentException(e.getMessage(), e);
        }
    }

    private static Path eplbPath(String flag, String value) {
        try {
            return EngineLoadFlags.eplbPath(flag, value);
        } catch (EngineLoadFlags.FlagException e) {
            throw new IllegalArgumentException(e.getMessage(), e);
        }
    }

    private static String eplbPolicy(String flag, String value) {
        try {
            return EngineLoadFlags.eplbPolicy(flag, value).label();
        } catch (EngineLoadFlags.FlagException e) {
            throw new IllegalArgumentException(e.getMessage(), e);
        }
    }

    /** The whole number of ranks a vLLM size flag takes, at least 1. */
    private static int count(String flag, String value) {
        int n;
        try {
            n = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(flag + " must be a whole number, got '" + value + "'", e);
        }
        if (n < 1) {
            throw new IllegalArgumentException(flag + " must be at least 1, got " + n);
        }
        return n;
    }

    /**
     * vLLM's {@code --eplb-config} JSON object, read and validated by the CLI's own reader ({@link
     * EngineLoadFlags#eplbConfig}), so both surfaces refuse the same keys and values: {@code policy},
     * {@code num_redundant_experts}, {@code window_size} and {@code step_interval} map to the balancer's
     * properties, and the keys this engine does not run refuse by name.
     */
    private void eplbConfig(String flag, String json) {
        EngineLoadFlags.EplbConfigJson parsed;
        try {
            parsed = EngineLoadFlags.eplbConfig(flag, json);
        } catch (EngineLoadFlags.FlagException e) {
            throw new IllegalArgumentException(e.getMessage(), e);
        }
        eplbConfigJson = parsed;
        if (parsed.policy() != null) {
            eplbPolicy = parsed.policy().label();
        }
    }

    // ------------------------------------------------------------------ helpers

    /** The value a launch sees for {@code key}: the launch flags, else the pre-set system property. */
    private static String effective(Map<String, String> properties, String key) {
        String value = properties.containsKey(key) ? properties.get(key) : System.getProperty(key);
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() ? null : trimmed;
    }

    private static boolean isOn(String value) {
        return value != null
                && (value.equalsIgnoreCase("on") || value.equalsIgnoreCase("true") || value.equals("1"));
    }

    private static int deviceCount(String devices) {
        int count = 0;
        for (String entry : devices.split(",", -1)) {
            if (!entry.trim().isEmpty()) {
                count++;
            }
        }
        return count;
    }

    private static int rpcCount(Map<String, String> properties) {
        String servers = effective(properties, EngineConfig.PROP_RPC);
        return servers == null ? 0 : deviceCount(servers);
    }

    private static String defaultDevices(int ranks) {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < ranks; i++) {
            if (i > 0) {
                out.append(',');
            }
            out.append(i);
        }
        return out.toString();
    }
}
