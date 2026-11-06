package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.demo.EngineLoadFlags.FlagException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.rpc.RpcServerSpec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * The multi-GPU load flags of {@link EngineLoadFlags} in llama.cpp's spellings (common/arg.cpp):
 * {@code --split-mode} / {@code -sm}, {@code --tensor-split} / {@code -ts}, {@code --device} / {@code -dev},
 * {@code --main-gpu} / {@code -mg} and {@code --rpc}, beside vLLM's context-parallel size flags
 * {@code --decode-context-parallel-size} / {@code -dcp}, {@code --prefill-context-parallel-size} / {@code -pcp} and
 * {@code --dcp-kv-cache-interleave-size}. They set the engine fields of {@code neuralfree.split.mode},
 * {@code neuralfree.tensor.split}, {@code neuralfree.gpu.devices} (a single device-name substring also sets
 * {@code neuralfree.device}, the GPU of a one-device load; a single device index picks it inside its backend's
 * device list, {@link EngineConfig#picksMainGpu()}), {@code neuralfree.main.gpu}, {@code neuralfree.rpc},
 * {@code neuralfree.prefill_context_parallel_size}. The flags are collected while the command line is read and
 * checked together when the engine configuration is built, so a refusal that spans several flags sees all of
 * them; every refusal names the flag. A device list of two or more GPUs without {@code --split-mode} splits by
 * layer, as llama.cpp's default split mode does, and so does an {@code --rpc} server (the remote GPU is one more
 * layer target); {@code --device none} runs the CPU engine (llama.cpp: no offload) and cannot carry remote
 * servers. A context-parallel size above 1 runs the context split over that many GPUs, as vLLM's sizes run over
 * that many ranks. Mutable parse state of one command line.
 */
final class SplitFlags {

    /** Reads the value of a flag from the command line. */
    interface Values {
        String value(String flag) throws FlagException;
    }

    private static final String SPLIT_MODE = "--split-mode";
    private static final String TENSOR_SPLIT = "--tensor-split";
    private static final String DEVICE = "--device";
    private static final String MAIN_GPU = "--main-gpu";
    private static final String RPC = "--rpc";
    private static final String DECODE_CONTEXT_PARALLEL = "--decode-context-parallel-size";
    private static final String PREFILL_CONTEXT_PARALLEL = "--prefill-context-parallel-size";
    private static final String DCP_KV_INTERLEAVE = "--dcp-kv-cache-interleave-size";
    /** llama.cpp's {@code --device} entry that keeps every layer off the GPUs. */
    private static final String NO_DEVICE = "none";

    private EngineConfig.SplitMode splitMode;
    private double[] tensorSplit;
    private List<String> devices;
    private List<RpcServerSpec> rpcServers;
    private boolean noDevice;
    private int mainGpu = -1;
    private Integer decodeContextParallel;
    private Integer prefillContextParallel;
    private Integer kvInterleaveSize;

    /**
     * Consumes {@code flag} when it is one of the split flags, reading its value through {@code values}.
     *
     * @return false for any other flag
     * @throws FlagException for a missing or invalid value, naming the flag
     */
    boolean parse(String flag, Values values) throws FlagException {
        switch (flag) {
            case "--split-mode": case "-sm": splitMode(flag, values.value(flag)); return true;
            case "--tensor-split": case "-ts": tensorSplit(flag, values.value(flag)); return true;
            case "--device": case "-device": case "-dev": devices(flag, values.value(flag)); return true;
            case "--main-gpu": case "-mg": mainGpu(flag, values.value(flag)); return true;
            case "--rpc": rpc(values.value(flag)); return true;
            case DECODE_CONTEXT_PARALLEL: case "-dcp":
                decodeContextParallel = rankSize(flag, values.value(flag)); return true;
            case PREFILL_CONTEXT_PARALLEL: case "-pcp":
                prefillContextParallel = rankSize(flag, values.value(flag)); return true;
            case DCP_KV_INTERLEAVE:
                kvInterleaveSize = rankSize(flag, values.value(flag)); return true;
            default: return false;
        }
    }

    /**
     * Applies the flags to {@code builder}, builds the configuration and checks the flags against it.
     *
     * @throws FlagException            naming the flag when the combination cannot run
     * @throws IllegalArgumentException from {@link EngineConfig.Builder#build()}, naming the setting
     */
    EngineConfig build(EngineConfig.Builder builder) throws FlagException {
        checkContextFlags();
        if (mainGpu >= 0 && devices != null && mainGpu >= devices.size()) {
            throw new FlagException(MAIN_GPU + " " + mainGpu + " is beyond the " + devices.size() + " GPU(s) of "
                    + DEVICE + " (indices 0.." + (devices.size() - 1) + ")");
        }
        apply(builder);
        EngineConfig built = builder.build();
        check(built);
        return built;
    }

    private void apply(EngineConfig.Builder builder) {
        EngineConfig.SplitMode mode = splitMode;
        boolean remotes = rpcServers != null && !rpcServers.isEmpty();
        int contextRanks = contextRankCount();
        if (mode == null && contextRanks > 0) {
            mode = EngineConfig.SplitMode.CONTEXT;
        }
        if (mode == null && ((devices != null && devices.size() > 1) || remotes)) {
            mode = EngineConfig.SplitMode.LAYER;
        }
        if (mode != null) {
            builder.splitMode(mode);
        }
        if (rpcServers != null) {
            builder.rpcServers(rpcServers);
        }
        if (noDevice) {
            builder.backend(EngineConfig.BackendMode.CPU);
        }
        if (devices != null) {
            builder.gpuDevices(devices);
            if (devices.size() == 1 && !EngineConfig.isDeviceIndex(devices.get(0))) {
                builder.deviceFilter(devices.get(0));
            }
        } else if (contextRanks > 0) {
            builder.gpuDevices(defaultDevices(contextRanks));
        }
        if (tensorSplit != null) {
            builder.tensorSplit(tensorSplit);
        }
        if (mainGpu >= 0) {
            builder.mainGpu(mainGpu);
        }
        if (prefillContextParallel != null && prefillContextParallel > 1) {
            builder.prefillContextParallelSize(prefillContextParallel);
        }
    }

    /**
     * The checks that need the effective configuration, the properties and the flags together. A layer or row split
     * on the CPU backend ({@code --cpu}, {@code --device none}) is refused by {@link EngineConfig.Builder#build()}.
     * Under split mode none, {@code --main-gpu} picks the one GPU from the {@code --device} list (or from every
     * GPU), as llama.cpp does ({@link EngineConfig#picksMainGpu()}).
     */
    private void check(EngineConfig cfg) throws FlagException {
        if (!cfg.rpcServers().isEmpty() && !cfg.backend().usesGpu()) {
            throw new FlagException(RPC + " adds remote GPU devices; it cannot run the CPU engine (" + DEVICE
                    + " none or the cpu backend)");
        }
        if (cfg.rowSplit()) {
            int listed = cfg.gpuDevices().size();
            if (tensorSplit != null && listed > 0 && tensorSplit.length != listed) {
                throw new FlagException(TENSOR_SPLIT + " gives " + tensorSplit.length + " share(s) for the "
                        + listed + " GPU(s) of " + DEVICE + "; give one share per device");
            }
            return;
        }
        if (cfg.tensorMode()) {
            if (tensorSplit != null) {
                throw new FlagException(TENSOR_SPLIT + " gives per-device shares, but " + SPLIT_MODE + "=tensor"
                        + " divides every attention head and feed-forward neuron evenly over the ranks; use "
                        + SPLIT_MODE + "=row for proportional feed-forward shards");
            }
            return;
        }
        if (cfg.contextMode()) {
            if (tensorSplit != null) {
                throw new FlagException(TENSOR_SPLIT + " gives per-device shares, but " + SPLIT_MODE + "=context"
                        + " divides the context positions evenly over the ranks; use " + SPLIT_MODE
                        + "=row for proportional feed-forward shards");
            }
            int contextRanks = contextRankCount();
            if (contextRanks > 0) {
                checkContextSplit(cfg, contextRanks);
            }
            if (kvInterleaveSize != null) {
                checkKvInterleave(cfg);
            }
            return;
        }
        if (kvInterleaveSize != null) {
            // a non-context launch runs no decode-context-parallel KV interleave the flag could configure
            throw new FlagException(DCP_KV_INTERLEAVE + " " + kvInterleaveSize + " configures the KV interleave"
                    + " of a decode-context-parallel split, but the launch runs a "
                    + (cfg.splitMode() == EngineConfig.SplitMode.NONE ? "single-GPU" : cfg.splitMode().llamaName())
                    + " split; give " + SPLIT_MODE + "=context, " + DECODE_CONTEXT_PARALLEL + " N or "
                    + PREFILL_CONTEXT_PARALLEL + " N");
        }
        if (!cfg.split()) {
            if (tensorSplit != null) {
                throw new FlagException(TENSOR_SPLIT + " sizes the layer spans of " + SPLIT_MODE + " layer; split"
                        + " mode none runs every layer on one GPU");
            }
            return;
        }
        int listed = cfg.gpuDevices().size();
        int total = listed + cfg.rpcServers().size();
        if (tensorSplit != null && listed > 0 && tensorSplit.length != total) {
            throw new FlagException(TENSOR_SPLIT + " gives " + tensorSplit.length + " share(s) for the " + total
                    + " device(s) of " + DEVICE + " and " + RPC + "; give one share per device");
        }
        if (mainGpu >= 0 && listed > 0 && mainGpu >= listed) {
            throw new FlagException(MAIN_GPU + " " + mainGpu + " is beyond the " + listed + " GPU(s) of " + DEVICE
                    + " (indices 0.." + (listed - 1) + ")");
        }
    }

    /** The tensor shares read so far, empty when no {@code --tensor-split} was given; a copy. */
    double[] tensorSplitShares() {
        return tensorSplit == null ? new double[0] : tensorSplit.clone();
    }

    /** The context rank count the vLLM size flags name: {@code -dcp}'s, else {@code -pcp}'s, else 0. */
    private int contextRankCount() {
        if (decodeContextParallel != null && decodeContextParallel > 1) {
            return decodeContextParallel;
        }
        if (prefillContextParallel != null && prefillContextParallel > 1) {
            return prefillContextParallel;
        }
        return 0;
    }

    /** The flag that named {@link #contextRankCount()}'s rank count. */
    private String contextRankFlag() {
        return decodeContextParallel != null && decodeContextParallel > 1
                ? DECODE_CONTEXT_PARALLEL : PREFILL_CONTEXT_PARALLEL;
    }

    /**
     * The context-parallel size flags against each other: a prefill size beside a decode size must name the same
     * rank count (the engine's prefill query split runs over the context split's own ranks), and a size beside an
     * explicit {@code --split-mode} must ask for the context split, as vLLM's sizes name the dcp-within-tp nesting
     * this engine does not run.
     */
    private void checkContextFlags() throws FlagException {
        int dcp = decodeContextParallel == null ? 1 : decodeContextParallel;
        int pcp = prefillContextParallel == null ? 1 : prefillContextParallel;
        if (dcp > 1 && pcp > 1 && dcp != pcp) {
            throw new FlagException(PREFILL_CONTEXT_PARALLEL + " " + pcp + " splits every prefill's query rows over"
                    + " the ranks of the context split " + DECODE_CONTEXT_PARALLEL + " " + dcp + " runs, so it"
                    + " must equal " + dcp);
        }
        if (dcp > 1 && splitMode != null && splitMode != EngineConfig.SplitMode.CONTEXT) {
            throw new FlagException(DECODE_CONTEXT_PARALLEL + " " + dcp + " runs a " + SPLIT_MODE + "=context"
                    + " split, but " + SPLIT_MODE + "=" + splitMode.llamaName() + " was given; the engine runs one"
                    + " split mode, not vLLM's dcp-within-tp nesting");
        }
        if (pcp > 1 && splitMode != null && splitMode != EngineConfig.SplitMode.CONTEXT) {
            throw new FlagException(PREFILL_CONTEXT_PARALLEL + " " + pcp + " splits every prefill's query rows over"
                    + " the ranks of a " + SPLIT_MODE + "=context split, but " + SPLIT_MODE + "="
                    + splitMode.llamaName() + " was given; the engine runs one split mode");
        }
    }

    /**
     * A context-parallel size above 1 against the configuration it built: one device per rank, and a prefill query
     * split needs the contiguous memory its context blocks live in, so a paged memory refuses. The rpc and CPU
     * combinations refuse from the configuration's own build ({@code EngineConfig.Builder#build()}), ahead of here.
     */
    private void checkContextSplit(EngineConfig cfg, int contextRanks) throws FlagException {
        String flag = contextRankFlag();
        int listed = cfg.gpuDevices().size();
        if (listed != contextRanks) {
            throw new FlagException(flag + " " + contextRanks + " name(s) " + contextRanks + " rank(s), but "
                    + DEVICE + " lists " + listed + " device(s); give one device per rank");
        }
        if (prefillContextParallel != null && prefillContextParallel > 1 && cfg.memory().paged()) {
            throw new FlagException(PREFILL_CONTEXT_PARALLEL + " " + prefillContextParallel + " splits every"
                    + " prefill's query rows at the context blocks of a contiguous memory, but the memory is paged"
                    + " (" + MemoryConfig.PROP_PARALLEL + " above 1 or " + MemoryConfig.PROP_KV_PAGED
                    + "=true): a paged context split partitions the page pools instead and carries no prefill"
                    + " query split; keep the prefill context-parallel size at 1");
        }
    }

    /**
     * {@code --dcp-kv-cache-interleave-size} against the configuration it built: the flag names the paged context
     * split's page-granular interleave (a page belongs to the rank owning its first stored position, one page per
     * rank in turn), so it takes a context split over a paged memory, and its value must equal the page size the
     * pools run ({@code neuralfree.kv.page.size}). Reached under the context mode only; {@link #check} refuses the
     * flag beside every other split.
     */
    private void checkKvInterleave(EngineConfig cfg) throws FlagException {
        if (!cfg.memory().paged()) {
            throw new FlagException(DCP_KV_INTERLEAVE + " " + kvInterleaveSize + " names the page-granular KV"
                    + " interleave of a paged context split, but the memory is contiguous; give "
                    + MemoryConfig.PROP_PARALLEL + " above 1 (--parallel) or " + MemoryConfig.PROP_KV_PAGED
                    + "=true");
        }
        if (kvInterleaveSize != cfg.memory().kvPageSize()) {
            throw new FlagException(DCP_KV_INTERLEAVE + " " + kvInterleaveSize + " does not match the page size "
                    + cfg.memory().kvPageSize() + " the paged KV pools run: the context split interleaves whole"
                    + " pages, one page per rank in turn; give " + kvInterleaveSize + " = the page size, or set "
                    + MemoryConfig.PROP_KV_PAGE_SIZE + "=" + kvInterleaveSize);
        }
    }

    /** The device selectors {@code 0..ranks-1} a context-parallel size implies without a {@code --device} list. */
    private static List<String> defaultDevices(int ranks) {
        List<String> out = new ArrayList<>(ranks);
        for (int i = 0; i < ranks; i++) {
            out.add(Integer.toString(i));
        }
        return Collections.unmodifiableList(out);
    }

    /** One vLLM context-parallel size: a whole number of ranks, at least 1. */
    private static int rankSize(String flag, String value) throws FlagException {
        int n;
        try {
            n = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new FlagException(flag + " must be a whole number of ranks, got '" + value + "'");
        }
        if (n < 1) {
            throw new FlagException(flag + " must be at least 1, got " + n);
        }
        return n;
    }

    /** The device selectors read so far, or null when none (or {@code --device none}) was given; a copy. */
    List<String> deviceSelectors() {
        return devices == null ? null : new ArrayList<>(devices);
    }

    /** Whether a {@code --device none} read so far keeps the model off every GPU. */
    boolean noDevice() {
        return noDevice;
    }

    /** The split mode read so far, or null when no {@code --split-mode} was given. */
    EngineConfig.SplitMode splitModeValue() {
        return splitMode;
    }

    /** The main GPU index read so far, or -1 when no {@code --main-gpu} was given. */
    int mainGpuValue() {
        return mainGpu;
    }

    private void splitMode(String flag, String value) throws FlagException {
        try {
            splitMode = EngineConfig.SplitMode.parse(value);
        } catch (IllegalArgumentException e) {
            throw new FlagException(flag + " " + value + ": " + e.getMessage());
        }
    }

    /** llama.cpp's list: proportions separated by {@code ,}, {@code /} or {@code ;}, every one a finite value above 0. */
    private void tensorSplit(String flag, String value) throws FlagException {
        String[] parts = value.trim().split("[,/;]+", -1);
        double[] shares = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            String part = parts[i].trim();
            try {
                shares[i] = Double.parseDouble(part);
            } catch (NumberFormatException e) {
                throw new FlagException(flag + " " + value + ": '" + part + "' is not a number");
            }
            if (shares[i] == 0.0) {
                throw new FlagException(flag + " " + value + ": a share of 0 leaves its GPU without layers; drop"
                        + " that GPU from " + DEVICE + " instead");
            }
            if (!(shares[i] > 0.0) || !Double.isFinite(shares[i])) {
                throw new FlagException(flag + " " + value + ": every share must be a finite value above 0, got "
                        + part);
            }
        }
        tensorSplit = shares;
    }

    /** A comma-separated device list: indices or name substrings, or {@code none} alone. */
    private void devices(String flag, String value) throws FlagException {
        List<String> out = new ArrayList<>();
        for (String entry : value.split(",", -1)) {
            String s = entry.trim();
            if (s.isEmpty()) {
                throw new FlagException(flag + " " + value + " has an empty device entry");
            }
            out.add(s);
        }
        boolean none = out.size() == 1 && out.get(0).equalsIgnoreCase(NO_DEVICE);
        for (String s : out) {
            if (!none && s.equalsIgnoreCase(NO_DEVICE)) {
                throw new FlagException(flag + " " + value + ": " + NO_DEVICE + " keeps the model off every GPU"
                        + " and cannot be listed with devices");
            }
        }
        if (none && rpcServers != null && !rpcServers.isEmpty()) {
            throw new FlagException(flag + " " + NO_DEVICE + " keeps the model off every GPU and cannot carry the"
                    + " remote servers of " + RPC);
        }
        noDevice = none;
        devices = none ? null : Collections.unmodifiableList(out);
    }

    /**
     * llama.cpp's {@code --rpc}: comma-separated {@code host:port[@key]} remote compute servers a layer split adds
     * as devices. An entry without an inline key reads {@code NEURALFREE_RPC_KEY}.
     */
    private void rpc(String value) throws FlagException {
        if (noDevice) {
            throw new FlagException(RPC + " cannot be combined with " + DEVICE + " " + NO_DEVICE
                    + ", which keeps the model off every GPU");
        }
        List<RpcServerSpec> out = new ArrayList<>();
        String shown = RpcServerSpec.masked(value);
        for (String entry : value.split(",", -1)) {
            String s = entry.trim();
            if (s.isEmpty()) {
                throw new FlagException(RPC + " " + shown + " has an empty server entry");
            }
            try {
                out.add(RpcServerSpec.parse(s));
            } catch (IllegalArgumentException e) {
                throw new FlagException(RPC + " " + shown + ": " + e.getMessage());
            }
        }
        rpcServers = Collections.unmodifiableList(out);
    }

    /** The remote servers read so far, or null when no {@code --rpc} was given; a copy. */
    List<RpcServerSpec> rpcServerSpecs() {
        return rpcServers == null ? null : new ArrayList<>(rpcServers);
    }

    private void mainGpu(String flag, String value) throws FlagException {
        try {
            int n = Integer.parseInt(value.trim());
            if (n < 0) {
                throw new FlagException(flag + " " + value + " must be a device index >= 0");
            }
            mainGpu = n;
        } catch (NumberFormatException e) {
            throw new FlagException(flag + " " + value + " is not a device index");
        }
    }

    /** The help lines of the flags, in the layout of {@link EngineLoadFlags#backendUsage()}. */
    static List<String> usage() {
        return Arrays.asList(
                "  --device, -dev <d1,d2,..>    GPUs to use: 0-based device indices of the backend's list or",
                "                               name substrings (see --list-devices); one device picks the GPU",
                "                               of the load, two or more split it by layer unless --split-mode",
                "                               says otherwise; none = no offload (the CPU engine)",
                "  --split-mode, -sm <m>        none (one GPU, default) | layer (each GPU holds a contiguous",
                "                               span of layers and their KV caches) | row (each GPU holds a",
                "                               slice of every dense feed-forward and an all-reduce sums the",
                "                               partials) | tensor (vLLM tensor parallelism: the attention heads,",
                "                               their KV caches and the feed-forward are divided over the ranks) |",
                "                               context (vLLM decode context parallelism: the ranks divide the",
                "                               context positions and merge their partial softmaxes)",
                "  --tensor-split, -ts <n0,n1,..>  Share of the model each GPU takes under layer (layer spans)",
                "                               or row (feed-forward neurons), e.g. 3,1 (default equal); a",
                "                               tensor split divides every head and neuron evenly",
                "  --main-gpu, -mg <i>          Index into the --device list of the GPU holding the embedding",
                "                               and the output head (default 0); with --split-mode none, the",
                "                               one GPU the model runs on",
                "  --decode-context-parallel-size, -dcp <n>",
                "                               vLLM decode context parallelism over n GPUs: --split-mode",
                "                               context over the first n device indices, or over the n",
                "                               devices --device lists; 1 is vLLM's default",
                "  --prefill-context-parallel-size, -pcp <n>",
                "                               vLLM prefill context parallelism: every prefill's query",
                "                               rows split over the context split's n ranks (equal to",
                "                               -dcp's); needs the contiguous memory",
                "  --dcp-kv-cache-interleave-size <n>",
                "                               vLLM's DCP KV interleave: the value must equal the page",
                "                               size of the paged memory the context split interleaves",
                "                               (one page per rank in turn); other values refuse",
                "  --rpc <h:p[@k],..>           Remote compute servers the layer split adds as devices",
                "                               (llama.cpp --rpc): each runs nftool rpc-server and contributes",
                "                               its GPU as a layer target; the shared key is inline or in",
                "                               NEURALFREE_RPC_KEY (neuralfree.rpc)");
    }

    @Override
    public String toString() {
        return "SplitFlags{mode=" + splitMode + ", devices=" + devices + (noDevice ? ", none" : "")
                + ", rpc=" + rpcServers + ", tensorSplit=" + Arrays.toString(tensorSplit) + ", mainGpu=" + mainGpu
                + '}';
    }
}
