package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.demo.EngineLoadFlags;
import com.eonmux.neuralfree.gguf.cpu.numa.CpuThreadPolicy;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The sweep a {@code nftool bench} run expands, llama.cpp llama-bench's {@code cmd_params}
 * ({@code tools/llama-bench/llama-bench.cpp}): a value list per parameter, filled by repeating flags or by
 * comma-separated lists whose members can be ranges ({@code first-last}, {@code first-last+step},
 * {@code first-last*mult}). {@link #instances()} lays the cross product out in the reference's order — the
 * engine settings outermost ({@code -ngl}, {@code -ncmoe}, {@code -sm}, {@code -lm}, {@code -mg}, {@code -dev},
 * {@code -ts}, {@code -ot}, {@code -embd}, {@code -b}, {@code -ub}, {@code -ctk}, {@code -ctv}, {@code -nkvo}),
 * then the depth, then the prompt, generation and pair shapes — so each model reload serves consecutive runs.
 * The {@code -hf} repositories resolve to their cached model paths here, as llama-bench downloads them before it
 * expands the sweep. Mutable while the flags parse, read-only after.
 */
final class BenchGrid {

    /** llama.cpp's defaults ({@code cmd_params_defaults}). */
    static final int DEFAULT_REPS = 5;
    static final int DEFAULT_N_PROMPT = 512;
    static final int DEFAULT_N_GEN = 128;
    static final int DEFAULT_N_DEPTH = 0;
    static final int DEFAULT_N_BATCH = 2048;
    static final int DEFAULT_N_UBATCH = 512;
    static final KvDType DEFAULT_KV_TYPE = KvDType.F16;
    static final int DEFAULT_NGL = -1;
    static final int DEFAULT_DELAY = 0;

    /** The engine configuration the sweep starts from: the {@code -Dneuralfree.*} properties. */
    final EngineConfig base;

    final List<String> models = new ArrayList<>();
    final List<Integer> nPrompt = new ArrayList<>();
    final List<Integer> nGen = new ArrayList<>();
    final List<List<Integer>> nPg = new ArrayList<>();
    final List<Integer> nDepth = new ArrayList<>();
    final List<Integer> nBatch = new ArrayList<>();
    final List<Integer> nUbatch = new ArrayList<>();
    final List<KvDType> typeK = new ArrayList<>();
    final List<KvDType> typeV = new ArrayList<>();
    final List<Integer> ngl = new ArrayList<>();
    final List<Integer> nCpuMoe = new ArrayList<>();
    final List<Boolean> noKvOffload = new ArrayList<>();
    final List<EngineConfig.SplitMode> splitModes = new ArrayList<>();
    final List<Integer> mainGpus = new ArrayList<>();
    final List<double[]> tensorSplits = new ArrayList<>();
    /** The {@code -dev} selections: empty for auto, {@code none}, or the engine's comma-joined selectors. */
    final List<String> devices = new ArrayList<>();
    /** The {@code -ot} sweep members: one per comma-separated group. */
    final List<BenchTensorOverrides> tensorOverrides = new ArrayList<>();
    final List<BenchLoadMode> loadModes = new ArrayList<>();
    final List<Boolean> embeddings = new ArrayList<>();
    /** The {@code -hf} repositories and their paired {@code -hff} files, before they resolve to model paths. */
    final List<String> hfRepos = new ArrayList<>();
    final List<String> hfFiles = new ArrayList<>();
    /** The {@code -hft} token, or null for the environment's token ({@code HF_TOKEN}, else the token file). */
    String hfToken;
    /** The CPU pool size every run reports; a single value, set before the pool starts. */
    int threads = defaultThreads();
    boolean threadsGiven;
    boolean devicesGiven;
    boolean overridesGiven;
    boolean loadModeGiven;
    int reps = DEFAULT_REPS;
    int delay = DEFAULT_DELAY;
    boolean progress;
    boolean noWarmup;
    /**
     * The device timing {@code --kernels} asks for: {@link KernelTiming.Mode#DISPATCH} times every dispatch with the
     * decode-step replay off, {@link KernelTiming.Mode#STEP} keeps the replay and times each graph launch whole.
     */
    KernelTiming.Mode kernels = KernelTiming.Mode.OFF;
    /** True when {@code --host-breakdown} reports where the host time of each test's decode steps goes. */
    boolean hostBreakdown;
    /** What each generated token reads back ({@code --tg-readout}). */
    BenchReadout readout = BenchReadout.LOGITS;
    BenchOutput output = BenchOutput.MARKDOWN;
    BenchOutput outputErr = BenchOutput.NONE;
    /** The model paths of the {@code -hf} repositories, once resolved. */
    private List<String> resolvedHf;

    /** A sweep over {@code base}'s engine settings until its flags change them. */
    BenchGrid(EngineConfig base) {
        this.base = base;
    }

    /**
     * The CPU pool size the runs report when {@code -t} is absent: the size {@link CpuParallel#threadCount()} gives
     * the pool, else (an invalid property, which fails when the pool starts) the count of
     * {@link CpuThreadPolicy#DEFAULT_POLICY} over the host's cores.
     */
    static int defaultThreads() {
        try {
            return CpuParallel.threadCount();
        } catch (IllegalArgumentException e) {
            return CpuThreadPolicy.host().threads(CpuThreadPolicy.DEFAULT_POLICY);
        }
    }

    /**
     * Parses llama.cpp's {@code parse_int_range}: comma-separated members of {@code first},
     * {@code first-last}, {@code first-last+step} or {@code first-last*mult}; the step forms walk the range
     * multiplicatively or additively.
     *
     * @param flag name of the flag being parsed, for the error messages
     * @throws BenchCommand.UsageException for a member that parses badly or walks nothing
     */
    static List<Integer> parseIntRange(String value, String flag) throws BenchCommand.UsageException {
        java.util.regex.Pattern member = java.util.regex.Pattern.compile("(\\d+)(?:-(\\d+)(?:([+*])(\\d+))?)?");
        List<Integer> result = new ArrayList<>();
        for (String part : value.split(",", -1)) {
            java.util.regex.Matcher m = member.matcher(part);
            if (!m.matches()) {
                throw new BenchCommand.UsageException(flag + " needs integers or ranges like '128-1024+128', not '"
                        + part + "'");
            }
            int first = Integer.parseInt(m.group(1));
            int last = m.group(2) != null ? Integer.parseInt(m.group(2)) : first;
            if (last < first) {
                throw new BenchCommand.UsageException(flag + " range '" + part + "' ends before it starts");
            }
            char op = m.group(3) != null ? m.group(3).charAt(0) : '+';
            int step = m.group(4) != null ? Integer.parseInt(m.group(4)) : 1;
            if (step < 1) {
                throw new BenchCommand.UsageException(flag + " step must be positive, got '" + part + "'");
            }
            int prev;
            for (int i = first; i <= last; ) {
                result.add(i);
                prev = i;
                i = op == '*' ? i * step : i + step;
                if (i <= prev) {
                    throw new BenchCommand.UsageException(
                            flag + " range '" + part + "' with step " + step + " never leaves " + prev);
                }
            }
        }
        return result;
    }

    /**
     * The runs of the sweep: the cross product in llama.cpp's reload-minimizing order. Empty value lists take
     * their defaults (the {@code -Dneuralfree.*} settings for the split and KV-offload parameters), and zero
     * values give no run ({@code -p 0}, {@code -n 0}, a {@code -pg 0,0}), like the reference skips them; a sweep
     * of only zeros is refused. {@code -ts} without {@code -sm}, and a {@code -dev} member listing two or more
     * devices, split by layer, llama-bench's default split mode.
     *
     * @throws BenchCommand.UsageException for no model, a run whose batch is smaller than its ubatch, no non-zero
     *                                     shape, or split and KV-offload settings the engine cannot run together
     */
    List<BenchInstance> instances() throws BenchCommand.UsageException {
        List<String> allModels = allModels();
        if (allModels.isEmpty()) {
            throw new BenchCommand.UsageException("bench needs a model path (-m or -hf)");
        }
        List<Integer> p = orDefault(nPrompt, DEFAULT_N_PROMPT);
        List<Integer> g = orDefault(nGen, DEFAULT_N_GEN);
        List<Integer> d = orDefault(nDepth, DEFAULT_N_DEPTH);
        List<Integer> b = orDefault(nBatch, DEFAULT_N_BATCH);
        List<Integer> ub = orDefault(nUbatch, DEFAULT_N_UBATCH);
        List<KvDType> tk = typeK.isEmpty() ? List.of(DEFAULT_KV_TYPE) : typeK;
        List<KvDType> tv = typeV.isEmpty() ? List.of(DEFAULT_KV_TYPE) : typeV;
        List<Integer> nl = orDefault(ngl, DEFAULT_NGL);
        List<Integer> ncmoe = orDefault(nCpuMoe, base.nCpuMoe());
        List<Boolean> kvo = noKvOffload.isEmpty() ? List.of(!base.kvOffload()) : noKvOffload;
        List<EngineConfig.SplitMode> sm = splitModeValues();
        List<BenchLoadMode> lm = loadModes.isEmpty() ? List.of(BenchLoadMode.AUTO) : loadModes;
        List<Integer> mg = mainGpus.isEmpty() ? List.of(base.mainGpu()) : mainGpus;
        List<String> dev = devices.isEmpty() ? List.of("") : devices;
        List<double[]> ts = tensorSplits.isEmpty() ? List.of(base.tensorSplit()) : tensorSplits;
        List<BenchTensorOverrides> ot = tensorOverrides.isEmpty() ? List.of(BenchTensorOverrides.NONE)
                : tensorOverrides;
        List<Boolean> embd = embeddings.isEmpty() ? List.of(Boolean.FALSE) : embeddings;
        List<BenchInstance> instances = new ArrayList<>();
        for (String m : allModels) {
            for (int n : nl) {
                for (int ncm : ncmoe) {
                    for (EngineConfig.SplitMode mode : sm) {
                        for (BenchLoadMode loadMode : lm) {
                            for (int gpu : mg) {
                                for (String device : dev) {
                                    for (double[] split : ts) {
                                        for (BenchTensorOverrides overrides : ot) {
                                            for (boolean embds : embd) {
                                                for (int nb : b) {
                                                    for (int nub : ub) {
                                                        for (KvDType k : tk) {
                                                            for (KvDType v : tv) {
                                                                for (boolean nkvo : kvo) {
                                                                    for (int nd : d) {
                                                                        shape(instances, m, p, g, nd, nb, nub, k, v,
                                                                                n, nkvo,
                                                                                effectiveSplitMode(mode, device),
                                                                                loadMode, gpu, device,
                                                                                split, ncm, overrides, embds);
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (instances.isEmpty()) {
            throw new BenchCommand.UsageException("bench needs a test shape: -p, -n or -pg with a non-zero value");
        }
        checkConfigs(instances);
        return instances;
    }

    /**
     * The models the sweep runs over: {@code -m} paths and the cached path of every {@code -hf} repository, in the
     * reference's order (the {@code -hf} models follow the {@code -m} ones). Resolves the repositories once.
     *
     * @throws BenchCommand.UsageException for a download failure or an {@code -hff} without its {@code -hf}
     */
    private List<String> allModels() throws BenchCommand.UsageException {
        if (resolvedHf == null) {
            resolveHf();
        }
        List<String> all = new ArrayList<>(models);
        all.addAll(resolvedHf);
        return all;
    }

    /** Downloads the {@code -hf} repositories (or reads their cache) into model paths, pairing {@code -hff}. */
    private void resolveHf() throws BenchCommand.UsageException {
        if (!hfRepos.isEmpty() && hfFiles.size() > hfRepos.size()) {
            throw new BenchCommand.UsageException("-hff/--hf-file " + hfFiles.get(hfRepos.size())
                    + " has no matching -hf/--hf-repo");
        }
        if (hfRepos.isEmpty() && !hfFiles.isEmpty()) {
            throw new BenchCommand.UsageException("-hff/--hf-file needs an -hf/--hf-repo to name a file in");
        }
        if (hfRepos.isEmpty() && hfToken != null) {
            throw new BenchCommand.UsageException("-hft/--hf-token needs an -hf/--hf-repo to download");
        }
        List<String> paths = new ArrayList<>();
        for (int i = 0; i < hfRepos.size(); i++) {
            String file = i < hfFiles.size() ? hfFiles.get(i) : null;
            try {
                paths.add(DownloadCommand.downloadModel(hfRepos.get(i), file, hfToken));
            } catch (DownloadCommand.UsageException e) {
                throw new BenchCommand.UsageException(e.getMessage());
            } catch (IOException e) {
                throw new BenchCommand.UsageException("cannot download '" + hfRepos.get(i) + "': " + e.getMessage());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new BenchCommand.UsageException("the download of '" + hfRepos.get(i) + "' was interrupted");
            }
        }
        resolvedHf = paths;
    }

    /**
     * The split modes of the sweep: the {@code -sm} values, or the base configuration's mode. A {@code -ts} value
     * without {@code -sm}, and a {@code -dev} list holding two or more devices, run a layer split, as llama-bench's
     * default mode does when it spreads a model over several devices.
     */
    private List<EngineConfig.SplitMode> splitModeValues() {
        if (!splitModes.isEmpty()) {
            return splitModes;
        }
        return List.of(tensorSplits.isEmpty() ? base.splitMode() : EngineConfig.SplitMode.LAYER);
    }

    /**
     * The split mode one run uses: the swept or explicit mode, or, when neither {@code -sm} nor {@code -ts} was
     * given, a layer split for a {@code -dev} member listing two or more devices, as llama.cpp's default mode does.
     */
    private EngineConfig.SplitMode effectiveSplitMode(EngineConfig.SplitMode mode, String device) {
        if (!splitModes.isEmpty() || !tensorSplits.isEmpty()) {
            return mode;
        }
        return device.indexOf(',') >= 0 ? EngineConfig.SplitMode.LAYER : mode;
    }

    /** Appends the prompt, generation and pair shapes of one engine setting combination, in the reference's order. */
    private void shape(List<BenchInstance> instances, String m, List<Integer> p, List<Integer> g,
                       int nd, int nb, int nub, KvDType k, KvDType v, int n, boolean nkvo,
                       EngineConfig.SplitMode mode, BenchLoadMode loadMode, int gpu, String device,
                       double[] split, int ncmoe, BenchTensorOverrides overrides, boolean embd)
            throws BenchCommand.UsageException {
        for (int np : p) {
            if (np > 0) {
                append(instances, m, np, 0, nd, nb, nub, k, v, n, nkvo, mode, loadMode, gpu, device, split, ncmoe,
                        overrides, embd);
            }
        }
        for (int ng : g) {
            if (ng > 0) {
                append(instances, m, 0, ng, nd, nb, nub, k, v, n, nkvo, mode, loadMode, gpu, device, split, ncmoe,
                        overrides, embd);
            }
        }
        for (List<Integer> pg : nPg) {
            if (pg.get(0) != 0 || pg.get(1) != 0) {
                append(instances, m, pg.get(0), pg.get(1), nd, nb, nub, k, v, n, nkvo, mode, loadMode, gpu, device,
                        split, ncmoe, overrides, embd);
            }
        }
    }

    /**
     * Appends one run, refusing a batch smaller than the ubatch it would feed and a depth prefill in the embedding
     * mode (the embedding pass evaluates a prompt from position 0, with no stored depth to continue from).
     */
    private void append(List<BenchInstance> instances, String m, int np, int ng, int nd, int nb, int nub,
                        KvDType k, KvDType v, int n, boolean nkvo, EngineConfig.SplitMode mode,
                        BenchLoadMode loadMode, int gpu, String device, double[] split, int ncmoe,
                        BenchTensorOverrides overrides, boolean embd) throws BenchCommand.UsageException {
        if (nb < nub) {
            throw new BenchCommand.UsageException("-b/--batch-size " + nb
                    + " must stay >= -ub/--ubatch-size " + nub + " (llama.cpp processes at most one ubatch per batch)");
        }
        if (embd && nd > 0) {
            throw new BenchCommand.UsageException("-d/--n-depth cannot be swept with -embd 1: the embedding pass"
                    + " evaluates a prompt from position 0, with no stored depth to continue from");
        }
        instances.add(new BenchInstance(m, np, ng, nd, nb, nub, k, v, n, nkvo, mode, gpu, split, threads,
                ncmoe, device, overrides, loadMode, embd));
    }

    /**
     * Builds every run's engine configuration once, so a combination of split and KV-offload flags the engine
     * refuses (a layer split on the CPU engine, a tensor split under split mode none, a main GPU beyond the
     * listed devices) is reported before any model loads.
     *
     * @throws BenchCommand.UsageException naming the flag or setting the engine refuses
     */
    private void checkConfigs(List<BenchInstance> instances) throws BenchCommand.UsageException {
        java.util.Set<String> seen = new java.util.HashSet<>();
        for (BenchInstance inst : instances) {
            if (!seen.add(inst.engineKey())) {
                continue;
            }
            MemoryConfig mc = MemoryConfig.builder()
                    .nCtx(Math.max(inst.nCtxNeeded(), 1))
                    .ubatch(inst.nUbatch)
                    .typeK(inst.typeK)
                    .typeV(inst.typeV)
                    .kvPaged(inst.embeddings)
                    .build();
            try {
                inst.engineConfig(base, mc);
            } catch (EngineLoadFlags.FlagException | IllegalArgumentException e) {
                throw new BenchCommand.UsageException(e.getMessage());
            }
        }
    }

    /** The list, or the one-value default when no flag gave values. */
    private static List<Integer> orDefault(List<Integer> list, int dflt) {
        return list.isEmpty() ? List.of(dflt) : list;
    }

    /**
     * The runs grouped by {@link BenchInstance#engineKey}, in encounter order, so each group shares one model
     * load sized for its largest context.
     */
    Map<String, List<BenchInstance>> groups(List<BenchInstance> instances) {
        Map<String, List<BenchInstance>> groups = new LinkedHashMap<>();
        for (BenchInstance inst : instances) {
            groups.computeIfAbsent(inst.engineKey(), key -> new ArrayList<>()).add(inst);
        }
        return groups;
    }
}
