package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.demo.EngineLoadFlags;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.cpu.numa.NumaStrategy;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.hub.HubClient;
import com.eonmux.neuralfree.gguf.hub.HubRef;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.stream.Stream;

/**
 * {@code nftool bench}: the parameter sweep of llama.cpp llama-bench ({@code tools/llama-bench/llama-bench.cpp})
 * on the NeuralFree engine. {@link BenchGrid} expands the {@code -p}, {@code -n}, {@code -pg} and {@code -d}
 * value grids over the engine settings ({@code -b}, {@code -ub}, {@code -ctk}, {@code -ctv}, {@code -ngl},
 * {@code -ncmoe}, {@code -sm}, {@code -lm}, {@code -mg}, {@code -dev}, {@code -ts}, {@code -ot}, {@code -embd},
 * {@code -nkvo});
 * runs that share one model load run together, the context sized for the largest shape of the group. Each run
 * warms up (a full prompt pass and one decode, {@code --no-warmup} skips), then repeats its timed section
 * {@code -r} times: the depth prefill runs untimed, and the clock covers the prompt pass and the decodes. In the
 * embedding mode ({@code -embd 1}) the timed prompt pass runs the model's embedding output instead, and the
 * models without a decoder loop may bench their prompt shapes. The samples land in a {@link BenchTest} whose
 * statistics the {@link BenchPrinters} write as csv, json, jsonl, Markdown or SQL. The engine settings
 * llama-bench sweeps that this engine does not have (flash attention, host-operation offload, the fit margins)
 * are refused by name, never ignored.
 */
final class BenchCommand {

    /** The command surface: {@code --help} and every error print it. */
    static final String USAGE =
            "nftool bench <model> [options]\n"
            + "options:\n"
            + "  -r, --repetitions <n>                     times each test repeats, default " + BenchGrid.DEFAULT_REPS + "\n"
            + "  --delay <seconds>                         pause between tests, default " + BenchGrid.DEFAULT_DELAY + "\n"
            + "  -o, --output <md|none|csv|json|jsonl|sql> output format on stdout, default md\n"
            + "  -oe, --output-err <md|none|csv|json|jsonl|sql>  output format on stderr, default none\n"
            + "  --progress                                print progress indicators to stderr\n"
            + "  --no-warmup                               skip the warmup pass of each test\n"
            + "  --kernels                                 time the GPU dispatches of every test (disables the\n"
            + "                                            decode-step replay) and print per-kernel device time\n"
            + "  --help, -h                                this text\n"
            + "test parameters (lists accept 'a,b,c' and ranges 'first-last+step' or 'first-last*mult'):\n"
            + "  -m, --model <path>                  model to bench, repeatable\n"
            + "  -hf, -hfr, --hf-repo <u>/<m>[:quant]\n"
            + "  -hff, --hf-file <file>              a Hugging Face Hub model, resolved (or read from the cache) into\n"
            + "                                      a model path; -hff names the file and overrides the quant tag\n"
            + "  -hft, --hf-token <token>            Hub access token (default: HF_TOKEN, else the token\n"
            + "                                      file HF_TOKEN_PATH names, else HF_HOME/token,\n"
            + "                                      as `hf auth login` writes it)\n"
            + "  -p, --n-prompt <n>                  prompt-processing tokens (pp), default " + BenchGrid.DEFAULT_N_PROMPT + "\n"
            + "  -n, --n-gen <n>                     token-generation tokens (tg), default " + BenchGrid.DEFAULT_N_GEN + "\n"
            + "  -pg <pp,tg>                         one test processing and generating, repeatable\n"
            + "  -d, --n-depth <n>                   tokens prefilled before each timed run, default " + BenchGrid.DEFAULT_N_DEPTH + "\n"
            + "  -b, --batch-size <n>                logical batch size, default " + BenchGrid.DEFAULT_N_BATCH + "\n"
            + "  -ub, --ubatch-size <n>              prefill chunk (the engine's ubatch), default " + BenchGrid.DEFAULT_N_UBATCH + "\n"
            + "  -ctk, --cache-type-k <t>            KV key type (f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0,\n"
            + "                                      q5_1, fp8_e4m3, fp8_e5m2, fp8_per_token_head,\n"
            + "                                      int8_per_token_head, int4_per_token_head), default f16\n"
            + "  -ctv, --cache-type-v <t>            KV value type, default f16\n"
            + "  -t, --threads <n>                   CPU pool size, set before the pool starts; one value only\n"
            + "  --numa <distribute|isolate|numactl>  CPU thread placement on a multi-node host, set before the\n"
            + "                                      pool starts; no effect with a single NUMA node\n"
            + "  -ngl, --n-gpu-layers <n>            0 runs the CPU engine, any other value the GPU engine, default -1\n"
            + "  -ncmoe, --n-cpu-moe <n>             the first <n> MoE layers keep their experts in host RAM, default 0\n"
            + "  -sm, --split-mode <none|layer|row|tensor|context>  how the model spreads over GPUs, default none\n"
            + "  -lm, --load-mode <m>                tensor reads: auto, none, mmap, mlock, mmap+mlock or dio\n"
            + "  -mg, --main-gpu <i>                 GPU of the embedding and the output head, default 0\n"
            + "  -dev, --device <d0/d1,..>           GPUs to use (indices or name substrings), none = no offload,\n"
            + "                                      auto = the configured devices; each member is one test\n"
            + "  -ts, --tensor-split <n0/n1,..>      share of each GPU for layer or row mode (one test per member),\n"
            + "                                      default equal\n"
            + "  -ot, --override-tensor <p=bt;..>    tensor placement overrides, pattern=buffer type; CPU, CUDA,\n"
            + "                                      ROCm, HIP, Metal, MTL, Vulkan and OpenCL buffer types map\n"
            + "                                      onto host/backend targets, as do the accelerator APIs\n"
            + "                                      (" + BenchTensorOverrides.acceleratorBufferNames()
            + "); each\n"
            + "                                      comma-separated group is one test\n"
            + "  -embd, --embeddings <0|1>           1 times the model's embedding pass over the prompt instead of\n"
            + "                                      the logits pass, default 0\n"
            + "  -nkvo, --no-kv-offload <0|1>        keep the KV caches in host RAM, default 0\n"
            + "Unsupported here: -nopo/--no-op-offload (operations follow their tensors), -fitt/--fit-target (no\n"
            + "per-device reserve margin), -fitc/--fit-ctx (no minimum-context floor) and -fa/--flash-attn\n"
            + "(attention always runs the fused kernels); llama-bench's remaining parameters are refused by name.\n"
            + "Tokens are drawn from a fixed-seed generator, so a sweep is reproducible. The depth prefill is not\n"
            + "timed; the reference reuses its cached state, this bench re-runs it per repetition.\n";

    /** An invalid command line or an unsupported sweep; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    private BenchCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code bench}. */
    static void run(String[] args) {
        try {
            bench(args, System.out, System.err);
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Parses the flags, sweeps the grid and writes every format the run selected; {@code --help} prints the
     * usage and returns without a run.
     *
     * @throws UsageException for an invalid command line or a sweep the engine cannot run
     * @throws IOException    for an unreadable model or a failing forward pass
     */
    static void bench(String[] args, Appendable out, Appendable err) throws IOException, UsageException {
        BenchGrid grid = parse(args);
        if (grid == null) {
            return;
        }
        List<BenchInstance> instances = grid.instances();
        BenchPrinters.BenchPrinter p = BenchPrinters.create(grid.output, out, grid);
        BenchPrinters.BenchPrinter pErr = BenchPrinters.create(grid.outputErr, err, grid);
        if (p != null) {
            p.printHeader(grid);
        }
        if (pErr != null) {
            pErr.printHeader(grid);
        }
        int count = instances.size();
        int idx = 0;
        String priorProfile = null;
        String priorReplay = null;
        if (grid.kernels) {
            priorProfile = System.setProperty(KernelTiming.PROFILE_PROPERTY, "true");
            priorReplay = System.setProperty(GpuForwardBackend.DECODE_REPLAY_PROPERTY, "false");
        }
        try {
            for (List<BenchInstance> group : grid.groups(instances).values()) {
                String priorReadMode = applyLoadMode(group);
                try {
                    try (EngineModel model = open(grid, group)) {
                        Forwarder f = new Forwarder(model, group.get(0), group);
                        for (BenchInstance inst : group) {
                            idx++;
                            runOne(grid, f, inst, idx, count, p, pErr, err);
                        }
                    }
                } finally {
                    restoreReadMode(priorReadMode);
                }
            }
        } finally {
            if (grid.kernels) {
                restoreProperty(KernelTiming.PROFILE_PROPERTY, priorProfile);
                restoreProperty(GpuForwardBackend.DECODE_REPLAY_PROPERTY, priorReplay);
            }
        }
        if (p != null) {
            p.printFooter();
        }
        if (pErr != null) {
            pErr.printFooter();
        }
    }

    /**
     * Parses llama-bench's flags into the sweep; returns null when {@code --help} printed the usage. Unknown
     * options and the options this engine cannot honour refuse by name; long flags accept underscores like the
     * reference replaces them.
     *
     * @throws UsageException naming the offending flag
     */
    static BenchGrid parse(String[] args) throws UsageException {
        for (String a : args) {
            if (a.equals("--help") || a.equals("-h")) {
                System.out.print(USAGE);
                return null;
            }
        }
        BenchGrid grid;
        try {
            grid = new BenchGrid(EngineConfig.fromSystemProperties());
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.startsWith("--")) {
                a = a.replace('_', '-');
            }
            switch (a) {
                case "-m":
                case "--model":
                    for (String m : value(a, args, i++).split(",")) {
                        grid.models.add(m);
                    }
                    break;
                case "-p":
                case "--n-prompt":
                    grid.nPrompt.addAll(BenchGrid.parseIntRange(value(a, args, i++), a));
                    break;
                case "-n":
                case "--n-gen":
                    grid.nGen.addAll(BenchGrid.parseIntRange(value(a, args, i++), a));
                    break;
                case "-pg":
                    grid.nPg.add(pgPair(value(a, args, i++), a));
                    break;
                case "-d":
                case "--n-depth":
                    grid.nDepth.addAll(BenchGrid.parseIntRange(value(a, args, i++), a));
                    break;
                case "-r":
                case "--repetitions":
                    grid.reps = positiveInt(value(a, args, i++), a);
                    break;
                case "-b":
                case "--batch-size":
                    grid.nBatch.addAll(BenchGrid.parseIntRange(value(a, args, i++), a));
                    break;
                case "-ub":
                case "--ubatch-size":
                    grid.nUbatch.addAll(BenchGrid.parseIntRange(value(a, args, i++), a));
                    break;
                case "-ctk":
                case "--cache-type-k":
                    grid.typeK.add(KvDType.parse(value(a, args, i++), a));
                    break;
                case "-ctv":
                case "--cache-type-v":
                    grid.typeV.add(KvDType.parse(value(a, args, i++), a));
                    break;
                case "-ngl":
                case "--n-gpu-layers":
                    grid.ngl.addAll(BenchGrid.parseIntRange(value(a, args, i++), a));
                    break;
                case "-sm":
                case "--split-mode":
                    for (String part : value(a, args, i++).split(",", -1)) {
                        grid.splitModes.add(splitMode(part.trim(), a));
                    }
                    break;
                case "-mg":
                case "--main-gpu":
                    // llama-bench assigns the parsed range here, unlike its appending list flags.
                    grid.mainGpus.clear();
                    grid.mainGpus.addAll(BenchGrid.parseIntRange(value(a, args, i++), a));
                    break;
                case "-ts":
                case "--tensor-split":
                    for (String part : value(a, args, i++).split(",", -1)) {
                        grid.tensorSplits.add(tensorSplit(part.trim(), a));
                    }
                    break;
                case "-nkvo":
                case "--no-kv-offload":
                    for (String part : value(a, args, i++).split(",", -1)) {
                        grid.noKvOffload.add(noKvOffload(part.trim(), a));
                    }
                    break;
                case "-ncmoe":
                case "--n-cpu-moe":
                    grid.nCpuMoe.addAll(BenchGrid.parseIntRange(value(a, args, i++), a));
                    break;
                case "-dev":
                case "--device":
                case "-device":
                    deviceSelections(grid, a, value(a, args, i++));
                    break;
                case "-ot":
                case "--override-tensor":
                    grid.tensorOverrides.addAll(BenchTensorOverrides.parse(a, value(a, args, i++)));
                    grid.overridesGiven = true;
                    break;
                case "-lm":
                case "--load-mode":
                    for (String part : value(a, args, i++).split(",", -1)) {
                        grid.loadModes.add(BenchLoadMode.parse(part.trim(), a));
                    }
                    grid.loadModeGiven = true;
                    break;
                case "-embd":
                case "--embeddings":
                    for (String part : value(a, args, i++).split(",", -1)) {
                        grid.embeddings.add(noKvOffload(part.trim(), a));
                    }
                    break;
                case "-t":
                case "--threads":
                    threads(grid, a, value(a, args, i++));
                    break;
                case "--numa":
                    numa(a, value(a, args, i++));
                    break;
                case "-hf":
                case "-hfr":
                case "--hf-repo":
                    hfRepos(grid, a, value(a, args, i++));
                    break;
                case "-hff":
                case "--hf-file":
                    for (String part : value(a, args, i++).split(",", -1)) {
                        if (part.trim().isEmpty()) {
                            throw new UsageException(a + " has an empty file name");
                        }
                        grid.hfFiles.add(part.trim());
                    }
                    break;
                case "-hft":
                case "--hf-token":
                    hfToken(grid, a, value(a, args, i++));
                    break;
                case "-nopo":
                case "--no-op-offload":
                    throw new UsageException(a + " is not supported: the engine places each operation with its"
                            + " tensors (host tensors compute on the host and device tensors on the device), so"
                            + " there are no host operations to offload separately");
                case "-fitt":
                case "--fit-target":
                    throw new UsageException(a + " is not supported: the engine fits the weights and the context to"
                            + " the device memory itself and has no per-device reserve margin to set"
                            + " (neuralfree.gpu.vram.limit caps the allocation instead)");
                case "-fitc":
                case "--fit-ctx":
                    throw new UsageException(a + " is not supported: the engine fits the largest context the memory"
                            + " holds, or the exact neuralfree.ctx, and has no minimum-context floor for the fit");
                case "-o":
                case "--output":
                    grid.output = BenchOutput.parse(value(a, args, i++), a);
                    break;
                case "-oe":
                case "--output-err":
                    grid.outputErr = BenchOutput.parse(value(a, args, i++), a);
                    break;
                case "--progress":
                    grid.progress = true;
                    break;
                case "--no-warmup":
                    grid.noWarmup = true;
                    break;
                case "--kernels":
                    grid.kernels = true;
                    break;
                case "--delay":
                    grid.delay = nonNegativeInt(value(a, args, i++), a);
                    break;
                case "-fa":
                case "--flash-attn":
                    throw new UsageException(a + " is not supported: attention always runs the fused"
                            + " flash-attention kernels, so off cannot be honored and on/auto change nothing");
                default:
                    if (a.startsWith("-")) {
                        throw new UsageException("unknown option " + a + " (run with --help for the options)");
                    }
                    grid.models.add(a);
            }
        }
        return grid;
    }

    /** One {@code -sm} value: llama.cpp's split mode name, parsed by the engine's own enum. */
    private static EngineConfig.SplitMode splitMode(String v, String flag) throws UsageException {
        try {
            return EngineConfig.SplitMode.parse(v);
        } catch (IllegalArgumentException e) {
            throw new UsageException(flag + " " + v + ": " + e.getMessage());
        }
    }

    /**
     * One {@code -ts} value: the shares {@link EngineLoadFlags} reads them, so the separator, count and share
     * checks are the engine's, not a second copy. Comma-separated values are separate sweep members, as in
     * llama-bench, so only {@code /} and {@code ;} separate shares here.
     */
    private static double[] tensorSplit(String v, String flag) throws UsageException {
        try {
            return EngineLoadFlags.tensorSplitShares(flag, v);
        } catch (EngineLoadFlags.FlagException e) {
            throw new UsageException(e.getMessage());
        }
    }

    /**
     * The device selections of one {@code -dev} value: llama-bench separates its sweep members by {@code ,} and
     * the devices of one member by {@code /}; each member goes through the engine's {@link SplitFlags} parser, so
     * {@code auto}, {@code none} and the selector lists are read and checked as every other tool checks them.
     */
    private static void deviceSelections(BenchGrid grid, String flag, String value) throws UsageException {
        for (String combo : value.split(",", -1)) {
            String member = combo.trim();
            if (member.isEmpty()) {
                throw new UsageException(flag + " has an empty device selection");
            }
            try {
                String canonical = EngineLoadFlags.deviceSelection(flag, member.replace('/', ','));
                grid.devices.add(canonical == null ? "" : canonical);
            } catch (EngineLoadFlags.FlagException e) {
                throw new UsageException(e.getMessage());
            }
        }
        grid.devicesGiven = true;
    }

    /**
     * One {@code -t} value: a single positive thread count, applied to the process-wide CPU pool before it starts
     * ({@code neuralfree.cpu.threads}). A list or a repeated flag is refused by name: the pool is sized once.
     */
    private static void threads(BenchGrid grid, String flag, String value) throws UsageException {
        List<Integer> values = BenchGrid.parseIntRange(value, flag);
        if (values.size() != 1 || grid.threadsGiven) {
            throw new UsageException(flag + " takes one thread count: the CPU pool is sized once before it starts"
                    + " (neuralfree.cpu.threads), so a sweep of thread counts needs separate processes");
        }
        int n = values.get(0);
        if (n < 1) {
            throw new UsageException(flag + " needs a positive thread count, not '" + value + "'");
        }
        System.setProperty(CpuParallel.THREADS_PROPERTY, Integer.toString(n));
        grid.threads = n;
        grid.threadsGiven = true;
    }

    /**
     * One {@code --numa} value, applied to the process-wide property before the CPU pool starts
     * ({@code neuralfree.numa}); the pool reads it once, so a sweep of strategies needs separate processes.
     */
    private static void numa(String flag, String value) throws UsageException {
        NumaStrategy strategy;
        try {
            strategy = NumaStrategy.parse(value);
        } catch (IllegalArgumentException e) {
            throw new UsageException(flag + ": " + e.getMessage());
        }
        System.setProperty(NumaStrategy.PROPERTY, strategy.label());
    }

    /** One {@code -hft} token, validated without ever printing it. */
    private static void hfToken(BenchGrid grid, String flag, String value) throws UsageException {
        if (HubClient.validToken(value) == null) {
            throw new UsageException(flag + " is not a valid Hugging Face token: hf_ followed by 34 to 253"
                    + " alphanumerics");
        }
        grid.hfToken = value.trim();
    }

    /** The {@code -hf} repositories of one value: comma-separated Hub repository expressions, each validated. */
    private static void hfRepos(BenchGrid grid, String flag, String value) throws UsageException {
        for (String part : value.split(",", -1)) {
            String spec = part.trim();
            if (spec.isEmpty()) {
                throw new UsageException(flag + " has an empty repository");
            }
            try {
                grid.hfRepos.add(HubRef.parse(spec).toString());
            } catch (IllegalArgumentException e) {
                throw new UsageException(flag + ": " + e.getMessage());
            }
        }
    }

    /** One {@code -nkvo} value: llama-bench's {@code <0|1>} boolean. */
    private static boolean noKvOffload(String v, String flag) throws UsageException {
        if (v.equals("0")) {
            return false;
        }
        if (v.equals("1")) {
            return true;
        }
        throw new UsageException(flag + " takes 0 or 1, not '" + v + "'");
    }

    /** The {@code pp,tg} pair of {@code -pg}: exactly two non-negative integers. */
    private static List<Integer> pgPair(String v, String flag) throws UsageException {
        String[] parts = v.split(",");
        if (parts.length != 2) {
            throw new UsageException(flag + " needs one pp,tg pair like '512,128', not '" + v + "'");
        }
        try {
            int pp = Integer.parseInt(parts[0].trim());
            int tg = Integer.parseInt(parts[1].trim());
            if (pp < 0 || tg < 0) {
                throw new NumberFormatException();
            }
            List<Integer> pair = new ArrayList<>();
            pair.add(pp);
            pair.add(tg);
            return pair;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs two non-negative integers, not '" + v + "'");
        }
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    /** A count of at least one, as llama.cpp parses its repetitions. */
    private static int positiveInt(String v, String flag) throws UsageException {
        try {
            int n = Integer.parseInt(v.trim());
            if (n < 1) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a positive integer, not '" + v + "'");
        }
    }

    /** A count of at least zero, as llama.cpp parses its delay. */
    private static int nonNegativeInt(String v, String flag) throws UsageException {
        try {
            int n = Integer.parseInt(v.trim());
            if (n < 0) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a non-negative integer, not '" + v + "'");
        }
    }

    /**
     * Applies the group's {@code -lm} read mode to the process-wide tensor read property before the model opens,
     * since every reader of tensor data consults it then.
     *
     * @return the property's previous value, to restore with {@link #restoreReadMode}
     */
    private static String applyLoadMode(List<BenchInstance> group) {
        String prior = System.getProperty(TensorReadMode.PROP_READ_MODE);
        TensorReadMode mode = group.get(0).loadMode.readMode();
        if (mode != null) {
            System.setProperty(TensorReadMode.PROP_READ_MODE, mode.flagName());
        }
        return prior;
    }

    /** Restores the tensor read property after a group's model closed. */
    private static void restoreReadMode(String prior) {
        if (prior == null) {
            System.clearProperty(TensorReadMode.PROP_READ_MODE);
        } else {
            System.setProperty(TensorReadMode.PROP_READ_MODE, prior);
        }
    }

    /**
     * Opens the group's model through the factory: the CPU engine for {@code -ngl 0} or {@code -dev none}, the
     * GPU engine (with the factory's CPU fallback) otherwise. The engine settings come from the
     * {@code -Dneuralfree.*} properties and the run's {@code -dev}, {@code -ncmoe}, {@code -sm}, {@code -mg},
     * {@code -ts}, {@code -ot} and {@code -nkvo} values ({@link BenchInstance#engineConfig}); the load's memory
     * carries the group's context — sized for its largest run — its prefill chunk ({@code -ub}) and its KV types
     * ({@code -ctk}, {@code -ctv}).
     */
    private static EngineModel open(BenchGrid grid, List<BenchInstance> group) throws IOException {
        BenchInstance first = group.get(0);
        int maxTokens = 0;
        for (BenchInstance inst : group) {
            maxTokens = Math.max(maxTokens, inst.nCtxNeeded());
        }
        MemoryConfig mc = MemoryConfig.builder()
                .nCtx(Math.max(maxTokens, 1))
                .ubatch(first.nUbatch)
                .typeK(first.typeK)
                .typeV(first.typeV)
                // the embedding pass reads every prompt row's hidden state through the batched path
                .kvPaged(first.embeddings)
                .build();
        EngineConfig cfg;
        try {
            cfg = first.engineConfig(grid.base, mc);
        } catch (EngineLoadFlags.FlagException e) {
            throw new IOException(e.getMessage(), e);
        }
        return LLMModelFactory.open(first.model, cfg);
    }

    /**
     * Runs one benchmark of the sweep: the progress lines, the pause, the warmup, the repetitions and the
     * printing of the finished {@link BenchTest}.
     *
     * @throws IOException when a forward pass fails, or the run's tokens exceed the model's context
     */
    private static void runOne(BenchGrid grid, Forwarder f, BenchInstance inst, int idx, int count,
                               BenchPrinters.BenchPrinter p, BenchPrinters.BenchPrinter pErr, Appendable err)
            throws IOException {
        if (grid.progress) {
            progress(err, "benchmark %d/%d: starting", idx, count);
        }
        if (grid.delay > 0) {
            try {
                Thread.sleep(grid.delay * 1000L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("the bench pause was interrupted", e);
            }
        }
        if (grid.kernels) {
            KernelTiming timing = gpuTiming(f);
            if (timing != null) {
                timing.reset();
            }
        }
        if (!grid.noWarmup) {
            if (inst.nPrompt > 0) {
                if (grid.progress) {
                    progress(err, "benchmark %d/%d: warmup prompt run", idx, count);
                }
                f.prefillPrompt(inst, 0);
            }
            if (inst.nGen > 0) {
                if (grid.progress) {
                    progress(err, "benchmark %d/%d: warmup generation run", idx, count);
                }
                f.decodeWarmup();
            }
            f.reset();
        }
        long[] samples = new long[grid.reps];
        for (int r = 0; r < grid.reps; r++) {
            final int rep = r + 1;
            samples[r] = f.timedRun(inst, grid.progress ? () -> progress(err, "benchmark %d/%d: run %d/%d",
                    idx, count, rep, grid.reps) : null);
        }
        BenchTest test = new BenchTest(inst, f.modelType, f.modelSize, f.modelNParams, f.backends, f.nCtx,
                samples);
        if (p != null) {
            p.printTest(test);
        }
        if (pErr != null) {
            pErr.printTest(test);
        }
        if (grid.kernels) {
            printKernelTiming(f, err);
        }
    }

    /**
     * The per-kernel device time the GPU engine's queue accumulated over this test's warmup and
     * repetitions, on stderr: each dispatch's timing events land there once the finish of the run
     * retired them. A CPU engine or a device without dispatch timing prints a note instead.
     */
    private static void printKernelTiming(Forwarder f, Appendable err) throws IOException {
        KernelTiming timing = gpuTiming(f);
        err.append("nftool bench: per-kernel device time, cumulative over the warmup and repetitions\n");
        if (timing == null || timing.isEmpty()) {
            err.append("  (the backend timed no dispatch; --kernels needs the GPU engine)\n");
            return;
        }
        for (String line : timing.format()) {
            err.append("  ").append(line).append('\n');
        }
    }

    /** The GPU engine's dispatch timing, or null for a CPU engine or a queue without timing. */
    private static KernelTiming gpuTiming(Forwarder f) {
        if (f.model.forwardBackend() instanceof GpuForwardBackend) {
            return ((GpuForwardBackend) f.model.forwardBackend()).context().queue().kernelTiming();
        }
        return null;
    }

    /** Restores a system property the {@code --kernels} run changed, removing it when unset before. */
    private static void restoreProperty(String name, String priorValue) {
        if (priorValue == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, priorValue);
        }
    }

    /** One progress line of the reference's {@code --progress} set, on stderr. */
    private static void progress(Appendable err, String line, Object... args) throws IOException {
        err.append("nftool bench: ").append(String.format(Locale.ROOT, line, args)).append('\n');
    }

    /** One progress line, whose writing may fail like any {@link Appendable} write. */
    @FunctionalInterface
    private interface Progress {

        /** Writes the line. */
        void line() throws IOException;
    }

    /**
     * One open model and the group of runs on it: the model facts every {@link BenchTest} of the group
     * carries and the forward calls of a run.
     */
    private static final class Forwarder {

        /** The seed of every token generator, so a sweep is reproducible. */
        private static final long TOKEN_SEED = 0L;

        private final EngineModel model;
        private final GGUFTokenizer tok;
        private final String modelType;
        private final long modelSize;
        private final long modelNParams;
        private final String backends;
        private final int nCtx;
        /** Whether this group's runs time the model's embedding output instead of the logits pass. */
        private final boolean embeddingMode;

        /** Reads the model facts once, refusing models without a decoder token loop a run needs. */
        Forwarder(EngineModel model, BenchInstance first, List<BenchInstance> group) throws IOException {
            this.model = model;
            this.tok = model.tokenizer();
            this.embeddingMode = first.embeddings;
            boolean hasMemory = model.forwardBackend().memory() != null;
            boolean hasLoop = hasTokenLoop(model.supportClass());
            boolean anyGen = false;
            for (BenchInstance inst : group) {
                anyGen |= inst.nGen > 0;
            }
            if (embeddingMode) {
                if (anyGen && (!hasMemory || !hasLoop)) {
                    throw new IOException("the embedding mode of " + model.architecture() + " ("
                            + model.supportClass() + ") runs no token generation; bench a prompt shape"
                            + " (-p or -pg pp,0)");
                }
            } else if (!hasMemory || !hasLoop) {
                throw new IOException("bench runs the decoder passes of a model with memory; "
                        + model.architecture() + " (" + model.supportClass() + ") has none");
            }
            int maxTokens = 0;
            for (BenchInstance inst : group) {
                maxTokens = Math.max(maxTokens, inst.nCtxNeeded());
            }
            this.nCtx = model.kvCacheContext();
            if (nCtx < maxTokens) {
                throw new IOException("the sweep needs a context of " + maxTokens
                        + " tokens (-p/-n/-d), but " + model.architecture() + " opened with " + nCtx);
            }
            this.modelType = model.name().equals(model.architecture())
                    ? model.architecture() : model.architecture() + " " + model.name();
            this.modelSize = sizeBytes(Paths.get(first.model));
            this.modelNParams = nParams(model);
            this.backends = model instanceof GGUFModelGPU
                    ? ((GGUFModelGPU) model).backend().backendType().toString() : "CPU";
        }

        /** Empties the memory between runs. */
        void reset() {
            model.forwardBackend().reset();
        }

        /** Prefills the run's prompt (or its warmup) at {@code pos0}, or runs the embedding pass there. */
        void prefillPrompt(BenchInstance inst, int pos0) throws IOException {
            try {
                if (embeddingMode) {
                    embeddingPass(inst.nPrompt, pos0);
                } else {
                    model.forwardBackend().prefill(promptTokens(inst.nPrompt, TOKEN_SEED), pos0);
                }
            } catch (ContextFullException e) {
                throw contextFull(inst, e);
            }
        }

        /** Decodes the warmup's one token at position 0, the first the generation would use. */
        void decodeWarmup() throws IOException {
            if (embeddingMode) {
                forwardToken(nextToken(new Random(TOKEN_SEED + 2L)), 0);
                return;
            }
            model.forwardBackend().decode(nextToken(new Random(TOKEN_SEED + 2L)), 0);
        }

        /**
         * One repetition: the untimed depth prefill on empty memory, then the timed prompt pass and the decodes.
         * In the embedding mode the prompt pass computes the model's embedding output for the prompt rows instead
         * of the last row's logits.
         *
         * @param progressLine optional progress callback, run before the timed section
         * @return the nanoseconds the timed section took
         */
        long timedRun(BenchInstance inst, Progress progressLine) throws IOException {
            model.forwardBackend().reset();
            try {
                if (inst.nDepth > 0) {
                    if (embeddingMode) {
                        forwardRows(promptTokens(inst.nDepth, TOKEN_SEED + 1L), 0, false);
                    } else {
                        model.forwardBackend().prefill(promptTokens(inst.nDepth, TOKEN_SEED + 1L), 0);
                    }
                }
                if (progressLine != null) {
                    progressLine.line();
                }
                long t0 = System.nanoTime();
                if (inst.embeddings) {
                    if (inst.nPrompt > 0) {
                        embeddingPass(inst.nPrompt, inst.nDepth);
                    }
                } else if (inst.nPrompt > 0) {
                    model.forwardBackend().prefill(promptTokens(inst.nPrompt, TOKEN_SEED), inst.nDepth);
                }
                if (inst.nGen > 0) {
                    Random random = new Random(TOKEN_SEED + 2L);
                    int pos = inst.nDepth + inst.nPrompt;
                    if (embeddingMode) {
                        forwardToken(nextToken(random), pos);
                        for (int i = 1; i < inst.nGen; i++) {
                            forwardToken(random.nextInt(model.vocabSize()), pos + i);
                        }
                    } else {
                        model.forwardBackend().decode(nextToken(random), pos);
                        for (int i = 1; i < inst.nGen; i++) {
                            model.forwardBackend().decode(random.nextInt(model.vocabSize()), pos + i);
                        }
                    }
                }
                return Math.max(1L, System.nanoTime() - t0);
            } catch (ContextFullException e) {
                throw contextFull(inst, e);
            } finally {
                model.forwardBackend().reset();
            }
        }

        /** The run's context overflow, naming the shapes and the context it needs. */
        private IOException contextFull(BenchInstance inst, ContextFullException e) {
            return new IOException("the run's " + inst.nCtxNeeded() + " tokens (-p/-n/-d) exceed the model's"
                    + " context of " + nCtx + ": " + e.getMessage(), e);
        }

        /**
         * The embedding pass over {@code n} prompt tokens at {@code pos0}: the model's embedding output — the rows
         * the pooling head reads — for every prompt row. A model without memory runs it from position 0 through
         * {@link EngineModel#encode}; a model with memory runs the batched hidden-row path over its paged memory,
         * which stores the prompt like the logits pass. The warmup and the timed repetitions share it.
         */
        private void embeddingPass(int n, int pos0) throws IOException {
            int[] tokens = promptTokens(n, TOKEN_SEED);
            if (model.forwardBackend().memory() == null) {
                if (pos0 != 0) {
                    throw new IOException("the embedding pass of " + model.architecture() + " starts at position 0;"
                            + " it has no memory to hold a depth prefill");
                }
                try {
                    model.encode(tokens, null);
                } catch (ContextFullException e) {
                    throw new IOException("the embedding pass of " + n + " prompt tokens exceeds the model's"
                            + " context of " + nCtx + ": " + e.getMessage(), e);
                } catch (UnsupportedOperationException | IllegalArgumentException e) {
                    throw new IOException("the embedding mode of " + model.architecture() + ": " + e.getMessage(), e);
                }
                return;
            }
            forwardRows(tokens, pos0, true);
        }

        /** One batched forward over the paged memory: the last row's logits, or every row's hidden states. */
        private void forwardRows(int[] tokens, int pos0, boolean hidden) throws IOException {
            ForwardBatch.Builder batch = ForwardBatch.builder(tokens.length);
            for (int i = 0; i < tokens.length; i++) {
                batch.add(tokens[i], pos0 + i, 0, hidden || i == tokens.length - 1);
            }
            try {
                if (hidden) {
                    model.forwardBackend().hiddenBatch(batch.build());
                } else {
                    model.forwardBackend().forwardBatch(batch.build());
                }
            } catch (ContextFullException e) {
                throw new IOException("the forward pass of " + tokens.length + " tokens at position " + pos0
                        + " exceeds the model's context of " + nCtx + ": " + e.getMessage(), e);
            } catch (RuntimeException e) {
                throw new IOException("the forward pass of " + tokens.length + " tokens at position " + pos0
                        + " failed: " + e.getMessage(), e);
            }
        }

        /** One batched decode row over the paged memory, for the embedding mode's generation shapes. */
        private void forwardToken(int token, int pos) throws IOException {
            ForwardBatch.Builder batch = ForwardBatch.builder(1);
            batch.add(token, pos, 0, true);
            try {
                model.forwardBackend().forwardBatch(batch.build());
            } catch (ContextFullException e) {
                throw new IOException("the decode at position " + pos + " exceeds the model's context of " + nCtx
                        + ": " + e.getMessage(), e);
            } catch (RuntimeException e) {
                throw new IOException("the decode at position " + pos + " failed: " + e.getMessage(), e);
            }
        }

        /**
         * {@code n} token ids for one pass: the BOS first when the vocab adds one, then fixed-seed random ids
         * (llama-bench draws random ids from the process generator; this bench fixes its own, so a sweep is
         * reproducible).
         */
        private int[] promptTokens(int n, long seed) {
            int[] tokens = new int[n];
            Random random = new Random(seed);
            int start = bosFirst() ? 1 : 0;
            if (start == 1) {
                tokens[0] = tok.bosId();
            }
            for (int i = start; i < tokens.length; i++) {
                tokens[i] = random.nextInt(model.vocabSize());
            }
            return tokens;
        }

        /** One token id: the BOS when the vocab adds one, else a random id. */
        private int nextToken(Random random) {
            return bosFirst() ? tok.bosId() : random.nextInt(model.vocabSize());
        }

        /** Whether the vocab's first token is the BOS (llama-bench's {@code llama_vocab_get_add_bos}). */
        private boolean bosFirst() {
            return tok.vocab().addBos() && tok.bosId() >= 0;
        }

        /** The bytes of the model's container: the file, or the shard files of a directory. */
        private static long sizeBytes(Path path) throws IOException {
            if (Files.isDirectory(path)) {
                try (Stream<Path> entries = Files.list(path)) {
                    return entries.filter(Files::isRegularFile)
                            .filter(p -> p.getFileName().toString().endsWith(".gguf"))
                            .mapToLong(p -> sizeOf(p)).sum();
                }
            }
            return sizeOf(path);
        }

        private static long sizeOf(Path p) {
            try {
                return Files.size(p);
            } catch (IOException e) {
                throw new IllegalStateException("cannot read the size of " + p, e);
            }
        }

        /** The parameter count: the elements of every tensor of the catalog. */
        private static long nParams(EngineModel model) {
            long n = 0;
            for (String name : model.spec().tensors().names()) {
                n += model.spec().tensors().info(name).numElements();
            }
            return n;
        }

        /** Whether models of {@code support} run a decoder-only token loop (LLMModelFactory's set). */
        private static boolean hasTokenLoop(SupportClass support) {
            switch (support) {
                case EMBED:
                case CLS:
                case NA:
                case DIFF:
                case ENC_DEC:
                case DRAFT:
                    return false;
                default:
                    return true;
            }
        }
    }
}
