package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import com.eonmux.neuralfree.demo.EngineLoadFlags;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryFullException;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * {@code nftool batched-bench}: the multi-sequence throughput matrix of llama.cpp ({@code
 * tools/batched-bench/batched-bench.cpp}). Every cell of the {@code -npp x -ntg x -npl} grid prefills
 * {@code npl} random-token prompts of {@code npp} tokens and runs {@code ntg} decode steps of {@code npl}
 * rows each — no sampling, exactly the reference's decode patterns — over the paged memory through
 * {@link ForwardBackend#forwardBatch}, and reports the prompt-phase and generation-phase times and
 * throughputs as the reference's Markdown table or, with {@code --output-format jsonl}, one JSON object
 * per cell.
 *
 * <p>The flags carry llama.cpp's names and shapes: {@code -npp}, {@code -ntg} and {@code -npl} take
 * comma-separated integer lists (repeatable, they append; {@code -npp} and {@code -ntg} also take 0, a
 * negative count is refused), {@code -pps} shares one prefilled prompt across
 * the sequences through the paged memory's page copy ({@code llama_memory_seq_cp}; the prompt phase then
 * prefills one prompt and the copy sits between the two measured phases, as the reference's unified KV
 * does), {@code -tgs} decodes one sequence after another instead of all in one step, {@code -c} and
 * {@code -ub} size the memory, {@code -s} seeds the random tokens. {@code -npp 0} runs no prompt phase: the
 * decode steps start at position 0 and the cell measures generation only, printing the reference's ~0 s
 * {@code T_PP} and 0.00 {@code S_PP}. {@code n_kv_max} is the context the
 * paged sequences share; a cell whose {@code n_kv} exceeds it, or whose {@code pp + tg} rows exceed one
 * sequence's cap, is skipped, as the reference skips it; the skip is reported on standard error.
 *
 * <p>The model opens through LLMDemo's load flags ({@link EngineLoadFlags}: {@code --backend}, {@code -ngl},
 * {@code --device}, {@code -ot}, {@code --cpu-moe}, {@code -fa}, the KV cache types and the rest) over the
 * {@code -Dneuralfree.*} properties, as llama.cpp's batched-bench takes the common load flags: by default on the
 * best GPU backend, whose paged device memory runs every batch through the page tables, with the CPU engine as the
 * fallback, or on the CPU engine with {@code --cpu}. The header names the backend the model opened on.
 */
final class BatchedBenchCommand {

    /** The warm-up decode of llama.cpp: 16 tokens into sequence 0 (batched-bench.cpp:100-112). */
    static final int WARMUP_TOKENS = 16;

    /** llama.cpp's default context (common_params n_ctx): 0, the context the model was trained with. */
    static final int DEFAULT_CTX = 0;

    /** The default seed of the random tokens: C {@code rand()}'s. */
    static final long DEFAULT_SEED = 1L;

    private static final String USAGE =
            "nftool batched-bench <model> [options]\n"
            + "options (llama.cpp batched-bench names):\n"
            + "  -npp n0,n1,...           prompt tokens per sequence (0 allowed: generation only), repeatable;"
            + " required\n"
            + "  -ntg n0,n1,...           generated tokens per sequence (0 allowed), repeatable; required\n"
            + "  -npl n0,n1,...           parallel sequences, repeatable; required\n"
            + "  -pps                     share one prefilled prompt across the sequences (page copy)\n"
            + "  -tgs                     decode the sequences one after another\n"
            + "  -c, --ctx <n>            context rows the sequences share (n_kv_max; default 0, the trained"
            + " context)\n"
            + "  -ub, --ubatch <n>        tokens per forward step (default " + MemoryConfig.DEFAULT_UBATCH + ")\n"
            + "  -s, --seed <n>           seed of the random tokens (default " + DEFAULT_SEED + ", C rand's)\n"
            + "  --output-format <f>      md (the table, default) or jsonl\n"
            + "  --kernels                time the GPU dispatches of every cell (disables the decode-step replay)\n"
            + "                           and print its per-kernel device time on standard error\n"
            + "  --help, -h               this text\n"
            + EngineLoadFlags.usageSection("engine load options (as LLMDemo's):")
            + "example: nftool batched-bench model.gguf -c 2048 -ub 512 -npp 128,256,512 -ntg 128,256"
            + " -npl 1,2,4,8\n"
            + "A cell whose rows do not fit the per-sequence context and the pools is skipped, as llama.cpp\n"
            + "skips it; the skip is reported on standard error. With -npp 0 no prompt phase runs: the decode\n"
            + "steps start at position 0 and the cell reports T_PP ~0 s with S_PP 0.00, as llama.cpp's does.\n"
            + "The model opens on the best GPU backend by\n"
            + "default (paged device memory), or on the CPU engine with --cpu.\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The parsed command line. Immutable. */
    static final class Plan {

        final Path model;
        /**
         * The load configuration: backend, device, placement and KV types ({@link EngineLoadFlags}); the bench's
         * memory flags apply on top ({@link #engineConfig()}).
         */
        final EngineConfig engine;
        /** The prompt-token counts of the grid ({@code -npp}). */
        final List<Integer> npp;
        /** The generated-token counts of the grid ({@code -ntg}). */
        final List<Integer> ntg;
        /** The parallel-sequence counts of the grid ({@code -npl}). */
        final List<Integer> npl;
        /** The largest {@code -npl} value: the sequence count the memory is sized for. */
        final int maxPl;
        final boolean ppShared;
        final boolean tgSeparate;
        final boolean jsonl;
        /** {@code --kernels}: each cell's per-kernel device time goes to standard error. */
        final boolean kernels;
        final int ctx;
        final int ubatch;
        final long seed;

        Plan(Path model, EngineConfig engine, List<Integer> npp, List<Integer> ntg, List<Integer> npl,
                boolean ppShared, boolean tgSeparate, boolean jsonl, boolean kernels, int ctx, int ubatch, long seed) {
            this.model = model;
            this.engine = engine;
            this.npp = List.copyOf(npp);
            this.ntg = List.copyOf(ntg);
            this.npl = List.copyOf(npl);
            this.maxPl = Collections.max(this.npl);
            this.ppShared = ppShared;
            this.tgSeparate = tgSeparate;
            this.jsonl = jsonl;
            this.kernels = kernels;
            this.ctx = ctx;
            this.ubatch = ubatch;
            this.seed = seed;
        }

        /**
         * The configuration the model opens with: the load configuration over a paged memory of {@link #maxPl}
         * sequences sharing {@code -c} rows, stepped {@code -ub} tokens at a time.
         */
        EngineConfig engineConfig() {
            return EngineLoadFlags.pagedConfig(engine, ctx, ubatch, maxPl);
        }
    }

    /** One measured cell of the grid: the figures of the reference's table row. Immutable. */
    static final class BenchRow {

        final int pp;
        final int tg;
        final int pl;
        /** The rows the cell holds in the KV memory (llama.cpp {@code n_kv}). */
        final long nKv;
        /** The prompt-phase time in seconds and its throughput in tokens per second. */
        final double tPP;
        final double speedPP;
        /** The generation-phase time in seconds and its throughput in tokens per second. */
        final double tTG;
        final double speedTG;
        final double t;
        final double speed;

        BenchRow(int pp, int tg, int pl, long nKv,
                double tPP, double speedPP, double tTG, double speedTG, double t, double speed) {
            this.pp = pp;
            this.tg = tg;
            this.pl = pl;
            this.nKv = nKv;
            this.tPP = tPP;
            this.speedPP = speedPP;
            this.tTG = tTG;
            this.speedTG = speedTG;
            this.t = t;
            this.speed = speed;
        }
    }

    private BatchedBenchCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code batched-bench}. */
    static void run(String[] args) {
        for (String a : args) {
            if (a.equals("--help") || a.equals("-h")) {
                System.out.print(USAGE);
                return;
            }
        }
        try {
            execute(parse(args), System.out, System.err);
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
     * Parses the flags; a first argument that is not a flag is the model path. The load flags
     * ({@link EngineLoadFlags}) start from the {@code -Dneuralfree.*} properties, as LLMDemo's do.
     *
     * @throws UsageException for an unknown flag, a missing value, a non-positive count, an invalid load flag or a
     *                        missing model path or {@code -npp}/{@code -ntg}/{@code -npl} list
     */
    static Plan parse(String[] args) throws UsageException {
        try {
            return parseFlags(args);
        } catch (EngineLoadFlags.FlagException e) {
            throw new UsageException(e.getMessage());
        }
    }

    /** {@link #parse} with the load flags' own exception, which {@link #parse} turns into a usage error. */
    private static Plan parseFlags(String[] args) throws UsageException, EngineLoadFlags.FlagException {
        EngineLoadFlags load = EngineLoadFlags.fromSystemProperties();
        Path model = null;
        List<Integer> npp = new ArrayList<>();
        List<Integer> ntg = new ArrayList<>();
        List<Integer> npl = new ArrayList<>();
        boolean ppShared = false;
        boolean tgSeparate = false;
        boolean jsonl = false;
        boolean kernels = false;
        int ctx = DEFAULT_CTX;
        int ubatch = MemoryConfig.DEFAULT_UBATCH;
        long seed = DEFAULT_SEED;
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "-npp":
                    npp.addAll(countList(a, value(a, args, i++), true));
                    break;
                case "-ntg":
                    ntg.addAll(countList(a, value(a, args, i++), true));
                    break;
                case "-npl":
                    npl.addAll(countList(a, value(a, args, i++), false));
                    break;
                case "-pps":
                    ppShared = true;
                    break;
                case "-tgs":
                    tgSeparate = true;
                    break;
                case "-c":
                case "--ctx":
                    ctx = countValue(a, value(a, args, i++), true);
                    break;
                case "-ub":
                case "--ubatch":
                    ubatch = intValue(a, value(a, args, i++));
                    break;
                case "-s":
                case "--seed":
                    seed = longValue(a, value(a, args, i++));
                    break;
                case "--output-format":
                    jsonl = outputFormat(a, value(a, args, i++));
                    break;
                case "--kernels":
                    kernels = true;
                    break;
                case "-m":
                case "--model":
                    model = Paths.get(value(a, args, i++));
                    break;
                default:
                    int last = load.parse(args, i);
                    if (last >= 0) {
                        i = last;
                        break;
                    }
                    if (a.startsWith("-")) {
                        throw new UsageException("unknown option " + a + " (run with --help for the options)");
                    }
                    if (model != null) {
                        throw new UsageException("too many file arguments; '" + a + "' follows " + model);
                    }
                    model = Paths.get(a);
            }
        }
        if (model == null) {
            throw new UsageException("batched-bench needs a model path");
        }
        if (npp.isEmpty() || ntg.isEmpty() || npl.isEmpty()) {
            List<String> missing = new ArrayList<>();
            if (npp.isEmpty()) {
                missing.add("-npp");
            }
            if (ntg.isEmpty()) {
                missing.add("-ntg");
            }
            if (npl.isEmpty()) {
                missing.add("-npl");
            }
            throw new UsageException("batched-bench needs " + String.join(", ", missing)
                    + " (comma-separated integer lists), as the llama.cpp usage shows");
        }
        return new Plan(model, load.build(), npp, ntg, npl, ppShared, tgSeparate, jsonl, kernels, ctx, ubatch,
                seed);
    }

    /**
     * Opens the model on the paged memory of the plan's backend, runs the grid, prints every measured cell to
     * {@code out} and every skipped one to {@code err}, and returns the measured cells in grid order.
     *
     * @throws IOException when the model cannot be read, opens without paged memory or a forward pass fails
     */
    static List<BenchRow> execute(Plan plan, Appendable out, Appendable err) throws IOException {
        if (!plan.kernels) {
            return executeGrid(plan, out, err);
        }
        String priorProfile = System.setProperty(KernelTiming.PROFILE_PROPERTY, "true");
        String priorReplay = System.setProperty(GpuForwardBackend.DECODE_REPLAY_PROPERTY, "false");
        try {
            return executeGrid(plan, out, err);
        } finally {
            BenchCommand.restoreProperty(KernelTiming.PROFILE_PROPERTY, priorProfile);
            BenchCommand.restoreProperty(GpuForwardBackend.DECODE_REPLAY_PROPERTY, priorReplay);
        }
    }

    /** {@link #execute} with the {@code --kernels} properties already in place. */
    private static List<BenchRow> executeGrid(Plan plan, Appendable out, Appendable err) throws IOException {
        EngineConfig cfg;
        try {
            cfg = plan.engineConfig();
        } catch (IllegalArgumentException e) {
            throw new IOException(e.getMessage(), e);
        }
        try (EngineModel model = LLMModelFactory.open(plan.model.toString(), cfg)) {
            PagedSequences paged = pagedMemory(model);
            ForwardBackend backend = model.forwardBackend();
            int nVocab = model.vocabSize();
            Random rng = new Random(plan.seed);
            warmup(backend, nVocab, rng);
            long nKvMax = paged.plan().nCtx();
            if (!plan.jsonl) {
                out.append(String.format(Locale.ROOT,
                        "nftool batched-bench: n_kv_max = %d, n_ubatch = %d, parallel = %d, is_pp_shared = %b,"
                                + " is_tg_separate = %b, backend = %s\n",
                        nKvMax, plan.ubatch, plan.maxPl, plan.ppShared, plan.tgSeparate,
                        EngineLoadFlags.backendLabel(model, cfg)));
                out.append("\n");
                tableHeader(out);
            }
            List<BenchRow> measured = new ArrayList<>();
            for (int pp : plan.npp) {
                for (int tg : plan.ntg) {
                    for (int pl : plan.npl) {
                        KernelTiming timing = plan.kernels ? BenchCommand.gpuTiming(backend) : null;
                        if (timing != null) {
                            timing.reset();
                        }
                        BenchRow row = runCell(plan, backend, paged, pp, tg, pl, nVocab, rng, err);
                        if (plan.kernels && row != null) {
                            BenchCommand.printKernelTiming(timing, String.format(Locale.ROOT,
                                    "nftool batched-bench: per-kernel device time of pp=%d tg=%d pl=%d", pp, tg, pl),
                                    err);
                        }
                        if (row != null) {
                            measured.add(row);
                            out.append(plan.jsonl ? jsonl(row, plan, nKvMax) : tableRow(row));
                        }
                    }
                }
            }
            if (!plan.jsonl) {
                out.append("\n");
            }
            return Collections.unmodifiableList(measured);
        }
    }

    // ------------------------------------------------------------------ the grid

    /**
     * Runs one grid cell, or returns null with a standard-error note when its rows do not fit the memory,
     * as llama.cpp skips the cell: before the run when {@code n_kv} exceeds the context, or when the paged
     * pools run out mid-cell (they allocate whole pages, so per-sequence partial pages can exhaust them below
     * {@code n_kv_max}); the memory resets for the next cell either way.
     */
    private static BenchRow runCell(Plan plan, ForwardBackend backend, PagedSequences paged,
            int pp, int tg, int pl, int nVocab, Random rng, Appendable err) throws IOException {
        long nKvMax = paged.plan().nCtx();
        int seqCap = paged.sequence(0).nCtx();
        long nKv = plan.ppShared ? (long) pp + (long) pl * tg : (long) pl * (pp + (long) tg);
        if (pl > paged.maxSequences() || (long) pp + tg > seqCap || nKv > nKvMax) {
            err.append(String.format(Locale.ROOT,
                    "nftool batched-bench: skipped pp=%d tg=%d pl=%d: n_kv %d does not fit"
                            + " (n_kv_max %d, per-sequence cap %d, %d sequences)\n",
                    pp, tg, pl, nKv, nKvMax, seqCap, paged.maxSequences()));
            return null;
        }
        backend.reset();
        double tPP;
        double tTG;
        try {
            tPP = prefillPhase(plan, backend, paged, pp, pl, nVocab, rng);
            tTG = decodePhase(plan, backend, pp, tg, pl, nVocab, rng);
        } catch (MemoryFullException e) {
            backend.reset();
            err.append(String.format(Locale.ROOT, "nftool batched-bench: skipped pp=%d tg=%d pl=%d: n_kv %d ran the"
                    + " paged pools out: %s\n", pp, tg, pl, nKv, e.getMessage()));
            return null;
        }
        double ppTokens = plan.ppShared ? pp : (double) pl * pp;
        double tgTokens = (double) pl * tg;
        double t = tPP + tTG;
        return new BenchRow(pp, tg, pl, nKv, tPP, speed(ppTokens, tPP), tTG, speed(tgTokens, tTG), t,
                speed(ppTokens + tgTokens, t));
    }

    /**
     * The reference's throughput formula, tokens over seconds. A phase with no tokens ({@code pp = 0} or
     * {@code tg = 0}) times ~0 s and can measure 0/0; its well-defined throughput is 0 tokens per second, which
     * is what the cell prints where the C value would be nan.
     */
    private static double speed(double tokens, double seconds) {
        return seconds > 0 ? tokens / seconds : 0;
    }

    /**
     * The prompt phase: prefills one prompt per sequence (output logits of each prompt's last row only) and
     * returns its time; with {@code -pps} it then copies the prompt pages to the other sequences, after the
     * measured time ends, as the reference copies after {@code t_pp_end}. With {@code pp = 0} no prompt phase
     * runs, as the reference's timed block over its empty batch does (tools/batched-bench): no forward
     * call, no page copy of an empty prompt, and the decode steps below start at position 0.
     */
    private static double prefillPhase(Plan plan, ForwardBackend backend, PagedSequences paged,
            int pp, int pl, int nVocab, Random rng) throws IOException {
        if (pp == 0) {
            return 0;
        }
        int seqs = plan.ppShared ? 1 : pl;
        ForwardBatch.Builder batch = ForwardBatch.builder(seqs * pp);
        for (int j = 0; j < seqs; j++) {
            for (int i = 0; i < pp; i++) {
                batch.add(rng.nextInt(nVocab), i, j, i == pp - 1);
            }
        }
        long t0 = System.nanoTime();
        try {
            backend.forwardBatch(batch.build());
        } catch (MemoryFullException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException("the prefill of " + seqs + " x " + pp + " tokens failed: " + e.getMessage(), e);
        }
        double seconds = (System.nanoTime() - t0) / 1e9;
        if (plan.ppShared) {
            for (int j = 1; j < pl; j++) {
                paged.copy(0, j, 0, pp);
            }
        }
        return seconds;
    }

    /**
     * The generation phase in the reference's two decode patterns: {@code tg} steps of one row per sequence
     * ({@code 0123 0123 ...}), or, with {@code -tgs}, every sequence's {@code tg} single-row steps one sequence
     * after another ({@code 000 ... 111 ... 222 ...}).
     */
    private static double decodePhase(Plan plan, ForwardBackend backend, int pp, int tg, int pl,
            int nVocab, Random rng) throws IOException {
        long t0 = System.nanoTime();
        try {
            if (plan.tgSeparate) {
                for (int j = 0; j < pl; j++) {
                    for (int i = 0; i < tg; i++) {
                        decodeStep(backend, pp + i, j, j + 1, nVocab, rng);
                    }
                }
            } else {
                for (int i = 0; i < tg; i++) {
                    decodeStep(backend, pp + i, 0, pl, nVocab, rng);
                }
            }
        } catch (MemoryFullException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IOException("the decode of tg=" + tg + " x pl=" + pl + " rows failed: " + e.getMessage(), e);
        }
        return (System.nanoTime() - t0) / 1e9;
    }

    /** One decode forward: one output row per sequence in {@code [fromSeq, toSeq)} at {@code pos}. */
    private static void decodeStep(ForwardBackend backend, int pos, int fromSeq, int toSeq,
            int nVocab, Random rng) {
        ForwardBatch.Builder step = ForwardBatch.builder(toSeq - fromSeq);
        for (int j = fromSeq; j < toSeq; j++) {
            step.add(rng.nextInt(nVocab), pos, j, true);
        }
        backend.forwardBatch(step.build());
    }

    /** The warm-up decode: 16 tokens into sequence 0, then the memory clears. */
    private static void warmup(ForwardBackend backend, int nVocab, Random rng) throws IOException {
        ForwardBatch.Builder batch = ForwardBatch.builder(WARMUP_TOKENS);
        for (int i = 0; i < WARMUP_TOKENS; i++) {
            batch.add(rng.nextInt(nVocab), i, 0, false);
        }
        try {
            backend.forwardBatch(batch.build());
        } catch (RuntimeException e) {
            throw new IOException("the warm-up decode failed: " + e.getMessage(), e);
        }
        backend.reset();
    }

    /**
     * The paged sequences of the model's memory, host or device, or a named refusal: the bench runs on paged memory
     * only, and a model whose memory has no page tables (or no memory at all) cannot hold its sequences.
     */
    private static PagedSequences pagedMemory(EngineModel model) throws IOException {
        ModelMemory memory = model.forwardBackend().memory();
        PagedSequences paged = memory == null ? null : memory.pagedSequences();
        if (paged != null) {
            return paged;
        }
        throw new IOException(model.architecture() + ": batched-bench runs on paged memory"
                + " (neuralfree.parallel > 1 or neuralfree.kv.paged=true), but the model opened on "
                + (memory == null ? "no memory" : memory.getClass().getSimpleName()));
    }

    // ------------------------------------------------------------------ output

    /** The reference's table header and rule. */
    private static void tableHeader(Appendable out) throws IOException {
        out.append(String.format(Locale.ROOT, "|%6s | %6s | %4s | %6s | %8s | %8s | %8s | %8s | %8s | %8s |\n",
                "PP", "TG", "B", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "T s", "S t/s"));
        out.append(String.format(Locale.ROOT,
                "|%6s-|-%6s-|-%4s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|-%8s-|\n",
                "------", "------", "----", "------", "--------", "--------", "--------", "--------",
                "--------", "--------"));
    }

    /** The reference's table row. */
    private static String tableRow(BenchRow r) {
        return String.format(Locale.ROOT,
                "|%6d | %6d | %4d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %8.3f | %8.2f |\n",
                r.pp, r.tg, r.pl, r.nKv, r.tPP, r.speedPP, r.tTG, r.speedTG, r.t, r.speed);
    }

    /**
     * One JSON object per cell, with the reference's keys; the run-level fields the reference repeats but the
     * engine has no equivalent of ({@code n_batch}, {@code flash_attn}, the thread counts) are left out.
     */
    private static String jsonl(BenchRow r, Plan plan, long nKvMax) {
        return String.format(Locale.ROOT,
                "{\"n_kv_max\": %d, \"n_ubatch\": %d, \"parallel\": %d, \"is_pp_shared\": %b, "
                        + "\"is_tg_separate\": %b, \"pp\": %d, \"tg\": %d, \"pl\": %d, \"n_kv\": %d, "
                        + "\"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f, "
                        + "\"t\": %f, \"speed\": %f}\n",
                nKvMax, plan.ubatch, plan.maxPl, plan.ppShared, plan.tgSeparate,
                r.pp, r.tg, r.pl, r.nKv, r.tPP, r.speedPP, r.tTG, r.speedTG, r.t, r.speed);
    }

    // ------------------------------------------------------------------ argument helpers

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    private static int intValue(String flag, String v) throws UsageException {
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

    private static long longValue(String flag, String v) throws UsageException {
        try {
            return Long.parseLong(v.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer, not '" + v + "'");
        }
    }

    /**
     * A comma-separated list of counts ({@code n0,n1,...}), empty parts dropped. {@code -npp} and {@code -ntg}
     * take 0, as llama.cpp's string-split lists do ({@code -npp 0} is the generation-only cell, {@code -ntg 0}
     * the prompt-only one); negatives are refused, and {@code -npl} counts sequences, so it stays positive.
     */
    private static List<Integer> countList(String flag, String v, boolean allowZero) throws UsageException {
        List<Integer> out = new ArrayList<>();
        for (String part : v.split(",")) {
            String p = part.trim();
            if (p.isEmpty()) {
                continue;
            }
            try {
                out.add(countValue(flag, p, allowZero));
            } catch (UsageException e) {
                throw new UsageException(flag + " needs comma-separated integers; " + e.getMessage());
            }
        }
        if (out.isEmpty()) {
            throw new UsageException(flag + " needs at least one integer");
        }
        return out;
    }

    /** One count: non-negative where 0 is a count ({@code -npp}, {@code -ntg}), positive where it is not. */
    private static int countValue(String flag, String v, boolean allowZero) throws UsageException {
        try {
            int n = Integer.parseInt(v.trim());
            if (n < (allowZero ? 0 : 1)) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + (allowZero
                    ? " needs a non-negative integer, not '" : " needs a positive integer, not '") + v + "'");
        }
    }

    /** {@code md} or {@code jsonl}, as llama.cpp's {@code --output-format}. */
    private static boolean outputFormat(String flag, String v) throws UsageException {
        String f = v.trim().toLowerCase(Locale.ROOT);
        if (f.equals("md")) {
            return false;
        }
        if (f.equals("jsonl")) {
            return true;
        }
        throw new UsageException(flag + " " + v + " is invalid; use md or jsonl");
    }
}
