package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.engine.batch.BatchStats;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

/**
 * {@code BatchedDemo}: one prompt decoded into {@code -np} parallel sequences at once (llama.cpp
 * {@code examples/batched/batched.cpp}) over a {@link BatchEngine}. The prompt is tokenized once and
 * submitted once per sequence; every sequence carries its own sampler chain (the one seed, as llama.cpp
 * builds every chain with it, so a fixed seed yields identical streams and a random one diverging
 * continuations), runs to the budget or an end-of-generation token, and the demo prints each stream and the
 * decode totals. {@code -n} counts the whole sequence, prompt included, exactly as the reference's
 * {@code n_predict} does here, and the required KV size ({@code n_kv_req}) is refused by name when it does
 * not fit the context.
 *
 * <p>The model opens through LLMDemo's load flags ({@link EngineLoadFlags}: {@code --backend}, {@code -ngl},
 * {@code --device}, {@code -ot}, {@code --cpu-moe}, {@code -fa}, the KV cache types and the rest) over the
 * {@code -Dneuralfree.*} properties, as llama.cpp's batched example takes the common load flags: by default on the
 * best GPU backend, whose paged device memory the batch engine runs over, with the CPU engine as the fallback, or on
 * the CPU engine with {@code --cpu}. Where the reference prefills the prompt rows once for all sequences, the engine
 * prefills per sequence; the shared-prefix behavior remains available through the memory's prefix cache.
 */
public final class BatchedDemo {

    /** batched.cpp's default prompt. */
    static final String DEFAULT_PROMPT = "Hello my name is";
    /** batched.cpp's default {@code n_predict}: the whole sequence, prompt included. */
    static final int DEFAULT_MAX_TOKENS = 32;
    /** llama.cpp's default {@code n_parallel} (common_params). */
    static final int DEFAULT_PARALLEL = 1;
    /** llama.cpp's sampler defaults (common_params). */
    static final float DEFAULT_TEMPERATURE = 0.80f;
    static final float DEFAULT_TOP_P = 0.95f;
    static final int DEFAULT_TOP_K = 40;

    private static final String USAGE =
            "Usage: BatchedDemo [model.gguf] [options]\n"
            + "options (llama.cpp batched names):\n"
            + "  -p, --prompt <text>      the prompt every sequence continues (default '" + DEFAULT_PROMPT
                    + "')\n"
            + "  -n, --max-tokens <n>     whole sequence length, prompt included (default "
                    + DEFAULT_MAX_TOKENS + ")\n"
            + "  -np, --parallel <n>      parallel sequences (default " + DEFAULT_PARALLEL + ")\n"
            + "  -c, --ctx <n>            context rows (0 = fit automatically)\n"
            + "  -ub, --ubatch <n>        tokens per forward step (default " + MemoryConfig.DEFAULT_UBATCH
                    + ")\n"
            + "  -s, --seed <n>           sampler seed of every chain (-1 = random)\n"
            + "  --temp <f>               temperature, 0 = greedy (default " + DEFAULT_TEMPERATURE + ")\n"
            + "  --top-k <n>              top-k, 0 = off (default " + DEFAULT_TOP_K + ")\n"
            + "  --top-p <f>              top-p (default " + DEFAULT_TOP_P + ")\n"
            + "  --help, -h               this text\n"
            + EngineLoadFlags.usageSection("engine load options (as LLMDemo's):")
            + "The model opens on the best GPU backend by default (paged device memory), or on the CPU engine\n"
            + "with --cpu.\n";

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
         * The load configuration: backend, device, placement and KV types ({@link EngineLoadFlags}); the demo's
         * memory flags apply on top ({@link #engineConfig()}).
         */
        final EngineConfig engine;
        final String prompt;
        /** The whole sequence length, prompt included ({@code n_predict}). */
        final int maxTokens;
        final int parallel;
        final int ctx;
        final int ubatch;
        final long seed;
        final float temperature;
        final int topK;
        final float topP;

        Plan(Path model, EngineConfig engine, String prompt, int maxTokens, int parallel, int ctx, int ubatch,
                long seed, float temperature, int topK, float topP) {
            this.model = model;
            this.engine = engine;
            this.prompt = prompt;
            this.maxTokens = maxTokens;
            this.parallel = parallel;
            this.ctx = ctx;
            this.ubatch = ubatch;
            this.seed = seed;
            this.temperature = temperature;
            this.topK = topK;
            this.topP = topP;
        }

        /**
         * The configuration the model opens with: the load configuration over a paged memory of {@link #parallel}
         * sequences sharing {@code -c} rows, stepped {@code -ub} tokens at a time.
         */
        EngineConfig engineConfig() {
            return EngineLoadFlags.pagedConfig(engine, ctx, ubatch, parallel);
        }
    }

    private BatchedDemo() {
    }

    public static void main(String[] args) {
        run(args, System.out);
    }

    /** Runs the demo, printing the streams and totals to {@code out}; errors exit nonzero. */
    static void run(String[] args, Appendable out) {
        for (String a : args) {
            if (a.equals("--help") || a.equals("-h")) {
                System.out.print(USAGE);
                return;
            }
        }
        try {
            simulate(parse(args), out);
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
     * @throws UsageException for an unknown flag, a missing value, a value out of range or an invalid load flag
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
        String prompt = DEFAULT_PROMPT;
        int maxTokens = DEFAULT_MAX_TOKENS;
        int parallel = DEFAULT_PARALLEL;
        int ctx = 0;
        int ubatch = MemoryConfig.DEFAULT_UBATCH;
        long seed = SamplerConfig.RANDOM_SEED;
        float temperature = DEFAULT_TEMPERATURE;
        int topK = DEFAULT_TOP_K;
        float topP = DEFAULT_TOP_P;
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "-p":
                case "--prompt":
                    prompt = value(a, args, i++);
                    break;
                case "-n":
                case "--max-tokens":
                    maxTokens = intValue(a, value(a, args, i++));
                    break;
                case "-np":
                case "--parallel":
                    parallel = intValue(a, value(a, args, i++));
                    break;
                case "-c":
                case "--ctx":
                    ctx = nonNegative(a, value(a, args, i++));
                    break;
                case "-ub":
                case "--ubatch":
                    ubatch = intValue(a, value(a, args, i++));
                    break;
                case "-s":
                case "--seed":
                    seed = longValue(a, value(a, args, i++));
                    break;
                case "--temp":
                    temperature = floatValue(a, value(a, args, i++), 0f, Float.MAX_VALUE);
                    break;
                case "--top-k":
                    topK = nonNegative(a, value(a, args, i++));
                    break;
                case "--top-p":
                    topP = floatValue(a, value(a, args, i++), 0f, 1f);
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
            throw new UsageException("BatchedDemo needs a model path");
        }
        return new Plan(model, load.build(), prompt, maxTokens, parallel, ctx, ubatch, seed, temperature, topK, topP);
    }

    /** Opens the model on the paged memory of the plan's backend and runs the parallel continuations of the prompt. */
    static void simulate(Plan plan, Appendable out) throws IOException {
        EngineConfig cfg;
        try {
            cfg = plan.engineConfig();
        } catch (IllegalArgumentException e) {
            throw new IOException(e.getMessage(), e);
        }
        try (EngineModel model = LLMModelFactory.open(plan.model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(model, cfg)) {
            int[] prompt = model.tokenizer().encode(plan.prompt, true);
            int[] budget = budget(plan, prompt, engine.contextLength());
            out.append(String.format(Locale.ROOT,
                    "BatchedDemo: max-tokens = %d, ctx = %d, ubatch = %d, parallel = %d, n_kv_req = %d,"
                            + " backend = %s\n",
                    plan.maxTokens, engine.contextLength(), plan.ubatch, plan.parallel, budget[1],
                    EngineLoadFlags.backendLabel(model, cfg)));
            generate(plan, engine, prompt, budget[0], out);
        } catch (UnsupportedOperationException e) {
            throw new IOException(e.getMessage(), e);
        }
    }

    /**
     * The generated-token budget and the required KV size: llama.cpp counts the prompt in {@code -n} and
     * discards the sample it draws at {@code n_cur == n_predict}, so every stream carries
     * {@code n_predict - P} tokens and the sequences need {@code P + (n_predict - P) * np} KV rows
     * ({@code n_kv_req}).
     *
     * @throws IOException when the prompt is empty, the budget does not exceed the prompt, or
     *                     {@code n_kv_req} does not fit the context (the reference's refusal)
     */
    private static int[] budget(Plan plan, int[] prompt, int nCtx) throws IOException {
        if (prompt.length == 0) {
            throw new IOException("the prompt tokenizes to no tokens");
        }
        if (plan.maxTokens <= prompt.length) {
            throw new IOException("--max-tokens " + plan.maxTokens + " counts the whole sequence (prompt"
                    + " included), as llama.cpp's -n does here; it must exceed the prompt's "
                    + prompt.length + " tokens");
        }
        long nKvReq = (long) prompt.length + (long) (plan.maxTokens - prompt.length) * plan.parallel;
        if (nKvReq > nCtx) {
            throw new IOException("n_kv_req (" + nKvReq + ") > n_ctx (" + nCtx
                    + "), the required KV cache size is not big enough: either reduce --parallel or"
                    + " increase --ctx");
        }
        return new int[] { plan.maxTokens - prompt.length, (int) Math.min(nKvReq, Integer.MAX_VALUE) };
    }

    /** Submits one sequence per parallel stream, drains the engine and prints the streams and the totals. */
    private static void generate(Plan plan, BatchEngine engine, int[] prompt, int budget, Appendable out)
            throws IOException {
        GenerationRequest request = GenerationRequest.builder()
                .maxNewTokens(budget)
                .sampler(sampler(plan))
                .build();
        StringBuilder[] streams = new StringBuilder[plan.parallel];
        BatchSequence[] handles = new BatchSequence[plan.parallel];
        if (plan.parallel > 1) {
            out.append("\nBatchedDemo: generating " + plan.parallel + " sequences ...\n");
        } else {
            out.append(plan.prompt);
        }
        BatchStats before = engine.stats();
        try {
            for (int i = 0; i < plan.parallel; i++) {
                if (plan.parallel > 1) {
                    streams[i] = new StringBuilder();
                    StringBuilder into = streams[i];
                    handles[i] = engine.submit(prompt, request,
                            (tokenId, piece) -> append(into, piece));
                } else {
                    handles[i] = engine.submit(prompt, request, (tokenId, piece) -> append(out, piece));
                }
            }
            engine.drain();
        } catch (ContextFullException e) {
            throw new IOException("the prompt of " + prompt.length + " tokens does not fit the context of "
                    + engine.contextLength() + " rows", e);
        }
        for (BatchSequence handle : handles) {
            if (handle.state() == BatchSequence.State.FAILED) {
                throw new IOException("a sequence failed: " + handle.error());
            }
        }
        totals(out, plan.prompt, engine, before, streams);
    }

    /** Appends one streamed piece; an output failure stops the generation as an unchecked error. */
    private static void append(Appendable out, String piece) {
        try {
            out.append(piece);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** The shared sampler chain configuration: the demo's flags with the one seed. */
    private static SamplerConfig sampler(Plan plan) {
        return SamplerConfig.defaults().toBuilder()
                .temperature(plan.temperature)
                .topK(plan.topK)
                .topP(plan.topP)
                .seed(plan.seed)
                .build();
    }

    /** The streams ({@code sequence i: prompt + generated}, as llama.cpp prints them) and the totals line. */
    private static void totals(Appendable out, String prompt, BatchEngine engine, BatchStats before,
            StringBuilder[] streams) throws IOException {
        BatchStats after = engine.stats();
        long decoded = after.generatedTokens() - before.generatedTokens();
        double seconds = after.decodeSeconds() - before.decodeSeconds();
        out.append('\n');
        if (streams.length > 1) {
            for (int i = 0; i < streams.length; i++) {
                out.append("sequence ").append(String.valueOf(i)).append(":\n\n").append(prompt)
                        .append(streams[i]).append("\n\n");
            }
        }
        out.append(String.format(Locale.ROOT, "BatchedDemo: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
                decoded, seconds, decoded / seconds));
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

    private static int nonNegative(String flag, String v) throws UsageException {
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

    private static long longValue(String flag, String v) throws UsageException {
        try {
            return Long.parseLong(v.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer, not '" + v + "'");
        }
    }

    private static float floatValue(String flag, String v, float min, float max) throws UsageException {
        try {
            float f = Float.parseFloat(v.trim());
            if (!(f >= min && f <= max)) {
                throw new NumberFormatException();
            }
            return f;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a number in [" + min + ", " + max + "], not '" + v + "'");
        }
    }
}
