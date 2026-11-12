package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.demo.EngineLoadFlags;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.PagedEvaluations;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * The pooling and batching flags {@code nftool embedding} and {@code nftool retrieval} share, in llama.cpp's
 * spellings (common/arg.cpp): {@code --pooling}, {@code -np/--parallel}, {@code -b/--batch-size},
 * {@code -ub/--ubatch-size}, {@code -c/--ctx-size}, and the model-load flags of {@link EngineLoadFlags}
 * ({@code -ngl}, {@code --gpu}, {@code --backend}, {@code -ot}, {@code -t}, {@code --kv-type}, ...). The flags start
 * from the {@code -Dneuralfree.*} property configuration; without a GPU request ({@code --gpu}, {@code --backend},
 * {@code -ngl} above 0 or {@code neuralfree.backend}) the model runs on the CPU engine.
 *
 * <p>The run embeds its prompts in llama.cpp's batches ({@link #windows}): a batch closes when the next prompt
 * would push it past the batch size or past the memory's sequence count, and every batch runs through
 * {@link EngineModel#encodeBatch}, one multi-sequence forward over a paged memory or packed memoryless forwards of
 * an encoder.</p>
 *
 * <p>One instance parses one command line; it is not thread-safe.</p>
 */
final class EmbeddingBatchFlags {

    /** llama.cpp's default {@code n_batch} (common/common.h). */
    static final int DEFAULT_BATCH_SIZE = 2048;

    /** The usage lines of the shared flags, newline-terminated. */
    static final String USAGE =
            "  --pooling <none|mean|cls|last|rank>\n"
            + "                                   override the model's pooling type (llama.cpp --pooling, vLLM's\n"
            + "                                   pooler pooling_type; rank needs the classifier head)\n"
            + "  --batch-size, -b <n>             max tokens per batch (llama.cpp -b, default " + DEFAULT_BATCH_SIZE
            + ", raised to the context)\n"
            + "  --ubatch-size, -ub <n>           tokens per forward step (llama.cpp -ub; a non-causal model takes\n"
            + "                                   the batch size, as llama.cpp sets it)\n"
            + "  --ctx-size, -c <n>               context length (llama.cpp -c; default: the model's)\n"
            + "  load flags: --gpu | --cpu | --backend <name> | -ngl <n> | -ot <pat>=<cpu|gpu> | -fa on |\n"
            + "  -t <n> | --kv-type <t> (the model runs on the CPU engine unless one of them asks for a GPU)\n";

    /** An invalid flag or value; the message names the flag. */
    static final class FlagException extends Exception {

        private static final long serialVersionUID = 1L;

        FlagException(String message) {
            super(message);
        }
    }

    /** The parsed flags of one run. Immutable. */
    static final class Settings {

        private final EngineConfig load;
        private final PoolingKind pooling;
        private final int batchSize;
        private final int parallel;
        private final boolean ubatchExplicit;

        Settings(EngineConfig load, PoolingKind pooling, int batchSize, int parallel, boolean ubatchExplicit) {
            this.load = load;
            this.pooling = pooling;
            this.batchSize = batchSize;
            this.parallel = parallel;
            this.ubatchExplicit = ubatchExplicit;
        }

        /**
         * The configuration of the flags: the load flags, {@code -c}, {@code -ub}, the {@code -np} sequences and
         * the pooling override. {@link #forModel} fits it to the resolved model.
         */
        EngineConfig load() { return load; }

        /** The pooling override, or null for the GGUF's pooling. */
        PoolingKind pooling() { return pooling; }

        /** {@code --batch-size} as given (llama.cpp {@code n_batch}). */
        int batchSize() { return batchSize; }

        /** {@code --parallel} as given, or 0 when the flag is absent. */
        int parallel() { return parallel; }

        /**
         * The batch size the run windows its prompts by: llama.cpp raises {@code n_batch} to the context when it is
         * smaller (embedding.cpp:122-126, retrieval.cpp:189), so no prompt that fits the context refuses.
         */
        int batchSize(ModelSpec spec) {
            int ctx = load.memory().nCtx() > 0 ? load.memory().nCtx() : spec.hparams().nCtxTrain();
            return Math.max(batchSize, ctx);
        }

        /**
         * The configuration the model opens with: {@link #load()} running {@code sequences} memory sequences
         * (llama.cpp {@code n_seq_max}; above 1 the memory is paged), and for a non-causal model the batch size as
         * its ubatch unless {@code -ub} set one, as llama.cpp's tools do (a non-causal attention runs its whole input
         * in one step). Sequences chosen without {@code --parallel} share one unified context, as llama.cpp's
         * embedding tool sets {@code kv_unified} when it picks them.
         *
         * @throws IllegalArgumentException when {@code sequences} is outside {@code [1, MemoryConfig.MAX_PARALLEL]}
         */
        EngineConfig forModel(ModelSpec spec, int sequences) {
            MemoryConfig.Builder memory = load.memory().toBuilder().parallel(sequences);
            if (parallel == 0 && sequences > 1) {
                memory.kvUnified(true);
            }
            if (!spec.hparams().causal() && !ubatchExplicit) {
                memory.ubatch(batchSize(spec));
            }
            return load.toBuilder().memory(memory.build()).build();
        }
    }

    private final EngineConfig base;
    private final MemoryConfig.Builder memory;
    private final EngineLoadFlags load;
    private PoolingKind pooling;
    private int batchSize = DEFAULT_BATCH_SIZE;
    private int parallel;
    private int ubatch;
    private int ctx;
    private boolean gpuAsked;

    private EmbeddingBatchFlags(EngineConfig base) {
        this.base = base;
        this.memory = base.memory().toBuilder();
        this.load = new EngineLoadFlags(base.toBuilder(), memory, base.moeCache(), base.eplb());
        this.gpuAsked = System.getProperty(EngineConfig.PROP_BACKEND) != null;
    }

    /**
     * Flags over the {@code -Dneuralfree.*} property configuration.
     *
     * @throws FlagException naming the property when one holds an invalid value
     */
    static EmbeddingBatchFlags fromSystemProperties() throws FlagException {
        try {
            return new EmbeddingBatchFlags(EngineConfig.fromSystemProperties());
        } catch (IllegalArgumentException e) {
            throw new FlagException(e.getMessage());
        }
    }

    /**
     * Parses the flag at {@code args[index]} with its value.
     *
     * @return the index of the flag's last argument, or -1 when {@code args[index]} is not one of these flags
     * @throws FlagException for a missing or invalid value, naming the flag
     */
    int parse(String[] args, int index) throws FlagException {
        String a = args[index];
        switch (a) {
            case "--pooling":
                pooling = poolingValue(a, value(a, args, index));
                return index + 1;
            case "-np":
            case "--parallel":
                parallel = intValue(a, value(a, args, index), 1, MemoryConfig.MAX_PARALLEL);
                return index + 1;
            case "-b":
            case "--batch-size":
                batchSize = intValue(a, value(a, args, index), 1, Integer.MAX_VALUE);
                return index + 1;
            case "-ub":
            case "--ubatch-size":
                ubatch = intValue(a, value(a, args, index), 1, Integer.MAX_VALUE);
                return index + 1;
            case "-c":
            case "--ctx-size":
                ctx = intValue(a, value(a, args, index), 0, Integer.MAX_VALUE);
                return index + 1;
            default:
                return loadFlag(args, index);
        }
    }

    /**
     * The settings of the parsed flags.
     *
     * @throws FlagException when the combined settings are invalid, naming the flag or the setting
     */
    Settings build() throws FlagException {
        if (ctx > 0) {
            memory.nCtx(ctx);
        }
        if (ubatch > 0) {
            memory.ubatch(ubatch);
        }
        memory.parallel(parallel > 0 ? parallel : 1);
        EngineConfig cfg;
        try {
            cfg = load.build();
        } catch (EngineLoadFlags.FlagException e) {
            throw new FlagException(e.getMessage());
        }
        EngineConfig.Builder b = cfg.toBuilder().pooling(pooling != null ? pooling : base.pooling());
        if (!gpuAsked) {
            b.backend(EngineConfig.BackendMode.CPU);
        }
        boolean ubatchExplicit = ubatch > 0 || System.getProperty(MemoryConfig.PROP_UBATCH) != null;
        return new Settings(b.build(), pooling != null ? pooling : base.pooling(), batchSize, parallel,
                ubatchExplicit);
    }

    /**
     * The {@code [from, to)} prompt ranges of llama.cpp's batches (embedding.cpp:262-283, retrieval.cpp:229-249): a
     * range closes when the next prompt would push it past {@code batchSize} tokens or past {@code maxSequences}
     * prompts.
     */
    static List<int[]> windows(List<int[]> inputs, int batchSize, int maxSequences) {
        List<int[]> out = new ArrayList<>();
        int from = 0;
        long tokens = 0;
        for (int i = 0; i < inputs.size(); i++) {
            int n = inputs.get(i).length;
            if (i > from && (tokens + n > batchSize || i - from >= maxSequences)) {
                out.add(new int[]{from, i});
                from = i;
                tokens = 0;
            }
            tokens += n;
        }
        if (from < inputs.size()) {
            out.add(new int[]{from, inputs.size()});
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * The pooled output row of every input in order (the RANK scores for a ranking model), embedded in llama.cpp's
     * batches ({@link #windows}) through {@link EngineModel#encodeBatch}: a batch holds up to the paged memory's
     * sequences, or the sequences one memoryless batch packs for an encoder planned without memory.
     *
     * @throws IOException when the model fails to encode
     */
    static List<float[]> encode(EngineModel model, List<int[]> inputs, int batchSize) throws IOException {
        ForwardBackend backend = model.forwardBackend();
        int maxSequences = PagedEvaluations.packsMemoryless(backend) ? backend.memorylessSequences()
                : backend.maxSequences();
        List<float[]> rows = new ArrayList<>(inputs.size());
        for (int[] w : windows(inputs, batchSize, maxSequences)) {
            Collections.addAll(rows, model.encodeBatch(inputs.subList(w[0], w[1]).toArray(new int[0][])));
        }
        return Collections.unmodifiableList(rows);
    }

    // ------------------------------------------------------------------ values

    /** One {@link EngineLoadFlags} flag; a GPU backend flag or {@code -ngl} above 0 asks for the GPU. */
    private int loadFlag(String[] args, int index) throws FlagException {
        int last;
        try {
            last = load.parse(args, index);
        } catch (EngineLoadFlags.FlagException e) {
            throw new FlagException(e.getMessage());
        }
        if (last >= 0) {
            gpuAsked |= asksForGpu(args[index], last > index ? args[last] : null);
        }
        return last;
    }

    /**
     * Whether a parsed load flag asks for a GPU: {@code --gpu}, a {@code --backend} other than cpu, or
     * {@code -ngl} of {@code auto}, {@code all} or a count above 0 (llama.cpp's {@code -ngl 0} keeps the model on
     * the host).
     */
    private static boolean asksForGpu(String flag, String value) {
        switch (flag) {
            case "--gpu":
                return true;
            case "--backend":
                return !"cpu".equalsIgnoreCase(value.trim());
            case "--gpu-layers":
            case "-ngl":
            case "--n-gpu-layers": {
                String v = value.trim().toLowerCase(Locale.ROOT);
                return !v.equals("0");
            }
            default:
                return false;
        }
    }

    private static PoolingKind poolingValue(String flag, String value) throws FlagException {
        try {
            return PoolingKind.parse(value);
        } catch (IllegalArgumentException e) {
            throw new FlagException(flag + ": " + e.getMessage());
        }
    }

    private static String value(String flag, String[] args, int index) throws FlagException {
        if (index + 1 >= args.length || args[index + 1] == null) {
            throw new FlagException(flag + " needs a value");
        }
        return args[index + 1];
    }

    private static int intValue(String flag, String s, int min, int max) throws FlagException {
        int n;
        try {
            n = Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            throw new FlagException(flag + " needs an integer, got '" + s + "'");
        }
        if (n < min || n > max) {
            throw new FlagException(flag + " " + n + " is outside [" + min + ", " + max + "]");
        }
        return n;
    }
}
