package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.demo.EngineLoadFlags;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.ImatrixCollector;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuMatmulTap;
import com.eonmux.neuralfree.gguf.quant.write.ImatrixFile;
import com.eonmux.neuralfree.gguf.quant.write.ImatrixStatistics;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * {@code nftool imatrix}: the activation-matrix collection of the quantize pipeline (llama.cpp
 * {@code tools/imatrix/imatrix.cpp}). It opens the model on the CPU engine ({@code -ngl 0}, the default) or the GPU
 * engine and collects the matmul inputs into an {@link ImatrixCollector} filtered like the reference's ask callback
 * (every routed expert product; dense products of at least 16 input rows over the block tensors, the output tensors
 * under {@code --process-output} and the NextN tensors under {@code --nextn}): the CPU matmuls through
 * {@link CpuMatVec#observeGlobally}, the device matmuls through {@link GpuMatmulTap}. The
 * whole concatenated corpus is tokenized once and runs as consecutive whole-context windows of {@code --ctx-size}
 * tokens decoded from position 0, {@code --batch-size / --ctx-size} windows per decode as parallel sequences
 * ({@link ImatrixRun}). At least {@code 2 * n_ctx} tokens are required; {@code --from-chunk} drops whole leading
 * chunks, {@code --chunks} caps the count. {@code --nextn} (or {@code --model-draft}) replays the NextN head over
 * the trunk's hidden rows and records {@code imatrix.n_layer_nextn}.
 *
 * <p>The counters and the saved metadata follow the reference: {@code imatrix.chunk_count} grows one chunk per
 * processed window (seeded by the counts of {@code --in-file} matrices combined in first, llama.cpp's
 * {@code load_imatrix}), {@code --output-frequency} rewrites the output every N chunks and
 * {@code --save-frequency} keeps {@code <out>.at_<chunk>} copies. {@code --ppl} (the default) reports the
 * corpus perplexity over the second half of every chunk through {@link ImatrixPerplexity};
 * {@code --show-statistics} prints the {@link ImatrixStatistics} report of one {@code --in-file} and exits
 * without a model. {@code --output-format dat} writes the legacy {@code .dat} form. With no {@code --file}
 * the tool combines the {@code --in-file} matrices and saves, exactly as the reference does.</p>
 */
final class ImatrixCommand {

    /** Default GGUF output path of llama.cpp ({@code params.out_file}). */
    static final String DEFAULT_OUT = "imatrix.gguf";

    /** The collection window of llama.cpp's imatrix example ({@code params.n_ctx} default 512). */
    static final int DEFAULT_CTX_SIZE = 512;

    /** llama.cpp's default output period ({@code params.n_out_freq}). */
    static final int DEFAULT_OUTPUT_FREQUENCY = 10;

    /** llama.cpp's default logical batch ({@code params.n_batch}): four 512-token sequences per decode. */
    static final int DEFAULT_BATCH_SIZE = 2048;

    private static final String USAGE =
            "nftool imatrix <model> [options]\n"
            + "options (llama.cpp imatrix names):\n"
            + "  --file <fname>                   calibration text, repeatable; tokenized into whole-context\n"
            + "                                   chunks over the corpus, at least 2 * --ctx-size tokens\n"
            + "  --in-file <fname>                imatrix to combine in (GGUF or .dat), repeatable; with no\n"
            + "                                   --file the tool combines and saves only\n"
            + "  --out-file <fname>               output imatrix, default " + DEFAULT_OUT + "\n"
            + "  --ctx-size <n>                   chunk window in tokens (llama.cpp n_ctx), default "
            + DEFAULT_CTX_SIZE + "\n"
            + "  --chunks <n>                     at most n chunks (-1 = all), default -1\n"
            + "  --chunk, --from-chunk <n>        drop the first n chunks of the corpus, default 0\n"
            + "  --output-frequency <n>           rewrite the output every n chunks, default "
            + DEFAULT_OUTPUT_FREQUENCY + "\n"
            + "  --save-frequency <n>             keep an imatrix copy <out>.at_<chunk> every n chunks, 0 = off\n"
            + "  --output-format {gguf,dat}       output form, default gguf\n"
            + "  --process-output                 also collect output.weight and token_embd.weight\n"
            + "  -b, --batch-size <n>             tokens per decode, default " + DEFAULT_BATCH_SIZE + "; below\n"
            + "                                   --ctx-size or a multiple of it, which runs batch / ctx\n"
            + "                                   chunks per decode as parallel sequences\n"
            + "  -ngl, --n-gpu-layers <n|auto|all> 0 (default) runs the CPU engine, else the GPU engine\n"
            + "  -ot, --override-tensor <pattern=cpu|gpu,...>, --cpu-moe, --n-cpu-moe <n>,\n"
            + "  --moe-cache <off|on|auto|MiB>, --moe-cache-mib <n>\n"
            + "                                   the GPU engine's weight placement, as nftool chat takes them\n"
            + "  --nextn                          also collect the NextN head, replayed over the trunk's\n"
            + "                                   hidden states (needs --batch-size <= --ctx-size)\n"
            + "  -md, --model-draft <fname>       collect the NextN head of an MTP draft file (implies --nextn)\n"
            + "  --ppl, --no-ppl                  report the corpus perplexity, default on\n"
            + "  --show-statistics                print the statistics report of the first --in-file and exit\n"
            + "  --parse-special                  parse special tokens in the calibration text\n"
            + "  --help, -h                       this text\n"
            + "The GPU engine reads every tapped matmul input back to the host. The output is replaced when it\n"
            + "exists, as llama.cpp saves.\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /**
     * The GPU engine's weight-placement flags the tool takes, as llama.cpp's imatrix takes the common load arguments:
     * the tensor overrides, the MoE expert placement and the MoE expert cache.
     */
    static final Set<String> PLACEMENT_FLAGS = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
            "-ot", "--override-tensor", "--cpu-moe", "-cmoe", "--n-cpu-moe", "-ncmoe", "--moe-cache",
            "--moe-cache-mib")));

    /** The parsed command line: every flag of the reference's imatrix example with its NF default. */
    static final class Params {
        final Path model;
        final List<Path> files;
        final List<Path> inFiles;
        final Path outFile;
        final int ctxSize;
        final int chunks;
        final int fromChunk;
        final int outputFrequency;
        final int saveFrequency;
        final boolean processOutput;
        final boolean nextn;
        final boolean ppl;
        final boolean legacyFormat;
        final boolean showStatistics;
        final boolean parseSpecial;
        final int batchSize;
        /** {@code -ngl}: 0 runs the CPU engine, else the GPU engine with this layer count (or auto/all). */
        final int nGpuLayers;
        /** {@code --model-draft}: the MTP draft whose NextN head is replayed, or null. */
        final Path modelDraft;
        /** The weight-placement flags with their values, in command-line order ({@link #PLACEMENT_FLAGS}). */
        final List<String> placement;

        Params(Path model, List<Path> files, List<Path> inFiles, Path outFile, int ctxSize, int chunks,
               int fromChunk, int outputFrequency, int saveFrequency, boolean processOutput, boolean nextn,
               boolean ppl, boolean legacyFormat, boolean showStatistics, boolean parseSpecial, int batchSize,
               int nGpuLayers, Path modelDraft, List<String> placement) {
            this.model = model;
            this.files = files;
            this.inFiles = inFiles;
            this.outFile = outFile;
            this.ctxSize = ctxSize;
            this.chunks = chunks;
            this.fromChunk = fromChunk;
            this.outputFrequency = outputFrequency;
            this.saveFrequency = saveFrequency;
            this.processOutput = processOutput;
            this.nextn = nextn;
            this.ppl = ppl;
            this.legacyFormat = legacyFormat;
            this.showStatistics = showStatistics;
            this.parseSpecial = parseSpecial;
            this.batchSize = batchSize;
            this.nGpuLayers = nGpuLayers;
            this.modelDraft = modelDraft;
            this.placement = Collections.unmodifiableList(new ArrayList<>(placement));
        }

        /** Parses and validates the flags; unknown or out-of-range values fail naming the flag. */
        static Params parse(String[] args) throws UsageException {
            Path model = null;
            List<Path> files = new ArrayList<>();
            List<Path> inFiles = new ArrayList<>();
            Path out = null;
            int ctxSize = DEFAULT_CTX_SIZE;
            int chunks = -1;
            int fromChunk = 0;
            int outputFrequency = DEFAULT_OUTPUT_FREQUENCY;
            int saveFrequency = 0;
            boolean processOutput = false;
            boolean nextn = false;
            boolean ppl = true;
            boolean legacyFormat = false;
            boolean showStatistics = false;
            boolean parseSpecial = false;
            int batchSize = DEFAULT_BATCH_SIZE;
            int nGpuLayers = 0;
            Path modelDraft = null;
            List<String> placement = new ArrayList<>();
            for (int i = 0; i < args.length; i++) {
                String a = args[i];
                if (a.equals("--help") || a.equals("-h")) {
                    return null;
                }
                switch (a) {
                    case "--file":
                        files.add(Paths.get(value(a, args, i++)));
                        break;
                    case "--in-file":
                        inFiles.add(Paths.get(value(a, args, i++)));
                        break;
                    case "--out-file":
                        out = Paths.get(value(a, args, i++));
                        break;
                    case "--ctx-size":
                        ctxSize = intValue(a, value(a, args, i++), 1);
                        break;
                    case "--chunks":
                        chunks = intValue(a, value(a, args, i++), -1);
                        if (chunks < 1 && chunks != -1) {
                            throw new UsageException("--chunks needs a positive integer or -1, not " + chunks);
                        }
                        break;
                    case "--chunk":
                    case "--from-chunk":
                        fromChunk = intValue(a, value(a, args, i++), 0);
                        break;
                    case "--output-frequency":
                        outputFrequency = intValue(a, value(a, args, i++), 1);
                        break;
                    case "--save-frequency":
                        saveFrequency = intValue(a, value(a, args, i++), 0);
                        break;
                    case "--output-format":
                        String format = value(a, args, i++).trim();
                        if (format.equals("gguf")) {
                            legacyFormat = false;
                        } else if (format.equals("dat")) {
                            legacyFormat = true;
                        } else {
                            throw new UsageException("--output-format needs gguf or dat, not '" + format + "'");
                        }
                        break;
                    case "--process-output":
                        processOutput = true;
                        break;
                    case "--nextn":
                        nextn = true;
                        break;
                    case "--ppl":
                        ppl = true;
                        break;
                    case "--no-ppl":
                        ppl = false;
                        break;
                    case "--show-statistics":
                        showStatistics = true;
                        break;
                    case "--parse-special":
                        parseSpecial = true;
                        break;
                    case "-b":
                    case "--batch-size":
                        batchSize = intValue(a, value(a, args, i++), 1);
                        break;
                    case "-ngl":
                    case "--gpu-layers":
                    case "--n-gpu-layers":
                        nGpuLayers = gpuLayers(a, value(a, args, i++));
                        break;
                    case "-md":
                    case "--model-draft":
                        modelDraft = Paths.get(value(a, args, i++));
                        break;
                    default:
                        if (PLACEMENT_FLAGS.contains(a)) {
                            i = placementFlag(args, i, placement);
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
            if (showStatistics && inFiles.isEmpty()) {
                throw new UsageException("--show-statistics needs an --in-file to describe");
            }
            if (model == null && !files.isEmpty()) {
                throw new UsageException("imatrix needs a model path to collect over");
            }
            if (files.isEmpty() && inFiles.isEmpty()) {
                throw new UsageException("imatrix needs at least one --file of calibration text, "
                        + "or --in-file matrices to combine");
            }
            for (Path f : files) {
                if (!Files.isReadable(f)) {
                    throw new UsageException("--file " + f + " is not a readable file");
                }
            }
            for (Path f : inFiles) {
                if (!Files.isReadable(f)) {
                    throw new UsageException("--in-file " + f + " is not a readable file");
                }
            }
            if (modelDraft != null && !Files.isReadable(modelDraft)) {
                throw new UsageException("--model-draft " + modelDraft + " is not a readable file");
            }
            return new Params(model, files, inFiles, out != null ? out : Paths.get(DEFAULT_OUT), ctxSize, chunks,
                    fromChunk, outputFrequency, saveFrequency, processOutput, nextn || modelDraft != null, ppl,
                    legacyFormat, showStatistics, parseSpecial, batchSize, nGpuLayers, modelDraft, placement);
        }

        /**
         * The engine configuration of the run over {@code base}: the placement flags applied through the shared
         * load flags ({@link EngineLoadFlags}).
         *
         * @throws UsageException when a placement flag does not fit the configuration, naming it
         */
        EngineConfig placed(EngineConfig base) throws UsageException {
            EngineLoadFlags load = EngineLoadFlags.over(base);
            String[] flags = placement.toArray(new String[0]);
            try {
                for (int i = 0; i < flags.length; i++) {
                    i = load.parse(flags, i);
                }
                return load.build();
            } catch (EngineLoadFlags.FlagException e) {
                throw new UsageException(e.getMessage());
            }
        }

        /**
         * Validates the placement flag at {@code args[i]} through the shared load flags and appends it with its
         * values to {@code into}; returns the index of its last argument.
         */
        private static int placementFlag(String[] args, int i, List<String> into) throws UsageException {
            try {
                int last = EngineLoadFlags.over(EngineConfig.builder().build()).parse(args, i);
                into.addAll(Arrays.asList(args).subList(i, last + 1));
                return last;
            } catch (EngineLoadFlags.FlagException e) {
                throw new UsageException(e.getMessage());
            }
        }
    }

    private ImatrixCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code imatrix}. */
    static void run(String[] args) {
        try {
            collect(args);
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
     * Parses the flags and runs the requested mode: the statistics report, the combine-only save, or the
     * collection over the corpus. {@code --help} prints the usage and returns null.
     *
     * @return the final output file, or null for {@code --help} and {@code --show-statistics}
     */
    static Path collect(String[] args) throws IOException, UsageException {
        Params params = Params.parse(args);
        if (params == null) {
            System.out.println(USAGE);
            return null;
        }
        List<ImatrixFile.Read> reads = readInFiles(params);
        if (params.showStatistics) {
            ImatrixFile.Read read = reads.get(0);
            System.out.println("\nComputing tensor statistics for " + read.file()
                    + " (" + read.entries().size() + " tensors)");
            System.out.print(ImatrixStatistics.compute(read.entries(), read.nLayerNextn()).format());
            return null;
        }
        int nLayerNextn = nLayerNextn(reads);
        List<ImatrixFile.Entry> combined = new ArrayList<>();
        List<String> datasets = new ArrayList<>();
        for (ImatrixFile.Read r : reads) {
            combined = ImatrixFile.combine(combined, scaleLegacy(r, params.ctxSize));
            datasets.addAll(r.datasets());
        }
        if (datasets.isEmpty()) {
            for (Path f : params.inFiles) {
                datasets.add(f.toString());
            }
        }
        if (params.files.isEmpty()) {
            return combineOnly(params, combined, datasets, nLayerNextn);
        }
        for (Path f : params.files) {
            datasets.add(f.toString());
        }
        return collectOverCorpus(params, combined, datasets, nLayerNextn);
    }

    /**
     * The {@code imatrix.n_layer_nextn} of the combined matrices (llama.cpp {@code load_imatrix}): the first
     * non-zero count; a later different one is warned about and ignored.
     */
    private static int nLayerNextn(List<ImatrixFile.Read> reads) {
        int n = 0;
        for (ImatrixFile.Read r : reads) {
            if (r.nLayerNextn() == 0) {
                continue;
            }
            if (n == 0) {
                n = r.nLayerNextn();
            } else if (n != r.nLayerNextn()) {
                System.err.println("nftool imatrix: warning: NextN layer count mismatch in " + r.file() + ": "
                        + r.nLayerNextn() + " != " + n + ", using " + n);
            }
        }
        return n;
    }

    /** Combines the {@code --in-file} matrices and saves them (llama.cpp's no-prompt path). */
    private static Path combineOnly(Params params, List<ImatrixFile.Entry> combined, List<String> datasets,
                                    int nLayerNextn) throws IOException {
        if (combined.isEmpty()) {
            throw new IOException("the --in-file matrices hold no statistics");
        }
        int chunkSize = params.ctxSize;
        int chunkCount = rowsToChunks(maxCount(combined), chunkSize);
        Path out = save(params, combined, datasets, chunkCount, -1, nLayerNextn);
        System.out.println("nftool imatrix: wrote " + out + " combined from " + params.inFiles.size()
                + " matrices, " + combined.size() + " tensors, " + chunkCount + " chunks of " + chunkSize);
        return out;
    }

    /** The collection run: tokenize the corpus, run the context windows, save the matrix and the report. */
    private static Path collectOverCorpus(Params params, List<ImatrixFile.Entry> combined, List<String> datasets,
                                          int inNextn) throws IOException, UsageException {
        try (ImatrixRun run = ImatrixRun.open(params)) {
            int[] tokens = tokenize(run.model().tokenizer(), params);
            int[] window = chunkWindow(tokens.length, params.ctxSize, params.fromChunk, params.chunks);
            int seed = rowsToChunks(maxCount(combined), params.ctxSize);
            int nLayerNextn = run.nLayerNextn() > 0 ? run.nLayerNextn() : inNextn;
            ImatrixPerplexity ppl = new ImatrixPerplexity();
            int chunkCount = run.run(tokens, window, seed, ppl, (collected, count, atChunk) ->
                    save(params, ImatrixFile.combine(collected, combined), datasets, count, atChunk, nLayerNextn));
            List<ImatrixFile.Entry> entries = ImatrixFile.combine(run.entries(), combined);
            if (entries.isEmpty()) {
                throw new IOException("no matmul activations were collected; the model ran no matmul");
            }
            Path out = save(params, entries, datasets, chunkCount, -1, nLayerNextn);
            System.out.println("nftool imatrix: wrote " + out + " from " + (chunkCount - seed)
                    + " chunks of " + params.ctxSize + " tokens, " + entries.size() + " tensors, "
                    + chunkCount + " chunks in total");
            String report = ppl.report();
            if (report != null) {
                System.out.println(report);
            }
            return out;
        }
    }

    /**
     * Writes the matrix in the requested format; {@code atChunk >= 0} names the copy {@code <out>.at_<n>}. The GGUF
     * form records {@code nLayerNextn} as {@code imatrix.n_layer_nextn}; the legacy form has no field for it.
     */
    private static Path save(Params params, List<ImatrixFile.Entry> entries, List<String> datasets,
                             int chunkCount, int atChunk, int nLayerNextn) throws IOException {
        Path out = atChunk >= 0 ? Paths.get(params.outFile + ".at_" + atChunk) : params.outFile;
        if (params.legacyFormat) {
            return ImatrixFile.writeLegacy(out, entries, chunkCount, params.ctxSize, datasets, null);
        }
        return ImatrixFile.writeGguf(out, entries, datasets, chunkCount, params.ctxSize, nLayerNextn);
    }

    /** Loads every {@code --in-file} in order. */
    private static List<ImatrixFile.Read> readInFiles(Params params) throws IOException {
        List<ImatrixFile.Read> reads = new ArrayList<>();
        for (Path f : params.inFiles) {
            reads.add(ImatrixFile.read(f));
        }
        return reads;
    }

    /** A read's entries; legacy sums and counts scale by the chunk size (llama.cpp {@code load_imatrix}). */
    private static List<ImatrixFile.Entry> scaleLegacy(ImatrixFile.Read read, int chunkSize) {
        if (!read.isLegacy()) {
            return read.entries();
        }
        List<ImatrixFile.Entry> out = new ArrayList<>(read.entries().size());
        for (ImatrixFile.Entry e : read.entries()) {
            float[] sums = e.sumSq();
            for (int j = 0; j < sums.length; j++) {
                sums[j] *= chunkSize;
            }
            long[] counts = e.counts();
            for (int i = 0; i < counts.length; i++) {
                counts[i] = Math.multiplyExact(counts[i], chunkSize);
            }
            out.add(new ImatrixFile.Entry(e.tensor(), e.nMat(), e.cols(), sums, counts));
        }
        return out;
    }

    /** The largest activation count of a set, the chunk seed of a combined matrix. */
    private static long maxCount(List<ImatrixFile.Entry> entries) {
        long max = 0;
        for (ImatrixFile.Entry e : entries) {
            for (long c : e.counts()) {
                max = Math.max(max, c);
            }
        }
        return max;
    }

    /** llama.cpp {@code rows_to_chunks}: half-up rounding of activation rows to whole chunks. */
    private static int rowsToChunks(long rows, int chunkSize) {
        return (int) ((rows + chunkSize / 2) / chunkSize);
    }

    /**
     * The corpus window plan: {@code {startToken, nChunk}}. The start drops {@code fromChunk} whole windows; the
     * corpus must hold at least two windows and, with a from-chunk, enough tokens left after the dropped ones
     * (llama.cpp {@code compute_imatrix}).
     *
     * @throws UsageException naming the flag when the corpus cannot fill the requested windows
     */
    static int[] chunkWindow(int totalTokens, int ctxSize, int fromChunk, int chunks) throws UsageException {
        if (fromChunk > 0 && (long) (fromChunk + 2) * ctxSize >= totalTokens) {
            throw new UsageException("--from-chunk " + fromChunk + " leaves not enough tokens: the corpus holds "
                    + totalTokens + ", dropping " + fromChunk + " chunks of " + ctxSize
                    + " needs more than " + (fromChunk + 2) * ctxSize);
        }
        int remaining = totalTokens - fromChunk * ctxSize;
        if (remaining < 2 * ctxSize) {
            throw new UsageException("the corpus needs at least " + (2 * ctxSize) + " tokens for a context of "
                    + ctxSize + " tokens, the calibration files tokenize to only " + remaining);
        }
        int max = remaining / ctxSize;
        int nChunk = chunks < 0 ? max : Math.min(chunks, max);
        if (nChunk < 1) {
            throw new UsageException("--chunks " + chunks + " exceeds the " + max + " chunks the corpus holds");
        }
        return new int[]{fromChunk * ctxSize, nChunk};
    }

    /** The whole corpus as one token sequence: the files concatenated, BOS added per the vocabulary. */
    private static int[] tokenize(GGUFTokenizer tok, Params params) throws IOException, UsageException {
        StringBuilder corpus = new StringBuilder();
        for (Path f : params.files) {
            corpus.append(new String(Files.readAllBytes(f), StandardCharsets.UTF_8));
        }
        int[] tokens = tok.encode(corpus.toString(), true, params.parseSpecial);
        if (tokens.length == 0) {
            throw new UsageException("--file tokenizes to no tokens; the corpus is empty");
        }
        return tokens;
    }

    /**
     * The name half of the reference's ask for dense products: block tensors, the output tensors under
     * {@code --process-output}, and the NextN projection under {@code --nextn}.
     */
    static boolean collectable(String name, boolean processOutput, boolean nextn) {
        if (name.startsWith("blk.")) {
            return true;
        }
        boolean isOutput = name.equals("output.weight") || name.equals("token_embd.weight")
                || name.equals("nextn.post_projection.weight");
        boolean isNextn = name.equals("nextn.pre_projection.weight");
        return (processOutput && isOutput) || (nextn && isNextn);
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    /**
     * The {@code -ngl} value: {@code auto}, {@code all} or a layer count of at least 0 (llama.cpp
     * {@code --n-gpu-layers}).
     */
    private static int gpuLayers(String flag, String v) throws UsageException {
        String t = v.trim();
        if (t.equals("auto")) {
            return EngineConfig.GPU_LAYERS_AUTO;
        }
        if (t.equals("all")) {
            return EngineConfig.GPU_LAYERS_ALL;
        }
        int n;
        try {
            n = Integer.parseInt(t);
        } catch (NumberFormatException e) {
            n = -1;
        }
        if (n < 0) {
            throw new UsageException(flag + " needs auto, all or a layer count of at least 0, not '" + v + "'");
        }
        return n;
    }

    /** An integer of the flag, at least {@code min} (as llama.cpp parses its counts). */
    private static int intValue(String flag, String v, int min) throws UsageException {
        try {
            int n = Integer.parseInt(v.trim());
            if (n < min) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer of at least " + min + ", not '" + v + "'");
        }
    }
}
