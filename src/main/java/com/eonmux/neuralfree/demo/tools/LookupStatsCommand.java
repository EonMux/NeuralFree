package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.engine.draft.NgramCache;
import com.eonmux.neuralfree.gguf.engine.draft.NgramCacheFile;
import com.eonmux.neuralfree.gguf.engine.draft.NgramDrafter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Locale;

/**
 * {@code nftool lookup-stats}: llama.cpp's {@code examples/lookup/lookup-stats.cpp}. The prompt tokenizes once and
 * then replays as consecutive contexts of {@code --ctx-size} tokens: each context drafts from the
 * {@code ngram-cache} routine ({@link NgramDrafter#draftCache}) over the caches the {@code -lcs} and {@code -lcd}
 * files hold, the prompt tokens stand in for the sampled ones, and the tool reports how many tokens were drafted
 * and how many a target would have accepted. Like the reference it reads the dynamic file when it exists and
 * writes nothing.
 */
final class LookupStatsCommand {

    /** llama.cpp's default draft length ({@code params.speculative.draft.n_max}). */
    static final int DEFAULT_DRAFT_MAX = 3;

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The counts and timings one replay measured (immutable). */
    static final class Stats {

        final int draftMax;
        /** The prompt tokens the reference counts as predicted ({@code n_input - n_input % n_ctx}). */
        final int nPredict;
        final int nDrafted;
        final int nAccepted;
        final double acceptPercent;
        final double draftFlatMs;
        final double draftMs;
        final double draftUsPerToken;
        final double draftTokensPerSecond;

        private Stats(int draftMax, int nPredict, int nDrafted, int nAccepted, double draftFlatNs,
                      double draftNs) {
            this.draftMax = draftMax;
            this.nPredict = nPredict;
            this.nDrafted = nDrafted;
            this.nAccepted = nAccepted;
            this.acceptPercent = nDrafted == 0 ? 0.0 : 100.0 * nAccepted / nDrafted;
            this.draftFlatMs = draftFlatNs / 1e6;
            this.draftMs = draftNs / 1e6;
            this.draftUsPerToken = nDrafted == 0 ? 0.0 : draftNs / 1e3 / nDrafted;
            this.draftTokensPerSecond = draftNs == 0.0 ? 0.0 : nDrafted / (draftNs / 1e9);
        }
    }

    private static final String USAGE =
            "nftool lookup-stats <model> [options]\n"
            + "options (llama.cpp lookup-stats names):\n"
            + "  -m, --model <file>                  the model whose tokenizer reads the prompt\n"
            + "  -p, --prompt <text>                 the text to replay\n"
            + "  -f, --file <fname>                  read the text from this file instead\n"
            + "  -lcs, --lookup-cache-static <file>  static lookup cache to read\n"
            + "  -lcd, --lookup-cache-dynamic <file> dynamic lookup cache to read; a missing file starts empty\n"
            + "  -c, --ctx-size <n>                  context per replay, default the model's training context\n"
            + "  --draft-max <n>                     largest draft, default " + DEFAULT_DRAFT_MAX + "\n"
            + "                                      (aliases --draft-n-max, --spec-draft-n-max)\n"
            + "  --help, -h                          this text\n"
            + "Replays the prompt as consecutive contexts and reports n_drafted, n_accept and the draft timings;\n"
            + "nothing is written to either cache file.\n";

    /** The parsed command line; {@link #parse} fills and validates it. */
    static final class Options {
        Path model;
        String prompt = "";
        Path staticCache;
        Path dynamicCache;
        int ctxSize;
        int draftMax = DEFAULT_DRAFT_MAX;
    }

    private LookupStatsCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code lookup-stats}. */
    static void run(String[] args) {
        try {
            execute(args, System.err);
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Parses the flags and replays the prompt; {@code null} for {@code --help}, which prints the usage. The
     * statistics print to {@code err} in the reference's format.
     *
     * @throws IOException when the model, a prompt file or a cache file cannot be read
     */
    static Stats execute(String[] args, Appendable err) throws IOException, UsageException {
        Options o = parse(args);
        if (o == null) {
            return null;
        }

        LookupToolSupport.TokenizerModel m = LookupToolSupport.open(o.model);
        int[] tokens = m.encode(o.prompt);
        int ctxSize = o.ctxSize > 0 ? o.ctxSize : m.contextLength();
        if (ctxSize <= 0) {
            throw new UsageException("lookup-stats needs -c/--ctx-size: the model carries no usable training context");
        }

        boolean dynamicMissing = o.dynamicCache != null && Files.notExists(o.dynamicCache);
        long flatStart = System.nanoTime();
        NgramCache staticCache = o.staticCache == null ? new NgramCache() : loadStatic(o.staticCache);
        NgramCache dynamicCache = o.dynamicCache == null || dynamicMissing ? new NgramCache()
                : NgramCacheFile.load(o.dynamicCache);
        long flatNs = System.nanoTime() - flatStart;
        if (dynamicMissing) {
            err.append("lookup-stats: the dynamic lookup cache " + o.dynamicCache + " does not exist; it starts"
                    + " empty\n");
        }

        Replay replay = replay(tokens, ctxSize, o.draftMax, staticCache, dynamicCache, err);
        int nInput = tokens.length;
        Stats stats = new Stats(o.draftMax, nInput - nInput % ctxSize, replay.drafted, replay.accepted, flatNs,
                replay.draftNs);
        print(stats, err);
        return stats;
    }

    /** The replay counters and the time drafting and cache updates took. */
    private static final class Replay {
        int drafted;
        int accepted;
        long draftNs;
    }

    /**
     * The reference's replay loop: every context of {@code ctxSize} tokens drafts and accepts against
     * {@code slice}, and its context cache merges into the dynamic cache at the chunk's end.
     */
    private static Replay replay(int[] tokens, int ctxSize, int draftMax, NgramCache staticCache,
                                 NgramCache dynamicCache, Appendable err) throws IOException {
        Replay r = new Replay();
        NgramCache context = new NgramCache();
        int nInput = tokens.length;
        for (int start = 0; start + ctxSize < nInput; start += ctxSize) {
            int[] slice = Arrays.copyOfRange(tokens, start, start + ctxSize);
            int[] pseudo = new int[ctxSize];
            int size = 0;
            pseudo[size++] = slice[0];
            while (size < ctxSize) {
                long t0 = System.nanoTime();
                int[] draft = NgramDrafter.draftCache(Arrays.copyOf(pseudo, size), draftMax,
                        NgramDrafter.DEFAULT_NGRAM_MIN, NgramDrafter.DEFAULT_NGRAM_MAX, context, dynamicCache,
                        staticCache);
                r.draftNs += System.nanoTime() - t0;
                r.drafted += draft.length;

                for (int j = 0; j < draft.length && size < ctxSize; j++) {
                    int groundTruth = slice[size];
                    if (groundTruth != draft[j]) {
                        break;
                    }
                    r.accepted++;
                    r.draftNs += accept(context, pseudo, size++, groundTruth);
                }
                if (size < ctxSize) {
                    int groundTruth = slice[size];
                    r.draftNs += accept(context, pseudo, size, groundTruth);
                    size++;
                }
            }
            dynamicCache.merge(context);
            context = new NgramCache();

            if (start > 0 && start / 100000 != (start - ctxSize) / 100000) {
                err.append(String.format(Locale.ROOT, "lookup-stats: %d/%d done%n", start, nInput));
            }
        }
        return r;
    }

    /**
     * Appends {@code token} at {@code index} of {@code pseudo} and adds the n-grams that end there to
     * {@code context}; returns the time taken.
     */
    private static long accept(NgramCache context, int[] pseudo, int index, int token) {
        pseudo[index] = token;
        long t0 = System.nanoTime();
        context.update(NgramDrafter.DEFAULT_NGRAM_MIN, NgramDrafter.DEFAULT_NGRAM_MAX, pseudo, index + 1, 1);
        return System.nanoTime() - t0;
    }

    /** The statistics in llama.cpp's {@code LOG_INF} format. */
    private static void print(Stats s, Appendable err) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append('\n');
        sb.append("n_draft      = ").append(s.draftMax).append('\n');
        sb.append("n_predict    = ").append(s.nPredict).append('\n');
        sb.append("n_drafted    = ").append(s.nDrafted).append('\n');
        sb.append(String.format(Locale.ROOT, "t_draft_flat = %.2f ms%n", s.draftFlatMs));
        sb.append(String.format(Locale.ROOT, "t_draft      = %.2f ms, %.2f us per token, %.2f tokens per second%n",
                s.draftMs, s.draftUsPerToken, s.draftTokensPerSecond));
        sb.append("n_accept     = ").append(s.nAccepted).append('\n');
        sb.append(String.format(Locale.ROOT, "accept       = %.3f%%%n", s.acceptPercent));
        err.append(sb);
    }

    /** The static cache file, which the reference requires to exist. */
    private static NgramCache loadStatic(Path file) throws IOException {
        if (!Files.isRegularFile(file)) {
            throw new IOException("-lcs/--lookup-cache-static " + file + " is not a readable file");
        }
        return NgramCacheFile.load(file);
    }

    /**
     * Parses and validates the flags; {@code null} for {@code --help}, which prints the usage.
     *
     * @throws UsageException naming the flag for every unsupported or contradictory option
     */
    static Options parse(String[] args) throws UsageException {
        Options o = new Options();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                System.out.print(USAGE);
                return null;
            }
            switch (a) {
                case "-m":
                case "--model":
                    o.model = model(a, o.model, value(a, args, i++));
                    break;
                case "-p":
                case "--prompt":
                    o.prompt = value(a, args, i++);
                    break;
                case "-f":
                case "--file": {
                    Path file = Paths.get(value(a, args, i++));
                    if (!Files.isReadable(file)) {
                        throw new UsageException(a + " " + file + " is not a readable file");
                    }
                    try {
                        o.prompt = LookupToolSupport.readPrompt(file);
                    } catch (IOException e) {
                        throw new UsageException(a + " " + file + " cannot be read: " + e.getMessage());
                    }
                    break;
                }
                case "-lcs":
                case "--lookup-cache-static":
                    o.staticCache = Paths.get(value(a, args, i++));
                    break;
                case "-lcd":
                case "--lookup-cache-dynamic":
                    o.dynamicCache = Paths.get(value(a, args, i++));
                    break;
                case "-c":
                case "--ctx-size":
                    o.ctxSize = intValue(a, value(a, args, i++));
                    break;
                case "--draft-max":
                case "--draft-n-max":
                case "--spec-draft-n-max":
                    o.draftMax = intValue(a, value(a, args, i++));
                    break;
                default:
                    if (a.startsWith("-")) {
                        throw new UsageException("unknown option " + a + " (run with --help for the options)");
                    }
                    o.model = model(a, o.model, a);
            }
        }
        if (o.model == null) {
            throw new UsageException("lookup-stats needs a model path");
        }
        return o;
    }

    /** {@code path} as the model, refusing a second model argument. */
    private static Path model(String flag, Path current, String path) throws UsageException {
        if (current != null) {
            throw new UsageException("too many model arguments; " + flag + " " + path + " follows " + current);
        }
        return Paths.get(path);
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    /** A non-negative integer value, refused by flag otherwise. */
    private static int intValue(String flag, String raw) throws UsageException {
        try {
            int n = Integer.parseInt(raw.trim());
            if (n < 0) {
                throw new UsageException(flag + " " + n + " is negative");
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer, got '" + raw + "'");
        }
    }
}
