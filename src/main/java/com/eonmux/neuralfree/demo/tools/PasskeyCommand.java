package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.tokenizer.IncrementalDetokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Locale;

/**
 * {@code nftool passkey}: the long-context retrieval test of llama.cpp's examples/passkey/passkey.cpp. A
 * pass key is hidden at position {@code --pos} of {@code --junk} filler sentences, the prompt is prefilled
 * in batches with the context shifted once it fills ({@code --keep} positions kept, one batch dropped, the
 * reference's shift policy), and the greedy answer to "What is the pass key?" is generated and checked.
 */
final class PasskeyCommand {

    /** The prompt the reference hides the key in. */
    static final String PROMPT_PREFIX = "There is an important info hidden inside a lot of irrelevant text."
            + " Find it and memorize them. I will quiz you about the important information there.";
    static final String PROMPT_SUFFIX = " What is the pass key? The pass key is";
    private static final String JUNK_SENTENCE = " The grass is green. The sky is blue. The sun is yellow."
            + " Here we go. There and back again.";

    /** The tokens left for the answer ({@code n_predict = 16}). */
    static final int N_PREDICT = 16;

    /** The default filler sentences of the reference ({@code n_junk = 250}). */
    static final int DEFAULT_JUNK = 250;

    /** The default sink positions kept through every shift ({@code n_keep = 32}). */
    static final int DEFAULT_KEEP = 32;

    /** The default forward batch, also the positions one shift drops ({@code n_batch}). */
    static final int DEFAULT_BATCH = 2048;

    private static final String USAGE =
            "nftool passkey <model> [options]\n"
            + "options (llama.cpp passkey names):\n"
            + "  --junk <n>                       filler sentences, default " + DEFAULT_JUNK + "\n"
            + "  --pos <n>                        where the pass key hides, 0-based; default drawn from the seed\n"
            + "  --keep <n>                       positions kept through every context shift, default "
            + DEFAULT_KEEP + "\n"
            + "  --batch-size <n>                 tokens per prefill batch and per shift, default "
            + DEFAULT_BATCH + "\n"
            + "  --ctx-size <n>                   context, default the training context plus --keep\n"
            + "  --seed <n>                       the pass key and its position, default 1\n"
            + "  --help, -h                       this text\n"
            + "The generated answer and whether it holds the pass key print to stdout; the run details go to\n"
            + "stderr.\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    private PasskeyCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code passkey}. */
    static void run(String[] args) {
        try {
            dispatch(args, EngineConfig.fromSystemProperties());
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
     * Parses the arguments, builds the prompt and runs the prefill, shift and answer loop. Package-private
     * so tests drive it with an explicit engine configuration.
     *
     * @return the generated answer text
     */
    static String dispatch(String[] args, EngineConfig cfg) throws IOException, UsageException {
        Options o = parse(args);
        if (o == null) {
            return "";
        }
        Mt19937 rng = new Mt19937(o.seed);
        int iPos = o.pos >= 0 ? o.pos : (int) rng.nextBounded(o.junk);
        int passkey = (int) rng.nextBounded(50000) + 1;
        String prompt = buildPrompt(o.junk, iPos, passkey);

        int keep = o.keep;
        EngineConfig effective = config(o, cfg, keep);
        try (GGUFModel m = open(o.model, effective)) {
            requireTokenLoop(m);
            ForwardBackend backend = m.forwardBackend();
            GGUFTokenizer tok = m.tokenizer();
            int nCtx = backend.memory().plan().nCtx();
            if (keep + 1 > nCtx) {
                throw new UsageException("--keep " + keep + " leaves no room in the " + nCtx
                        + " token context");
            }
            if (o.batchSize > nCtx - keep) {
                throw new UsageException("--batch-size " + o.batchSize + " exceeds the " + (nCtx - keep)
                        + " positions one shift can drop from the " + nCtx + " token context");
            }
            return runPasskey(m, backend, tok, o, prompt, iPos, passkey, keep, nCtx);
        }
    }

    /** The prompt of the reference: the prefix, the filler sentences with the key at {@code iPos}, the suffix. */
    static String buildPrompt(int junk, int iPos, int passkey) {
        StringBuilder prompt = new StringBuilder(PROMPT_PREFIX);
        for (int i = 0; i < junk; i++) {
            if (i == iPos) {
                prompt.append(" The pass key is ").append(passkey).append(". Remember it. ")
                        .append(passkey).append(" is the pass key.");
            }
            prompt.append(JUNK_SENTENCE);
        }
        return prompt.append(PROMPT_SUFFIX).toString();
    }

    /** The fill, shift and answer loop over one open model; prints the run details and the verdict. */
    private static String runPasskey(GGUFModel m, ForwardBackend backend, GGUFTokenizer tok, Options o,
            String prompt, int iPos, int passkey, int keep, int nCtx) throws IOException {
        int[] tokens = tok.encode(prompt);
        int nBatch = o.batchSize;
        int fillHorizon = nCtx - keep;
        System.err.printf(Locale.ROOT, "nftool passkey: prompt tokens: %d, context: %d, batch: %d,"
                + " junk: %d, pass key inserted at %d / %d%n", tokens.length, nCtx, nBatch, o.junk,
                iPos, o.junk);
        System.err.printf(Locale.ROOT, "nftool passkey: passkey = %d, inserted at position %d / %d%n",
                passkey, iPos, o.junk);

        // fill the context with the prompt; once it fills, drop one batch after the sink and continue,
        // the reference's shift loop
        int next = Math.min(fillHorizon, tokens.length);
        float[] logits = prefill(backend, tokens, 0, next, nBatch);
        while (next < tokens.length) {
            backend.shiftContext(keep, nBatch);
            int nPast = nPast(backend);
            int end = Math.min(next + nBatch, tokens.length);
            logits = prefill(backend, Arrays.copyOfRange(tokens, next, end), nPast, end - next, nBatch);
            next = end;
        }
        // free room for the answer, the reference's final shift
        int discard = nPast(backend) - fillHorizon + N_PREDICT;
        if (discard > 0) {
            backend.shiftContext(keep, discard);
        }
        String answer = generate(backend, tok, logits, nPast(backend), tokens.length);
        String digits = answer.replaceAll("[^0-9]", "");
        boolean recovered = digits.contains(String.valueOf(passkey));
        System.out.println("nftool passkey: passkey = " + passkey + ", position " + iPos + " / " + o.junk);
        System.out.println("nftool passkey: answer: " + answer);
        System.out.println("nftool passkey: passkey recovered: " + recovered);
        return answer;
    }

    /**
     * Prefills {@code count} tokens at {@code start} in batches of {@code nBatch}, logging every covered
     * range as the reference's fill loop does.
     *
     * @return the logits of the last prefilled row, or {@code null} when nothing was prefilled
     */
    private static float[] prefill(ForwardBackend backend, int[] tokens, int start, int count, int nBatch) {
        float[] logits = null;
        for (int off = 0; off < count; ) {
            int n = Math.min(nBatch, count - off);
            logits = backend.prefill(Arrays.copyOfRange(tokens, off, off + n), start + off);
            off += n;
            System.err.printf(Locale.ROOT, "nftool passkey: processed: [%6d, %6d)%n", start + off - n,
                    start + off);
        }
        return logits;
    }

    /** Greedy answer loop: at most {@link #N_PREDICT} tokens, stopping at the end of generation. */
    private static String generate(ForwardBackend backend, GGUFTokenizer tok, float[] logits,
            int nPast, int promptTokens) throws IOException {
        IncrementalDetokenizer detok = new IncrementalDetokenizer(t -> tok.decode(t));
        System.err.print(PROMPT_SUFFIX);
        StringBuilder answer = new StringBuilder();
        int nCur = promptTokens;
        int nLen = promptTokens + N_PREDICT;
        while (nCur <= nLen) {
            int id = ForwardBackend.argmax(logits);
            if (tok.isStopToken(id) || nCur == nLen) {
                break;
            }
            String piece = detok.next(id);
            answer.append(piece);
            System.out.print(piece);
            logits = backend.decode(id, nPast++);
            nCur++;
        }
        answer.append(detok.flush());
        System.out.println();
        return answer.toString();
    }

    /** The memory's stored length, the position the next tokens continue at. */
    private static int nPast(ForwardBackend backend) {
        return backend.memory().sequence().nPast();
    }

    /** The parsed command line; {@link #parse} fills and validates it. */
    private static final class Options {
        Path model;
        int junk = DEFAULT_JUNK;
        int pos = -1;
        int keep = DEFAULT_KEEP;
        int batchSize = DEFAULT_BATCH;
        int ctxSize = -1;
        long seed = 1;
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
                case "--junk":
                    o.junk = intValue(a, value(a, args, i++));
                    break;
                case "--pos":
                    o.pos = intValue(a, value(a, args, i++));
                    break;
                case "--keep":
                    o.keep = intValue(a, value(a, args, i++));
                    break;
                case "--batch-size":
                    o.batchSize = intValue(a, value(a, args, i++));
                    break;
                case "--ctx-size":
                    o.ctxSize = intValue(a, value(a, args, i++));
                    break;
                case "--seed":
                    o.seed = longValue(a, value(a, args, i++));
                    break;
                default:
                    if (a.startsWith("-")) {
                        throw new UsageException("unknown option " + a + " (run with --help for the options)");
                    }
                    if (o.model != null) {
                        throw new UsageException("too many file arguments; '" + a + "' follows " + o.model);
                    }
                    o.model = Paths.get(a);
            }
        }
        if (o.model == null) {
            throw new UsageException("passkey needs a model path");
        }
        if (o.junk < 1) {
            throw new UsageException("--junk needs at least one filler sentence, not " + o.junk);
        }
        if (o.pos >= o.junk) {
            throw new UsageException("--pos " + o.pos + " hides the pass key past the " + o.junk
                    + " filler sentences; use a position below the junk count");
        }
        if (o.batchSize < 1) {
            throw new UsageException("--batch-size needs a positive count, not " + o.batchSize);
        }
        return o;
    }

    /** The engine config: the requested context, the training one plus the sink when none is given. */
    private static EngineConfig config(Options o, EngineConfig base, int keep)
            throws IOException, UsageException {
        int ctx = o.ctxSize;
        if (ctx == 0) {
            throw new UsageException("--ctx-size 0 fits no passkey prompt; give a positive context");
        }
        if (ctx < 0) {
            ctx = trainingContext(o.model) + keep;
        }
        MemoryConfig.Builder mem = base.memory().toBuilder().nCtx(ctx);
        mem.ubatch(Math.min(o.batchSize, ctx));
        return base.toBuilder().memory(mem.build()).build();
    }

    /**
     * The training context of the model file ({@code n_ctx_train}), read from the metadata the engine loads
     * its hparams from.
     *
     * @throws IOException when the file carries no context length
     */
    static int trainingContext(Path model) throws IOException {
        try (ModelSource src = ModelSources.open(model.toString(), ModelSourceOptions.defaults())) {
            String arch = src.getString("general.architecture");
            Object ctx = arch != null ? src.getMetadata().get(arch + ".context_length") : null;
            if (ctx instanceof Number) {
                return ((Number) ctx).intValue();
            }
        }
        throw new IOException(model + " carries no context_length in its metadata; pass --ctx-size");
    }

    /** Opens the model through the factory, as the demos open theirs. */
    private static GGUFModel open(Path model, EngineConfig cfg) throws IOException {
        EngineModel opened = LLMModelFactory.open(model.toString(), cfg);
        if (!(opened instanceof GGUFModel)) {
            throw new IOException("the passkey test runs the token engine, but " + model + " opened as "
                    + opened.getClass().getSimpleName());
        }
        return (GGUFModel) opened;
    }

    /** The passkey loop runs a decoder with memory, as the imatrix collector does. */
    private static void requireTokenLoop(GGUFModel m) throws IOException {
        SupportClass support = m.supportClass();
        switch (support) {
            case EMBED:
            case CLS:
            case NA:
            case DIFF:
            case ENC_DEC:
            case DRAFT:
                throw new IOException("the passkey test runs the decoder prefill of a model with memory; "
                        + m.architecture() + " (" + support + ") has none");
            default:
                return;
        }
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    /** A non-negative count of the flag. */
    private static int intValue(String flag, String v) throws UsageException {
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

    /** A non-negative seed of the flag. */
    private static long longValue(String flag, String v) throws UsageException {
        try {
            long n = Long.parseLong(v.trim());
            if (n < 0) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a non-negative integer, not '" + v + "'");
        }
    }
}
