package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EmbeddingNorm;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.heads.ClassifierHead;
import com.eonmux.neuralfree.gguf.ops.heads.EncodeGate;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * {@code nftool embedding}: llama.cpp's embedding example (examples/embedding/embedding.cpp). The prompt
 * ({@code --prompt} or {@code --file}, escapes processed) splits into one prompt per {@code --embd-separator}
 * occurrence, every prompt tokenizes with special tokens and embeds on one loaded model, the rows normalize per
 * {@code --embd-normalize} (llama.cpp {@code common_embd_normalize}, {@link EmbeddingNorm}), and the output takes
 * llama.cpp's shapes through {@link EmbeddingOutput}. A RANK-pooling model scores each prompt; a prompt holding
 * the {@code --cls-separator} is a query/document pair joined by the model's rerank prompt or its EOS/SEP tokens.
 *
 * <p>Prompts embed in llama.cpp's batches ({@link EmbeddingBatchFlags}): a batch of at most {@code --batch-size}
 * tokens over at most {@code --parallel} sequences runs as one multi-sequence forward over a paged memory, its
 * pages freed before the next batch, which produces the same per-sequence embeddings as llama.cpp's
 * {@code batch_decode}. Without {@code --parallel} the memory holds every prompt (up to
 * {@link MemoryConfig#MAX_PARALLEL}) in one unified context, as llama.cpp's embedding tool does. Per-token output
 * (pooling {@code NONE}) prints one row per token across all prompts, as llama.cpp's {@code n_embd_count}
 * accounting does.</p>
 */
final class EmbeddingCommand {

    /** llama.cpp's default classification pair separator ({@code cls_sep}). */
    static final String DEFAULT_CLS_SEPARATOR = "\t";

    private static final String USAGE =
            "nftool embedding <model> [options]\n"
            + "options (llama.cpp embedding names):\n"
            + "  --prompt, -p <text>              the prompts; the --embd-separator occurrences split them\n"
            + "  --file, -f <fname>               read the prompts from a file (one trailing newline dropped)\n"
            + "  --escape, -e | --no-escape       process the \\n \\r \\t \\' \\\" \\\\ escapes of the prompt\n"
            + "                                   (default: on)\n"
            + "  --embd-normalize <n>             normalisation: -1 none, 0 max absolute int16, 1 taxicab,\n"
            + "                                   2 euclidean (default), >2 p-norm\n"
            + "  --embd-output-format <format>    empty = default, array = [[],[]...], json = openai style,\n"
            + "                                   json+ = json plus the cosine similarity matrix, raw = one\n"
            + "                                   whitespace-delimited embedding per line\n"
            + "  --embd-separator <string>        separator between prompts, default a newline\n"
            + "  --cls-separator <string>         separator of a reranker's query/document pair, default a tab\n"
            + "  --parallel, -np <n>              sequences per batch (llama.cpp -np; default: every prompt, up\n"
            + "                                   to " + MemoryConfig.MAX_PARALLEL + ", in one unified context)\n"
            + EmbeddingBatchFlags.USAGE
            + "  --verbose-prompt                 print every prompt's tokens before embedding\n"
            + "  --help, -h                       this text\n"
            + "Every prompt embeds from position 0 in its own sequence, so the rows match llama.cpp's per-sequence\n"
            + "embeddings. Results print to stdout; the run summary and warnings go to stderr.\n";

    private EmbeddingCommand() {
    }

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The embedding options of one run (immutable); the prompt has its escapes processed. */
    static final class Options {

        final String model;
        final String prompt;
        final int normalize;
        final String format;
        final String separator;
        final String clsSeparator;
        final boolean verbosePrompt;
        final EmbeddingBatchFlags.Settings batch;

        Options(String model, String prompt, int normalize, String format, String separator, String clsSeparator,
                boolean verbosePrompt, EmbeddingBatchFlags.Settings batch) {
            this.model = model;
            this.prompt = prompt;
            this.normalize = normalize;
            this.format = format;
            this.separator = separator;
            this.clsSeparator = clsSeparator;
            this.verbosePrompt = verbosePrompt;
            this.batch = batch;
        }

        String model() { return model; }
        String prompt() { return prompt; }
        int normalize() { return normalize; }
        String format() { return format; }
        String separator() { return separator; }
        String clsSeparator() { return clsSeparator; }
        boolean verbosePrompt() { return verbosePrompt; }
        /** The pooling, batching and load flags. */
        EmbeddingBatchFlags.Settings batch() { return batch; }
        /** The configuration of the flags, before {@link EmbeddingBatchFlags.Settings#forModel} fits it. */
        EngineConfig load() { return batch.load(); }
        /** The pooling override, or null for the GGUF's pooling. */
        PoolingKind pooling() { return batch.pooling(); }
        int batchSize() { return batch.batchSize(); }
        /** {@code --parallel}, or 0 when absent. */
        int parallel() { return batch.parallel(); }
    }

    /** Runs the tool with the launcher's arguments after {@code embedding}. */
    static void run(String[] args) {
        try {
            Options o = parse(args);
            if (o != null) {
                execute(o);
            }
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | UnsupportedOperationException | IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Opens the model on the parsed configuration, embeds every prompt and prints the rows in the selected format.
     * The spec resolves before the load: it refuses a pooling override the model cannot serve before any weight
     * moves, and sizes the memory's sequences ({@link #sequences}) and a non-causal model's ubatch
     * ({@link EmbeddingBatchFlags.Settings#forModel}).
     *
     * @throws IOException for an unreadable model or a prompt longer than the model's context or the batch size
     */
    static void execute(Options o) throws IOException {
        List<String> prompts = splitLines(o.prompt(), o.separator());
        ModelSpec spec = LLMModelFactory.inspect(o.model(), o.load());
        int batchSize = o.batch().batchSize(spec);
        EngineConfig cfg = o.batch().forModel(spec, sequences(o.parallel(), prompts.size(), spec));
        try (EngineModel model = LLMModelFactory.open(o.model(), cfg)) {
            List<int[]> inputs = tokenizePrompts(model, prompts, o.clsSeparator(), batchSize);
            StringBuilder log = new StringBuilder(lastTokenWarnings(model.tokenizer(), inputs));
            if (o.verbosePrompt()) {
                PromptDump dump = new PromptDump(model.tokenizer());
                for (int i = 0; i < inputs.size(); i++) {
                    dump.append(log, i, prompts.get(i), inputs.get(i));
                }
            }
            System.err.print(log);
            List<float[]> rows = embedRows(model, inputs, o.normalize(), batchSize);
            System.err.println("[nftool embedding] prompts: " + prompts.size() + ", pooling: " + model.poolingType()
                    + ", embeddings: " + rows.size() + " x " + rows.get(0).length + ", normalize: " + o.normalize());
            StringBuilder out = new StringBuilder();
            EmbeddingOutput.render(out, rows, prompts, model.poolingType(), classLabels(model), o.format(),
                    o.normalize());
            System.out.print(out);
        }
    }

    /**
     * The memory sequences of a run: {@code --parallel} when given; else every prompt in its own sequence, up to
     * {@link MemoryConfig#MAX_PARALLEL} (llama.cpp's embedding tool takes {@code n_seq_max} sequences when
     * {@code -np} is 1); one for a model that embeds per token (pooling {@code NONE}), which runs no batches.
     */
    static int sequences(int parallel, int prompts, ModelSpec spec) {
        if (parallel > 0) {
            return parallel;
        }
        if (spec.head().pooling() == PoolingKind.NONE) {
            return 1;
        }
        return Math.max(1, Math.min(prompts, MemoryConfig.MAX_PARALLEL));
    }

    /** The model's classifier labels for RANK pooling, empty otherwise. */
    private static String[] classLabels(EngineModel model) {
        return model.poolingType() == LLMModel.PoolingType.RANK ? model.classLabels() : new String[0];
    }

    /**
     * Tokenizes every prompt as llama.cpp's embedding example does: with special tokens added and parsed
     * ({@code common_tokenize(prompt, true, true)}), and for RANK pooling a prompt holding {@code clsSeparator}
     * becomes its reranker input ({@link #rankPrompt}). A model without an embedding output refuses before the first
     * prompt tokenizes, as llama.cpp refuses its encoder-decoders. A pooled prompt of more than {@code batchSize}
     * tokens refuses as llama.cpp's exceeds-batch error does (embedding.cpp:209-212).
     *
     * @throws IOException                   when a prompt holds no tokens, or more than the model's training
     *                                       context or the batch size fits
     * @throws UnsupportedOperationException when the model has no embedding output
     */
    static List<int[]> tokenizePrompts(EngineModel model, List<String> prompts, String clsSeparator, int batchSize)
            throws IOException {
        EncodeGate.checkEncode(model.spec());
        boolean perToken = model.poolingType() == LLMModel.PoolingType.NONE;
        if (perToken) {
            EncodeGate.checkEncodeTokens(model.spec());
        }
        GGUFTokenizer tok = model.tokenizer();
        boolean rank = model.poolingType() == LLMModel.PoolingType.RANK;
        int cap = model.contextLength();
        List<int[]> inputs = new ArrayList<>(prompts.size());
        for (int p = 0; p < prompts.size(); p++) {
            String prompt = prompts.get(p);
            String text = rank && prompt.contains(clsSeparator) ? rankPrompt(tok, prompt, clsSeparator) : prompt;
            int[] tokens = tok.encode(text, true, true);
            if (tokens.length == 0) {
                throw new IOException("prompt " + p + " ('" + prompt + "') tokenizes to no tokens");
            }
            if (tokens.length > cap) {
                throw new IOException("prompt " + p + " tokenizes to " + tokens.length
                        + " tokens, more than the model's training context (" + cap + ") fits");
            }
            if (!perToken && tokens.length > batchSize) {
                throw new IOException("number of tokens in input line " + p + " (" + tokens.length
                        + ") exceeds batch size (" + batchSize + "), increase batch size (--batch-size) and re-run");
            }
            inputs.add(tokens);
        }
        return inputs;
    }

    /**
     * The reranker text of a prompt holding the classification separator (embedding.cpp:178-203): with the model's
     * {@code rerank} prompt, its {@code {query}} and {@code {document}} placeholders filled with the first two
     * pieces; otherwise every piece joined by the EOS text (when the vocabulary adds EOS) and the SEP text (when it
     * adds SEP).
     */
    static String rankPrompt(GGUFTokenizer tok, String prompt, String clsSeparator) {
        List<String> pairs = splitLines(prompt, clsSeparator);
        String template = tok.rerankTemplate();
        if (template != null) {
            return template.replace(ClassifierHead.RERANK_QUERY, pairs.get(0))
                    .replace(ClassifierHead.RERANK_DOCUMENT, pairs.get(1));
        }
        SpecialTokens special = tok.specialTokens();
        String joint = tokenText(tok, tok.vocab().addEos(), special.eos())
                + tokenText(tok, tok.vocab().addSep(), special.sep());
        return String.join(joint, pairs);
    }

    private static String tokenText(GGUFTokenizer tok, boolean added, int id) {
        return added && id != SpecialTokens.NULL ? tok.getToken(id) : "";
    }

    /**
     * llama.cpp's warning for every input whose last token is neither SEP nor EOS (embedding.cpp:216-222): the
     * vocabulary should set {@code tokenizer.ggml.add_eos_token}. Empty when every input ends on one.
     */
    static String lastTokenWarnings(GGUFTokenizer tok, List<int[]> inputs) {
        int sep = tok.specialTokens().sep();
        int eos = tok.eosId();
        StringBuilder out = new StringBuilder();
        for (int[] inp : inputs) {
            int last = inp.length == 0 ? SpecialTokens.NULL : inp[inp.length - 1];
            if (inp.length == 0 || (last != sep && last != eos)) {
                out.append("main: last token in the prompt is not SEP or EOS\n");
                out.append("main: 'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header\n");
            }
        }
        return out.toString();
    }

    /**
     * Embeds every tokenized prompt: one normalized row per token for pooling {@code NONE}, one per prompt otherwise
     * (the RANK scores for a ranking model). The pooled prompts run in llama.cpp's batches
     * ({@link EmbeddingBatchFlags#encode}, embedding.cpp:262-283).
     *
     * @throws IOException when the model fails to encode
     */
    static List<float[]> embedRows(EngineModel model, List<int[]> inputs, int embdNorm, int batchSize)
            throws IOException {
        List<float[]> rows = new ArrayList<>();
        if (model.poolingType() == LLMModel.PoolingType.NONE) {
            for (int[] tokens : inputs) {
                for (float[] row : model.encodeTokens(tokens)) {
                    rows.add(EmbeddingNorm.normalize(row, embdNorm));
                }
            }
            return Collections.unmodifiableList(rows);
        }
        for (float[] row : EmbeddingBatchFlags.encode(model, inputs, batchSize)) {
            rows.add(EmbeddingNorm.normalize(row, embdNorm));
        }
        return Collections.unmodifiableList(rows);
    }

    /**
     * The prompt lines of llama.cpp's {@code split_lines}: every separator occurrence starts a new line, empty
     * lines stay (they embed as their special tokens alone), and the text after the last separator is the last
     * line.
     *
     * @throws IllegalArgumentException for an empty separator
     */
    static List<String> splitLines(String text, String separator) {
        if (separator.isEmpty()) {
            throw new IllegalArgumentException("the prompt separator is empty");
        }
        List<String> lines = new ArrayList<>();
        int start = 0;
        int end = text.indexOf(separator);
        while (end >= 0) {
            lines.add(text.substring(start, end));
            start = end + separator.length();
            end = text.indexOf(separator, start);
        }
        lines.add(text.substring(start));
        return lines;
    }

    /**
     * Parses the flags (llama.cpp's embedding option names). The last of {@code --prompt} and {@code --file} sets
     * the prompt, whose escapes are processed afterwards unless {@code --no-escape} is given, as llama.cpp's
     * {@code common_params_parse} does. The pooling, batching and load flags parse through
     * {@link EmbeddingBatchFlags}.
     *
     * @return the options, or null after printing the usage for {@code --help}
     * @throws UsageException naming the offending flag
     */
    static Options parse(String[] args) throws UsageException {
        String model = null;
        String prompt = null;
        String format = "";
        String separator = "\n";
        String clsSeparator = DEFAULT_CLS_SEPARATOR;
        int normalize = 2;
        boolean escape = true;
        boolean verbosePrompt = false;
        EmbeddingBatchFlags batch = batchFlags();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--help":
                case "-h":
                    System.out.println(USAGE);
                    return null;
                case "--prompt":
                case "-p":
                    prompt = value(a, args, i++);
                    break;
                case "--file":
                case "-f":
                    prompt = readPromptFile(a, value(a, args, i++));
                    break;
                case "--escape":
                case "-e":
                    escape = true;
                    break;
                case "--no-escape":
                    escape = false;
                    break;
                case "--embd-normalize":
                    normalize = normalizeValue(a, value(a, args, i++));
                    break;
                case "--embd-output-format":
                    format = formatValue(a, value(a, args, i++));
                    break;
                case "--embd-separator":
                    separator = nonEmpty(a, value(a, args, i++));
                    break;
                case "--cls-separator":
                    clsSeparator = nonEmpty(a, value(a, args, i++));
                    break;
                case "--verbose-prompt":
                    verbosePrompt = true;
                    break;
                default: {
                    int last = batchFlag(batch, args, i);
                    if (last < 0) {
                        model = positional(a, model);
                    } else {
                        i = last;
                    }
                }
            }
        }
        if (model == null) {
            throw new UsageException("embedding needs a model path");
        }
        if (prompt == null) {
            throw new UsageException("embedding needs --prompt or --file");
        }
        String text = escape ? CvectorGenerator.processEscapes(prompt) : prompt;
        return new Options(model, text, normalize, format, separator, clsSeparator, verbosePrompt,
                batchSettings(batch));
    }

    private static EmbeddingBatchFlags batchFlags() throws UsageException {
        try {
            return EmbeddingBatchFlags.fromSystemProperties();
        } catch (EmbeddingBatchFlags.FlagException e) {
            throw new UsageException(e.getMessage());
        }
    }

    /** One {@link EmbeddingBatchFlags} flag: the index of its last argument, or -1 for any other token. */
    private static int batchFlag(EmbeddingBatchFlags batch, String[] args, int i) throws UsageException {
        try {
            return batch.parse(args, i);
        } catch (EmbeddingBatchFlags.FlagException e) {
            throw new UsageException(e.getMessage());
        }
    }

    private static EmbeddingBatchFlags.Settings batchSettings(EmbeddingBatchFlags batch) throws UsageException {
        try {
            return batch.build();
        } catch (EmbeddingBatchFlags.FlagException e) {
            throw new UsageException(e.getMessage());
        }
    }

    /** llama.cpp {@code -f}: the file's text with one trailing newline dropped. */
    private static String readPromptFile(String flag, String name) throws UsageException {
        Path f = Paths.get(name);
        if (!Files.isRegularFile(f) || !Files.isReadable(f)) {
            throw new UsageException(flag + " '" + name + "' is not a readable file");
        }
        try {
            String text = Files.readString(f, StandardCharsets.UTF_8);
            return text.endsWith("\n") ? text.substring(0, text.length() - 1) : text;
        } catch (IOException e) {
            throw new UsageException(flag + " '" + name + "' cannot be read: " + e.getMessage());
        }
    }

    private static int normalizeValue(String flag, String s) throws UsageException {
        int n = intValue(flag, s);
        if (n < -1) {
            throw new UsageException(flag + " " + n + " is invalid; use -1, 0, 1, 2 or a larger p");
        }
        return n;
    }

    private static String formatValue(String flag, String format) throws UsageException {
        if (!EmbeddingOutput.FORMATS.contains(format)) {
            throw new UsageException(flag + " '" + format + "' is invalid; use array, json, json+ or raw");
        }
        return format;
    }

    private static String nonEmpty(String flag, String s) throws UsageException {
        if (s.isEmpty()) {
            throw new UsageException(flag + " needs a non-empty string");
        }
        return s;
    }

    private static String positional(String arg, String model) throws UsageException {
        if (arg.startsWith("-")) {
            throw new UsageException("unknown option " + arg + " (run with --help for the options)");
        }
        if (model != null) {
            throw new UsageException("too many file arguments; '" + arg + "' follows " + model);
        }
        return arg;
    }

    private static String value(String flag, String[] args, int index) throws UsageException {
        // The value consumes the next argv token even when it starts with '-' (llama.cpp consumes it), so
        // --embd-normalize -1 parses.
        if (index + 1 >= args.length || args[index + 1] == null) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    private static int intValue(String flag, String s) throws UsageException {
        try {
            return Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer, got '" + s + "'");
        }
    }
}
