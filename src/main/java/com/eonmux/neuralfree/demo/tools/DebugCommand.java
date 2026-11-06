package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.EmbeddingNorm;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.gpu.diag.TensorTraceObserver;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * {@code nftool debug}: llama.cpp's debug example (examples/debug/debug.cpp). The prompt tokenizes with special
 * tokens, its tokens print with their pieces, and the run either prints every forward stage's tensor (the
 * {@code --tensor-filter} patterns select stages, {@code --abort-on-nan} stops on the first NaN) or saves the final
 * logits or the embeddings to the files the reference writes ({@code --save-logits}):
 *
 * <pre>
 *    &lt;dir&gt;/llamacpp-&lt;model&gt;.bin             raw little-endian floats
 *    &lt;dir&gt;/llamacpp-&lt;model&gt;.txt             one "index: value" line per float
 *    &lt;dir&gt;/llamacpp-&lt;model&gt;-prompt.txt      the prompt, its token count and its token ids
 *    &lt;dir&gt;/llamacpp-&lt;model&gt;-tokens.bin      the token ids, little-endian int32
 * </pre>
 *
 * <p>{@code --embedding} saves embeddings instead of logits, the name gaining {@code -embeddings}. A pooled model
 * saves its single pooled row, a model that pools {@code NONE} one row per token, through the model's embedding
 * output or, for a generative model without one, its final hidden rows. {@code --embd-normalize} normalizes each
 * row as llama.cpp's {@code common_embd_normalize}. The per-stage trace reads the GPU execution stages; a model on
 * the CPU backend is refused by name.</p>
 */
final class DebugCommand {

    /** llama.cpp's default logits output directory ({@code params.logits_output_dir}). */
    static final String DEFAULT_OUTPUT_DIR = "data";

    private static final String USAGE =
            "nftool debug <model> [options]\n"
            + "options (llama.cpp debug names):\n"
            + "  --model, -m <path>               the model to run\n"
            + "  --prompt, -p <text>              the prompt to evaluate\n"
            + "  --file, -f <fname>               read the prompt from a file (one trailing newline dropped)\n"
            + "  --save-logits                    save the final logits (or the embeddings) to files instead of\n"
            + "                                   printing the per-op tensor trace\n"
            + "  --logits-output-dir <dir>        directory of the saved files, default " + DEFAULT_OUTPUT_DIR + "\n"
            + "  --embedding, --embeddings        save embeddings instead of logits\n"
            + "  --embd-normalize <n>             embedding normalization: -1 none, 0 max absolute, 1 taxicab,\n"
            + "                                   2 euclidean (default), >2 p-norm\n"
            + "  --tensor-filter <regex>          print only stages whose label starts with the pattern; repeatable\n"
            + "  --abort-on-nan                   stop with a message on the first stage that sums to NaN\n"
            + "  --steps <n>                      forward steps to trace, default 0 (every step)\n"
            + "  --ctx-size, -c <n>               the context the prompt must fit\n"
            + "  --batch-size, -b <n>             forward ubatch of the engine\n"
            + "  --escape, -e | --no-escape       process the \\n \\r \\t \\' \\\" \\\\ escapes of the prompt\n"
            + "                                   (default: on)\n"
            + "  --help, -h                       this text\n"
            + "The trace reads the GPU execution stages; a CPU backend has none and is refused by name.\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The parsed command line; {@link #parse} fills and validates it. */
    static final class Options {
        Path model;
        String prompt = "";
        boolean saveLogits;
        Path outputDir = Paths.get(DEFAULT_OUTPUT_DIR);
        boolean outputDirSet;
        boolean embedding;
        int embdNormalize = EmbeddingNorm.EUCLIDEAN;
        boolean embdNormalizeSet;
        final List<String> filters = new ArrayList<>();
        boolean abortOnNan;
        int steps;
        boolean stepsSet;
        int ctxSize = -1;
        int batchSize = -1;
        boolean escape = true;
    }

    private DebugCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code debug}. */
    static void run(String[] args) {
        try {
            Options o = parse(args);
            if (o != null) {
                dispatch(o, EngineConfig.fromSystemProperties(), System.out, System.err);
            }
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (TensorTraceObserver.NonFiniteException e) {
            System.err.println(e.getMessage());
            System.exit(0);
        } catch (IOException | UnsupportedOperationException | IllegalArgumentException | IllegalStateException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /** Opens the model, prints the tokenized prompt and runs the selected mode. */
    static void dispatch(Options o, EngineConfig base, PrintStream out, PrintStream err)
            throws IOException, UsageException {
        try (EngineModel model = LLMModelFactory.open(o.model.toString(), config(o, base))) {
            int[] tokens = model.tokenizer().encode(o.prompt, true);
            if (tokens.length == 0) {
                throw new IOException("the prompt tokenizes to no tokens; give --prompt or --file");
            }
            printTokenizedPrompt(model, tokens, o.prompt, out);
            if (o.saveLogits) {
                save(model, tokens, o, out, err);
            } else {
                ForwardTrace.trace(model, tokens, o.filters, o.abortOnNan, o.steps, out);
            }
        }
    }

    /** llama.cpp's {@code print_tokenized_prompt}: the BOS policy, the text, the ids and each token's piece. */
    static void printTokenizedPrompt(EngineModel model, int[] tokens, String prompt, Appendable out)
            throws IOException {
        out.append("Model add_bos: ").append(Boolean.toString(model.tokenizer().vocab().addBos())).append('\n');
        out.append("Input prompt: \"").append(prompt).append("\"\n");
        out.append("Token ids (").append(Integer.toString(tokens.length)).append("):\n");
        for (int id : tokens) {
            out.append(new String(model.tokenizer().pieceWithSpecial(id), StandardCharsets.UTF_8)).append('(')
                    .append(Integer.toString(id)).append(") ");
        }
        out.append('\n');
    }

    /** Computes the logits or embeddings and writes the four files of the reference. */
    private static void save(EngineModel model, int[] tokens, Options o, PrintStream out, PrintStream err)
            throws IOException {
        float[] values;
        String suffix;
        if (o.embedding) {
            values = flatten(embeddingRows(model, tokens, o.embdNormalize));
            suffix = "-embeddings";
        } else {
            values = lastLogits(model, tokens);
            suffix = "";
        }
        Path base = o.outputDir.resolve("llamacpp-" + modelName(o.model) + suffix);
        Path parent = base.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Files.write(base.resolveSibling(base.getFileName() + ".bin"), floatsLe(values));
        Files.write(base.resolveSibling(base.getFileName() + ".txt"), floatLines(values));
        Files.write(base.resolveSibling(base.getFileName() + "-prompt.txt"), promptLines(o.prompt, tokens));
        Files.write(base.resolveSibling(base.getFileName() + "-tokens.bin"), intsLe(tokens));
        err.println("Data saved to " + base + ".bin");
        err.println("Data saved to " + base + ".txt");
        err.println("Prompt saved to " + base + "-prompt.txt");
        err.println("Tokens saved to " + base + "-tokens.bin");
    }

    /** The logits of the prompt's last token ({@code llama_get_logits_ith(tokens.size() - 1)}). */
    static float[] lastLogits(EngineModel model, int[] tokens) {
        ForwardBackend backend = model.forwardBackend();
        backend.reset();
        float[][] rows = backend.forwardRows(tokens, 0, new int[]{tokens.length - 1});
        return rows[0].clone();
    }

    /**
     * The embedding rows of the prompt: one pooled row when the model pools, else one row per token. A generative
     * model without a per-token embedding output uses its final hidden rows ({@code llama_get_embeddings}).
     */
    static float[][] embeddingRows(EngineModel model, int[] tokens, int normalize) throws IOException {
        float[][] rows;
        if (model.poolingType() != LLMModel.PoolingType.NONE) {
            rows = new float[][]{model.encode(tokens)};
        } else {
            try {
                rows = model.encodeTokens(tokens);
            } catch (UnsupportedOperationException e) {
                rows = model.forwardBackend().hiddenRows(tokens, null);
            }
        }
        float[][] normalized = new float[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            normalized[i] = EmbeddingNorm.normalize(rows[i], normalize);
        }
        return normalized;
    }

    private static float[] flatten(float[][] rows) {
        int width = rows.length == 0 ? 0 : rows[0].length;
        long total = (long) rows.length * width;
        if (total > Integer.MAX_VALUE - 8L) {
            throw new IllegalStateException("the embeddings hold " + rows.length + " x " + width
                    + " values, more than one Java array");
        }
        float[] values = new float[(int) total];
        for (int i = 0; i < rows.length; i++) {
            if (rows[i].length != width) {
                throw new IllegalStateException("embedding row " + i + " holds " + rows[i].length
                        + " values, not " + width);
            }
            System.arraycopy(rows[i], 0, values, i * width, width);
        }
        return values;
    }

    /** The model file's name without its extension, as llama.cpp's {@code path.stem()} gives it. */
    static String modelName(Path model) {
        Path name = model.getFileName();
        String text = name == null ? "" : name.toString();
        int dot = text.lastIndexOf('.');
        if (dot > 0) {
            text = text.substring(0, dot);
        }
        return text.isEmpty() ? "model" : text;
    }

    private static byte[] floatsLe(float[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : values) {
            bytes.putFloat(v);
        }
        return bytes.array();
    }

    private static byte[] intsLe(int[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : values) {
            bytes.putInt(v);
        }
        return bytes.array();
    }

    /** One {@code index: value} line per value, the reference's {@code file << i << ": " << value << '\n'}. */
    private static byte[] floatLines(float[] values) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            sb.append(i).append(": ").append(String.format(Locale.ROOT, "%.6g", values[i])).append('\n');
        }
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }

    /** The reference's {@code -prompt.txt}: the prompt, its token count and the comma-separated ids. */
    private static byte[] promptLines(String prompt, int[] tokens) {
        StringBuilder sb = new StringBuilder();
        sb.append("prompt: ").append(prompt).append('\n');
        sb.append("n_tokens: ").append(tokens.length).append('\n');
        sb.append("token ids: ");
        for (int i = 0; i < tokens.length; i++) {
            sb.append(tokens[i]);
            if (i + 1 < tokens.length) {
                sb.append(", ");
            }
        }
        sb.append('\n');
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }

    /** The engine config: the command line's context and batch over the base configuration. */
    private static EngineConfig config(Options o, EngineConfig base) {
        MemoryConfig.Builder mem = base.memory().toBuilder();
        if (o.ctxSize > 0) {
            mem.nCtx(o.ctxSize);
        }
        if (o.batchSize > 0) {
            mem.ubatch(o.batchSize);
        }
        return base.toBuilder().memory(mem.build()).build();
    }

    /**
     * Parses and validates the flags (llama.cpp's debug option names); {@code null} for {@code --help}.
     *
     * @throws UsageException naming the flag for every unsupported or contradictory option
     */
    static Options parse(String[] args) throws UsageException {
        Options o = new Options();
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--help":
                case "-h":
                    System.out.print(USAGE);
                    return null;
                case "--model":
                case "-m":
                    o.model = Paths.get(value(a, args, i++));
                    break;
                case "--prompt":
                case "-p":
                    o.prompt = value(a, args, i++);
                    break;
                case "--file":
                case "-f":
                    o.prompt = readPromptFile(a, value(a, args, i++));
                    break;
                case "--save-logits":
                    o.saveLogits = true;
                    break;
                case "--logits-output-dir":
                    o.outputDir = Paths.get(value(a, args, i++));
                    o.outputDirSet = true;
                    break;
                case "--embedding":
                case "--embeddings":
                    o.embedding = true;
                    break;
                case "--embd-normalize":
                    o.embdNormalize = normalization(a, value(a, args, i++));
                    o.embdNormalizeSet = true;
                    break;
                case "--tensor-filter":
                    o.filters.add(value(a, args, i++));
                    break;
                case "--abort-on-nan":
                    o.abortOnNan = true;
                    break;
                case "--steps":
                    o.steps = nonNegative(a, value(a, args, i++));
                    o.stepsSet = true;
                    break;
                case "--ctx-size":
                case "-c":
                    o.ctxSize = positive(a, value(a, args, i++));
                    break;
                case "--batch-size":
                case "-b":
                    o.batchSize = positive(a, value(a, args, i++));
                    break;
                case "--escape":
                case "-e":
                    o.escape = true;
                    break;
                case "--no-escape":
                    o.escape = false;
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
            throw new UsageException("debug needs a model path");
        }
        if (o.escape) {
            o.prompt = CvectorGenerator.processEscapes(o.prompt);
        }
        if (o.saveLogits) {
            if (!o.filters.isEmpty()) {
                throw new UsageException("--tensor-filter shapes the per-op trace; --save-logits writes files"
                        + " instead and prints no tensors");
            }
            if (o.abortOnNan) {
                throw new UsageException("--abort-on-nan shapes the per-op trace; --save-logits writes files"
                        + " instead and traces nothing");
            }
            if (o.stepsSet) {
                throw new UsageException("--steps shapes the per-op trace; --save-logits writes files instead and"
                        + " traces nothing");
            }
        } else {
            if (o.outputDirSet) {
                throw new UsageException("--logits-output-dir holds the files of --save-logits; add --save-logits");
            }
            if (o.embedding) {
                throw new UsageException("--embedding saves embeddings with --save-logits; add --save-logits");
            }
        }
        if (o.embdNormalizeSet && !o.embedding) {
            throw new UsageException("--embd-normalize normalizes saved embeddings; add --embedding");
        }
        return o;
    }

    /** The prompt file's text with one trailing newline dropped, as llama.cpp's {@code read_file} does. */
    private static String readPromptFile(String flag, String name) throws UsageException {
        Path file = Paths.get(name);
        if (!Files.isReadable(file)) {
            throw new UsageException(flag + " " + file + " is not a readable file");
        }
        try {
            String text = new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
            return text.endsWith("\n") ? text.substring(0, text.length() - 1) : text;
        } catch (IOException e) {
            throw new UsageException(flag + " " + file + " cannot be read: " + e.getMessage());
        }
    }

    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    private static int positive(String flag, String v) throws UsageException {
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

    private static int normalization(String flag, String v) throws UsageException {
        try {
            int n = Integer.parseInt(v.trim());
            if (n < EmbeddingNorm.NONE) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs -1 (none) or a non-negative norm, not '" + v + "'");
        }
    }
}
