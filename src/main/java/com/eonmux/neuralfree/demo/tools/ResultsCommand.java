package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.GgufEncoder;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.ModelMemory;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Locale;

/**
 * {@code nftool results}: llama.cpp's results tool (tools/results/results.cpp). One run tokenizes the prompt,
 * evaluates every token and writes the logits of every position to a GGUF file whose {@code path_model} key names the
 * model and whose {@code tokens} (I32) and {@code logits} (F32, token-major as llama.cpp stores them) tensors carry
 * the run. {@code --check} recomputes the run and compares it against a stored file by normalized mean squared error,
 * {@code mse(stored, computed) / mse(stored, 0)}, failing above {@value #NMSE_TOLERANCE}.
 *
 * <p>The file layout is llama.cpp's byte for byte, so each tool reads the other's file. A check refuses a file whose
 * {@code path_model} names another model, whose tensors have another shape or type, or whose tokens differ before
 * it compares any logit.</p>
 */
final class ResultsCommand {

    /** The largest NMSE of a passing check (llama.cpp's tolerance). */
    static final double NMSE_TOLERANCE = 1e-6;

    /** The GGUF key naming the model a results file belongs to. */
    static final String KEY_PATH_MODEL = "path_model";

    /** The tensor names of a results file. */
    static final String TOKENS_TENSOR = "tokens";
    static final String LOGITS_TENSOR = "logits";

    /** The data alignment of a results file (llama.cpp's GGUF default). */
    static final int ALIGNMENT = 32;

    private static final String USAGE =
            "nftool results <model> [options]\n"
            + "options (llama.cpp results names):\n"
            + "  --model, -m <path>               the model to run (a GGUF file or Hugging Face checkpoint)\n"
            + "  --prompt, -p <text>              the prompt to evaluate\n"
            + "  --file, -f <fname>               read the prompt from a file (one trailing newline dropped)\n"
            + "  --output, -o <fname>             the results file to write, or to check against\n"
            + "  --check                          compare the run against the results file instead of writing it\n"
            + "  --ctx-size, -c <n>               the context the prompt must fit\n"
            + "  --batch-size, -b <n>             forward ubatch of the engine\n"
            + "  --help, -h                       this text\n"
            + "The metric is the normalized mean squared error (NMSE), with a tolerance of " + NMSE_TOLERANCE + ".\n";

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
        Path output;
        boolean check;
        int ctxSize = -1;
        int batchSize = -1;
    }

    private ResultsCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code results}. */
    static void run(String[] args) {
        try {
            int code = dispatch(args, EngineConfig.fromSystemProperties(), System.out, System.err);
            if (code != 0) {
                System.exit(code);
            }
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | UnsupportedOperationException | IllegalArgumentException | IllegalStateException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Parses the arguments, runs the model and writes or checks the results file.
     *
     * @return 0 when the run was written or the check passed, 1 when the check failed
     */
    static int dispatch(String[] args, EngineConfig base, Appendable out, Appendable err)
            throws IOException, UsageException {
        Options o = parse(args);
        if (o == null) {
            return 0;
        }
        try (EngineModel model = LLMModelFactory.open(o.model.toString(), config(o, base))) {
            int[] tokens = tokenize(model, o.prompt);
            float[] logits = logits(model, tokens, o.ctxSize);
            if (o.check) {
                return check(model, tokens, logits, o, out, err);
            }
            write(o.output, o.model.toString(), tokens, logits, model.vocabSize());
            err.append("nftool results: writing results to ").append(o.output.toString()).append('\n');
            out.append("OK\n");
            return 0;
        }
    }

    /** Tokenizes the prompt with special tokens, as llama.cpp's {@code common_tokenize(..., true)}. */
    static int[] tokenize(EngineModel model, String prompt) throws IOException {
        int[] tokens = model.tokenizer().encode(prompt, true);
        if (tokens.length == 0) {
            throw new IOException("the prompt tokenizes to no tokens; give --prompt or --file");
        }
        return tokens;
    }

    /**
     * The logits of every token of the run, token-major: one vocabulary row per token, in prompt order
     * (llama.cpp's {@code llama_get_logits_ith} over every position).
     */
    static float[] logits(EngineModel model, int[] tokens, int ctxSize) throws IOException {
        ForwardBackend backend = model.forwardBackend();
        ModelMemory memory = backend.memory();
        if (memory != null && tokens.length > memory.plan().nCtx()) {
            throw new IOException("the prompt tokenizes to " + tokens.length + " tokens, more than the context of "
                    + memory.plan().nCtx() + "; raise it with --ctx-size");
        }
        int[] rows = new int[tokens.length];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = i;
        }
        backend.reset();
        float[][] logits = backend.forwardRows(tokens, 0, rows);
        int nVocab = model.vocabSize();
        long total = (long) tokens.length * nVocab;
        if (total > Integer.MAX_VALUE - 8L) {
            throw new IOException("the run holds " + tokens.length + " x " + nVocab + " logits, more than one"
                    + " Java array; evaluate a shorter prompt");
        }
        float[] flat = new float[(int) total];
        for (int i = 0; i < tokens.length; i++) {
            if (logits[i].length != nVocab) {
                throw new IOException("the model returned " + logits[i].length + " logits for token " + i
                        + ", not its vocabulary of " + nVocab);
            }
            System.arraycopy(logits[i], 0, flat, i * nVocab, nVocab);
        }
        return flat;
    }

    /** Compares the run against the stored file: its model, tokens and per-token logits, then the NMSE. */
    private static int check(EngineModel model, int[] tokens, float[] logits, Options o, Appendable out,
                             Appendable err) throws IOException {
        try (GGUFFileParser file = new GGUFFileParser(o.output.toString())) {
            String storedModel = file.getString(KEY_PATH_MODEL);
            if (storedModel == null) {
                throw new IOException(o.output + " has no '" + KEY_PATH_MODEL + "' key; it is not a results file");
            }
            if (!storedModel.equals(o.model.toString())) {
                throw new IOException(o.output + " holds the results of '" + storedModel + "', not of '"
                        + o.model + "'");
            }
            GGUFTensorInfo tokensInfo = requireTensor(file, TOKENS_TENSOR, GGMLType.I32,
                    new long[]{tokens.length});
            GGUFTensorInfo logitsInfo = requireTensor(file, LOGITS_TENSOR, GGMLType.F32,
                    new long[]{tokens.length, model.vocabSize()});
            int[] storedTokens = readInts(file, tokensInfo);
            for (int i = 0; i < tokens.length; i++) {
                if (storedTokens[i] != tokens[i]) {
                    throw new IOException(o.output + " holds token " + storedTokens[i] + " at position " + i
                            + ", but the prompt tokenizes to " + tokens[i]);
                }
            }
            float[] storedLogits = readFloats(file, logitsInfo);
            double value = nmse(storedLogits, logits);
            err.append("nftool results: NMSE=").append(String.format(Locale.ROOT, "%.3e", value)).append('\n');
            if (value > NMSE_TOLERANCE) {
                out.append("FAIL\n");
                return 1;
            }
            out.append("OK\n");
            return 0;
        }
    }

    private static GGUFTensorInfo requireTensor(GGUFFileParser file, String name, GGMLType type, long[] dims)
            throws IOException {
        GGUFTensorInfo info = file.getTensor(name);
        if (info == null) {
            throw new IOException(file.getFilePath() + " has no '" + name + "' tensor; it is not a results file");
        }
        if (info.type() != type) {
            throw new IOException(file.getFilePath() + ": tensor '" + name + "' is " + info.type() + ", not " + type);
        }
        if (!Arrays.equals(info.dims(), dims)) {
            throw new IOException(file.getFilePath() + ": tensor '" + name + "' has shape "
                    + Arrays.toString(info.dims()) + ", not " + Arrays.toString(dims));
        }
        return info;
    }

    private static int[] readInts(GGUFFileParser file, GGUFTensorInfo info) throws IOException {
        ByteBuffer bytes = ByteBuffer.wrap(file.readTensorData(info)).order(ByteOrder.LITTLE_ENDIAN);
        int[] values = new int[(int) info.numElements()];
        for (int i = 0; i < values.length; i++) {
            values[i] = bytes.getInt();
        }
        return values;
    }

    private static float[] readFloats(GGUFFileParser file, GGUFTensorInfo info) throws IOException {
        ByteBuffer bytes = ByteBuffer.wrap(file.readTensorData(info)).order(ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[(int) info.numElements()];
        for (int i = 0; i < values.length; i++) {
            values[i] = bytes.getFloat();
        }
        return values;
    }

    /**
     * The normalized mean squared error {@code mse(a, b) / mse(a, 0)}, llama.cpp's metric: {@code a} is the stored
     * run, {@code b} the computed one. A stored run of all zeros divides by zero and yields NaN.
     */
    static double nmse(float[] a, float[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("NMSE of " + a.length + " and " + b.length + " values");
        }
        double mseAB = 0.0;
        double mseA0 = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = (double) a[i] - b[i];
            mseAB += d * d;
            mseA0 += (double) a[i] * a[i];
        }
        return mseAB / mseA0;
    }

    /** Writes the results file: the model's path, then the tokens (I32) and the logits (F32). */
    static void write(Path out, String modelPath, int[] tokens, float[] logits, int nVocab) throws IOException {
        Path target = out.toAbsolutePath();
        Path parent = target.getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        Path tmp = target.resolveSibling(target.getFileName() + ".tmp");
        GGUFTensorInfo tokensInfo = new GGUFTensorInfo(TOKENS_TENSOR, new long[]{tokens.length}, GGMLType.I32, 0);
        long logitsOffset = alignUp(tokensInfo.sizeInBytes(), ALIGNMENT);
        GGUFTensorInfo logitsInfo = new GGUFTensorInfo(LOGITS_TENSOR, new long[]{tokens.length, nVocab},
                GGMLType.F32, logitsOffset);
        GGUFTensorInfo.validate("results file '" + target + "'", tokensInfo);
        GGUFTensorInfo.validate("results file '" + target + "'", logitsInfo);
        try (FileChannel channel = FileChannel.open(tmp, StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            GgufEncoder enc = new GgufEncoder(channel, target.toString());
            enc.writeHeader(channel, 2, 1);
            enc.writeKv(KEY_PATH_MODEL, GGUFValueType.STRING, null, modelPath);
            enc.writeTensorInfo(tokensInfo, 0);
            enc.writeTensorInfo(logitsInfo, logitsOffset);
            enc.align(ALIGNMENT);
            enc.writeData(intsLe(tokens));
            enc.align(ALIGNMENT);
            enc.writeData(floatsLe(logits));
            enc.align(ALIGNMENT);
            enc.flush();
        } catch (IOException | RuntimeException e) {
            Files.deleteIfExists(tmp);
            throw e;
        }
        Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
    }

    /** {@code value} rounded up to the next multiple of {@code alignment}. */
    private static long alignUp(long value, int alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }

    private static byte[] intsLe(int[] values) {        ByteBuffer bytes = ByteBuffer.allocate(values.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int v : values) {
            bytes.putInt(v);
        }
        return bytes.array();
    }

    private static byte[] floatsLe(float[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : values) {
            bytes.putFloat(v);
        }
        return bytes.array();
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
     * Parses and validates the flags (llama.cpp's results option names); {@code null} for {@code --help}.
     *
     * @throws UsageException naming the flag for every unsupported or missing option
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
                case "--output":
                case "--output-file":
                case "-o":
                    o.output = Paths.get(value(a, args, i++));
                    break;
                case "--check":
                    o.check = true;
                    break;
                case "--ctx-size":
                case "-c":
                    o.ctxSize = positive(a, value(a, args, i++));
                    break;
                case "--batch-size":
                case "-b":
                    o.batchSize = positive(a, value(a, args, i++));
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
            throw new UsageException("results needs a model path");
        }
        if (o.output == null) {
            throw new UsageException("results needs --output <file>: the file to write, or to check against");
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
}
