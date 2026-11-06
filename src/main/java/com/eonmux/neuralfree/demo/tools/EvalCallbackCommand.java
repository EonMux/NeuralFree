package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.gpu.diag.TensorTraceObserver;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code nftool eval-callback}: llama.cpp's eval-callback example (examples/eval-callback/eval-callback.cpp). The
 * prompt tokenizes with special tokens, its ids print, and the forward pass runs with a
 * {@link TensorTraceObserver} on every forward device, so each stage's tensor prints as the engine produces it —
 * the counterpart of the reference's {@code ggml_backend_sched_eval_callback} over the computation graph.
 *
 * <p>{@code --tensor-filter} patterns select the stages by label ({@code embedding}, {@code mixer-0},
 * {@code logits}, ...), {@code --abort-on-nan} stops on the first stage whose sum is NaN, and {@code --steps}
 * limits how many forward steps report. The trace reads the GPU execution stages; a model that opened on the CPU
 * backend has none and is refused by name.</p>
 */
final class EvalCallbackCommand {

    private static final String USAGE =
            "nftool eval-callback <model> [options]\n"
            + "options (llama.cpp eval-callback names):\n"
            + "  --model, -m <path>               the model to run\n"
            + "  --prompt, -p <text>              the prompt to evaluate\n"
            + "  --file, -f <fname>               read the prompt from a file (one trailing newline dropped)\n"
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
        final List<String> filters = new ArrayList<>();
        boolean abortOnNan;
        int steps;
        int ctxSize = -1;
        int batchSize = -1;
        boolean escape = true;
    }

    private EvalCallbackCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code eval-callback}. */
    static void run(String[] args) {
        try {
            Options o = parse(args);
            if (o != null) {
                dispatch(o, EngineConfig.fromSystemProperties(), System.out);
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

    /** Opens the model, prints the input tokens and traces the prompt's forward pass. */
    static void dispatch(Options o, EngineConfig base, PrintStream out) throws IOException, UsageException {
        try (EngineModel model = LLMModelFactory.open(o.model.toString(), config(o, base))) {
            int[] tokens = model.tokenizer().encode(o.prompt, true);
            if (tokens.length == 0) {
                throw new IOException("there are no input tokens to process; give --prompt or --file");
            }
            out.println("number of input tokens = " + tokens.length);
            for (int id : tokens) {
                out.println("  " + id);
            }
            ForwardTrace.trace(model, tokens, o.filters, o.abortOnNan, o.steps, out);
        }
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
     * Parses and validates the flags (llama.cpp's eval-callback option names); {@code null} for {@code --help}.
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
                case "--tensor-filter":
                    o.filters.add(value(a, args, i++));
                    break;
                case "--abort-on-nan":
                    o.abortOnNan = true;
                    break;
                case "--steps":
                    o.steps = nonNegative(a, value(a, args, i++));
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
            throw new UsageException("eval-callback needs a model path");
        }
        if (o.escape) {
            o.prompt = CvectorGenerator.processEscapes(o.prompt);
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
}
