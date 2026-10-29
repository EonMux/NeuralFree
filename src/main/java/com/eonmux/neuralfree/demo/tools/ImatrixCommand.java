package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.ImatrixCollector;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.engine.ContextFullException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.quant.write.ImatrixFile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * {@code nftool imatrix}: the activation-matrix collection of the quantize pipeline (PARITY.md P5, llama.cpp
 * {@code tools/imatrix/imatrix.cpp}). It opens the model on the CPU engine — the collection taps the CPU matmul
 * inputs through {@link CpuMatVec#observeGlobally} and an {@link ImatrixCollector} — prefills every non-empty line
 * of every {@code --file} from position 0 (one prefill per line, cache reset between lines, like llama.cpp's
 * per-chunk decode windows reset when the text restarts), and writes the statistics through
 * {@link ImatrixFile#writeGguf}. The entry key of every tensor is its stored GGUF name, the names the quantize
 * tools read back.
 *
 * <p>The chunk accounting follows llama.cpp: {@code --chunk-size} is the tokens-per-chunk value of the collection
 * and {@code imatrix.chunk_count} counts the chunks of {@code ceil(tokens / chunk-size)} over the tokenized lines;
 * the summed statistics themselves are per activation row and identical whatever the chunking.</p>
 */
final class ImatrixCommand {

    /** Default GGUF output path of llama.cpp ({@code params.out_file}). */
    static final String DEFAULT_OUT = "imatrix.gguf";

    /** llama.cpp's default chunk size ({@code params.n_tokens}). */
    static final int DEFAULT_CHUNK_SIZE = 512;

    private static final String USAGE =
            "nftool imatrix <model> [options]\n"
            + "options (llama.cpp imatrix names):\n"
            + "  --file <fname>                   calibration text, repeatable; at least one is required\n"
            + "  --out-file <fname>               output GGUF imatrix, default " + DEFAULT_OUT + "\n"
            + "  --chunk-size <n>                 tokens per collection chunk, default " + DEFAULT_CHUNK_SIZE + "\n"
            + "  --help, -h                       this text\n"
            + "The model runs on the CPU engine: the collector taps the CPU matmul inputs. The output is replaced\n"
            + "when it exists, as llama.cpp saves.\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
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
     * Parses the flags, collects over the calibration files and writes the imatrix GGUF.
     *
     * @throws UsageException for an invalid command line, a missing calibration file or a model without a decoder
     *                        prefill loop
     * @throws IOException     for an unreadable model, file or output
     */
    private static void collect(String[] args) throws IOException, UsageException {
        Path model = null;
        Path out = null;
        List<Path> files = new ArrayList<>();
        int chunkSize = DEFAULT_CHUNK_SIZE;
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                System.out.println(USAGE);
                return;
            }
            switch (a) {
                case "--file":
                    files.add(Paths.get(value(a, args, i++)));
                    break;
                case "--out-file":
                    out = Paths.get(value(a, args, i++));
                    break;
                case "--chunk-size":
                    chunkSize = intValue(a, value(a, args, i++));
                    break;
                default:
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
            throw new UsageException("imatrix needs a model path");
        }
        if (files.isEmpty()) {
            throw new UsageException("imatrix needs at least one --file of calibration text");
        }
        for (Path f : files) {
            if (!Files.isReadable(f)) {
                throw new UsageException("--file " + f + " is not a readable file");
            }
        }

        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .build();
        try (GGUFModel m = open(model.toString(), cfg)) {
            SupportClass support = m.supportClass();
            if (m.forwardBackend().memory() == null || !hasTokenLoop(support)) {
                throw new UsageException("the collector runs the decoder matmuls of a model with memory; "
                        + m.architecture() + " (" + support + ") has none");
            }
            CpuForwardContext ctx = m.forwardContext();
            if (ctx == null) {
                throw new UsageException(m.architecture() + " has no host forward context to collect over");
            }
            CpuWeights weights = ctx.weights();
            ImatrixCollector collector = new ImatrixCollector(w -> weights.describe(w).name());
            GGUFTokenizer tok = m.tokenizer();
            long chunks = 0;
            long lines = 0;
            CpuMatVec.observeGlobally(collector);
            try {
                for (Path f : files) {
                    List<String> text = Files.readAllLines(f, StandardCharsets.UTF_8);
                    for (int li = 0; li < text.size(); li++) {
                        String line = text.get(li);
                        if (line.isEmpty()) {
                            continue;
                        }
                        int[] tokens = tok.encode(line);
                        if (tokens.length == 0) {
                            continue;
                        }
                        try {
                            m.forwardBackend().prefill(tokens, 0);
                        } catch (ContextFullException e) {
                            throw new IOException(f + " line " + (li + 1) + " holds " + tokens.length
                                    + " tokens, more than the model's context fits: " + e.getMessage(), e);
                        }
                        m.forwardBackend().reset();
                        lines++;
                        chunks += (tokens.length + chunkSize - 1) / chunkSize;
                    }
                }
            } finally {
                CpuMatVec.observeGlobally(null);
            }
            if (collector.tensorCount() == 0) {
                throw new IOException("no matmul activations were collected; the model ran no matmul");
            }
            List<String> datasets = new ArrayList<>();
            for (Path f : files) {
                datasets.add(f.toString());
            }
            Path written = ImatrixFile.writeGguf(out != null ? out : Paths.get(DEFAULT_OUT),
                    collector.entries(), datasets, (int) Math.min(chunks, Integer.MAX_VALUE), chunkSize);
            System.out.println("nftool imatrix: wrote " + written + " from " + lines + " lines, "
                    + collector.tensorCount() + " tensors, " + chunks + " chunks of " + chunkSize);
        }
    }

    /** Opens the model through the factory, keeping the GPU refusal the config's CPU backend names. */
    private static GGUFModel open(String path, EngineConfig cfg) throws IOException {
        com.eonmux.neuralfree.gguf.engine.EngineModel opened = LLMModelFactory.open(path, cfg);
        if (!(opened instanceof GGUFModel)) {
            throw new IOException("the imatrix collection runs on the CPU engine, but " + path
                    + " opened as " + opened.getClass().getSimpleName());
        }
        return (GGUFModel) opened;
    }

    /** Whether models of {@code support} run a decoder-only token loop (LLMModelFactory's set). */
    private static boolean hasTokenLoop(SupportClass support) {
        switch (support) {
            case EMBED:
            case CLS:
            case NA:
            case DIFF:
            case ENC_DEC:
            case DRAFT:
                return false;
            default:
                return true;
        }
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    /** A positive count of the flag, as llama.cpp parses its chunk size. */
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
}
