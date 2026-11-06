package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.write.QuantizeFtype;
import com.eonmux.neuralfree.gguf.quant.write.QuantizeParams;
import com.eonmux.neuralfree.gguf.quant.write.QuantizeTool;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * {@code nftool quantize}: the offline GGUF→GGUF requantization of llama.cpp's {@code llama-quantize}
 * (tools/quantize/quantize.cpp). Flags first ({@code --allow-requantize},
 * {@code --leave-output-tensor}, {@code --pure}, {@code --imatrix}, {@code --include-weights},
 * {@code --exclude-weights}, {@code --output-tensor-type}, {@code --token-embedding-type},
 * {@code --tensor-type}, {@code --tensor-type-file}, {@code --prune-layers}, {@code --keep-split},
 * {@code --override-kv}, {@code --dry-run}, {@code --max-buffer-size}), then the positionals
 * {@code model [out] type [nthreads]} — the output name defaults to {@code ggml-model-<type>.gguf}
 * beside the input, as the reference derives it, and the recipes are the {@link QuantizeFtype} names.
 */
final class QuantizeCommand {

    private static final String USAGE =
            "nftool quantize [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix file]\n"
            + "       [--include-weights name] [--exclude-weights name] [--output-tensor-type type]\n"
            + "       [--token-embedding-type type] [--tensor-type name=type] [--tensor-type-file file]\n"
            + "       [--prune-layers L0,L1,...] [--keep-split] [--override-kv key=type:value] [--dry-run]\n"
            + "       [--max-buffer-size MiB]\n"
            + "       model.gguf [model-quant.gguf] type [nthreads]\n\n"
            + "  type: one of " + QuantizeFtype.names() + "\n"
            + "  COPY copies every tensor unchanged; the default output name is ggml-model-<type>.gguf\n"
            + "  beside the input. --dry-run prints the sizes and writes nothing.\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    private QuantizeCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code quantize}. */
    static void run(String[] args) {
        try {
            runParsed(parse(args));
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /** The parsed command line. */
    static final class Parsed {
        Path input;
        Path out;
        QuantizeParams params;
    }

    /** Runs the parsed command line and prints what was written. */
    private static void runParsed(Parsed p) throws IOException {
        QuantizeTool.Result result = QuantizeTool.run(p.input, p.out, p.params);
        if (result.files.size() == 1) {
            System.out.println("nftool quantize: wrote " + result.files.get(0));
        } else if (result.files.size() > 1) {
            StringBuilder names = new StringBuilder();
            for (int i = 0; i < result.files.size(); i++) {
                if (i > 0) {
                    names.append(", ");
                }
                names.append(result.files.get(i));
            }
            System.out.println("nftool quantize: wrote " + result.files.size() + " shards: " + names);
        }
    }

    /**
     * Parses the flags then the positionals, in llama.cpp's order: flags until the first non-{@code --}
     * argument, the model, an optional output path, the type (a recipe name or a numeric
     * {@code LLAMA_FTYPE_*} id), an optional thread count.
     *
     * @throws UsageException naming the flag or the positional that went wrong
     */
    static Parsed parse(String[] args) throws UsageException {
        QuantizeParams.Builder params = QuantizeParams.builder();
        List<String> includes = new ArrayList<>();
        List<String> excludes = new ArrayList<>();
        List<QuantizeParams.TensorType> tensorTypes = new ArrayList<>();
        boolean dryRun = false;
        boolean outGiven = false;
        int i = 0;
        for (; i < args.length && args[i].startsWith("--"); i++) {
            String a = args[i];
            switch (a) {
                case "--allow-requantize":
                    params.allowRequantize(true);
                    break;
                case "--leave-output-tensor":
                    params.quantizeOutputTensor(false);
                    break;
                case "--pure":
                    params.pure(true);
                    break;
                case "--keep-split":
                    params.keepSplit(true);
                    break;
                case "--dry-run":
                    dryRun = true;
                    params.dryRun(true);
                    break;
                case "--imatrix":
                    params.imatrixFile(Paths.get(value(a, args, i++)));
                    break;
                case "--include-weights":
                    includes.add(value(a, args, i++));
                    break;
                case "--exclude-weights":
                    excludes.add(value(a, args, i++));
                    break;
                case "--output-tensor-type":
                    params.outputTensorType(ggmlType(a, value(a, args, i++)));
                    break;
                case "--token-embedding-type":
                    params.tokenEmbeddingType(ggmlType(a, value(a, args, i++)));
                    break;
                case "--tensor-type":
                    tensorTypes.add(tensorType(value(a, args, i++)));
                    break;
                case "--tensor-type-file":
                    tensorTypes.addAll(tensorTypeFile(value(a, args, i++)));
                    break;
                case "--prune-layers":
                    try {
                        params.pruneLayers(value(a, args, i++));
                    } catch (IllegalArgumentException e) {
                        throw new UsageException(e.getMessage());
                    }
                    break;
                case "--override-kv":
                    try {
                        params.kvOverride(value(a, args, i++));
                    } catch (IllegalArgumentException e) {
                        throw new UsageException(e.getMessage());
                    }
                    break;
                case "--max-buffer-size":
                    // rows are quantized one at a time, so the working set of one tensor stays a row
                    // wide, far below every positive buffer bound; the value is validated like the
                    // reference validates it
                    intValue(a, value(a, args, i++));
                    break;
                case "--help":
                case "-h":
                    System.out.print(USAGE);
                    System.exit(0);
                    break;
                default:
                    throw new UsageException("unknown option " + a);
            }
        }
        if (args.length - i < 2) {
            throw new UsageException("quantize needs a model path and a type");
        }
        if (!includes.isEmpty() && !excludes.isEmpty()) {
            throw new UsageException("--include-weights and --exclude-weights cannot be used together");
        }
        params.includeWeights(includes);
        params.excludeWeights(excludes);
        params.tensorTypes(tensorTypes);

        Parsed p = new Parsed();
        p.input = Paths.get(args[i++]);
        String first = args[i++];
        QuantizeFtype ftype;
        try {
            ftype = QuantizeFtype.parse(first);
        } catch (IllegalArgumentException e) {
            // the argument is not a type, so it is the output path: <input> <output> <type>
            outGiven = true;
            p.out = Paths.get(first);
            if (i >= args.length) {
                throw new UsageException("missing ftype after " + p.out);
            }
            try {
                ftype = QuantizeFtype.parse(args[i++]);
            } catch (IllegalArgumentException e2) {
                throw new UsageException("invalid ftype '" + args[i - 1] + "'");
            }
        }
        params.ftype(ftype);
        if (i < args.length) {
            params.nthread(intValue("nthreads", args[i++]));
        }
        if (i < args.length) {
            throw new UsageException("too many arguments; '" + args[i] + "' follows the thread count");
        }
        try {
            p.params = params.build();
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
        if (!dryRun && !outGiven) {
            p.out = defaultOut(p.input, ftype);
        }
        return p;
    }

    /**
     * One {@code --tensor-type name=type} entry: the name lower-cases into a regex pattern, as the
     * reference's {@code parse_tensor_type} prepares it.
     */
    private static QuantizeParams.TensorType tensorType(String entry) throws UsageException {
        int eq = entry.indexOf('=');
        if (eq <= 0) {
            throw new UsageException("malformed tensor type '" + entry + "': use <tensor name>=<ggml type>");
        }
        if (eq == entry.length() - 1) {
            throw new UsageException("malformed tensor type '" + entry + "': no ggml type after '='");
        }
        String name = entry.substring(0, eq).toLowerCase(Locale.ROOT);
        GGMLType type = ggmlType("--tensor-type", entry.substring(eq + 1));
        Pattern pattern;
        try {
            pattern = Pattern.compile(name);
        } catch (PatternSyntaxException e) {
            throw new UsageException("tensor type pattern '" + name + "' does not compile: " + e.getMessage());
        }
        return new QuantizeParams.TensorType(pattern, name, type);
    }

    /** The {@code --tensor-type-file} entries: whitespace-separated {@code name=type} pairs. */
    private static List<QuantizeParams.TensorType> tensorTypeFile(String file) throws UsageException {
        Path path = Paths.get(file);
        if (!Files.isReadable(path)) {
            throw new UsageException("--tensor-type-file " + file + " is not a readable file");
        }
        String text;
        try {
            text = Files.readString(path);
        } catch (IOException e) {
            throw new UsageException("--tensor-type-file " + file + " cannot be read: " + e.getMessage());
        }
        List<QuantizeParams.TensorType> types = new ArrayList<>();
        for (String entry : text.split("\\s+")) {
            if (!entry.isEmpty()) {
                types.add(tensorType(entry));
            }
        }
        return types;
    }

    /** The reference's default output name: {@code ggml-model-<type>.gguf} beside the input. */
    private static Path defaultOut(Path input, QuantizeFtype ftype) {
        Path parent = input.toAbsolutePath().getParent();
        String name = "ggml-model-" + ftype.recipeName().toLowerCase(Locale.ROOT) + ".gguf";
        return parent == null ? Path.of(name) : parent.resolve(name);
    }

    private static GGMLType ggmlType(String flag, String v) throws UsageException {
        try {
            return QuantizeParams.parseGgmlType(v);
        } catch (IllegalArgumentException e) {
            throw new UsageException(flag + ": " + e.getMessage());
        }
    }

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
}
