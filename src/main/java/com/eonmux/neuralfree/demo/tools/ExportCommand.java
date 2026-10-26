package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Locale;

/**
 * {@code nftool export}: exports any {@link ModelSource}, resolved from its path exactly as the demos
 * resolve a model ({@link ModelSources#open}, which routes a GGUF file, a split's first file, a Hugging
 * Face checkpoint and the rest), to a GGUF file through
 * {@link GgufWriter#write(ModelSource, Path, GgufWriter.Options)} (PARITY.md WP TL-1). Positionals first
 * ({@code nftool export <model> <out.gguf>}), then the flags: {@code --mode native|compat|compat-exact},
 * {@code --overwrite}, {@code --vocab-only}, {@code --split-max-tensors <n>} and
 * {@code --split-max-size <n>} (bytes, or with a {@code k}/{@code m}/{@code g} suffix as llama.cpp's
 * split tool takes it). With a split limit the written files are the llama.cpp shard names derived from
 * {@code out} ({@link com.eonmux.neuralfree.gguf.GgufTensorSink#shards}) and the written paths are
 * printed. Flags parse in the {@code LLMDemoOptions} style; {@code -Dneuralfree.*} properties still
 * apply to the model's open.
 */
final class ExportCommand {

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    private static final String USAGE =
            "nftool export <model> <out.gguf> [options]\n"
            + "options:\n"
            + "  --mode <native|compat|compat-exact>   export mode, default native\n"
            + "  --overwrite                           replace the output (or every derived shard)\n"
            + "  --vocab-only                          metadata and tokenizer keys, no tensors\n"
            + "  --split-max-tensors <n>               largest tensor count of one shard\n"
            + "  --split-max-size <n>                  largest shard size in bytes, e.g. 2000m\n"
            + "  --help, -h                            this text\n";

    private ExportCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code export}. */
    static void run(String[] args) {
        try {
            List<Path> written = export(args);
            if (written != null) {
                StringBuilder names = new StringBuilder();
                for (int i = 0; i < written.size(); i++) {
                    if (i > 0) {
                        names.append(", ");
                    }
                    names.append(written.get(i));
                }
                System.out.println("nftool export: wrote " + names);
            }
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
     * Parses the flags and exports; {@code null} for {@code --help}, which prints the usage and returns
     * nothing written.
     */
    private static List<Path> export(String[] args) throws IOException, UsageException {
        Path model = null;
        Path out = null;
        GgufWriter.Options options = new GgufWriter.Options(GgufWriter.Mode.NATIVE);
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                System.out.println(USAGE);
                return null;
            }
            switch (a) {
                case "--mode": {
                    String name = value(a, args, i++).trim().toUpperCase(Locale.ROOT).replace('-', '_');
                    try {
                        options = options.mode(GgufWriter.Mode.valueOf(name));
                    } catch (IllegalArgumentException e) {
                        throw new UsageException(a + " is invalid; use native, compat or compat-exact");
                    }
                    break;
                }
                case "--overwrite":
                    options = options.overwrite(true);
                    break;
                case "--vocab-only":
                    options = options.vocabOnly(true);
                    break;
                case "--split-max-tensors":
                    options = options.splitMaxTensors(longValue(a, value(a, args, i++)));
                    break;
                case "--split-max-size":
                    options = options.splitMaxBytes(splitSize(a, value(a, args, i++)));
                    break;
                default:
                    if (a.startsWith("-")) {
                        throw new UsageException("unknown option " + a + " (run with --help for the options)");
                    }
                    if (model == null) {
                        model = Paths.get(a);
                    } else if (out == null) {
                        out = Paths.get(a);
                    } else {
                        throw new UsageException("too many file arguments; '" + a
                                + "' follows " + model + " and " + out);
                    }
            }
        }
        if (model == null) {
            throw new UsageException("export needs a model path and an output path");
        }
        if (out == null) {
            throw new UsageException("export needs an output path after " + model);
        }
        try (ModelSource src = ModelSources.open(model.toString(),
                ModelSourceOptions.fromSystemProperties())) {
            return GgufWriter.write(src, out, options);
        }
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    /** A non-negative count of the flag, as llama.cpp parses one. */
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

    /**
     * A byte size: a non-negative integer with an optional {@code k}/{@code m}/{@code g} suffix (a power
     * of 1024 like llama.cpp's {@code --split-max-size} N, e.g. {@code 2000m}), never one that overflows;
     * {@code 0} keeps the no-limit meaning.
     */
    private static long splitSize(String flag, String v) throws UsageException {
        String s = v.trim().toLowerCase(Locale.ROOT);
        long factor = 1;
        if (!s.isEmpty()) {
            char suffix = s.charAt(s.length() - 1);
            if (suffix == 'k') {
                factor = 1L << 10;
            } else if (suffix == 'm') {
                factor = 1L << 20;
            } else if (suffix == 'g') {
                factor = 1L << 30;
            }
            if (factor != 1) {
                s = s.substring(0, s.length() - 1);
            }
        }
        long n;
        try {
            n = Long.parseLong(s.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a size in bytes, e.g. 2000000 or 2000m, not '" + v + "'");
        }
        if (n < 0 || factor != 1 && n > Long.MAX_VALUE / factor) {
            throw new UsageException(flag + " '" + v + "' overflows a byte size");
        }
        return n * factor;
    }
}
