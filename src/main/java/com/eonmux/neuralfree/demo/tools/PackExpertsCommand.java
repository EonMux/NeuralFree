package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.packed.PackedExpertEntry;
import com.eonmux.neuralfree.gguf.packed.PackedExpertView;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * {@code nftool pack-experts}: rewrites a model's routed-expert tensors into Avalon's packed-expert
 * layout, one I8 {@code blk.N.ffn_exps_packed} tensor per MoE layer described by its
 * {@code avalon.experts_packed.blk.N} metadata, byte for byte as Avalon's {@code pack-experts.py}
 * writes it, so NeuralFree and Avalon load the same files. The model is resolved from its path
 * exactly as the demos resolve one ({@link ModelSources#open}, which routes a GGUF file, a split's
 * first file and the rest) and packed through {@link PackedExpertView#packedViewIfMoe}; {@link GgufWriter} then
 * streams the bytes, so every GGUF rule, the shard naming and the atomic {@code .tmp} twin of an
 * export apply unchanged. Positionals first ({@code nftool pack-experts <model> <out.gguf>}), then
 * the flags {@code --overwrite}, {@code --split-max-tensors <n>} and {@code --split-max-size <n>}
 * (bytes, or with a {@code k}/{@code m}/{@code g} suffix). A model without routed experts, and a
 * layer with only some of the three expert tensors, is refused with the layer named. Flags parse in
 * the {@code ExportCommand} style; {@code -Dneuralfree.*} properties still apply to the model's
 * open.
 */
final class PackExpertsCommand {

    private static final String USAGE =
            "nftool pack-experts <model> <out.gguf> [options]\n"
            + "options:\n"
            + "  --overwrite                           replace the output (or every derived shard)\n"
            + "  --split-max-tensors <n>               largest tensor count of one shard\n"
            + "  --split-max-size <n>                  largest shard size in bytes, e.g. 2000m\n"
            + "  --help, -h                            this text\n";

    private PackExpertsCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code pack-experts}. */
    static void run(String[] args) {
        try {
            List<Path> written = pack(args);
            if (written != null) {
                StringBuilder names = new StringBuilder();
                for (int i = 0; i < written.size(); i++) {
                    if (i > 0) {
                        names.append(", ");
                    }
                    names.append(written.get(i));
                }
                System.out.println("nftool pack-experts: wrote " + names);
            }
        } catch (ExportCommand.UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Parses the flags and packs; {@code null} for {@code --help}, which prints the usage and
     * returns nothing written. Package-private for the command's tests.
     */
    static List<Path> pack(String[] args) throws IOException, ExportCommand.UsageException {
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
                case "--overwrite":
                    options = options.overwrite(true);
                    break;
                case "--split-max-tensors":
                    options = options.splitMaxTensors(longValue(a, value(a, args, i++)));
                    break;
                case "--split-max-size":
                    options = options.splitMaxBytes(splitSize(a, value(a, args, i++)));
                    break;
                default:
                    if (a.startsWith("-")) {
                        throw new ExportCommand.UsageException("unknown option " + a
                                + " (run with --help for the options)");
                    }
                    if (model == null) {
                        model = Paths.get(a);
                    } else if (out == null) {
                        out = Paths.get(a);
                    } else {
                        throw new ExportCommand.UsageException("too many file arguments; '" + a
                                + "' follows " + model + " and " + out);
                    }
            }
        }
        if (model == null) {
            throw new ExportCommand.UsageException("pack-experts needs a model path and an output path");
        }
        if (out == null) {
            throw new ExportCommand.UsageException("pack-experts needs an output path after " + model);
        }
        try (ModelSource src = ModelSources.open(model.toString(), ModelSourceOptions.fromSystemProperties())) {
            ModelSource packed = PackedExpertView.packedViewIfMoe(src);
            if (packed == src) {
                throw new IOException("model " + src.getFilePath() + " has no routed experts"
                        + " (no blk.N.ffn_{up,gate,down}_exps.weight tensors); nothing to pack");
            }
            List<Path> files = GgufWriter.write(packed, out, options);
            packedLayers(packed);
            return files;
        }
    }

    /** Prints one line per packed layer: its entry layout, as {@code pack-experts.py} reports it. */
    private static void packedLayers(ModelSource packed) {
        if (!(packed instanceof PackedExpertView)) {
            return;
        }
        for (Map.Entry<Integer, PackedExpertEntry> e : ((PackedExpertView) packed).entries().entrySet()) {
            System.out.println("blk." + e.getKey() + ": " + e.getValue());
        }
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws ExportCommand.UsageException {
        if (index + 1 >= args.length) {
            throw new ExportCommand.UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    /** A non-negative count of the flag, as llama.cpp parses one. */
    private static long longValue(String flag, String v) throws ExportCommand.UsageException {
        try {
            long n = Long.parseLong(v.trim());
            if (n < 0) {
                throw new NumberFormatException();
            }
            return n;
        } catch (NumberFormatException e) {
            throw new ExportCommand.UsageException(flag + " needs a non-negative integer, not '" + v + "'");
        }
    }

    /**
     * A byte size: a non-negative integer with an optional {@code k}/{@code m}/{@code g} suffix (a power
     * of 1024 like llama.cpp's {@code --split-max-size} N, e.g. {@code 2000m}), never one that overflows;
     * {@code 0} keeps the no-limit meaning.
     */
    private static long splitSize(String flag, String v) throws ExportCommand.UsageException {
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
            throw new ExportCommand.UsageException(flag + " needs a size in bytes, e.g. 2000000 or 2000m, not '"
                    + v + "'");
        }
        if (n < 0 || factor != 1 && n > Long.MAX_VALUE / factor) {
            throw new ExportCommand.UsageException(flag + " '" + v + "' overflows a byte size");
        }
        return n * factor;
    }
}
