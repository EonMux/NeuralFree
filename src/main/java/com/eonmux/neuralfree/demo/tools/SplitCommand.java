package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSources;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Locale;

/**
 * {@code nftool split}: the GGUF splitter of llama.cpp's {@code gguf-split} (tools/gguf-split/gguf-split.cpp,
 * {@code --split}), cutting one GGUF file into the llama.cpp shard files
 * {@code <out-prefix>-<shard, %05d>-of-<shardCount, %05d>.gguf} through {@link GgufWriter} in
 * {@link GgufWriter.Mode#NATIVE} — every tensor byte is copied as it is stored, shard 0 keeps the
 * metadata and every shard the {@code split.*} keys {@link GGUFFileParser#loadShards()} reads back. The
 * input may itself be the first file of a split, which the writer merges before it re-splits. Flags first
 * ({@code --split-max-tensors}, {@code --split-max-size}, {@code --no-tensor-first-split}, {@code --dry-run},
 * {@code --overwrite}), then the positionals {@code GGUF_IN GGUF_OUT}: the tensor limit defaults to
 * gguf-split's 128 tensors per shard, the two limits are mutually exclusive as they are there, and a
 * single-file result (a table smaller than the limit) writes {@code GGUF_OUT} itself. A shard bigger than
 * the byte limit is a named error, and {@code --dry-run} prints the files and their tensor counts and
 * bytes without writing anything.
 */
final class SplitCommand {

    private static final String USAGE =
            "nftool split [--split] [--split-max-tensors <n>] [--split-max-size <n>]\n"
            + "            [--no-tensor-first-split] [--dry-run] [--overwrite]\n"
            + "            GGUF_IN GGUF_OUT\n"
            + "options:\n"
            + "  --split                               the operation of this command, as gguf-split names it\n"
            + "  --split-max-tensors <n>               largest tensor count of one shard (default: 128)\n"
            + "  --split-max-size <n>                  largest shard size in bytes, e.g. 2000m; exclusive of\n"
            + "                                        --split-max-tensors\n"
            + "  --no-tensor-first-split               the first shard of a split carries no tensors\n"
            + "  --dry-run                             print the files the split writes, write nothing\n"
            + "  --overwrite                           replace an existing output (or every derived shard)\n"
            + "  --help, -h                            this text\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The command line {@link #parse} fills; read-only once parsed. */
    private static final class Request {
        Path input;
        Path out;
        long splitMaxTensors = 128;
        long splitMaxBytes;
        boolean tensorLimitGiven;
        boolean noTensorFirstSplit;
        boolean dryRun;
        boolean overwrite;
        boolean help;
    }

    private SplitCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code split}. */
    static void run(String[] args) {
        try {
            List<Path> written = split(args);
            if (written != null) {
                StringBuilder names = new StringBuilder();
                for (int i = 0; i < written.size(); i++) {
                    if (i > 0) {
                        names.append(", ");
                    }
                    names.append(written.get(i));
                }
                System.out.println("nftool split: wrote " + names);
            }
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | RuntimeException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Parses the flags and splits; {@code null} for {@code --help} and {@code --dry-run}, which print and
     * write nothing.
     */
    static List<Path> split(String[] args) throws IOException, UsageException {
        Request r = parse(args);
        if (r.help) {
            System.out.println(USAGE);
            return null;
        }
        validate(r);
        GgufWriter.Options options = new GgufWriter.Options(GgufWriter.Mode.NATIVE)
                .overwrite(r.overwrite)
                .splitMaxTensors(r.splitMaxTensors)
                .splitMaxBytes(r.splitMaxBytes)
                .smallFirstShard(r.noTensorFirstSplit);
        try (GGUFFileParser src = ModelSources.parseGguf(r.input.toString())) {
            if (r.dryRun) {
                printPlan(GgufWriter.plan(src, r.out, options));
                return null;
            }
            return GgufWriter.write(src, r.out, options);
        }
    }

    // ------------------------------------------------------------------ parsing

    /** The command line as a {@link Request}; every flag value checked as it is read. */
    private static Request parse(String[] args) throws UsageException {
        Request r = new Request();
        for (int i = 0; i < args.length && !r.help; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                r.help = true;
            } else if (a.startsWith("-")) {
                i = flag(r, a, args, i);
            } else if (r.input == null) {
                r.input = Paths.get(a);
            } else if (r.out == null) {
                r.out = Paths.get(a);
            } else {
                throw new UsageException("too many file arguments; '" + a + "' follows " + r.input + " and "
                        + r.out);
            }
        }
        return r;
    }

    /** Applies the flag {@code a} at {@code args[i]}; the index of its last argument. */
    private static int flag(Request r, String a, String[] args, int i) throws UsageException {
        switch (a) {
            case "--split":
                return i;
            case "--merge":
                throw new UsageException("--merge is not an option of nftool split; run 'nftool merge"
                        + " <first-shard.gguf> <out.gguf>'");
            case "--no-tensor-first-split":
                r.noTensorFirstSplit = true;
                return i;
            case "--dry-run":
                r.dryRun = true;
                return i;
            case "--overwrite":
                r.overwrite = true;
                return i;
            case "--split-max-tensors":
                r.splitMaxTensors = longValue(a, value(a, args, i));
                r.tensorLimitGiven = true;
                return i + 1;
            case "--split-max-size":
                r.splitMaxBytes = splitSize(a, value(a, args, i));
                return i + 1;
            default:
                throw new UsageException("unknown option " + a + " (run with --help for the options)");
        }
    }

    /** Refuses a missing positional, the two limits together, and an output that is the input. */
    private static void validate(Request r) throws UsageException {
        if (r.input == null) {
            throw new UsageException("split needs an input path and an output path");
        }
        if (r.out == null) {
            throw new UsageException("split needs an output path after " + r.input);
        }
        if (r.tensorLimitGiven && r.splitMaxBytes != 0) {
            throw new UsageException("either --split-max-tensors or --split-max-size can be specified,"
                    + " but not both");
        }
        if (sameFile(r.input, r.out)) {
            throw new UsageException("the output '" + r.out + "' is the input; name another output");
        }
    }

    /** {@code true} when both paths name the same file (the same path, or a hardlink pair). */
    private static boolean sameFile(Path a, Path b) {
        if (a.toAbsolutePath().normalize().equals(b.toAbsolutePath().normalize())) {
            return true;
        }
        try {
            return Files.exists(a) && Files.exists(b) && Files.isSameFile(a, b);
        } catch (IOException e) {
            return false;
        }
    }

    // ------------------------------------------------------------------ prints

    /** The plan of a {@code --dry-run} split: the files it would write, with their tensor counts and bytes. */
    private static void printPlan(List<GgufWriter.PlannedShard> plan) {
        System.out.println("nftool split: dry run, not writing files; the split writes");
        for (GgufWriter.PlannedShard shard : plan) {
            System.out.println("  " + shard);
        }
    }

    // ------------------------------------------------------------------ values

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
     * of 1024 like {@link ExportCommand}'s {@code --split-max-size}, e.g. {@code 2000m}), never one that
     * overflows; {@code 0} keeps the no-limit meaning.
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
