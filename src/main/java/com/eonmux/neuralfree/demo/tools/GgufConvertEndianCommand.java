package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.edit.GgufEndianConverter;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * {@code nftool convert-endian}: rewrites a GGUF file in the other byte order, the operation of llama.cpp's
 * {@code gguf_convert_endian.py}. The order is detected from the file's version field, every multi-byte field of
 * the header, key/value store and tensor infos is written in the target order and every multi-byte tensor block
 * field is byte-swapped. Without {@code --out} the file is converted in place through a temporary twin and an
 * atomic rename; {@code --dry-run} validates the tensor types and writes nothing.
 *
 * <p>NeuralFree reads little-endian files, as ggml writes them; a big-endian result is for a consumer that reads
 * that order. {@code native} names the host's own order.</p>
 */
final class GgufConvertEndianCommand {

    private static final String USAGE =
            "nftool convert-endian <model.gguf> <big|little|native> [options]\n"
            + "options:\n"
            + "  --out <file>          write the converted file to <file> instead of converting in place\n"
            + "  --overwrite           replace an existing --out file\n"
            + "  --dry-run             check the tensor types and write nothing\n"
            + "  --help, -h            this text\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The parsed command line. */
    static final class Parsed {
        String model;
        String order;
        Path out;
        boolean overwrite;
        boolean dryRun;
        boolean help;
    }

    private GgufConvertEndianCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code convert-endian}. */
    static void run(String[] args) {
        try {
            execute(parse(args), System.out);
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | RuntimeException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /** Parses the two positionals and the flags. */
    static Parsed parse(String[] args) throws UsageException {
        Parsed p = new Parsed();
        for (int i = 0; i < args.length && !p.help; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                p.help = true;
            } else if (a.equals("--overwrite")) {
                p.overwrite = true;
            } else if (a.equals("--dry-run")) {
                p.dryRun = true;
            } else if (a.equals("--out")) {
                p.out = Paths.get(value(a, args, i));
                i++;
            } else if (a.startsWith("-")) {
                throw new UsageException("unknown option " + a + " (run with --help for the options)");
            } else if (p.model == null) {
                p.model = a;
            } else if (p.order == null) {
                p.order = a;
            } else {
                throw new UsageException("too many arguments; '" + a + "' follows the order");
            }
        }
        if (p.help) {
            return p;
        }
        if (p.model == null || p.order == null) {
            throw new UsageException("convert-endian needs a GGUF file and an order (big, little or native)");
        }
        if (GgufEndianConverter.Endian.named(p.order) == null) {
            throw new UsageException("the order '" + p.order + "' is not big, little or native");
        }
        return p;
    }

    /** Converts the file and prints what happened; {@code --help} prints the usage instead. */
    static void execute(Parsed p, Appendable out) throws IOException {
        if (p.help) {
            out.append(USAGE);
            return;
        }
        Path input = Paths.get(p.model);
        GgufEndianConverter.Endian target = GgufEndianConverter.Endian.named(p.order);
        GgufEndianConverter.Info info = p.dryRun
                ? GgufEndianConverter.checkConvertible(input)
                : GgufEndianConverter.inspect(input);
        out.append("* Host is ").append(GgufEndianConverter.Endian.host().name())
                .append(" endian, GGUF file seems to be ").append(info.fileEndian().name()).append(" endian\n");
        if (info.fileEndian() == target) {
            out.append("* File is already ").append(target.name()).append(" endian. Nothing to do.\n");
            return;
        }
        if (target != GgufEndianConverter.Endian.host()) {
            out.append("* Warning: the requested endianness differs from the host's; this machine cannot load")
                    .append(" the result.\n");
        }
        if (p.dryRun) {
            out.append("* Preparing to convert from ").append(info.fileEndian().name()).append(" to ")
                    .append(target.name()).append(" (dry run, nothing written)\n");
            return;
        }
        Path output = p.out == null ? input : p.out;
        out.append("* Preparing to convert from ").append(info.fileEndian().name()).append(" to ")
                .append(target.name()).append('\n');
        GgufEndianConverter.convert(input, target, output, p.overwrite);
        out.append("* Completion: wrote ").append(output.toString()).append(" in ").append(target.name())
                .append(" endian\n");
    }

    /** The next argument, or the flag's own error: a flag with no value has no default to invent. */
    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }
}
