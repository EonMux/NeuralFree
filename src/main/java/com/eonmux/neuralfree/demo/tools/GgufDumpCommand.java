package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.inspect.GgufDump;

import java.io.IOException;
import java.nio.file.Paths;

/**
 * {@code nftool dump}: the metadata and tensor table of a GGUF file, the report of llama.cpp's
 * {@code gguf_dump.py}. The text form lists every key/value pair with its GGUF type, element count and value,
 * then every tensor with its element count, four-dimension shape, storage type and name; {@code --json} gives
 * the same content as JSON, {@code --json-array} includes full array values in it, and {@code --no-tensors} stops
 * after the key/value store. {@code --data-offset} and {@code --data-alignment} print one number, as the
 * reference's flags of those names do.
 *
 * <p>The reference's {@code --markdown} report is not provided; {@code --json} carries the same facts for a
 * renderer. A file in big-endian order is refused by the reader with its endianness message; {@code nftool
 * convert-endian} turns it into the little-endian order NeuralFree reads.</p>
 */
final class GgufDumpCommand {

    private static final String USAGE =
            "nftool dump <model.gguf> [options]\n"
            + "options:\n"
            + "  --no-tensors          dump the key/value store only\n"
            + "  --json                print the report as JSON\n"
            + "  --json-array          include full array values in the JSON report\n"
            + "  --data-offset         print the file offset of the tensor data and nothing else\n"
            + "  --data-alignment      print the data alignment and nothing else\n"
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
        GgufDump.Options options = GgufDump.Options.text();
        boolean dataOffset;
        boolean dataAlignment;
        boolean help;
    }

    private GgufDumpCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code dump}. */
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

    /** Parses the flags and the model path. */
    static Parsed parse(String[] args) throws UsageException {
        Parsed p = new Parsed();
        for (int i = 0; i < args.length && !p.help; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                p.help = true;
            } else if (a.equals("--no-tensors")) {
                p.options = p.options.noTensors(true);
            } else if (a.equals("--json")) {
                p.options = p.options.json(true);
            } else if (a.equals("--json-array")) {
                p.options = p.options.jsonArray(true);
            } else if (a.equals("--data-offset")) {
                p.dataOffset = true;
            } else if (a.equals("--data-alignment")) {
                p.dataAlignment = true;
            } else if (a.equals("--markdown")) {
                throw new UsageException("--markdown is not supported by nftool dump; use --json");
            } else if (a.startsWith("-")) {
                throw new UsageException("unknown option " + a + " (run with --help for the options)");
            } else if (p.model == null) {
                p.model = a;
            } else {
                throw new UsageException("too many file arguments; '" + a + "' follows " + p.model);
            }
        }
        if (!p.help) {
            if (p.model == null) {
                throw new UsageException("dump needs a GGUF file");
            }
            if (p.options.isJsonArray() && !p.options.isJson()) {
                throw new UsageException("--json-array applies to --json; add --json");
            }
        }
        return p;
    }

    /** Writes the requested report; {@code --help} prints the usage instead. */
    static void execute(Parsed p, Appendable out) throws IOException {
        if (p.help) {
            out.append(USAGE);
            return;
        }
        try (GGUFFileParser src = ModelSources.parseGguf(p.model)) {
            if (p.options.isJson()) {
                GgufDump.write(src, p.options, out);
            } else if (p.dataOffset) {
                out.append(Long.toString(src.getDataOffset())).append('\n');
            } else if (p.dataAlignment) {
                out.append(Long.toString(src.getAlignment())).append('\n');
            } else {
                GgufDump.text(src, p.options, out);
            }
        } catch (IOException e) {
            throw new IOException("cannot dump '" + Paths.get(p.model) + "': " + e.getMessage(), e);
        }
    }
}
