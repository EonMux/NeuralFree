package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.edit.GgufMetadataEditor;

import java.io.IOException;

/**
 * {@code nftool set-metadata}: changes one scalar metadata value of a GGUF file in place, the operation of
 * llama.cpp's {@code gguf_set_metadata.py}. The key keeps its stored GGUF type and byte width, so only the value
 * bytes change; the text is parsed as that type (unsigned for the UINT types). {@code --dry-run} prints the
 * planned change without writing it.
 *
 * <p>A key the file does not have is refused — the reference tool changes values, it never adds keys — and so are
 * STRING and ARRAY values (their length is part of the file layout) and {@code general.alignment} (the tensor
 * offsets depend on it). Adding keys or changing those values is {@code nftool new-metadata}.</p>
 */
final class GgufSetMetadataCommand {

    private static final String USAGE =
            "nftool set-metadata <model.gguf> <key> <value> [--dry-run]\n"
            + "  Changes one scalar value in place, keeping the key's GGUF type and byte width.\n"
            + "  options:\n"
            + "    --dry-run             print the change, write nothing\n"
            + "    --help, -h            this text\n";

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
        String key;
        String value;
        boolean dryRun;
        boolean help;
    }

    private GgufSetMetadataCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code set-metadata}. */
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

    /** Parses the flags and the three positionals. */
    static Parsed parse(String[] args) throws UsageException {
        Parsed p = new Parsed();
        for (int i = 0; i < args.length && !p.help; i++) {
            String a = args[i];
            if (a.equals("--help") || a.equals("-h")) {
                p.help = true;
            } else if (a.equals("--dry-run")) {
                p.dryRun = true;
            } else if (a.startsWith("-")) {
                throw new UsageException("unknown option " + a + " (run with --help for the options)");
            } else if (p.model == null) {
                p.model = a;
            } else if (p.key == null) {
                p.key = a;
            } else if (p.value == null) {
                p.value = a;
            } else {
                throw new UsageException("too many arguments; '" + a + "' follows the value");
            }
        }
        if (!p.help && (p.model == null || p.key == null || p.value == null)) {
            throw new UsageException("set-metadata needs a model, a metadata key and a value");
        }
        return p;
    }

    /** Plans and (unless {@code --dry-run}) writes the change. */
    static void execute(Parsed p, Appendable out) throws IOException {
        if (p.help) {
            out.append(USAGE);
            return;
        }
        GgufMetadataEditor.Change change;
        try (GGUFFileParser src = ModelSources.parseGguf(p.model)) {
            change = GgufMetadataEditor.plan(src, p.key, p.value);
            out.append("* Preparing to change field '").append(change.key()).append("' from ")
                    .append(change.oldText()).append(" to ").append(change.newText()).append('\n');
            if (change.isNoOp()) {
                out.append("- Key '").append(change.key()).append("' already set to requested value ")
                        .append(change.newText()).append('\n');
                return;
            }
            if (p.dryRun) {
                out.append("- Dry run, not writing\n");
                return;
            }
            GgufMetadataEditor.apply(src, change);
        }
        out.append("* Field changed. Successful completion.\n");
    }
}
