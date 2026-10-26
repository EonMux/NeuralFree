package com.eonmux.neuralfree.demo.tools;

import java.util.Arrays;
import java.util.List;

/**
 * The {@code nftool} launcher: one subcommand surface for every offline tool of the engine
 * (PARITY.md §5.17), routed by its first argument, as llama.cpp routes its CMake tool targets. Each
 * tool is a class of this package; {@code export} is the first one (PARITY.md WP TL-1). The
 * {@code runNfTool} Gradle task runs this main with its arguments.
 *
 * <pre>
 *    nftool quantize | imatrix | perplexity | results | bench | batched-bench | split | merge |
 *           export-lora | export | download | tokenize | dump | hash | embedding | retrieval | cvector
 * </pre>
 *
 * <p>The later tools route into their WP's command class as it lands; until then they fail with the
 * named error here, and {@code nftool} without a command prints this surface. Every command reads its
 * model the way the demos do, through {@code ModelSources.open}.</p>
 */
public final class NfTool {

    /** The subcommands of the launcher, in the order llama.cpp documents them. */
    static final List<String> COMMANDS = List.of("quantize", "imatrix", "perplexity", "results", "bench",
            "batched-bench", "split", "merge", "export-lora", "export", "download", "tokenize", "dump",
            "hash", "embedding", "retrieval", "cvector");

    private NfTool() {
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            usageError("nftool needs a command: " + COMMANDS);
        }
        String command = args[0];
        String[] rest = Arrays.copyOfRange(args, 1, args.length);
        switch (command) {
            case "export":
                ExportCommand.run(rest);
                return;
            case "--help":
            case "-h":
                usage(System.out);
                return;
            default:
                if (COMMANDS.contains(command)) {
                    // Explicit refusal, never a stub: the tool's WP owns its landing (§5.17).
                    usageError(command + " is not implemented yet; it arrives with its own PARITY.md work "
                            + "package; until then the only tool behind nftool is export");
                }
                usageError("nftool has no command '" + command + "': " + COMMANDS);
        }
    }

    /** The command surface as {@code --help} and every error print it. */
    static void usage(Appendable out) {
        try {
            out.append("nftool <command> [options]\n");
            out.append("commands: " + String.join(" | ", COMMANDS) + "\n");
            out.append("implemented: export\n");
            out.append("run 'nftool export --help' for the export options.\n");
        } catch (java.io.IOException e) {
            throw new IllegalStateException("cannot print the nftool usage", e);
        }
    }

    /** Prints the command surface to stderr and exits with llama.cpp's usage exit code. */
    static void usageError(String message) {
        System.err.println("Error: " + message);
        usage(System.err);
        System.exit(2);
    }
}
