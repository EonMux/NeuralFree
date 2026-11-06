package com.eonmux.neuralfree.demo.tools;

import java.util.Arrays;
import java.util.List;

import com.eonmux.neuralfree.demo.OpenAiChatCommand;
import com.eonmux.neuralfree.demo.OpenAiCompleteCommand;
import com.eonmux.neuralfree.demo.tools.eval.EvalCommand;
import com.eonmux.neuralfree.demo.tools.serve.BenchLatencyCommand;
import com.eonmux.neuralfree.demo.tools.serve.BenchStartupCommand;
import com.eonmux.neuralfree.demo.tools.serve.BenchSweepCommand;
import com.eonmux.neuralfree.demo.tools.serve.BenchThroughputCommand;
import com.eonmux.neuralfree.demo.tools.serve.ServeBenchCommand;

/**
 * The {@code nftool} launcher: one subcommand surface for every tool of the engine — the offline tools
 * and the clients of a running server ({@code chat}, {@code complete}, the {@code bench-*} commands) —
 * routed by its first argument, as llama.cpp routes its CMake tool targets. Each tool is a command class
 * of this package or of the demo package. The {@code runNfTool} Gradle task runs this main with its
 * arguments, and the distribution's multi-call launcher ({@code com.eonmux.neuralfree.cli.Launcher})
 * routes its tool commands here.
 *
 * <pre>
 *    nftool quantize | imatrix | perplexity | passkey | results | debug | eval-callback | bench | batched-bench |
 *           finetune | split | merge |
 *           export-lora | export | download | slice | tokenize | dump | hash | set-metadata | new-metadata |
 *           convert-endian | embedding | retrieval |
 *           lookup-create | lookup-merge | lookup-stats | cvector |
 *           pack-experts | fit-params | chat | complete | eval |
 *           bench-serve | bench-throughput | bench-latency | bench-sweep | bench-startup | rpc-server
 * </pre>
 *
 * <p>A command of the surface without an implementation ({@link #IMPLEMENTED} lists the ones that have
 * one) is refused by name, and {@code nftool} without a command prints this surface. Every command that loads a
 * model reads it the way the demos do, through {@code ModelSources.open}; the clients of a running server send
 * their requests over HTTP instead.</p>
 */
public final class NfTool {

    /** The subcommands of the launcher, in the order llama.cpp documents them. */
    public static final List<String> COMMANDS = List.of("quantize", "imatrix", "perplexity", "finetune", "passkey",
            "results", "debug", "eval-callback",
            "bench", "batched-bench", "split", "merge", "export-lora", "export", "download", "slice", "tokenize",
            "dump", "hash", "set-metadata", "new-metadata", "convert-endian", "embedding", "retrieval",
            "lookup-create", "lookup-merge", "lookup-stats",
            "cvector", "pack-experts", "fit-params", "chat", "complete", "eval",
            "bench-serve", "bench-throughput", "bench-latency", "bench-sweep", "bench-startup", "rpc-server");

    /** The subcommands that run; the rest of {@link #COMMANDS} are refused by name. */
    public static final List<String> IMPLEMENTED = List.of("export", "download", "slice", "imatrix", "perplexity",
            "finetune", "passkey", "results", "debug", "eval-callback", "quantize", "split", "merge", "cvector",
            "export-lora",
            "bench", "fit-params",
            "embedding", "retrieval", "lookup-create", "lookup-merge", "lookup-stats",
            "batched-bench", "pack-experts", "tokenize", "dump", "hash", "set-metadata", "new-metadata",
            "convert-endian", "chat", "complete", "eval",
            "bench-serve", "bench-throughput", "bench-latency", "bench-sweep", "bench-startup", "rpc-server");

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
            case "export-lora":
                ExportLoraCommand.run(rest);
                return;
            case "download":
                DownloadCommand.run(rest);
                return;
            case "slice":
                SliceCommand.run(rest);
                return;
            case "imatrix":
                ImatrixCommand.run(rest);
                return;
            case "perplexity":
                PerplexityCommand.run(rest);
                return;
            case "finetune":
                FinetuneCommand.run(rest);
                return;
            case "passkey":
                PasskeyCommand.run(rest);
                return;
            case "results":
                ResultsCommand.run(rest);
                return;
            case "debug":
                DebugCommand.run(rest);
                return;
            case "eval-callback":
                EvalCallbackCommand.run(rest);
                return;
            case "quantize":
                QuantizeCommand.run(rest);
                return;
            case "split":
                SplitCommand.run(rest);
                return;
            case "merge":
                MergeCommand.run(rest);
                return;
            case "cvector":
                CvectorGenerator.run(rest);
                return;
            case "bench":
                BenchCommand.run(rest);
                return;
            case "embedding":
                EmbeddingCommand.run(rest);
                return;
            case "retrieval":
                RetrievalCommand.run(rest);
                return;
            case "lookup-create":
                LookupCreateCommand.run(rest);
                return;
            case "lookup-merge":
                LookupMergeCommand.run(rest);
                return;
            case "lookup-stats":
                LookupStatsCommand.run(rest);
                return;
            case "batched-bench":
                BatchedBenchCommand.run(rest);
                return;
            case "pack-experts":
                PackExpertsCommand.run(rest);
                return;
            case "tokenize":
                TokenizeCommand.run(rest);
                return;
            case "dump":
                GgufDumpCommand.run(rest);
                return;
            case "hash":
                GgufHashCommand.run(rest);
                return;
            case "set-metadata":
                GgufSetMetadataCommand.run(rest);
                return;
            case "new-metadata":
                GgufNewMetadataCommand.run(rest);
                return;
            case "convert-endian":
                GgufConvertEndianCommand.run(rest);
                return;
            case "fit-params":
                FitParamsCommand.run(rest);
                return;
            case "chat":
                OpenAiChatCommand.run(rest);
                return;
            case "complete":
                OpenAiCompleteCommand.run(rest);
                return;
            case "eval":
                EvalCommand.run(rest);
                return;
            case "bench-serve":
                ServeBenchCommand.run(rest);
                return;
            case "bench-throughput":
                BenchThroughputCommand.run(rest);
                return;
            case "bench-latency":
                BenchLatencyCommand.run(rest);
                return;
            case "bench-sweep":
                BenchSweepCommand.run(rest);
                return;
            case "bench-startup":
                BenchStartupCommand.run(rest);
                return;
            case "rpc-server":
                RpcServerCommand.run(rest);
                return;
            case "--help":
            case "-h":
                usage(System.out);
                return;
            default:
                if (COMMANDS.contains(command)) {
                    usageError("nftool " + command + " is not implemented; the implemented tools are "
                            + String.join(", ", IMPLEMENTED));
                }
                usageError("nftool has no command '" + command + "': " + COMMANDS);
        }
    }

    /** The command surface as {@code --help} and every error print it. */
    static void usage(Appendable out) {
        try {
            out.append("nftool <command> [options]\n");
            out.append("commands: " + String.join(" | ", COMMANDS) + "\n");
            out.append("implemented: " + String.join(", ", IMPLEMENTED) + "\n");
            out.append("run 'nftool <command> --help' for the options of an implemented command.\n");
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
