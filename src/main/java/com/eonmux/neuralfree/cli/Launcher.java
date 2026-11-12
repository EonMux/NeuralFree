package com.eonmux.neuralfree.cli;

import com.eonmux.neuralfree.NeuralFree;
import com.eonmux.neuralfree.demo.LLMDemo;
import com.eonmux.neuralfree.demo.tools.NfTool;
import com.eonmux.neuralfree.gguf.hub.HubCache;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The multi-call launcher of the NeuralFree CLI distribution: one entry point — the {@code
 * neuralfree} start script, {@code java -jar}, or the Gradle run tasks — that routes its first
 * argument to a command, as llama.cpp's app binary routes {@code serve}/{@code cli}/{@code
 * quantize}/{@code perplexity} and vLLM's CLI routes its subcommands. The inference commands run
 * here ({@code cli}, {@code bench}), the tools — the offline ones and the clients of a running server
 * ({@code chat}, {@code complete}) — delegate to {@link NfTool} (its surface is the one the tools
 * themselves document), {@code serve} launches the server module over its packaged jar
 * and {@code run-batch} that jar's offline batch runner ({@link ServeCommand}), and the
 * self-describing commands print and exit ({@code version}, {@code help}, {@code collect-env},
 * {@code --list-devices}, {@code --cache-list}). Aliases follow the
 * references ({@code server} for {@code serve}, {@code client} for {@code cli}, {@code get} for
 * {@code download}); a command may also be spelled as a {@code --flag}. An unknown command is
 * refused with the surface, never guessed. The commands that run the engine in this JVM, and
 * {@code collect-env}, which reports the CPU kernels this JVM selects, relaunch it once with
 * {@code --add-modules=jdk.incubator.vector} when the start-up JVM did not resolve the module
 * ({@link JvmVectorModule}), so the packaged start scripts, {@code java -jar} and the Gradle run tasks
 * all get the vector CPU kernels on Java 17 or later.
 */
public final class Launcher {

    /** One command of the surface: its name, what it does, its aliases and whether it is hidden. */
    public static final class Command {

        private final String name;
        private final String description;
        private final List<String> aliases;
        private final boolean hidden;

        Command(String name, String description, List<String> aliases, boolean hidden) {
            this.name = name;
            this.description = description;
            this.aliases = aliases;
            this.hidden = hidden;
        }

        /** The command's name as the surface prints it. */
        public String name() {
            return name;
        }

        /** The one-line description of the command. */
        public String description() {
            return description;
        }

        /** The alternative names that route to the same command. */
        public List<String> aliases() {
            return aliases;
        }

        /** Whether the default {@code help} listing omits the command. */
        public boolean hidden() {
            return hidden;
        }
    }

    /** The command surface, in the references' order: serve first, then the tools, then the meta commands. */
    static final List<Command> COMMANDS = Arrays.asList(
            new Command("serve", "HTTP API server (OpenAI, Anthropic and llama.cpp surfaces)",
                    Arrays.asList("server"), false),
            new Command("run-batch", "Run a file of OpenAI batch requests offline (local or http(s) files)",
                    new ArrayList<>(), false),
            new Command("rpc-server", "Serve one GPU as a remote compute device for a layer split",
                    new ArrayList<>(), false),
            new Command("cli", "Interactive chat and one-shot generation, embeddings, reranking",
                    Arrays.asList("client"), false),
            new Command("chat", "Stream chat completions from a running OpenAI-compatible server",
                    new ArrayList<>(), false),
            new Command("complete", "Stream text completions from a running OpenAI-compatible server",
                    new ArrayList<>(), false),
            new Command("bench", "Benchmark prompt processing and text generation", new ArrayList<>(), false),
            new Command("bench-serve", "Serve an online workload and report TTFT, TPOT, ITL and throughput",
                    new ArrayList<>(), false),
            new Command("bench-throughput", "Send an offline workload at once and report throughput",
                    new ArrayList<>(), false),
            new Command("bench-latency", "Measure the latency of a single batch of requests",
                    new ArrayList<>(), false),
            new Command("bench-sweep", "Sweep server and benchmark parameters over repeated runs",
                    new ArrayList<>(), false),
            new Command("bench-startup", "Measure the server's cold and warm startup time",
                    new ArrayList<>(), false),
            new Command("bench-compare", "Compare two sets of benchmark results and flag regressions",
                    new ArrayList<>(), false),
            new Command("eval", "Evaluate task accuracy against a running server (gsm8k, AIME)",
                    new ArrayList<>(), false),
            new Command("download", "Download a model from the Hugging Face Hub", Arrays.asList("get"), false),
            new Command("slice", "Build a small test slice of a Hugging Face safetensors model",
                    new ArrayList<>(), true),
            new Command("cache-list", "Show the models in the hub cache", new ArrayList<>(), false),
            new Command("collect-env", "Collect environment information for a bug report",
                    new ArrayList<>(), false),
            new Command("quantize", "Quantize a model", new ArrayList<>(), false),
            new Command("perplexity", "Compute model perplexity and KL divergence", new ArrayList<>(), false),
            new Command("finetune", "Fine-tune a GGUF model on a text corpus", new ArrayList<>(), false),
            new Command("imatrix", "Compute an importance matrix for quantization", new ArrayList<>(), false),
            new Command("cvector", "Compute control vectors", new ArrayList<>(), false),
            new Command("export", "Export a checkpoint to GGUF", new ArrayList<>(), false),
            new Command("version", "Show version", new ArrayList<>(), false),
            new Command("help", "Show available commands", new ArrayList<>(), false),
            // the training framework's entry, reachable as it was when the jar's Main-Class named it
            new Command("train", "Run the training framework demo", new ArrayList<>(), true),
            // nftool's further tools, listed by `help all`; nftool refuses the ones without an implementation
            new Command("passkey", "Passkey retrieval test of long-context recall", new ArrayList<>(), true),
            new Command("results", "Check logits against a stored results file (NMSE)", new ArrayList<>(), true),
            new Command("debug", "Print per-op tensors, or dump logits/embeddings", new ArrayList<>(), true),
            new Command("eval-callback", "Print every forward stage's tensor", new ArrayList<>(), true),
            new Command("batched-bench", "Benchmark batched decoding performance", new ArrayList<>(), true),
            new Command("split", "Split a GGUF into shards", new ArrayList<>(), true),
            new Command("merge", "Merge GGUF shards into one file", new ArrayList<>(), true),
            new Command("export-lora", "Merge LoRA adapters into a model, or convert a PEFT adapter to GGUF",
                    new ArrayList<>(), true),
            new Command("tokenize", "Tokenize a text", new ArrayList<>(), true),
            new Command("dump", "Dump a GGUF file's metadata and tensors", new ArrayList<>(), true),
            new Command("hash", "Hash a GGUF file", new ArrayList<>(), true),
            new Command("set-metadata", "Change one scalar metadata value of a GGUF file in place",
                    new ArrayList<>(), true),
            new Command("new-metadata", "Copy a GGUF file with metadata added, changed or removed",
                    new ArrayList<>(), true),
            new Command("convert-endian", "Convert a GGUF file between little- and big-endian order",
                    new ArrayList<>(), true),
            new Command("embedding", "Compute embeddings", new ArrayList<>(), true),
            new Command("retrieval", "Compute retrieval scores", new ArrayList<>(), true),
            new Command("lookup-create", "Build a static n-gram lookup cache from a prompt",
                    new ArrayList<>(), true),
            new Command("lookup-merge", "Merge n-gram lookup cache files", new ArrayList<>(), true),
            new Command("lookup-stats", "Measure n-gram lookup draft acceptance over a prompt",
                    new ArrayList<>(), true),
            new Command("pack-experts", "Pack a MoE model's experts into Avalon's packed-expert layout",
                    new ArrayList<>(), true),
            new Command("fit-params", "Fit the parameters of a device load to the free memory and print them",
                    new ArrayList<>(), true));

    /** The flags that act and exit wherever they appear as the first argument, as llama.cpp's common args do. */
    static final List<String> VERSION_FLAGS = Arrays.asList("--version");
    static final List<String> LIST_DEVICES_FLAGS = Arrays.asList("--list-devices");
    static final List<String> CACHE_LIST_FLAGS = Arrays.asList("-cl", "--cache-list");
    static final List<String> HELP_FLAGS = Arrays.asList("--help", "-h", "help");

    private Launcher() {
    }

    public static void main(String[] args) {
        String first = args.length > 0 ? args[0] : "help";
        if (VERSION_FLAGS.contains(first)) {
            System.out.println(versionLine());
            return;
        }
        if (LIST_DEVICES_FLAGS.contains(first)) {
            DeviceListing.print(System.out);
            return;
        }
        if (CACHE_LIST_FLAGS.contains(first)) {
            cacheList();
            return;
        }
        if (HELP_FLAGS.contains(first)) {
            usage(System.out, args.length > 1 && args[1].equals("all"));
            return;
        }
        Command command = commandOf(first);
        if (command == null) {
            System.err.println("Error: unknown command '" + first + "'");
            usage(System.err, false);
            System.exit(2);
            return;
        }
        if (runsEngineInProcess(command)) {
            // the start scripts and java -jar may not have resolved the incubator module: run this
            // launcher again with it so the command's CPU kernels are the vector ones
            Integer relaunched = JvmVectorModule.relaunchIfNeeded(
                    JvmVectorModule.simdEnabled(System.getProperties()), Launcher.class.getName(), args);
            if (relaunched != null) {
                System.exit(relaunched);
                return;
            }
        }
        String[] rest = Arrays.copyOfRange(args, 1, args.length);
        route(command, rest);
    }

    /**
     * Whether the command runs the engine in this JVM, so it needs the Vector API module resolved here.
     * {@code collect-env} counts as one: it reports the CPU kernels this JVM selects, which are the engine
     * commands' only when the module is resolved alike. The server commands ({@code serve}, {@code run-batch})
     * start their own JVM with the module, the OpenAI client commands ({@code chat}, {@code complete}) talk HTTP
     * to another process, and the other self-describing commands never load the CPU kernels.
     */
    static boolean runsEngineInProcess(Command command) {
        switch (command.name()) {
            case "serve":
            case "run-batch":
            case "chat":
            case "complete":
            case "version":
            case "help":
            case "cache-list":
                return false;
            default:
                return true;
        }
    }

    /** The command named or aliased by {@code arg}, its {@code --flag} spelling accepted, or null. */
    static Command commandOf(String arg) {
        String name = arg.startsWith("--") ? arg.substring(2) : arg;
        for (Command c : COMMANDS) {
            if (c.name().equals(name) || c.aliases().contains(name)) {
                return c;
            }
        }
        return null;
    }

    /** Runs one command. */
    private static void route(Command command, String[] rest) {
        switch (command.name()) {
            case "serve":
                launchServerJar(rest, null);
                return;
            case "run-batch":
                launchServerJar(rest, ServeCommand.BATCH_RUNNER_CLASS);
                return;
            case "cli":
                LLMDemo.main(rest);
                return;
            case "cache-list":
                cacheList();
                return;
            case "collect-env":
                CollectEnv.run();
                return;
            case "version":
                System.out.println(versionLine());
                return;
            case "train":
                NeuralFree.main(rest);
                return;
            case "help":
                usage(System.out, rest.length > 0 && rest[0].equals("all"));
                return;
            default:
                // the tools: NfTool is their surface and refuses, by name, the ones without an implementation
                String[] toolArgs = new String[rest.length + 1];
                toolArgs[0] = command.name();
                System.arraycopy(rest, 0, toolArgs, 1, rest.length);
                NfTool.main(toolArgs);
        }
    }

    /** Runs {@code mainClass} of the server jar (its {@code Main-Class} when null) and exits with its code. */
    private static void launchServerJar(String[] rest, String mainClass) {
        try {
            ServeCommand.launch(rest, mainClass);
        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /** Prints the hub cache listing of the default cache root and exits non-zero when it cannot be read. */
    private static void cacheList() {
        try {
            CacheListing.print(System.out, new HubCache(HubCache.defaultRoot()));
        } catch (IOException | IllegalStateException | IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /** The version line: the program name and the version {@link Version} reports. */
    public static String versionLine() {
        return "neuralfree " + Version.current();
    }

    /** The surface: usage, the visible commands (or every command with {@code all}) and the global flags. */
    static void usage(Appendable out, boolean showAll) {
        try {
            out.append("Usage: neuralfree <command> [options]\n");
            out.append("\nAvailable commands:\n");
            for (Command c : COMMANDS) {
                if (showAll || !c.hidden()) {
                    out.append(String.format("  %-15s %s%n", c.name(), c.description()));
                }
            }
            if (!showAll) {
                out.append("\nRun 'neuralfree help all' to show additional commands.\n");
            }
            out.append("\nFlags:\n");
            out.append("  --version       show version and exit\n");
            out.append("  --list-devices  print list of available devices and exit\n");
            out.append("  -cl, --cache-list\n");
            out.append("                  show list of models in cache and exit\n");
            out.append("\nRun 'neuralfree <command> --help' for command-specific usage.\n");
        } catch (IOException e) {
            throw new IllegalStateException("cannot print the launcher usage", e);
        }
    }
}
