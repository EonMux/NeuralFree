package com.eonmux.neuralfree.demo.tools.eval;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code nftool eval}: the task-accuracy evaluation of a running OpenAI-compatible server, the
 * reference's {@code llama-eval.py}. A math suite ({@code gsm8k}, {@code aime}, {@code aime2025} or
 * {@code aime2026}) is shuffled into cases ({@code --n_cases} may exceed one pass), each case's
 * prompt is sent to one of the comma-separated servers (each with its own thread count and
 * optional bearer token), the reply is graded by the regex, LLM or CLI grader, and the verdicts are
 * saved as the reference's state JSON with a readable HTML report beside it. An interrupted run
 * resumes with {@code --resume}.
 */
public final class EvalCommand {

    /** The command surface: {@code --help} and every error print it. */
    static final String USAGE =
            "nftool eval [options]\n"
            + "options:\n"
            + "  -h, --help                    this text\n"
            + "servers:\n"
            + "  --server <urls>               comma-separated server URLs, default http://localhost:8033\n"
            + "  --server-name <names>         comma-separated display names, in --server order\n"
            + "  --threads <counts>            comma-separated concurrent requests per server, default 32\n"
            + "  --api-key <keys>              bearer token for every request: one key for all servers or\n"
            + "                                one per --server; defaults to $LLAMA_API_KEY\n"
            + "dataset:\n"
            + "  --dataset <name>              aime (default), aime2025, aime2026 or gsm8k\n"
            + "  --dataset-source <source>     a local .jsonl/.json/.csv/.tsv/.parquet file or dataset\n"
            + "                                directory, or a Hub repository id, instead of the canonical one\n"
            + "  --n_cases <n>                 cases to run, default the whole suite once; more than one\n"
            + "                                pass reshuffles each pass and reports pass@k\n"
            + "  --seed <n>                    shuffling seed, default 1234\n"
            + "sampling (sent with every request):\n"
            + "  --n_predict <n>               maximum tokens per reply, default -1 (no limit)\n"
            + "  --temperature <t>  --top-k <k>  --top-p <p>  --min-p <p>\n"
            + "  --reasoning-effort <effort>   e.g. low, medium, high\n"
            + "graders:\n"
            + "  --grader-type <type>          llm (default), regex or cli\n"
            + "  --grader-server <url>         endpoint of the LLM grader, default the first --server\n"
            + "  --grader-model <id>           model of the LLM grader, default --model\n"
            + "  --grader-api-key <key>        bearer token of the LLM grader, default the key of its server\n"
            + "  --grader-script <path>        executable of the CLI grader; it receives --answer and\n"
            + "                                --expected and its exit status is the verdict\n"
            + "run:\n"
            + "  --model <id>                  model name the requests carry, default llama\n"
            + "  --retries <n>                 resends of a failed or empty request, default 5\n"
            + "  --retry-delay <s>             seconds before the first resend, doubled each time, default 10\n"
            + "  --output <path>               state JSON, default " + EvalOptions.DEFAULT_OUTPUT + "\n"
            + "  --resume                      continue an existing state file\n"
            + "  --verbose                     print each case's reply and answer\n"
            + "Not supported (refused by name): --grader-type exec and the agentic/code suites, which\n"
            + "execute generated code; this tool evaluates text answers of the math suites.\n";

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    private EvalCommand() {
    }

    /** Runs the tool with the launcher's arguments after {@code eval}. */
    public static void run(String[] args) {
        try {
            EvalOptions opts = parse(args);
            if (opts == null) {
                return;
            }
            execute(opts, System.out, System.err);
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IllegalArgumentException | IllegalStateException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Parses the flags into the options; null when {@code --help} printed the usage. Long flags
     * accept underscores where the reference uses them.
     *
     * @throws UsageException naming the offending flag
     */
    static EvalOptions parse(String[] args) throws UsageException {
        String serverFlag = "http://localhost:8033";
        String nameFlag = "";
        String threadsFlag = "32";
        String apiKeyFlag = null;
        String dataset = "aime";
        String datasetSource = null;
        Integer nCases = null;
        long seed = 1234L;
        int nPredict = -1;
        Double temperature = null;
        Integer topK = null;
        Double topP = null;
        Double minP = null;
        String reasoningEffort = null;
        int retries = 5;
        double retryDelay = 10.0;
        String model = null;
        boolean verbose = false;
        Path output = Paths.get(EvalOptions.DEFAULT_OUTPUT);
        String graderType = null;
        String graderScript = null;
        String graderServer = "";
        String graderModel = "";
        String graderApiKey = null;
        boolean resume = false;

        for (int i = 0; i < args.length; i++) {
            String typed = args[i];
            String a = typed;
            if (a.equals("--help") || a.equals("-h")) {
                System.out.print(USAGE);
                return null;
            }
            if (a.startsWith("--")) {
                a = a.replace('_', '-');
            }
            switch (a) {
                case "--server":
                    serverFlag = value(typed, args, i);
                    i++;
                    break;
                case "--server-name":
                    nameFlag = value(typed, args, i);
                    i++;
                    break;
                case "--threads":
                    threadsFlag = value(typed, args, i);
                    i++;
                    break;
                case "--api-key":
                    apiKeyFlag = value(typed, args, i);
                    i++;
                    break;
                case "--dataset":
                    dataset = value(typed, args, i);
                    i++;
                    break;
                case "--dataset-source":
                    datasetSource = value(typed, args, i);
                    i++;
                    break;
                case "--n-cases":
                    nCases = positiveInt(value(typed, args, i), typed);
                    i++;
                    break;
                case "--seed":
                    seed = longValue(value(typed, args, i), typed);
                    i++;
                    break;
                case "--n-predict":
                    nPredict = intValue(value(typed, args, i), typed);
                    i++;
                    break;
                case "--temperature":
                    temperature = doubleValue(value(typed, args, i), typed);
                    i++;
                    break;
                case "--top-k":
                    topK = nonNegativeInt(value(typed, args, i), typed);
                    i++;
                    break;
                case "--top-p":
                    topP = doubleValue(value(typed, args, i), typed);
                    i++;
                    break;
                case "--min-p":
                    minP = doubleValue(value(typed, args, i), typed);
                    i++;
                    break;
                case "--reasoning-effort":
                    reasoningEffort = value(typed, args, i);
                    i++;
                    break;
                case "--retries":
                    retries = nonNegativeInt(value(typed, args, i), typed);
                    i++;
                    break;
                case "--retry-delay":
                    retryDelay = nonNegativeDouble(value(typed, args, i), typed);
                    i++;
                    break;
                case "--model":
                    model = value(typed, args, i);
                    i++;
                    break;
                case "--verbose":
                    verbose = true;
                    break;
                case "--output":
                    output = path(value(typed, args, i), typed);
                    i++;
                    break;
                case "--grader-type":
                    graderType = value(typed, args, i);
                    i++;
                    break;
                case "--grader-script":
                    graderScript = value(typed, args, i);
                    i++;
                    break;
                case "--grader-server":
                    graderServer = value(typed, args, i);
                    i++;
                    break;
                case "--grader-model":
                    graderModel = value(typed, args, i);
                    i++;
                    break;
                case "--grader-api-key":
                    graderApiKey = value(typed, args, i);
                    i++;
                    break;
                case "--resume":
                    resume = true;
                    break;
                default:
                    if (!a.startsWith("--")) {
                        throw new UsageException("unexpected argument '" + a + "'");
                    }
                    throw new UsageException("unknown option '" + a + "'");
            }
        }

        if (graderType == null) {
            graderType = "llm";
        }
        if ("exec".equals(graderType)) {
            throw new UsageException("--grader-type exec is only meaningful for the code suites"
                    + " (humaneval, classeval, agentic), not the math suites; use regex, llm or cli");
        }
        if (!EvalGrader.TYPES.contains(graderType)) {
            throw new UsageException("--grader-type is one of " + EvalGrader.TYPES + ", not '"
                    + graderType + "'");
        }

        List<EvalServer> servers = servers(serverFlag, nameFlag, threadsFlag, apiKeyFlag);
        String resolvedGraderServer = graderServer == null || graderServer.isEmpty()
                ? servers.get(0).url() : stripSlashes(graderServer);
        String resolvedGraderKey = graderApiKey;
        if (resolvedGraderKey == null) {
            for (EvalServer server : servers) {
                if (server.url().equals(resolvedGraderServer)) {
                    resolvedGraderKey = server.apiKey();
                    break;
                }
            }
        }
        String resolvedGraderModel = graderModel == null || graderModel.isEmpty() ? model
                : graderModel;
        EvalSampling sampling = new EvalSampling(temperature, topK, topP, minP, reasoningEffort,
                nPredict);
        try {
            return EvalOptions.builder()
                    .servers(servers)
                    .dataset(dataset)
                    .datasetSource(datasetSource)
                    .nCases(nCases)
                    .seed(seed)
                    .sampling(sampling)
                    .retries(retries)
                    .retryDelay(retryDelay)
                    .model(model)
                    .verbose(verbose)
                    .output(output)
                    .graderType(graderType)
                    .graderScript(graderScript)
                    .graderServer(resolvedGraderServer)
                    .graderServerExplicit(graderServer != null && !graderServer.isEmpty())
                    .graderModel(resolvedGraderModel)
                    .graderApiKey(resolvedGraderKey)
                    .resume(resume)
                    .build();
        } catch (IllegalArgumentException e) {
            throw new UsageException(e.getMessage());
        }
    }

    /** The servers of the three comma-separated lists, refused when their counts disagree. */
    private static List<EvalServer> servers(String serverFlag, String nameFlag, String threadsFlag,
                                            String apiKeyFlag) throws UsageException {
        List<String> urls = split(serverFlag);
        List<String> threadTexts = split(threadsFlag);
        if (urls.isEmpty()) {
            throw new UsageException("--server must name at least one server URL");
        }
        if (urls.size() != threadTexts.size()) {
            throw new UsageException("--server (" + urls.size() + " URLs) and --threads ("
                    + threadTexts.size() + " values) must have the same count");
        }
        List<String> names;
        if (nameFlag == null || nameFlag.trim().isEmpty()) {
            names = urls;
        } else {
            names = split(nameFlag);
            if (names.size() != urls.size()) {
                throw new UsageException("--server-name (" + names.size() + " names) and --server ("
                        + urls.size() + " URLs) must have the same count");
            }
        }
        String keyText = apiKeyFlag != null ? apiKeyFlag : System.getenv("LLAMA_API_KEY");
        List<String> keys = new ArrayList<>();
        if (keyText != null && !keyText.trim().isEmpty()) {
            keys = split(keyText);
            if (keys.size() == 1) {
                String one = keys.get(0);
                keys = new ArrayList<>();
                for (int i = 0; i < urls.size(); i++) {
                    keys.add(one);
                }
            } else if (keys.size() != urls.size()) {
                throw new UsageException("--api-key (" + keys.size() + " keys) must be a single key"
                        + " or match --server (" + urls.size() + " URLs)");
            }
        }
        List<EvalServer> servers = new ArrayList<>();
        for (int i = 0; i < urls.size(); i++) {
            int threads = positiveInt(threadTexts.get(i), "--threads");
            String key = keys.isEmpty() ? null : keys.get(i);
            try {
                servers.add(new EvalServer(urls.get(i), threads, names.get(i), key));
            } catch (IllegalArgumentException e) {
                throw new UsageException(e.getMessage());
            }
        }
        return servers;
    }

    /** A comma-separated list's non-empty, trimmed entries. */
    private static List<String> split(String text) {
        List<String> out = new ArrayList<>();
        if (text == null) {
            return out;
        }
        for (String part : text.split(",")) {
            String trimmed = part.trim();
            if (!trimmed.isEmpty()) {
                out.add(trimmed);
            }
        }
        return out;
    }

    private static String stripSlashes(String url) {
        String out = url.trim();
        while (out.endsWith("/")) {
            out = out.substring(0, out.length() - 1);
        }
        return out;
    }

    /**
     * Runs one evaluation: loads the suite, resumes or lays out the tasks, builds the grader and
     * drives the run. Results are written to the state file as each case completes.
     *
     * @throws IllegalArgumentException naming a mismatched or missing state file
     * @throws IllegalStateException    when the dataset or a server cannot be read
     */
    static void execute(EvalOptions opts, Appendable out, Appendable err) {
        EvalDataset dataset = EvalDatasets.load(opts.dataset(), opts.datasetSource());
        EvalState state;
        boolean fresh;
        if (Files.exists(opts.output())) {
            fresh = false;
            append(out, "Loading existing eval state from " + opts.output() + "\n");
            EvalState loaded = EvalState.load(opts.output(), dataset, opts.sampling().nPredict());
            if (loaded.modelName() != null && !loaded.modelName().equals(opts.model())) {
                throw new IllegalArgumentException("the eval state " + opts.output()
                        + " was recorded for model '" + loaded.modelName() + "', but --model is '"
                        + opts.model() + "'");
            }
            loaded.printAllTasks(out);
            loaded.printExistingSummary(out);
            if (loaded.isComplete()) {
                return;
            }
            append(out, "\n");
            if (!opts.resume()) {
                append(out, "Evaluation incomplete. Run with --resume to continue.\n");
                return;
            }
            List<EvalState.TaskRef> pending = loaded.pendingTasks();
            append(out, "Resuming from " + pending.size() + " pending tasks\n");
            loaded.setTasks(pending);
            state = loaded;
        } else {
            fresh = true;
            if (opts.resume()) {
                throw new IllegalArgumentException("no existing eval state found to resume: "
                        + opts.output());
            }
            state = new EvalState(dataset, opts.model(), opts.sampling(), opts.output());
            state.setupTasks(opts.nCases(), opts.seed());
            state.dump();
            state.printAllTasks(out);
        }
        EvalGrader grader = grader(opts, dataset.type(), err, fresh);
        EvalRunner runner = new EvalRunner(opts.servers(), grader, opts.model(), opts.retries(),
                opts.retryDelay(), out, err);
        runner.evaluate(state, opts.verbose());
        if (state.hasMultipleSamples()) {
            state.printPassAtK(out);
        }
        append(out, "\nEval state dumped to " + opts.output() + "\n");
    }

    /** The grader the options name; {@code warnSameServer} prints the reference's note on a fresh run. */
    private static EvalGrader grader(EvalOptions opts, String datasetType, Appendable err,
                                     boolean warnSameServer) {
        if ("regex".equals(opts.graderType())) {
            return EvalGrader.regex(datasetType);
        }
        if ("cli".equals(opts.graderType())) {
            return EvalGrader.cli(datasetType, opts.graderScript());
        }
        if ("llm".equals(opts.graderType())) {
            if (warnSameServer && !opts.graderServerExplicit()) {
                append(err, "Warning: Using same server for LLM grader (no --grader-server"
                        + " specified)\n");
            }
            EvalServer server = new EvalServer(opts.graderServer(), 1, "grader",
                    opts.graderApiKey());
            EvalClient client = new EvalClient(server, opts.graderModel(), 0, 0.0, null);
            return EvalGrader.llm(datasetType, client, opts.graderModel());
        }
        throw new IllegalArgumentException("--grader-type is one of " + EvalGrader.TYPES + ", not '"
                + opts.graderType() + "'");
    }

    private static String value(String flag, String[] args, int index) throws UsageException {
        if (index + 1 >= args.length) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }

    private static int positiveInt(String value, String flag) throws UsageException {
        int n = intValue(value, flag);
        if (n < 1) {
            throw new UsageException(flag + " needs a positive integer, not '" + value + "'");
        }
        return n;
    }

    private static int nonNegativeInt(String value, String flag) throws UsageException {
        int n = intValue(value, flag);
        if (n < 0) {
            throw new UsageException(flag + " needs a non-negative integer, not '" + value + "'");
        }
        return n;
    }

    private static int intValue(String value, String flag) throws UsageException {
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer, not '" + value + "'");
        }
    }

    private static long longValue(String value, String flag) throws UsageException {
        try {
            return Long.parseLong(value.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs an integer, not '" + value + "'");
        }
    }

    private static double doubleValue(String value, String flag) throws UsageException {
        double parsed;
        try {
            parsed = Double.parseDouble(value.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(flag + " needs a number, not '" + value + "'");
        }
        if (Double.isNaN(parsed) || Double.isInfinite(parsed)) {
            throw new UsageException(flag + " needs a finite number, not '" + value + "'");
        }
        return parsed;
    }

    private static double nonNegativeDouble(String value, String flag) throws UsageException {
        double d = doubleValue(value, flag);
        if (d < 0 || Double.isNaN(d)) {
            throw new UsageException(flag + " needs a non-negative number, not '" + value + "'");
        }
        return d;
    }

    private static Path path(String value, String flag) throws UsageException {
        if (value == null || value.trim().isEmpty()) {
            throw new UsageException(flag + " needs a file path");
        }
        try {
            return Paths.get(value);
        } catch (InvalidPathException e) {
            throw new UsageException(flag + " is not a path: " + value);
        }
    }

    private static void append(Appendable sink, String text) {
        try {
            sink.append(text);
        } catch (IOException e) {
            throw new IllegalStateException("cannot write to the output", e);
        }
    }
}
