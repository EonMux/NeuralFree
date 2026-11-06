package com.eonmux.neuralfree.demo;

import java.io.PrintStream;

/**
 * The options of the OpenAI client subcommands {@code chat} and {@code complete} (vLLM's
 * {@code entrypoints/cli/openai.py}): the query options both share ({@code --url}, {@code --model-name},
 * {@code --api-key}, {@code -q/--quick} and {@code --stats}) and the one flag of the command that has it
 * ({@code --system-prompt} for {@code chat}, {@code --max-tokens} for {@code complete}). An option the command
 * does not have is refused by name. Immutable.
 */
final class OpenAiClientOptions {

    /** The OpenAI API base URL of a server started by {@code neuralfree serve} (vLLM's {@code --url} default). */
    static final String DEFAULT_URL = "http://localhost:8000/v1";

    /** The key sent when neither {@code --api-key} nor the environment names one, as the vLLM CLI sends it. */
    static final String DEFAULT_API_KEY = "EMPTY";

    private final String url;
    private final String modelName;
    private final String apiKey;
    private final String quick;
    private final boolean stats;
    private final String systemPrompt;
    private final Integer maxTokens;

    private OpenAiClientOptions(String url, String modelName, String apiKey, String quick, boolean stats,
                                String systemPrompt, Integer maxTokens) {
        this.url = url;
        this.modelName = modelName;
        this.apiKey = apiKey;
        this.quick = quick;
        this.stats = stats;
        this.systemPrompt = systemPrompt;
        this.maxTokens = maxTokens;
    }

    /** The OpenAI API base URL every request is appended to. */
    String url() {
        return url;
    }

    /** The {@code --model-name}, or null when the first model the server lists is used. */
    String modelName() {
        return modelName;
    }

    /**
     * The bearer key: {@code --api-key}, else {@code OPENAI_API_KEY}, else {@code NEURALFREE_API_KEY}, else
     * {@value #DEFAULT_API_KEY}.
     */
    String apiKey() {
        if (apiKey != null && !apiKey.isEmpty()) {
            return apiKey;
        }
        String env = System.getenv("OPENAI_API_KEY");
        if (env != null && !env.isEmpty()) {
            return env;
        }
        env = System.getenv("NEURALFREE_API_KEY");
        return env == null || env.isEmpty() ? DEFAULT_API_KEY : env;
    }

    /** The {@code -q/--quick} message or prompt, or null for the interactive loop. */
    String quick() {
        return quick;
    }

    /** Whether {@code --stats} asks for the TTFT and TPS of every answer. */
    boolean stats() {
        return stats;
    }

    /** The {@code --system-prompt} of {@code chat}, or null. */
    String systemPrompt() {
        return systemPrompt;
    }

    /** The {@code --max-tokens} of {@code complete}, or null when the server's default applies. */
    Integer maxTokens() {
        return maxTokens;
    }

    /** An invalid command line; the message names the option. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /**
     * The options of {@code args} for {@code chat} or {@code complete}. A {@code --name=value} spelling is
     * accepted.
     *
     * @return the options; null after printing the usage for {@code --help}
     * @throws UsageException for an option of the other command, an unknown option or argument, a missing value,
     *                        or a value that is not the option's kind
     */
    static OpenAiClientOptions parse(String[] args, boolean chat, PrintStream out) throws UsageException {
        String command = chat ? "chat" : "complete";
        State state = new State();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (arg.equals("-h") || arg.equals("--help")) {
                usage(chat, out);
                return null;
            }
            int eq = arg.indexOf('=');
            boolean inline = arg.startsWith("--") && eq > 2;
            String option = inline ? arg.substring(0, eq) : arg;
            if (!known(option)) {
                throw new UsageException(arg.startsWith("-")
                        ? command + " has no option '" + arg + "'"
                        : command + " takes no positional argument '" + arg + "'; -q/--quick sends one message");
            }
            if (option.equals("--stats")) {
                if (inline) {
                    throw new UsageException(option + " takes no value");
                }
                state.stats = true;
                continue;
            }
            String value = inline ? arg.substring(eq + 1) : value(args, i + 1, option);
            if (!inline) {
                i++;
            }
            state.apply(chat, option, value);
        }
        return new OpenAiClientOptions(state.url, state.modelName, state.apiKey, state.quick, state.stats,
                state.systemPrompt, state.maxTokens);
    }

    /** The options as they are read; {@link #parse} freezes them into the immutable value. */
    private static final class State {

        String url = DEFAULT_URL;
        String modelName;
        String apiKey;
        String quick;
        boolean stats;
        String systemPrompt;
        Integer maxTokens;

        /** Reads one option and its value; the other command's flag is refused by name. */
        void apply(boolean chat, String option, String value) throws UsageException {
            switch (option) {
                case "--url":
                    url = value;
                    break;
                case "--model-name":
                    modelName = value;
                    break;
                case "--api-key":
                    apiKey = value;
                    break;
                case "-q":
                case "--quick":
                    quick = value;
                    break;
                case "--system-prompt":
                    if (!chat) {
                        throw new UsageException("complete has no option '--system-prompt'; it is chat's");
                    }
                    systemPrompt = value;
                    break;
                case "--max-tokens":
                    if (chat) {
                        throw new UsageException("chat has no option '--max-tokens'; it is complete's");
                    }
                    maxTokens = whole(option, value);
                    break;
                default:
                    throw new IllegalStateException("unhandled option " + option);
            }
        }
    }

    /** Whether {@code option} is one of the flags either command reads. */
    private static boolean known(String option) {
        switch (option) {
            case "--url":
            case "--model-name":
            case "--api-key":
            case "-q":
            case "--quick":
            case "--system-prompt":
            case "--max-tokens":
            case "--stats":
                return true;
            default:
                return false;
        }
    }

    /** The value of an option, refusing a missing one. */
    private static String value(String[] args, int index, String option) throws UsageException {
        if (index >= args.length) {
            throw new UsageException(option + " needs a value");
        }
        return args[index];
    }

    /** A whole-number option value. */
    private static Integer whole(String option, String raw) throws UsageException {
        try {
            return Integer.valueOf(raw.trim());
        } catch (NumberFormatException e) {
            throw new UsageException(option + " must be a whole number, got '" + raw + "'");
        }
    }

    /** The command's usage, as {@code --help} and every usage error print it. */
    static void usage(boolean chat, PrintStream out) {
        String command = chat ? "chat" : "complete";
        out.println("neuralfree " + command + " [options]");
        out.println(chat ? "Generate chat completions via a running OpenAI-compatible server."
                : "Generate text completions via a running OpenAI-compatible server.");
        out.println("  --url URL            the OpenAI API base URL (default " + DEFAULT_URL + ")");
        out.println("  --model-name NAME    the model name of every request (default: the first model the");
        out.println("                       server lists on /models)");
        out.println("  --api-key KEY        the bearer key (default: OPENAI_API_KEY, NEURALFREE_API_KEY,");
        out.println("                       else " + DEFAULT_API_KEY + ")");
        if (chat) {
            out.println("  --system-prompt TEXT the system message that opens the conversation");
            out.println("  -q, --quick MESSAGE  send one message, print the answer and exit");
        } else {
            out.println("  --max-tokens N       the maximum number of tokens to generate per output sequence");
            out.println("  -q, --quick PROMPT   complete one prompt, print the output and exit");
        }
        out.println("  --stats              print TTFT and TPS after each answer");
        out.println("  -h, --help           this text");
    }
}
