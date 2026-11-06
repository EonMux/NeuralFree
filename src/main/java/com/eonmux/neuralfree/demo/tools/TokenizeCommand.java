package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

/**
 * {@code nftool tokenize}: llama.cpp's tokenize tool (tools/tokenize/tokenize.cpp). The text —
 * {@code --prompt}, {@code --file} (read verbatim, so a trailing newline stays a token) or
 * {@code --stdin} (read after the model opens, and takes precedence) — tokenizes on the loaded
 * vocabulary only: the model's weights are never read, as llama.cpp loads it with {@code vocab_only}.
 * The vocabulary's BOS policy applies unless {@code --no-bos}, special tokens in the text parse unless
 * {@code --no-parse-special}, and the escapes of every source process unless {@code --no-escape}.
 * The ids print as a Python-parseable list ({@code --ids}) or as one {@code %6d -> 'piece'} line per
 * token, the piece rendered as {@code common_token_to_piece}; {@code --show-count} prints the total.
 */
final class TokenizeCommand {

    private static final String USAGE =
            "nftool tokenize <model> [options]\n"
            + "options (llama.cpp tokenize names):\n"
            + "  --model, -m <path>               the model (a GGUF file or Hugging Face checkpoint); only the\n"
            + "                                   vocabulary is read, the weights never are\n"
            + "  --prompt, -p <text>              the text to tokenize\n"
            + "  --file, -f <fname>               read the text from a file, its content kept verbatim (a\n"
            + "                                   trailing newline is itself a token)\n"
            + "  --stdin                          read the text from standard input (takes precedence over\n"
            + "                                   -f/--file and -p/--prompt)\n"
            + "  --ids                            print the ids as a Python-parseable list like [1, 2, 3]\n"
            + "  --show-count                     print the total number of tokens\n"
            + "  --no-bos                         do not add a BOS token to the text, even when the\n"
            + "                                   vocabulary normally uses one\n"
            + "  --no-parse-special               do not parse special tokens (chat, tool, etc)\n"
            + "  --escape, -e | --no-escape       process the \\n \\r \\t \\' \\\" \\\\ escapes of the text\n"
            + "                                   (default: on)\n"
            + "  --help, -h                       this text\n"
            + "example usage:\n"
            + "  nftool tokenize -m model.gguf -p \"Hello world\"\n"
            + "  nftool tokenize -m model.gguf -f prompt.txt --ids\n"
            + "  cat prompt.txt | nftool tokenize -m model.gguf --stdin --show-count\n"
            + "Results print to stdout; the usage and errors go to stderr.\n";

    private TokenizeCommand() {
    }

    /** An invalid command line; the message names the flag. */
    static final class UsageException extends Exception {

        private static final long serialVersionUID = 1L;

        UsageException(String message) {
            super(message);
        }
    }

    /** The tokenize options of one run (immutable); the prompt text has its escapes processed. */
    static final class Options {

        final String model;
        /** The text of {@code --prompt} or the verbatim content of {@code --file}; null when neither was given. */
        final String prompt;
        final boolean stdin;
        final boolean ids;
        final boolean showCount;
        final boolean noBos;
        final boolean parseSpecial;
        /** Whether the text read from stdin processes its escapes in the run. */
        final boolean escape;

        Options(String model, String prompt, boolean stdin, boolean ids, boolean showCount, boolean noBos,
                boolean parseSpecial, boolean escape) {
            this.model = model;
            this.prompt = prompt;
            this.stdin = stdin;
            this.ids = ids;
            this.showCount = showCount;
            this.noBos = noBos;
            this.parseSpecial = parseSpecial;
            this.escape = escape;
        }

        String model() { return model; }
        String prompt() { return prompt; }
        boolean stdin() { return stdin; }
        boolean ids() { return ids; }
        boolean showCount() { return showCount; }
        boolean noBos() { return noBos; }
        boolean parseSpecial() { return parseSpecial; }
        boolean escape() { return escape; }
    }

    /** Runs the tool with the launcher's arguments after {@code tokenize}. */
    static void run(String[] args) {
        try {
            Options o = parse(args);
            if (o != null) {
                execute(o);
            }
        } catch (UsageException e) {
            System.err.println("Error: " + e.getMessage());
            System.err.print(USAGE);
            System.exit(2);
        } catch (IOException | UnsupportedOperationException | IllegalArgumentException | IllegalStateException e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Opens the model's vocabulary, reads stdin when asked (as llama.cpp does, after the model opens, so a
     * bad model exits before the read) and prints the tokens of the text.
     *
     * @throws IOException              when the model cannot be opened or stdin cannot be read
     * @throws IllegalArgumentException when the model's tokenizer is not supported
     */
    static void execute(Options o) throws IOException {
        execute(o, System.in, System.out);
    }

    /** The run over an explicit stdin and output, as the tests drive it. */
    static void execute(Options o, InputStream stdin, Appendable out) throws IOException {
        String prompt = o.prompt();
        try (ModelSource source = ModelSources.open(o.model())) {
            GGUFTokenizer tok = new GGUFTokenizer(source);
            if (o.stdin()) {
                String text = readAll(stdin);
                prompt = o.escape() ? CvectorGenerator.processEscapes(text) : text;
            }
            int[] tokens = tokenize(tok, prompt, o);
            out.append(render(tokens, tok, o));
        }
    }

    /**
     * llama.cpp's {@code common_tokenize} of the tool: the vocabulary's BOS applies unless
     * {@code --no-bos} suppressed it, and {@code --no-parse-special} turns the parsing of special tokens off.
     */
    static int[] tokenize(GGUFTokenizer tok, String prompt, Options o) {
        boolean addSpecial = tok.vocab().addBos() && !o.noBos();
        return tok.encode(prompt, addSpecial, o.parseSpecial());
    }

    /**
     * The output of the tool: with {@code --ids} the list {@code [1, 2, 3]}, otherwise one
     * {@code %6d -> 'piece'} line per token with the piece of {@code token_to_piece(special = true)};
     * {@code --show-count} appends the total.
     */
    static String render(int[] tokens, GGUFTokenizer tok, Options o) {
        StringBuilder out = new StringBuilder();
        if (o.ids()) {
            out.append('[');
            for (int i = 0; i < tokens.length; i++) {
                if (i > 0) {
                    out.append(", ");
                }
                out.append(tokens[i]);
            }
            out.append("]\n");
        } else {
            for (int id : tokens) {
                out.append(String.format(Locale.ROOT, "%6d -> '%s'\n", id,
                        new String(tok.pieceWithSpecial(id), StandardCharsets.UTF_8)));
            }
        }
        if (o.showCount()) {
            out.append("Total number of tokens: ").append(tokens.length).append('\n');
        }
        return out.toString();
    }

    /**
     * Parses the flags (llama.cpp's tokenize option names). An explicit {@code --file} answers any
     * {@code --prompt}, as llama.cpp's {@code prompt_file} does, whatever the order; its content is read
     * verbatim, its trailing newline kept. The escapes of the text process afterwards unless
     * {@code --no-escape}; the text read from stdin processes them in the run. {@code --stdin} needs no text.
     *
     * @return the options, or null after printing the usage for {@code --help}
     * @throws UsageException naming the offending flag
     */
    static Options parse(String[] args) throws UsageException {
        String model = null;
        String prompt = null;
        String fileText = null;
        boolean file = false;
        boolean stdin = false;
        boolean ids = false;
        boolean showCount = false;
        boolean noBos = false;
        boolean parseSpecial = true;
        boolean escape = true;
        for (int i = 0; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--help":
                case "-h":
                    System.out.println(USAGE);
                    return null;
                case "--model":
                case "-m":
                    model = value(a, args, i++);
                    break;
                case "--prompt":
                case "-p":
                    prompt = value(a, args, i++);
                    break;
                case "--file":
                case "-f":
                    fileText = readPromptFile(a, value(a, args, i++));
                    file = true;
                    break;
                case "--stdin":
                    stdin = true;
                    break;
                case "--ids":
                    ids = true;
                    break;
                case "--show-count":
                    showCount = true;
                    break;
                case "--no-bos":
                    noBos = true;
                    break;
                case "--no-parse-special":
                    parseSpecial = false;
                    break;
                case "--escape":
                case "-e":
                    escape = true;
                    break;
                case "--no-escape":
                    escape = false;
                    break;
                default: {
                    if (a.startsWith("-")) {
                        throw new UsageException("unknown option " + a + " (run with --help for the options)");
                    }
                    if (model != null) {
                        throw new UsageException("too many file arguments; '" + a + "' follows " + model);
                    }
                    model = a;
                }
            }
        }
        if (model == null) {
            throw new UsageException("tokenize needs a model path");
        }
        String text = fileText != null ? fileText : prompt;
        if (text != null && escape) {
            text = CvectorGenerator.processEscapes(text);
        }
        // llama.cpp's check: the text must be there, but an explicitly given empty file or --stdin passes
        if (!stdin && !file && (text == null || text.isEmpty())) {
            throw new UsageException("tokenize needs one of --stdin, --file or --prompt");
        }
        return new Options(model, text, stdin, ids, showCount, noBos, parseSpecial, escape);
    }

    /**
     * llama.cpp tokenize's {@code -f}: the file's content verbatim — unlike the other tools, one trailing
     * newline is not dropped, because a trailing newline is itself a token.
     */
    private static String readPromptFile(String flag, String name) throws UsageException {
        Path f = Paths.get(name);
        if (!Files.isRegularFile(f) || !Files.isReadable(f)) {
            throw new UsageException(flag + " '" + name + "' is not a readable file");
        }
        try {
            return Files.readString(f, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UsageException(flag + " '" + name + "' cannot be read: " + e.getMessage());
        }
    }

    /** The whole of {@code in} as UTF-8 text, llama.cpp's {@code stdin_buffer << std::cin.rdbuf()}. */
    private static String readAll(InputStream in) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int read;
        while ((read = in.read(buffer)) != -1) {
            bytes.write(buffer, 0, read);
        }
        return bytes.toString(StandardCharsets.UTF_8.name());
    }

    private static String value(String flag, String[] args, int index) throws UsageException {
        // The value consumes the next argv token even when it starts with '-' (llama.cpp consumes it), so
        // -p -x parses.
        if (index + 1 >= args.length || args[index + 1] == null) {
            throw new UsageException(flag + " needs a value");
        }
        return args[index + 1];
    }
}
