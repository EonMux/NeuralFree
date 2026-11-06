package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.StopReason;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * The interactive completion mode ({@code -i}, {@code -if}, {@code --in-prefix}, {@code --in-suffix},
 * {@code --in-prefix-bos}) of llama.cpp tools/completion/completion.cpp: raw completion with no chat template
 * over one growing token context. The prompt (escapes processed unless {@code --no-escape}) is completed, then
 * control returns to the user on a reverse prompt ({@code -r}), at the end of generation (which injects the first
 * reverse prompt), at the token budget or on Ctrl-C; each input enters the context as {@code --in-prefix}, the
 * input and {@code --in-suffix}. Before a generation that would leave fewer than {@value #CONTEXT_MARGIN}
 * positions free, the older half of the context after a leading BOS shifts out (completion.cpp:567-588).
 */
final class CompletionRepl {

    /** Positions the context keeps free (completion.cpp {@code n_ctx - 4}). */
    static final int CONTEXT_MARGIN = 4;

    /** What one round of input asks the loop to do. */
    private enum Input { GENERATE, EXIT }

    private final TurnGenerator generator;
    private final GenerationRequest request;
    private final boolean interactive;
    private final boolean interactiveFirst;
    private final boolean multiline;
    private final boolean showAllChannels;
    private final boolean escape;
    private final String prompt;
    private final String inPrefix;
    private final String inSuffix;
    private final boolean inPrefixBos;
    private final List<String> antiprompts;
    private final int capacity;
    private final GGUFTokenizer tokenizer;
    private final BufferedReader in;
    private final PrintStream out;
    private final Transcript transcript;
    private final Interrupts interrupts;
    private final List<Integer> context = new ArrayList<>();

    private CompletionRepl(LLMModel model, TurnGenerator generator, LLMDemoOptions opts, BufferedReader in,
                           PrintStream out, Transcript transcript, Interrupts interrupts) {
        this.generator = generator;
        this.request = opts.generation;
        this.interactive = opts.interactive || opts.interactiveFirst;
        this.interactiveFirst = opts.interactiveFirst;
        this.multiline = opts.multilineInput;
        this.showAllChannels = opts.showAllChannels;
        this.escape = opts.escape;
        this.prompt = escaped(opts.prompt == null ? "" : opts.prompt);
        this.inPrefix = escaped(opts.inPrefix == null ? "" : opts.inPrefix);
        this.inSuffix = escaped(opts.inSuffix == null ? "" : opts.inSuffix);
        this.inPrefixBos = opts.inPrefixBos;
        this.antiprompts = opts.generation.stopStrings();
        this.capacity = model.kvCacheContext();
        this.tokenizer = model.tokenizer();
        this.in = in;
        this.out = out;
        this.transcript = transcript;
        this.interrupts = interrupts;
    }

    /** Runs the completion mode on the standard streams and the process interrupt state. */
    static void run(LLMModel model, TurnGenerator generator, LLMDemoOptions opts) throws IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        Transcript transcript = opts.outFile == null ? null : Transcript.open(opts.outFile);
        try {
            run(model, generator, opts, in, System.out, transcript, Interrupts.system());
        } finally {
            if (transcript != null) {
                transcript.close();
            }
        }
    }

    /**
     * Runs the completion mode over the given streams.
     *
     * @param transcript the {@code -o} transcript, or null
     * @throws IllegalArgumentException when the prompt does not fit the context
     */
    static void run(LLMModel model, TurnGenerator generator, LLMDemoOptions opts, BufferedReader in,
                    PrintStream out, Transcript transcript, Interrupts interrupts) throws IOException {
        if (model == null || generator == null || opts == null || in == null || out == null || interrupts == null) {
            throw new IllegalArgumentException("model, generator, options, streams and interrupts are required");
        }
        new CompletionRepl(model, generator, opts, in, out, transcript, interrupts).loop();
    }

    private void loop() throws IOException {
        startContext();
        if (interactive) {
            printIntro();
        }
        out.print(prompt);
        out.flush();
        MultilineInput input = new MultilineInput(in);
        boolean waitForInput = interactiveFirst;
        while (true) {
            if (waitForInput && inputRound(input) == Input.EXIT) {
                return;
            }
            if (context.isEmpty()) {
                waitForInput = true;
                continue;
            }
            waitForInput = interactive;
            GenerationResult result = generate();
            if (!continueAfter(result)) {
                return;
            }
        }
    }

    /** The prompt's tokens with the vocabulary's BOS rule, a lone BOS for an empty prompt (completion.cpp:281-300). */
    private void startContext() {
        appendTokens(tokenizer.encode(prompt, true, true));
        if (context.isEmpty() && tokenizer.bosId() >= 0) {
            context.add(tokenizer.bosId());
        }
        if (context.size() > capacity - CONTEXT_MARGIN) {
            throw new IllegalArgumentException("the prompt is too long (" + context.size() + " tokens, max "
                    + (capacity - CONTEXT_MARGIN) + ")");
        }
    }

    private void printIntro() {
        out.println("== Running in interactive mode. ==");
        out.println(" - Press Ctrl+C to interject at any time.");
        if (multiline) {
            out.println(" - To return control to the AI, end your input with '\\'.");
            out.println(" - To return control without starting a new line, end your input with '/'.");
        } else {
            out.println(" - Press Return to return control to the AI.");
            out.println(" - To return control without starting a new line, end your input with '/'.");
            out.println(" - If you want to submit another line, end your input with '\\'.");
        }
        out.println();
    }

    /**
     * One user input (completion.cpp:803-905): the BOS of {@code --in-prefix-bos}, the shown prefix, the input,
     * the shown suffix. Enter on an empty line passes control back without adding tokens.
     */
    private Input inputRound(MultilineInput input) throws IOException {
        if (inPrefixBos && tokenizer.bosId() >= 0) {
            context.add(tokenizer.bosId());
        }
        out.print(inPrefix);
        out.flush();
        String buffer;
        interrupts.awaitingInput(true);
        try {
            buffer = input.read(multiline);
        } finally {
            interrupts.awaitingInput(false);
        }
        if (interrupts.requested()) {
            interrupts.clear();
            return Input.EXIT;
        }
        if (buffer == null) {
            out.println("EOF by user");
            return Input.EXIT;
        }
        if (buffer.isEmpty()) {
            return Input.GENERATE;
        }
        out.print(inSuffix);
        out.flush();
        String text = escaped(buffer);
        appendTokens(tokenizer.encode(inPrefix, false, true));
        appendTokens(tokenizer.encode(text, false, false));
        appendTokens(tokenizer.encode(inSuffix, false, true));
        if (transcript != null) {
            transcript.userTurn(text);
        }
        return Input.GENERATE;
    }

    /**
     * Generates over the whole context and appends the generated tokens (a matched reverse prompt's tokens among
     * them, its text shown after the output).
     */
    private GenerationResult generate() throws IOException {
        shiftContext();
        int[] tokens = new int[context.size()];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = context.get(i);
        }
        ChannelPrinter printer = new ChannelPrinter(out, showAllChannels, false);
        GenerationRequest turn = request.toBuilder().cancellation(interrupts::requested).build();
        GenerationResult result;
        try {
            result = generator.generate(tokens, turn, printer, null);
        } finally {
            interrupts.clear();
        }
        appendTokens(result.tokens());
        if (result.stopReason() == StopReason.STOP_STRING) {
            out.print(result.stopString());
        }
        out.flush();
        if (transcript != null) {
            transcript.assistant(printer.content(), printer.reasoning());
        }
        return result;
    }

    /** What the end of one generation means: another input round (true) or the end of the session (false). */
    private boolean continueAfter(GenerationResult result) {
        StopReason reason = result.stopReason();
        if (!interactive) {
            out.println(reason == StopReason.EOS ? " [end of text]" : "");
            return false;
        }
        if (reason == StopReason.EOS) {
            if (!antiprompts.isEmpty()) {
                String first = antiprompts.get(0);
                appendTokens(tokenizer.encode(first, false, true));
                out.print(first);
            }
            out.println();
        } else if (reason == StopReason.CONTEXT_FULL) {
            out.println();
            out.println("[context full: the older half of the context shifts out before the next generation]");
        } else if (reason == StopReason.CANCELLED) {
            out.println();
        }
        out.flush();
        return true;
    }

    /**
     * Shifts the older half of the context after a leading BOS out until {@value #CONTEXT_MARGIN} positions are
     * free, as completion.cpp's context shift with the default {@code n_keep} discards {@code (n_past - n_keep)/2}.
     */
    private void shiftContext() {
        int keep = !context.isEmpty() && tokenizer.bosId() >= 0 && context.get(0) == tokenizer.bosId() ? 1 : 0;
        int dropped = 0;
        while (context.size() > capacity - CONTEXT_MARGIN) {
            int discard = Math.max(1, (context.size() - keep) / 2);
            context.subList(keep, keep + discard).clear();
            dropped += discard;
        }
        if (dropped > 0) {
            out.println();
            out.println("[context shift: " + dropped + " tokens dropped]");
        }
    }

    private void appendTokens(int[] tokens) {
        for (int t : tokens) {
            context.add(t);
        }
    }

    /** {@code text} with its escapes processed, unless {@code --no-escape}. */
    private String escaped(String text) {
        return escape ? processEscapes(text) : text;
    }

    /**
     * The escapes of llama.cpp {@code string_process_escapes}: {@code \n \r \t \' \" \\} and {@code \xHH};
     * an unknown escape stays as written.
     */
    static String processEscapes(String input) {
        StringBuilder sb = new StringBuilder(input.length());
        for (int i = 0; i < input.length(); i++) {
            char c = input.charAt(i);
            if (c != '\\' || i + 1 >= input.length()) {
                sb.append(c);
                continue;
            }
            char e = input.charAt(++i);
            switch (e) {
                case 'n': sb.append('\n'); break;
                case 'r': sb.append('\r'); break;
                case 't': sb.append('\t'); break;
                case '\'': sb.append('\''); break;
                case '"': sb.append('"'); break;
                case '\\': sb.append('\\'); break;
                case 'x':
                    int value = i + 2 < input.length() ? hexPair(input.charAt(i + 1), input.charAt(i + 2)) : -1;
                    if (value >= 0) {
                        sb.append((char) value);
                        i += 2;
                    } else {
                        sb.append('\\').append('x');
                    }
                    break;
                default:
                    sb.append('\\').append(e);
            }
        }
        return sb.toString();
    }

    /** The value of two hex digits, or -1 when either is not one. */
    private static int hexPair(char high, char low) {
        int h = Character.digit(high, 16);
        int l = Character.digit(low, 16);
        return h >= 0 && l >= 0 ? h * 16 + l : -1;
    }
}
