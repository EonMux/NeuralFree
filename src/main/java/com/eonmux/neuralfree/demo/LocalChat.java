package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.LLMModel;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

/**
 * The {@link ChatBackend} of a locally loaded model: each turn renders the conversation through the model's chat
 * template, generates with the {@link TurnGenerator} (Ctrl-C cancels through the request's cancellation hook) and
 * streams the routed channels through a {@link ChannelPrinter}; the prompt renders with the {@code enable_thinking}
 * of {@code --thinking} or {@code --reasoning}, then the {@code --chat-template-kwargs}, {@code --reasoning-effort} and
 * {@code --reasoning-preserve} variables ({@link #chatRequest}), and the request carries those template variables
 * ({@link #templated}). When the rendered
 * prompt fills the KV cache the oldest exchanges after the system message are left out of that turn's prompt. Media
 * files are refused: the text chat path has no media pipeline.
 */
final class LocalChat implements ChatBackend {

    /** The template variable of {@code --reasoning-effort}. */
    static final String REASONING_EFFORT = "reasoning_effort";

    private static final int STATS_INPUT_CHARS = 50;
    private static final int STATS_OUTPUT_CHARS = 100;

    private final LLMModel model;
    private final TurnGenerator generator;
    private final GenerationRequest request;
    /** The {@code enable_thinking} switch, null for the template's own default. */
    private final Boolean thinking;
    private final Map<String, Object> templateKwargs;
    private final boolean debugPrompt;
    private final boolean showAllChannels;
    private final PrintStream out;
    private final Consumer<String> stats;

    /**
     * A backend over {@code model} with the generation settings of {@code opts}.
     *
     * @param stats receives each turn's statistics lines ({@code --stats-file})
     */
    LocalChat(LLMModel model, TurnGenerator generator, LLMDemoOptions opts, PrintStream out,
              Consumer<String> stats) {
        if (model == null || generator == null || opts == null || out == null || stats == null) {
            throw new IllegalArgumentException("model, generator, options, output and stats sink are required");
        }
        this.model = model;
        this.generator = generator;
        this.request = opts.generation;
        this.thinking = opts.thinking;
        this.templateKwargs = opts.templateKwargs;
        this.debugPrompt = opts.debugPrompt;
        this.showAllChannels = opts.showAllChannels;
        this.out = out;
        this.stats = stats;
    }

    /** Runs the console chat of {@code model} on the standard streams and the process interrupt state. */
    static void run(LLMModel model, TurnGenerator generator, LLMDemoOptions opts, Consumer<String> stats)
            throws IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        Transcript transcript = opts.outFile == null ? null : Transcript.open(opts.outFile);
        try {
            LocalChat backend = new LocalChat(model, generator, opts, System.out, stats);
            new ChatSession(backend, opts, in, System.out, transcript, Interrupts.system()).run();
        } finally {
            if (transcript != null) {
                transcript.close();
            }
        }
    }

    @Override
    public void printBanner(PrintStream banner) {
        GenerationRequest shown = generator.effective(request);
        SamplerConfig sampler = shown.sampler();
        banner.println("model      : " + model.architecture());
        banner.println("modalities : text");
        banner.println(String.format(Locale.ROOT,
                "settings   : temperature=%.2f, top_p=%.2f, top_k=%d, min_p=%.3f, typical_p=%.2f, max_tokens=%d,"
                        + " repeat_penalty=%.2f, presence_penalty=%.2f, frequency_penalty=%.2f, repeat_last_n=%d,"
                        + " seed=%d%s%s",
                sampler.temperature(), sampler.topP(), sampler.topK(), sampler.minP(), sampler.typicalP(),
                shown.maxNewTokens(), sampler.repeatPenalty(), sampler.presencePenalty(),
                sampler.frequencyPenalty(), sampler.penaltyLastN(), sampler.seed(),
                shown.ignoreEos() ? ", ignore_eos" : "",
                shown.stopStrings().isEmpty() ? "" : ", stop=" + shown.stopStrings()));
    }

    @Override
    public String mediaRefusal(String kind) {
        return "the local chat has no " + kind + " pipeline; chat with a server that serves " + kind
                + " input (--server-base)";
    }

    @Override
    public ChatAnswer answer(List<Map<String, Object>> messages, BooleanSupplier cancelled) throws IOException {
        GGUFTokenizer tokenizer = model.tokenizer();
        List<Map<String, String>> fitted = textMessages(messages);
        String fullPrompt;
        int[] promptTokens;
        int capacity = model.kvCacheContext();
        int dropped = 0;
        try {
            fullPrompt = tokenizer.formatChat(chatRequest(fitted, thinking, templateKwargs));
            promptTokens = tokenizer.encode(fullPrompt);
            while (promptTokens.length >= capacity && dropOldestExchange(fitted)) {
                dropped++;
                fullPrompt = tokenizer.formatChat(chatRequest(fitted, thinking, templateKwargs));
                promptTokens = tokenizer.encode(fullPrompt);
            }
        } catch (IllegalArgumentException | UnsupportedOperationException e) {
            return ChatAnswer.failed("", "", "the chat template cannot render the conversation: " + e.getMessage());
        }
        if (promptTokens.length >= capacity) {
            return ChatAnswer.failed("", "", "the prompt of " + promptTokens.length + " tokens fills the KV cache of "
                    + capacity + " positions even without the earlier turns");
        }
        if (dropped > 0) {
            out.println("[context full: the oldest " + dropped + " exchange(s) are left out of this turn]");
        }
        if (debugPrompt) {
            PromptDebug.print(out, fullPrompt, promptTokens, tokenizer);
        }
        ChannelPrinter printer = new ChannelPrinter(out, showAllChannels, debugPrompt);
        GenerationRequest turn = templated(request, thinking, templateKwargs).toBuilder().cancellation(cancelled)
                .build();
        GenerationResult result = generator.generate(promptTokens, turn, printer, fullPrompt);
        out.println();
        if (result.stopReason() == StopReason.CANCELLED) {
            out.println("[interrupted]");
        }
        logTurn(promptTokens.length, result, fitted.get(fitted.size() - 1).get("content"), printer.content());
        return ChatAnswer.of(printer.content(), printer.reasoning());
    }

    @Override
    public void reset() {
        generator.reset();
    }

    /**
     * The chat request the console renders {@code messages} with: the {@code enable_thinking} of {@code --thinking}
     * or {@code --reasoning}, unset when {@code thinking} is null ({@code --reasoning auto}), and the template
     * variables ({@code --chat-template-kwargs}, {@code --reasoning-effort}, {@code --reasoning-preserve}) bound after
     * it, so a kwarg of the same name wins as in llama.cpp. A {@code reasoning_effort} variable is also the request's
     * reasoning effort, which the template binds with its aliases ({@code TemplateCaps.applyReasoningEffort}) as
     * llama.cpp binds a {@code reasoning_effort} kwarg.
     *
     * @throws IllegalArgumentException for a malformed message or kwarg value
     */
    static ChatRequest chatRequest(List<? extends Map<String, ?>> messages, Boolean thinking,
                                   Map<String, ?> templateKwargs) {
        Object effort = templateKwargs.get(REASONING_EFFORT);
        return ChatRequest.builder().messages(messages).enableThinking(thinking).templateKwargs(templateKwargs)
                .reasoningEffort(effort instanceof String && !((String) effort).isEmpty() ? (String) effort : null)
                .build();
    }

    /**
     * The single-turn prompt of {@code -p}: a system message when {@code systemPrompt} is non-empty, then the user
     * message, rendered as {@link #chatRequest} renders a conversation.
     */
    static String singleTurnPrompt(GGUFTokenizer tokenizer, String systemPrompt, String userMessage,
                                   Boolean thinking, Map<String, ?> templateKwargs) {
        List<Map<String, String>> messages = new ArrayList<>(2);
        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            messages.add(textMessage("system", systemPrompt));
        }
        messages.add(textMessage("user", userMessage));
        return tokenizer.formatChat(chatRequest(messages, thinking, templateKwargs));
    }

    private static Map<String, String> textMessage(String role, String content) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    /**
     * {@code request} carrying the chat template variables the console renders its prompts with
     * ({@link #chatRequest}): the {@code enable_thinking} of {@code --thinking} or {@code --reasoning} (none when
     * {@code thinking} is null), then the {@code --chat-template-kwargs}, {@code --reasoning-effort} and
     * {@code --reasoning-preserve} variables, so the reasoning parser selects its configuration as the server does
     * for the same request. The request's other variables stay.
     */
    static GenerationRequest templated(GenerationRequest request, Boolean thinking, Map<String, ?> templateKwargs) {
        Map<String, Object> variables = new LinkedHashMap<>(request.templateVariables());
        if (thinking != null) {
            variables.put("enable_thinking", thinking);
        }
        variables.putAll(templateKwargs);
        return request.toBuilder().templateVariables(variables).build();
    }

    /**
     * The messages as the template's string maps. Only text content reaches here, since {@link #mediaRefusal}
     * refuses every media kind; a list content is refused by name.
     */
    private static List<Map<String, String>> textMessages(List<Map<String, Object>> messages) {
        List<Map<String, String>> out = new ArrayList<>(messages.size());
        for (Map<String, Object> m : messages) {
            Map<String, String> copy = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : m.entrySet()) {
                if (!(e.getValue() instanceof String)) {
                    throw new IllegalArgumentException("message field '" + e.getKey()
                            + "' is not text; the local chat takes text messages only");
                }
                copy.put(e.getKey(), (String) e.getValue());
            }
            out.add(copy);
        }
        return out;
    }

    /**
     * Removes the oldest user/assistant exchange after any system message, keeping the final user message.
     * Returns false when nothing but the system message and the final user message is left.
     */
    private static boolean dropOldestExchange(List<Map<String, String>> messages) {
        int first = !messages.isEmpty() && "system".equals(messages.get(0).get("role")) ? 1 : 0;
        if (messages.size() - first <= 1) {
            return false;
        }
        int count = Math.min(2, messages.size() - first - 1);
        for (int i = 0; i < count; i++) {
            messages.remove(first);
        }
        return true;
    }

    /** Prints the turn's statistics line and hands it, with the clipped input and output, to the stats sink. */
    private void logTurn(int promptTokens, GenerationResult result, String input, String response) {
        String line = statLine(promptTokens, result);
        out.println("[" + line + "]");
        stats.accept(line);
        stats.accept("  Input: " + clip(input, STATS_INPUT_CHARS));
        stats.accept("  Output: " + clip(response, STATS_OUTPUT_CHARS));
        stats.accept("");
    }

    private static String clip(String text, int max) {
        return text.length() > max ? text.substring(0, max) + "..." : text;
    }

    /** Prompt size, generated tokens, stop reason and the prefill / decode rates of one generation. */
    static String statLine(int promptTokens, GenerationResult result) {
        GenerationResult.Timings t = result.timings();
        double totalSec = t.totalNanos() / 1e9;
        double totalSpeed = totalSec > 0 ? result.tokenCount() / totalSec : 0;
        if (!t.split()) {
            return String.format(Locale.ROOT,
                    "Prompt: %d tokens | Generated: %d tokens (%s) | Total: %.2fs (%.2f tok/s)",
                    promptTokens, result.tokenCount(), result.stopReason(), totalSec, totalSpeed);
        }
        return String.format(Locale.ROOT,
                "Prompt: %d tokens (%d evaluated) | Generated: %d tokens (%s) | Prefill: %.2fs (%.2f tok/s)"
                        + " | Decode: %.2fs (%.2f tok/s) | Total: %.2fs (%.2f tok/s)",
                promptTokens, t.promptTokens(), result.tokenCount(), result.stopReason(), t.promptNanos() / 1e9,
                rate(t.promptTokensPerSecond()), t.generationNanos() / 1e9, rate(t.generationTokensPerSecond()),
                totalSec, totalSpeed);
    }

    private static double rate(double r) {
        return Double.isNaN(r) ? 0 : r;
    }
}
