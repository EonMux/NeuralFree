package com.eonmux.neuralfree.demo;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.BooleanSupplier;

/**
 * The {@link ChatBackend} of client mode ({@code --server-base}, llama.cpp tools/cli/cli-context.cpp over
 * cli-client.cpp): every answer streams from a running server's {@code /v1/chat/completions}, with the reasoning
 * channel shown and the server's timings printed. The request carries the conversation, {@code stream} and
 * {@code timings_per_token} as the reference sends them, plus the generation settings given on the command line or
 * through properties (the sampler settings, seed, max tokens, stop strings, ignore-eos, logit biases and the
 * grammar or JSON-schema constraint; {@link #settingsFields}), so no given flag is dropped. Media files are staged
 * when the server's {@code /props} reports the modality.
 */
final class RemoteChat implements ChatBackend {

    private final ChatClient client;
    private final String model;
    private final ChatClient.Props props;
    /** The request fields of the generation settings ({@link #settingsFields}). */
    private final Map<String, Object> settings;
    private final boolean showReasoning;
    private final PrintStream out;

    /** A backend over {@code client} for {@code model}, with the generation settings of {@code opts}. */
    RemoteChat(ChatClient client, String model, ChatClient.Props props, LLMDemoOptions opts, PrintStream out) {
        if (client == null || model == null || props == null || opts == null || out == null) {
            throw new IllegalArgumentException("client, model, props, options and output are required");
        }
        this.client = client;
        this.model = model;
        this.props = props;
        this.settings = settingsFields(opts.generation, opts.thinking, opts.templateKwargs);
        this.showReasoning = opts.showAllChannels;
        this.out = out;
    }

    /** Runs client mode on the standard streams and the process interrupt state; returns the exit code. */
    static int run(LLMDemoOptions opts) throws IOException, InterruptedException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        return run(opts, in, System.out, Interrupts.system());
    }

    /**
     * Connects to {@code opts.serverBase}, waits for its health, selects the model (the only one, else the
     * number the user picks), reads the properties and runs the chat. Returns 0 after the chat, 1 when the server
     * cannot be reached or offers no model.
     */
    static int run(LLMDemoOptions opts, BufferedReader in, PrintStream out, Interrupts interrupts)
            throws IOException, InterruptedException {
        ChatClient client = new ChatClient(opts.serverBase, apiKey(opts));
        out.println("Connecting to server at " + client.base());
        String healthError = client.waitHealth(interrupts::requested);
        if (healthError != null) {
            interrupts.clear();
            out.println("Error: " + healthError);
            return 1;
        }
        List<ChatClient.ModelInfo> models;
        try {
            models = client.models();
        } catch (IOException e) {
            out.println("Error: " + e.getMessage());
            return 1;
        }
        String model;
        interrupts.awaitingInput(true);
        try {
            model = selectModel(models, in, out);
        } finally {
            interrupts.awaitingInput(false);
        }
        if (model == null) {
            return 1;
        }
        ChatClient.Props props;
        try {
            props = client.props(model);
        } catch (IOException e) {
            props = ChatClient.Props.NONE;
        }
        Transcript transcript = opts.outFile == null ? null : Transcript.open(opts.outFile);
        try {
            RemoteChat backend = new RemoteChat(client, model, props, opts, out);
            new ChatSession(backend, opts, in, out, transcript, interrupts).run();
        } finally {
            if (transcript != null) {
                transcript.close();
            }
        }
        return 0;
    }

    /** The {@code --api-key}, else the {@code NEURALFREE_API_KEY} environment value, or null. */
    static String apiKey(LLMDemoOptions opts) {
        if (opts.apiKey != null && !opts.apiKey.isEmpty()) {
            return opts.apiKey;
        }
        String env = System.getenv("NEURALFREE_API_KEY");
        return env == null || env.isEmpty() ? null : env;
    }

    /**
     * The model of the session (cli-context.cpp list_and_ask_models): the only one listed, or the one the user
     * picks by number; null when the list is empty or the user gives no answer.
     */
    static String selectModel(List<ChatClient.ModelInfo> models, BufferedReader in, PrintStream out)
            throws IOException {
        if (models.isEmpty()) {
            out.println("Error: the server lists no models on /v1/models");
            return null;
        }
        if (models.size() == 1) {
            return models.get(0).id;
        }
        out.println("\nAvailable models:");
        for (int i = 0; i < models.size(); i++) {
            out.println("  " + (i + 1) + ". " + models.get(i).display());
        }
        out.println();
        while (true) {
            out.print("Select model by number: ");
            out.flush();
            String line = in.readLine();
            if (line == null || line.trim().isEmpty()) {
                return null;
            }
            try {
                int index = Integer.parseInt(line.trim());
                if (index >= 1 && index <= models.size()) {
                    String model = models.get(index - 1).id;
                    out.println("Selected model: " + model);
                    return model;
                }
            } catch (NumberFormatException e) {
                // not a number: the error below asks again
            }
            out.println("Error: Invalid selection. Please enter a valid number.");
        }
    }

    @Override
    public void printBanner(PrintStream banner) {
        banner.println("server     : " + client.base());
        banner.println("model      : " + displayName());
        StringBuilder modalities = new StringBuilder("text");
        if (props.vision) {
            modalities.append(", vision");
        }
        if (props.audio) {
            modalities.append(", audio");
        }
        if (props.video) {
            modalities.append(", video");
        }
        banner.println("modalities : " + modalities);
    }

    /** The model alias of {@code /props}, else its model file name, else the selected model id. */
    private String displayName() {
        if (!props.modelAlias.isEmpty()) {
            return props.modelAlias;
        }
        if (!props.modelPath.isEmpty()) {
            return Paths.get(props.modelPath).getFileName().toString();
        }
        return model;
    }

    @Override
    public String mediaRefusal(String kind) {
        boolean supported = kind.equals("image") ? props.vision : kind.equals("audio") ? props.audio
                : kind.equals("video") && props.video;
        if (supported) {
            return null;
        }
        String modality = kind.equals("image") ? "vision" : kind;
        return "the server does not report " + modality + " input in /props";
    }

    @Override
    public ChatAnswer answer(List<Map<String, Object>> messages, BooleanSupplier cancelled) throws IOException {
        StreamPrinter printer = new StreamPrinter(out, showReasoning);
        String error;
        try {
            error = client.streamChat(requestBody(messages), printer, cancelled);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            error = "interrupted while waiting for the server";
        }
        out.println();
        if (!error.isEmpty()) {
            return ChatAnswer.failed(printer.content(), printer.reasoning(), error);
        }
        if (cancelled.getAsBoolean()) {
            out.println("[interrupted]");
        }
        if (printer.sawTimings) {
            out.println(String.format(Locale.ROOT, "[ Prompt: %.1f t/s | Generation: %.1f t/s ]",
                    printer.promptPerSecond, printer.predictedPerSecond));
        }
        return ChatAnswer.of(printer.content(), printer.reasoning());
    }

    @Override
    public void reset() {
        // the server keeps no per-conversation state between requests
    }

    /** The chat request body: the reference fields, the model, then the given generation settings. */
    Map<String, Object> requestBody(List<Map<String, Object>> messages) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("messages", messages);
        body.put("stream", true);
        body.put("timings_per_token", true);
        if (!model.isEmpty()) {
            body.put("model", model);
        }
        body.putAll(settings);
        return body;
    }

    /**
     * The request fields of the generation settings given on the command line or through properties, in the
     * llama.cpp server's names: the sampler settings ({@link #samplerFields}), the seed, max tokens, stop strings,
     * ignore-eos, the logit biases, the grammar constraint, the reasoning budget, and the
     * {@code chat_template_kwargs}: {@code enable_thinking} when {@code thinking} is true, then the
     * {@code --chat-template-kwargs}, {@code --reasoning}, {@code --reasoning-effort} and {@code --reasoning-preserve}
     * variables over it ({@code templateKwargs}). Settings nobody gave stay the server's.
     *
     * @throws IllegalArgumentException when the constraint has no request field or its schema is not JSON
     */
    static Map<String, Object> settingsFields(GenerationRequest generation, Boolean thinking,
                                              Map<String, Object> templateKwargs) {
        Map<String, Object> f = new LinkedHashMap<>();
        SamplerConfig s = generation.sampler();
        samplerFields(f, s);
        if (s.seeded()) {
            f.put("seed", s.seed());
        }
        if (generation.maxNewTokensExplicit()) {
            f.put("max_tokens", generation.maxNewTokens());
        }
        if (!generation.stopStrings().isEmpty()) {
            f.put("stop", generation.stopStrings());
        }
        if (generation.ignoreEos()) {
            f.put("ignore_eos", true);
        }
        if (!generation.logitBias().isEmpty()) {
            Map<String, Object> bias = new LinkedHashMap<>();
            for (Map.Entry<Integer, Float> e : generation.logitBias().entrySet()) {
                bias.put(Integer.toString(e.getKey()), e.getValue());
            }
            f.put("logit_bias", bias);
        }
        if (generation.constraint() != null) {
            constraintField(f, generation.constraint());
        }
        if (generation.reasoningBudget() != GenerationRequest.REASONING_BUDGET_DISABLED) {
            f.put("reasoning_budget_tokens", generation.reasoningBudget());
        }
        String budgetMessage = generation.reasoningBudgetMessage();
        if (budgetMessage != null && !budgetMessage.isEmpty()) {
            f.put("reasoning_budget_message", budgetMessage);
        }
        Map<String, Object> kwargs = new LinkedHashMap<>();
        if (Boolean.TRUE.equals(thinking)) {
            kwargs.put("enable_thinking", true);
        }
        kwargs.putAll(templateKwargs);
        if (!kwargs.isEmpty()) {
            f.put("chat_template_kwargs", kwargs);
        }
        return Collections.unmodifiableMap(f);
    }

    /**
     * The sampler settings: those the config tracks as explicit when they are, every other one when it differs
     * from the engine default ({@link SamplerConfig#defaults()}, llama.cpp's), so a given flag is never dropped.
     */
    private static void samplerFields(Map<String, Object> f, SamplerConfig s) {
        putIfExplicit(f, s, SamplerConfig.Setting.TEMPERATURE, "temperature", s.temperature());
        putIfExplicit(f, s, SamplerConfig.Setting.TOP_K, "top_k", s.topK());
        putIfExplicit(f, s, SamplerConfig.Setting.TOP_P, "top_p", s.topP());
        putIfExplicit(f, s, SamplerConfig.Setting.MIN_P, "min_p", s.minP());
        putIfExplicit(f, s, SamplerConfig.Setting.TYPICAL_P, "typical_p", s.typicalP());
        putIfExplicit(f, s, SamplerConfig.Setting.REPEAT_PENALTY, "repeat_penalty", s.repeatPenalty());
        putIfExplicit(f, s, SamplerConfig.Setting.PENALTY_LAST_N, "repeat_last_n", s.penaltyLastN());
        putIfExplicit(f, s, SamplerConfig.Setting.PRESENCE_PENALTY, "presence_penalty", s.presencePenalty());
        putIfExplicit(f, s, SamplerConfig.Setting.FREQUENCY_PENALTY, "frequency_penalty", s.frequencyPenalty());
        putIfExplicit(f, s, SamplerConfig.Setting.MIROSTAT, "mirostat", s.mirostat());
        putIfExplicit(f, s, SamplerConfig.Setting.MIROSTAT_TAU, "mirostat_tau", s.mirostatTau());
        putIfExplicit(f, s, SamplerConfig.Setting.MIROSTAT_ETA, "mirostat_eta", s.mirostatEta());
        putIfExplicit(f, s, SamplerConfig.Setting.XTC_PROBABILITY, "xtc_probability", s.xtcProbability());
        putIfExplicit(f, s, SamplerConfig.Setting.XTC_THRESHOLD, "xtc_threshold", s.xtcThreshold());
        SamplerConfig d = SamplerConfig.defaults();
        putIfChanged(f, "min_keep", s.minKeep(), d.minKeep());
        putIfChanged(f, "top_n_sigma", s.topNSigma(), d.topNSigma());
        putIfChanged(f, "dynatemp_range", s.dynatempRange(), d.dynatempRange());
        putIfChanged(f, "dynatemp_exponent", s.dynatempExponent(), d.dynatempExponent());
        putIfChanged(f, "dry_multiplier", s.dryMultiplier(), d.dryMultiplier());
        putIfChanged(f, "dry_base", s.dryBase(), d.dryBase());
        putIfChanged(f, "dry_allowed_length", s.dryAllowedLength(), d.dryAllowedLength());
        putIfChanged(f, "dry_penalty_last_n", s.dryPenaltyLastN(), d.dryPenaltyLastN());
        putIfChanged(f, "dry_sequence_breakers", s.drySequenceBreakers(), d.drySequenceBreakers());
        putIfChanged(f, "adaptive_target", s.adaptiveTarget(), d.adaptiveTarget());
        putIfChanged(f, "adaptive_decay", s.adaptiveDecay(), d.adaptiveDecay());
        if (!s.order().equals(d.order())) {
            List<String> names = new ArrayList<>(s.order().size());
            for (StageKind kind : s.order()) {
                names.add(kind.label());
            }
            f.put("samplers", names);
        }
    }

    /**
     * The constraint's request field: {@code grammar} for GBNF or Lark text (the server tells them apart as the
     * CLI does), {@code json_schema} (as a JSON value) for a schema; the CLI gives no other kind, and any other is
     * refused by name.
     */
    private static void constraintField(Map<String, Object> f, ConstraintSpec c) {
        switch (c.kind()) {
            case GBNF:
            case LARK:
                f.put("grammar", c.text());
                return;
            case JSON_SCHEMA:
                try {
                    f.put("json_schema", Json.parse(c.text()));
                } catch (JsonException e) {
                    throw new IllegalArgumentException("--json-schema is not valid JSON: " + e.getMessage(), e);
                }
                return;
            default:
                throw new IllegalArgumentException("client mode sends grammar and JSON-schema constraints only, not "
                        + c.kind().name().toLowerCase(Locale.ROOT));
        }
    }

    private static void putIfExplicit(Map<String, Object> f, SamplerConfig s, SamplerConfig.Setting setting,
                                      String field, Number value) {
        if (s.isExplicit(setting)) {
            f.put(field, value);
        }
    }

    private static void putIfChanged(Map<String, Object> f, String field, Object value, Object engineDefault) {
        if (!Objects.equals(value, engineDefault)) {
            f.put(field, value);
        }
    }

    /**
     * Prints the streamed deltas, the reasoning under a header at each channel change unless hidden
     * ({@code --content-only}), and collects the turn's text.
     */
    private static final class StreamPrinter implements ChatClient.Deltas {
        private final PrintStream out;
        private final boolean showReasoning;
        private final StringBuilder content = new StringBuilder();
        private final StringBuilder reasoning = new StringBuilder();
        private boolean inReasoning;
        private boolean sawTimings;
        private double promptPerSecond;
        private double predictedPerSecond;

        StreamPrinter(PrintStream out, boolean showReasoning) {
            this.out = out;
            this.showReasoning = showReasoning;
        }

        @Override
        public void reasoning(String text) {
            reasoning.append(text);
            if (!showReasoning) {
                return;
            }
            if (!inReasoning) {
                out.print("[reasoning]\n");
                inReasoning = true;
            }
            out.print(text);
            out.flush();
        }

        @Override
        public void content(String text) {
            if (inReasoning) {
                out.print("\n[content]\n");
                inReasoning = false;
            }
            content.append(text);
            out.print(text);
            out.flush();
        }

        @Override
        public void timings(double prompt, double predicted) {
            sawTimings = true;
            promptPerSecond = prompt;
            predictedPerSecond = predicted;
        }

        String content() {
            return content.toString();
        }

        String reasoning() {
            return reasoning.toString();
        }
    }
}
