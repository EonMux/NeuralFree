package com.eonmux.neuralfree.gguf.chat.autoparser;

import com.eonmux.neuralfree.gguf.MiniJinja;
import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.chat.TemplateCaps;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * One compiled chat template plus the probe context {@link AutoParserAnalyzer} renders variants with: llama.cpp's
 * {@code apply_template} over {@code template_params} ({@code common/chat-auto-parser-helpers.cpp}).
 *
 * <p>A render binds the same variables the serving formatter does ({@link com.eonmux.neuralfree.gguf.chat.ChatFormatter}):
 * the probe messages in the content and argument forms the template reads ({@link ChatMessages#prepare}), the
 * offered tools (an empty list when none), empty BOS and EOS texts, the generation-prompt and thinking switches,
 * the reasoning-history aliases and the local date. A template that raises yields {@code null}, which the analyzer
 * treats as a failed comparison. Immutable and thread-safe after construction.
 */
final class TemplateProbe {

    private static final Logger LOGGER = LogManager.getLogger(TemplateProbe.class);
    private static final DateTimeFormatter DATETIME_FORMAT = DateTimeFormatter.ofPattern("MMM dd yyyy", Locale.ENGLISH);
    private static final DateTimeFormatter DATE_STRING_FORMAT = DateTimeFormatter.ofPattern("dd MMM yyyy", Locale.ENGLISH);

    private final String source;
    private final MiniJinja.Template template;
    private final TemplateCaps caps;

    private TemplateProbe(String source, MiniJinja.Template template, TemplateCaps caps) {
        this.source = source;
        this.template = template;
        this.caps = caps;
    }

    /**
     * Compiles {@code source} and probes its capabilities.
     *
     * @throws IllegalArgumentException when the template does not parse
     */
    static TemplateProbe of(String source) {
        if (source == null || source.isEmpty()) {
            throw new IllegalArgumentException("chat template is null or empty");
        }
        MiniJinja.Template compiled = MiniJinja.compile(source);
        return new TemplateProbe(source, compiled, TemplateCaps.probe(compiled));
    }

    /** The template source. */
    String source() {
        return source;
    }

    /** The template's capabilities. */
    TemplateCaps caps() {
        return caps;
    }

    /**
     * Renders one variant, or returns null when the template raises.
     *
     * @param messages            probe messages in the OpenAI shape
     * @param tools               probe tools in the template shape, or null for none
     * @param addGenerationPrompt whether the template appends its generation prompt
     * @param enableThinking      the {@code enable_thinking} variable
     */
    String render(List<Map<String, Object>> messages, List<Map<String, Object>> tools, boolean addGenerationPrompt,
                  boolean enableThinking) {
        Map<String, Object> context = new LinkedHashMap<>();
        context.put("messages", ChatMessages.prepare(new ArrayList<>(messages), source, caps));
        context.put("add_generation_prompt", addGenerationPrompt);
        context.put("bos_token", "");
        context.put("eos_token", "");
        context.put("tools", tools == null ? Collections.<Map<String, Object>>emptyList() : tools);
        context.put("enable_thinking", enableThinking);
        TemplateCaps.applyPreserveReasoning(context, true);
        LocalDate today = LocalDate.now();
        context.put("datetime", DATETIME_FORMAT.format(today));
        context.put("date_string", DATE_STRING_FORMAT.format(today));
        try {
            return template.render(context);
        } catch (RuntimeException e) {
            LOGGER.debug("Autoparser probe render failed: {}", e.getMessage());
            return null;
        }
    }

    /**
     * The two renders of a differential comparison, or null when either template application fails:
     * variant A of {@code messagesA}, then variant B of {@code messagesB}. The other inputs are shared.
     */
    VariantPair compare(List<Map<String, Object>> messagesA, List<Map<String, Object>> messagesB,
                        List<Map<String, Object>> tools, boolean addGenerationPrompt, boolean enableThinking) {
        String a = render(messagesA, tools, addGenerationPrompt, enableThinking);
        String b = render(messagesB, tools, addGenerationPrompt, enableThinking);
        if (a == null || b == null) {
            return null;
        }
        return new VariantPair(a, b);
    }

    /** Two successful renders and the diff between them. Immutable. */
    static final class VariantPair {
        private final String left;
        private final String right;
        private final MarkerDiff.Diff diff;

        VariantPair(String left, String right) {
            this.left = left;
            this.right = right;
            this.diff = MarkerDiff.split(left, right);
        }

        /** Variant A's output. */
        String outputA() {
            return left;
        }

        /** Variant B's output. */
        String outputB() {
            return right;
        }

        /** The diff from A to B. */
        MarkerDiff.Diff diff() {
            return diff;
        }
    }
}
