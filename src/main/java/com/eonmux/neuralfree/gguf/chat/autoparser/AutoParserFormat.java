package com.eonmux.neuralfree.gguf.chat.autoparser;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParser;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParserRegistry;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GrammarTriggers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The {@code autoparser} tool parser: a template with no hand-written parser gets its tool-call syntax from
 * differential analysis ({@link AutoParserAnalyzer}), reads it with the derived PEG ({@link AutoParserToolParser})
 * and is constrained by the same PEG's grammar ({@link AutoParserPeg}). It is llama.cpp's fallback for every
 * template its {@code common_chat_template_detect_params_init} does not recognize
 * ({@code common/chat-auto-parser-generator.cpp}), exposed here as a named parser and as the {@code auto} format's
 * last resort.
 *
 * <p>An analysis is derived once per template source and kept (failures too, so a bad template is not re-probed on
 * every request). A parsed request is a {@link ToolParser}; the guiding grammar is the forcing form for
 * {@code tool_choice: required} or a named function, and the lazy form behind the format's call opener for
 * {@code tool_choice: auto}. The reasoning the analysis found is available to the reasoning-parser selection
 * ({@link #reasoningFilter}). Immutable; the cache is synchronized.
 */
public final class AutoParserFormat implements ToolParser.Factory {

    /** The registry name and the name {@code --tool-call-parser} selects. */
    public static final String NAME = "autoparser";


    /** The analysis this factory is bound to; null for the registry row, which resolves one per model. */
    private final AutoParserAnalysis analysis;

    private AutoParserFormat(AutoParserAnalysis analysis) {
        this.analysis = analysis;
    }

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public List<String> aliases() {
        return Collections.emptyList();
    }

    @Override
    public ToolParser create(List<ChatTool> tools) {
        if (analysis == null) {
            throw new IllegalArgumentException("tool parser '" + NAME + "' analyses the loaded model's chat"
                    + " template; it needs the model's tokenizer");
        }
        if (!analysis.hasTools()) {
            throw new IllegalArgumentException("tool parser '" + NAME + "' needs a chat template with tool-call"
                    + " syntax; the analysis found none");
        }
        return new AutoParserToolParser(analysis, tools == null ? Collections.<ChatTool>emptyList() : tools);
    }

    @Override
    public ToolParser.Factory forModel(GGUFTokenizer tokenizer) {
        return forModelAnalysis(tokenizer);
    }

    /**
     * The factory of a loaded model's template, or a row that refuses by name when the analysis finds no tool-call
     * syntax.
     *
     * @throws IllegalArgumentException when the model's template cannot be analysed or has no tool-call syntax
     */
    public static ToolParser.Factory forModelAnalysis(GGUFTokenizer tokenizer) {
        if (tokenizer == null) {
            throw new IllegalArgumentException("tool parser '" + NAME + "' needs the model's tokenizer");
        }
        AutoParserAnalysis analysis = analysisOf(tokenizer.chatTemplate());
        if (analysis == null || !analysis.hasTools()) {
            throw new IllegalArgumentException("tool parser '" + NAME + "' found no tool-call syntax in the model's"
                    + " chat template");
        }
        return new AutoParserFormat(analysis);
    }

    /**
     * The autoparser of {@code template}, or null when the template cannot be analysed or writes no tool-call
     * syntax. Used as {@code auto} detection's last resort, after the hand-written formats.
     */
    public static ToolParser.Factory forTemplate(String template) {
        AutoParserAnalysis analysis = analysisOf(template);
        if (analysis == null || !analysis.hasTools()) {
            return null;
        }
        return new AutoParserFormat(analysis);
    }

    /**
     * The reasoning filter the template's analysis implies, or null when it found no reasoning markers.
     *
     * @throws IllegalArgumentException for a template whose markers are inconsistent with the tag filter
     */
    public static ReasoningFilter reasoningFilter(GGUFTokenizer tokenizer) {
        return tokenizer == null ? null : reasoningFilter(analysisOf(tokenizer.chatTemplate()));
    }

    /** The reasoning filter of an analysis, or null when it found no reasoning markers. */
    static ReasoningFilter reasoningFilter(AutoParserAnalysis analysis) {
        if (analysis == null || !analysis.reasoning().present()) {
            return null;
        }
        String start = analysis.reasoning().start().trim();
        String end = analysis.reasoning().end().trim();
        if (end.isEmpty()) {
            return null;
        }
        List<String> interrupts = new ArrayList<>();
        String opener = analysis.tools().callOpener();
        if (!opener.isEmpty()) {
            interrupts.add(opener);
        }
        return start.isEmpty()
                ? ReasoningFilter.implicitReasoning(end, interrupts)
                : ReasoningFilter.thinkTags(Collections.singletonList(start), Collections.singletonList(end),
                        interrupts);
    }

    /**
     * The forcing grammar of the request's tools in the template's own syntax: every call is a call of a declared
     * tool with its parameter schema.
     */
    public ConstraintSpec forcing(List<ChatTool> tools, boolean parallel) {
        if (analysis == null || tools == null || tools.isEmpty()) {
            return null;
        }
        return ConstraintSpec.gbnf(AutoParserPeg.grammar(analysis.tools(), tools, parallel));
    }

    /**
     * The lazy grammar of the request's tools: free output until one of {@code triggers} (the format's call
     * openers, and the OpenAI wrapper when the template's model emits it), a tool section from there on.
     */
    public ConstraintSpec lazy(List<ChatTool> tools, List<String> triggers) {
        if (analysis == null || tools == null || tools.isEmpty() || triggers == null || triggers.isEmpty()) {
            return null;
        }
        String opener = analysis.tools().callOpener();
        if (opener.isEmpty() || !triggers.contains(opener)) {
            return null;
        }
        ConstraintSpec grammar = ConstraintSpec.gbnf(AutoParserPeg.grammar(analysis.tools(), tools, true));
        List<String> words = new ArrayList<>(triggers);
        if (analysis.tools().openaiWrapperTrigger() && !words.contains(AutoParserToolParser.OPENAI_WRAPPER)) {
            words.add(AutoParserToolParser.OPENAI_WRAPPER);
        }
        return grammar.withTriggers(GrammarTriggers.lazyWords(words));
    }

    /**
     * The structured-output grammar of a JSON schema in this template's own syntax: the schema's value bare or in
     * a {@code ```json} fence, the form llama.cpp's autoparser constrains {@code json_schema} /
     * {@code response_format} requests to. Null for a factory without an analysis.
     */
    public ConstraintSpec responseFormat(String schemaJson) {
        if (analysis == null || schemaJson == null || schemaJson.isEmpty()) {
            return null;
        }
        return ConstraintSpec.gbnf(AutoParserPeg.responseFormatGrammar(schemaJson));
    }

    /** The literal stop strings the template's format adds, in order. */
    public List<String> additionalStops() {
        return analysis == null ? Collections.<String>emptyList() : analysis.additionalStops();
    }

    /**
     * The autoparser of {@code template} when the configured tool parser selects it: {@code auto} when no
     * hand-written format recognizes the template, and {@code autoparser} by name for any analysable template
     * (even one a hand-written row reads, or one without tool-call syntax, whose markers a structured-output
     * request and a continued final message still use). Null for an explicitly named other parser or, under
     * {@code auto}, a hand-written format.
     */
    public static AutoParserFormat selected(String configuredParser, String template) {
        if (template == null || template.isEmpty()) {
            return null;
        }
        String configured = configuredParser == null ? "auto" : configuredParser.trim();
        boolean auto = configured.isEmpty() || configured.equalsIgnoreCase("auto");
        if (!auto && !configured.equalsIgnoreCase(NAME)) {
            return null;
        }
        if (!auto) {
            // the request names the autoparser explicitly: it applies even to a template a hand-written row reads
            AutoParserAnalysis analysis = analysisOf(template);
            return analysis == null ? null : new AutoParserFormat(analysis);
        }
        ToolParser.Factory detected = ToolParserRegistry.detect(template);
        if (detected instanceof AutoParserFormat) {
            return (AutoParserFormat) detected;
        }
        if (detected != null) {
            return null;
        }
        AutoParserAnalysis analysis = analysisOf(template);
        return analysis == null ? null : new AutoParserFormat(analysis);
    }

    /** The analysis of {@code template} (resolved), or null when it does not parse. */
    static AutoParserAnalysis analysisOf(String template) {
        if (template == null || template.isEmpty()) {
            return null;
        }
        return AutoParser.analyze(ChatTemplateSource.resolve(template));
    }

    /** The analysis this factory is bound to, or null for the unbound registry row. */
    public AutoParserAnalysis analysis() {
        return analysis;
    }

    @Override
    public String toString() {
        return "ToolParser.Factory(" + NAME + (analysis == null ? "" : ", analysed") + ")";
    }
}
