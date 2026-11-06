package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.chat.TemplateCaps;
import com.eonmux.neuralfree.gguf.chat.autoparser.AutoParserAnalysis;
import com.eonmux.neuralfree.gguf.chat.reasoning.ReasoningParsers;
import com.eonmux.neuralfree.gguf.chat.response.BeamGrammar;
import com.eonmux.neuralfree.gguf.chat.response.ResponseTemplate;
import com.eonmux.neuralfree.gguf.chat.response.TemplateParsing;
import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.chat.tools.ToolStrict;
import com.eonmux.neuralfree.gguf.engine.InfillPrompt;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;

import java.util.List;
import java.util.Objects;

/**
 * The chat, tokenizer and tool surfaces of one {@link EngineService}: rendering with the model's chat template, the
 * reasoning filter and tool routing, the response-template parsing and {@code beam} grammar, the vocabulary
 * encode/decode helpers, and the llama.cpp {@code /infill} prompt. The service owns the model, its dual encoder and
 * the configuration these read.
 */
final class EngineChat {

    /** llama.cpp's default logical batch, the share of the context an infill's prefix and suffix fit. */
    private static final int INFILL_BATCH = 2048;

    private final EngineService owner;
    private volatile ReasoningFilter reasoning;

    EngineChat(EngineService owner) {
        this.owner = Objects.requireNonNull(owner, "owner");
    }

    /**
     * Renders the chat request through the model's template ({@code GGUFTokenizer.formatChat}) over the
     * autoparser's analysis: a continued final message is written around the derived reasoning markers.
     */
    String formatChat(ChatRequest request, AutoParserAnalysis continuation) {
        return owner.primary().model().tokenizer().formatChat(request, continuation);
    }

    /** {@link #formatChat} for the multimodal path: media parts render as media markers. */
    String formatChat(ChatRequest request, boolean mediaMarkers, AutoParserAnalysis continuation) {
        return owner.primary().model().tokenizer().formatChat(request, mediaMarkers, continuation);
    }

    /**
     * The reasoning filter the engine routes output with: the configured parser
     * ({@code neuralfree.reasoning.parser}), else the chat template's.
     */
    ReasoningFilter reasoning() {
        owner.requireLoaded();
        ReasoningFilter current = reasoning;
        if (current == null) {
            current = ReasoningParsers.forModel(owner.config().reasoningParser(), owner.config().toolParser(),
                    owner.primary().model().tokenizer());
            reasoning = current;
        }
        return current;
    }

    /** How the tools of {@code request} reach its output under the configured reasoning filter. */
    ToolRouting toolRouting(ChatRequest request, String renderedPrompt, ParallelToolCalls parallelToolCalls) {
        return toolRouting(request, renderedPrompt, parallelToolCalls, reasoning());
    }

    /** {@link #toolRouting} over a per-request reasoning filter. */
    ToolRouting toolRouting(ChatRequest request, String renderedPrompt, ParallelToolCalls parallelToolCalls,
                            ReasoningFilter filter) {
        owner.requireLoaded();
        return ToolRouting.plan(request, owner.config().toolParser(), owner.primary().model().tokenizer(),
                filter, renderedPrompt, parallelToolCalls, owner.config().toolStrict());
    }

    /** The response-template parsing a request answers under ({@code hf} or {@code beam}), or null. */
    TemplateParsing templateParsing(ReasoningFilter filter) {
        if (filter.responseTemplate() != null) {
            return filter.responseTemplate();
        }
        String toolParser = owner.config().toolParser();
        if (!TemplateParsing.names(toolParser)) {
            return null;
        }
        owner.requireLoaded();
        return TemplateParsing.forModel(toolParser, owner.primary().model().tokenizer(),
                "--tool-call-parser " + toolParser, ResponseTemplate.TOOL_CALLS);
    }

    /** The grammar the {@code beam} parser holds a chat request's output to, or null. */
    BeamGrammar beamGrammar(TemplateParsing parsing, ChatRequest chat, String renderedPrompt,
                            ParallelToolCalls parallelToolCalls, ConstraintSpec output, ReasoningFilter filter,
                            boolean includeReasoning) {
        if (parsing == null || !parsing.beam()) {
            return null;
        }
        return BeamGrammar.derive(parsing, BeamGrammar.Request.of(chat)
                .prompt(renderedPrompt)
                .parallelToolCalls(parallelToolCalls)
                .functionStrict(owner.config().toolStrict().atLeast(ToolStrict.FUNCTION))
                .parameterStrict(owner.config().toolStrict().atLeast(ToolStrict.PARAMETER))
                .output(output)
                .reasoningGated(includeReasoning && filter.parsesReasoning())
                .toolParsing(TemplateParsing.names(owner.config().toolParser()))
                .alwaysConstrain(owner.config().beamAlwaysConstrain()));
    }

    /** The model's chat template source, or null when it carries none. */
    String chatTemplate() {
        if (owner.dual() != null) {
            return null;
        }
        return owner.primary().model().tokenizer().chatTemplate();
    }

    /** The capabilities of the model's chat template, probed as llama.cpp probes them. */
    TemplateCaps chatTemplateCaps() {
        return owner.primary().model().tokenizer().chatTemplateCaps();
    }

    /** The {@code parallel_tool_calls} default of the model's template. */
    boolean defaultParallelToolCalls() {
        return chatTemplateCaps().supportsParallelToolCalls();
    }

    /** The {@code parallel_tool_calls} default of the template {@code request} renders with. */
    boolean defaultParallelToolCalls(ChatRequest request) {
        return chatTemplateCaps(request).supportsParallelToolCalls();
    }

    /** The capabilities of the template {@code request} renders with. */
    TemplateCaps chatTemplateCaps(ChatRequest request) {
        return owner.primary().model().tokenizer().chatTemplateCaps(request);
    }

    /** The vocabulary size, the bound of a token id. */
    int vocabSize() {
        if (owner.dual() != null) {
            return owner.dual().vocabSize();
        }
        return owner.primary().model().tokenizer().vocabSize();
    }

    /** Decodes token ids to text ({@code GGUFTokenizer.decode}). */
    String decode(int[] tokens) {
        if (owner.dual() != null) {
            return owner.dual().tokenizer().decode(tokens);
        }
        return owner.primary().model().tokenizer().decode(tokens);
    }

    /** Tokenizes text ({@code GGUFTokenizer.encode(text, addSpecial, parseSpecial)}). */
    int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        if (owner.dual() != null) {
            return owner.dual().tokenizer().encode(text, addSpecial, parseSpecial);
        }
        return owner.primary().model().tokenizer().encode(text, addSpecial, parseSpecial);
    }

    /** Decodes one token id to its text (the logprobs rendering). */
    String tokenText(int tokenId) {
        if (owner.dual() != null) {
            return owner.dual().tokenizer().decode(tokenId);
        }
        return owner.primary().model().tokenizer().decode(tokenId);
    }

    /**
     * The raw bytes one token renders to, special tokens included
     * ({@code GGUFTokenizer.pieceWithSpecial}): the source text a token covers, for the render endpoints'
     * character offsets.
     */
    byte[] tokenPiece(int tokenId) {
        if (owner.dual() != null) {
            return owner.dual().tokenizer().pieceWithSpecial(tokenId);
        }
        return owner.primary().model().tokenizer().pieceWithSpecial(tokenId);
    }

    /** The BOS token a raw prompt starts with when the vocabulary adds one, else -1. */
    int addedBos() {
        GGUFTokenizer tokenizer = owner.tokenizer();
        return tokenizer.vocab().addBos() ? tokenizer.bosId() : -1;
    }

    /** The FIM tokens the model lacks for {@code /infill}, or null when it has them all. */
    String infillMissingTokens() {
        return InfillPrompt.missingTokens(owner.primary().model().tokenizer());
    }

    /** The fill-in-the-middle prompt of llama.cpp's {@code /infill}. */
    int[] infillPrompt(int[] prefix, int[] suffix, List<InfillPrompt.Chunk> extra, int[] prompt, int nPredict) {
        int nCtx = owner.maxModelLength();
        return InfillPrompt.build(owner.primary().model().tokenizer(), prefix, suffix, extra, prompt,
                Math.min(INFILL_BATCH, nCtx), nPredict, nCtx, owner.options().spmInfill());
    }
}
