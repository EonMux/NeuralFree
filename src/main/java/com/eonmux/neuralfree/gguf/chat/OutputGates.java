package com.eonmux.neuralfree.gguf.chat;

import com.eonmux.neuralfree.gguf.chat.response.ResponseTemplate;
import com.eonmux.neuralfree.gguf.chat.response.TemplateParsing;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.grammar.ConstraintGate;
import com.eonmux.neuralfree.gguf.grammar.GrammarTriggers;

import java.util.Arrays;

/**
 * Where an output constraint starts for a model that reasons first, so a {@code response_format} or guided grammar
 * holds the answer and not the reasoning (vLLM advances the structured-output grammar only after the reasoning
 * parser sees the end of reasoning; llama.cpp builds the schema behind the thinking section):
 *
 * <ul>
 *   <li>a thinking-tag model whose rendered prompt leaves reasoning open: after its reasoning end tag;</li>
 *   <li>a response-template model ({@code hf}, {@code beam}) whose reasoning is not over at the prompt's end
 *       ({@link TemplateParsing#reasoningEnded}): after its thinking region's literal closer; a closer only a pattern
 *       spells, or a repeating thinking region, cannot gate and is refused by name;</li>
 *   <li>gpt-oss: after the {@code <|channel|>final} header, an optional {@code <|constrain|>} type included;</li>
 *   <li>Muse: after the header of the message addressed to the user;</li>
 *   <li>otherwise none: the constraint holds from the first token.</li>
 * </ul>
 */
public final class OutputGates {

    private OutputGates() {
    }

    /**
     * The gate of a chat completion, or null when the constraint holds from the first token.
     *
     * @param reasoning      the model's reasoning filter
     * @param renderedPrompt the rendered prompt; null leaves a thinking-tag model ungated
     * @param chatTemplate   the chat template, for the harmony dialect; may be null
     */
    public static ConstraintGate forChat(ReasoningFilter reasoning, String renderedPrompt, String chatTemplate) {
        switch (reasoning.mode()) {
            case THINK_TAGS:
                if (reasoning.responseTemplate() != null) {
                    return templateGate(reasoning.responseTemplate(), renderedPrompt);
                }
                return opensReasoning(reasoning, renderedPrompt)
                        ? ConstraintGate.afterWords(reasoning.endTags()) : null;
            case HARMONY_CHANNELS:
                return HarmonyDialect.of(chatTemplate) == HarmonyDialect.MUSE ? MUSE_FINAL : GPT_OSS_FINAL;
            default:
                return null;
        }
    }

    /**
     * The gate of a response-template model: none without reasoning parsing or once its reasoning is over, else its
     * thinking closer.
     *
     * @throws IllegalArgumentException when the thinking region repeats or closes only with a pattern
     */
    private static ConstraintGate templateGate(TemplateParsing parsing, String renderedPrompt) {
        if (!parsing.parsesReasoning() || parsing.reasoningEnded(renderedPrompt)) {
            return null;
        }
        ResponseTemplate.Field thinking = parsing.template().field(ResponseTemplate.THINKING);
        if (thinking.repeats() || thinking.closeLiterals() == null) {
            throw new IllegalArgumentException("an output constraint after the reasoning needs the response_template's"
                    + " thinking region to close once with a literal closer; the " + parsing.name() + " parser's"
                    + " thinking region " + (thinking.repeats() ? "repeats" : "closes with a pattern"));
        }
        return ConstraintGate.afterWords(thinking.closeLiterals());
    }

    /** Whether the rendered prompt leaves a thinking-tag model inside its reasoning block. */
    public static boolean opensReasoning(ReasoningFilter reasoning, String renderedPrompt) {
        if (reasoning.mode() != ReasoningFilter.Mode.THINK_TAGS || renderedPrompt == null) {
            return false;
        }
        return reasoning.newState((channel, tokenId, text) -> { }, renderedPrompt).inReasoning();
    }

    /** gpt-oss's final answer header (the {@code response-format} rule of llama.cpp's {@code parsers/gpt-oss.cpp}). */
    private static final ConstraintGate GPT_OSS_FINAL = new ConstraintGate(
            GrammarTriggers.lazyWords(Arrays.asList("<|channel|>final")),
            "\"<|channel|>final\" ( [ \\t\\n]* \"<|constrain|>\"? [A-Za-z0-9_-]+ )? \"<|message|>\"");

    /**
     * Muse's final answer header, its {@code to=user} recipient optional (llama.cpp {@code parsers/muse-spark.cpp}):
     * the addressed header anywhere, the bare one after a later {@code <|start|>assistant} or at the very start of the
     * output, where the prompt left the assistant header open.
     */
    private static final ConstraintGate MUSE_FINAL = new ConstraintGate(
            GrammarTriggers.builder(GrammarTriggers.Mode.LAZY)
                    .word(" to=user<|message|>")
                    .word("<|start|>assistant<|message|>")
                    .pattern("^(<\\|message\\|>)")
                    .build(),
            "( \"<|start|>assistant\" )? ( \" to=user\" )? \"<|message|>\"");
}
