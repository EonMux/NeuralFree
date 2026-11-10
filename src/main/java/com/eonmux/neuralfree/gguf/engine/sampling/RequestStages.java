package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.grammar.CompiledConstraint;
import com.eonmux.neuralfree.gguf.grammar.ConstraintCompiler;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.grammar.GbnfBuilder;
import com.eonmux.neuralfree.gguf.grammar.GrammarVocab;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The request-driven sampling stages: {@link AllowedTokensStage} for {@code allowed_token_ids},
 * {@link BadWordsStage} for {@code bad_words} and {@link MinTokensStage} for {@code min_tokens}, built from one
 * {@link GenerationRequest} for the sampler's Java-API extra stages ({@code Sampler.create(config, bannedTokens,
 * extra, vocab)}), plus {@link #constraintStage} for the request's grammar constraint ({@code
 * GenerationRequest#constraint}, a {@link GrammarStage}, or a {@link TriggeredGrammarStage} when the compiled
 * constraint is trigger-gated). Empty when the request sets none of them. The list order is the constraint
 * head's masking order (vLLM applies the masks to the logits before sampling in the same order): the allowed
 * ids, then the bad words, then the min-tokens ban, which yields when the masks before it left nothing but
 * end-of-generation candidates.
 */
public final class RequestStages {

    private RequestStages() {
    }

    /** The request's stages; empty when it sets none of {@code allowedTokenIds}, {@code badWords} or {@code minTokens}. */
    public static List<SamplerStage> of(GenerationRequest request, SamplerVocab vocab) {
        List<SamplerStage> stages = new ArrayList<>(3);
        if (request.allowedTokenIds().length > 0) {
            stages.add(new AllowedTokensStage(request.allowedTokenIds()));
        }
        BadWordsStage bad = BadWordsStage.of(request.badWords(), vocab);
        if (bad != null) {
            stages.add(bad);
        }
        if (request.minTokens() > 0) {
            stages.add(new MinTokensStage(request.minTokens(), vocab));
        }
        return stages;
    }

    /**
     * {@link #of} plus the reasoning-budget stage when the request arms one; {@code filter} names the tags the
     * budget counts over and forces (an armed request whose tags resolve to nothing refuses by name), and arms it at
     * the output's start when it starts a generation after {@code renderedPrompt} inside reasoning.
     *
     * @param renderedPrompt the prompt as the reasoning parser sees it, or null when unknown
     */
    public static List<SamplerStage> of(GenerationRequest request, SamplerVocab vocab, ReasoningFilter filter,
                                        String renderedPrompt) {
        List<SamplerStage> stages = of(request, vocab);
        ReasoningBudgetStage budget = ReasoningBudgetStage.of(filter, request, vocab, renderedPrompt);
        if (budget != null) {
            stages.add(budget);
        }
        return stages;
    }

    /**
     * The stage of the request's grammar {@code constraint}, or null when there is none: the spec compiles over
     * {@code grammarVocab} ({@link GbnfBuilder#compile}, so {@code <token>} references resolve and an invalid or
     * untranslatable spec throws), a trigger-gated compiled grammar runs as a {@link TriggeredGrammarStage} (its
     * free-text exclusions included) and a plain one as a {@link GrammarStage}, both reading their pieces from
     * {@code vocab}.
     */
    public static SamplerStage constraintStage(ConstraintSpec constraint, GrammarVocab grammarVocab,
                                               SamplerVocab vocab) {
        if (constraint == null) {
            return null;
        }
        return constraintStage(GbnfBuilder.compile(constraint, grammarVocab), vocab);
    }

    /**
     * {@link #constraintStage(ConstraintSpec, GrammarVocab, SamplerVocab)} compiling through {@code compiler}, which
     * hands back the engine's remembered compilation of an equal spec ({@link ConstraintCompiler}).
     */
    public static SamplerStage constraintStage(ConstraintSpec constraint, ConstraintCompiler compiler,
                                               SamplerVocab vocab) {
        if (constraint == null) {
            return null;
        }
        return constraintStage(compiler.compile(constraint), vocab);
    }

    /**
     * The stage of an already compiled constraint: a trigger-gated grammar runs as a {@link TriggeredGrammarStage}
     * with the constraint's free-text exclusions ({@link CompiledConstraint#freeExcludes},
     * {@link CompiledConstraint#freeExcludeTokens}), a plain one as a {@link GrammarStage}; both read their pieces
     * from {@code vocab}. Every engine path builds its constraint stage here, so the single-sequence and batched
     * engines enforce the same constraint.
     */
    public static SamplerStage constraintStage(CompiledConstraint compiled, SamplerVocab vocab) {
        Objects.requireNonNull(compiled, "compiled");
        Objects.requireNonNull(vocab, "vocab");
        return compiled.gated()
                ? new TriggeredGrammarStage(compiled.grammar(), compiled.triggers(), vocab,
                        compiled.freeExcludes(), compiled.freeExcludeTokens())
                : new GrammarStage(compiled.grammar(), vocab);
    }
}
