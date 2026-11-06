package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.kvtransfer.KvTransferParams;
import com.eonmux.neuralfree.gguf.engine.lora.LoraSelection;
import com.eonmux.neuralfree.gguf.engine.sampling.LogitsProcessorArgs;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;

/**
 * One autoregressive generation: how many tokens, how to sample them and when to stop. Immutable (the cancellation
 * hook is a caller-owned supplier that the generation polls once per token).
 *
 * <p>The token budget is explicit when {@link Builder#maxNewTokens} set it; otherwise it is a fallback
 * ({@link #DEFAULT_MAX_NEW_TOKENS}, or {@link Builder#fallbackMaxNewTokens}) that a model's
 * {@code max_new_tokens} default replaces ({@link #withModelDefaults}).
 */
public final class GenerationRequest {

    /** {@link #maxNewTokens()} value meaning "until end of generation or a full context" (llama.cpp {@code -n -1}). */
    public static final int UNLIMITED = -1;
    /** Default new-token budget when neither the request nor the model's sampling defaults give one. */
    public static final int DEFAULT_MAX_NEW_TOKENS = 512;
    /** {@link #reasoningBudget()} value meaning "no thinking-token budget" (llama.cpp {@code reasoning_budget_tokens: -1}). */
    public static final int REASONING_BUDGET_DISABLED = -1;
    /** {@link #promptLogprobs()} value meaning "every vocabulary token" (vLLM's {@code prompt_logprobs: -1}). */
    public static final int PROMPT_LOGPROBS_ALL = -1;
    /** The most explicit token ids one step may score (vLLM's {@code MAX_LOGPROB_TOKEN_IDS}). */
    public static final int MAX_LOGPROB_TOKEN_IDS = 128;
    /** The {@link #promptLogprobTokenIds()} entry that pads a row; it scores {@code -inf}. */
    public static final int PROMPT_TOKEN_ID_PAD = -1;
    /** {@link #nKeep()} value meaning "keep the whole prompt when the context shifts" (llama.cpp {@code n_keep: -1}). */
    public static final int KEEP_PROMPT_ALL = -1;
    /** {@link #grpAttnN()} value of llama.cpp's default: group attention off. */
    public static final int GRP_ATTN_OFF = 1;
    /** Default group-attention width (llama.cpp {@code --grp-attn-w}). */
    public static final int DEFAULT_GRP_ATTN_W = 512;

    private static final BooleanSupplier NEVER = new BooleanSupplier() {
        @Override
        public boolean getAsBoolean() {
            return false;
        }

        @Override
        public String toString() {
            return "never";
        }
    };

    private final int maxNewTokens;
    private final boolean maxNewTokensExplicit;
    private final SamplerConfig sampler;
    private final List<String> stopStrings;
    private final List<String> badWords;
    private final boolean ignoreEos;
    private final BooleanSupplier cancellation;
    private final int logprobs;
    private final int minTokens;
    private final Map<Integer, Float> logitBias;
    private final int reasoningBudget;
    private final String reasoningBudgetMessage;
    private final boolean reasoningControl;
    private final List<String> reasoningBudgetStartTags;
    private final List<String> reasoningBudgetEndTags;
    private final ConstraintSpec constraint;
    private final int[] stopTokenIds;
    private final int[] allowedTokenIds;
    private final int promptLogprobs;
    private final LogprobsMode logprobsMode;
    private final int[] logprobTokenIds;
    private final int promptLogprobStart;
    private final int[][] promptLogprobTokenIds;
    private final int nKeep;
    private final int nDiscard;
    private final int grpAttnN;
    private final int grpAttnW;
    private final int nCacheReuse;
    private final boolean includeStopString;
    private final boolean skipSpecialTokens;
    private final boolean spacesBetweenSpecialTokens;
    private final RepetitionDetection repetitionDetection;
    private final int priority;
    private final String cacheSalt;
    private final LoraSelection lora;
    private final Map<String, Object> templateVariables;
    private final LogitsProcessorArgs logitsProcessorArgs;
    private final int beamWidth;
    private final float lengthPenalty;
    /** The request's disaggregated-transfer parameters (vLLM's {@code kv_transfer_params}), or null. */
    private final KvTransferParams kvTransfer;

    private GenerationRequest(Builder b) {
        this.maxNewTokens = b.maxNewTokens;
        this.maxNewTokensExplicit = b.maxNewTokensExplicit;
        this.sampler = b.sampler;
        this.stopStrings = Collections.unmodifiableList(new ArrayList<>(b.stopStrings));
        this.badWords = Collections.unmodifiableList(new ArrayList<>(b.badWords));
        this.ignoreEos = b.ignoreEos;
        this.cancellation = b.cancellation;
        this.logprobs = b.logprobs;
        this.minTokens = b.minTokens;
        this.logitBias = Collections.unmodifiableMap(new LinkedHashMap<>(b.logitBias));
        this.reasoningBudget = b.reasoningBudget;
        this.reasoningBudgetMessage = b.reasoningBudgetMessage;
        this.reasoningControl = b.reasoningControl;
        this.reasoningBudgetStartTags = Collections.unmodifiableList(new ArrayList<>(b.reasoningBudgetStartTags));
        this.reasoningBudgetEndTags = Collections.unmodifiableList(new ArrayList<>(b.reasoningBudgetEndTags));
        this.constraint = b.constraint;
        this.stopTokenIds = b.stopTokenIds;
        this.allowedTokenIds = b.allowedTokenIds;
        this.promptLogprobs = b.promptLogprobs;
        this.logprobsMode = b.logprobsMode;
        this.logprobTokenIds = b.logprobTokenIds;
        this.promptLogprobStart = b.promptLogprobStart;
        this.promptLogprobTokenIds = b.promptLogprobTokenIds;
        this.nKeep = b.nKeep;
        this.nDiscard = b.nDiscard;
        this.grpAttnN = b.grpAttnN;
        this.grpAttnW = b.grpAttnW;
        this.nCacheReuse = b.nCacheReuse;
        this.includeStopString = b.includeStopString;
        this.skipSpecialTokens = b.skipSpecialTokens;
        this.spacesBetweenSpecialTokens = b.spacesBetweenSpecialTokens;
        this.repetitionDetection = b.repetitionDetection;
        this.priority = b.priority;
        this.cacheSalt = b.cacheSalt;
        this.lora = b.lora;
        this.templateVariables = Collections.unmodifiableMap(new LinkedHashMap<>(b.templateVariables));
        this.logitsProcessorArgs = b.logitsProcessorArgs;
        this.beamWidth = b.beamWidth;
        this.lengthPenalty = b.lengthPenalty;
        this.kvTransfer = b.kvTransfer;
    }

    /**
     * {@link #DEFAULT_MAX_NEW_TOKENS} new tokens with {@link SamplerConfig#defaults()}, all fallbacks a model's
     * sampling defaults replace.
     */
    public static GenerationRequest defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    /** New tokens to generate at most, or {@link #UNLIMITED}. */
    public int maxNewTokens() { return maxNewTokens; }
    public boolean unlimited() { return maxNewTokens == UNLIMITED; }
    /** Whether {@link Builder#maxNewTokens} set the token budget, so a model's default does not replace it. */
    public boolean maxNewTokensExplicit() { return maxNewTokensExplicit; }
    /** The request's disaggregated prefill/decode parameters (vLLM's {@code kv_transfer_params}), or null. */
    public KvTransferParams kvTransfer() { return kvTransfer; }
    public SamplerConfig sampler() { return sampler; }
    /** Strings that end generation when they appear in the decoded output; excluded from the result text. */
    public List<String> stopStrings() { return stopStrings; }
    /** Whether end-of-generation tokens are sampled through instead of stopping (llama.cpp {@code ignore_eos}). */
    public boolean ignoreEos() { return ignoreEos; }
    /** Polled once per generated token; true stops generation with {@link StopReason#CANCELLED}. */
    public BooleanSupplier cancellation() { return cancellation; }

    /** Alternatives to report per generated token (the sampled token included), or 0 when off. */
    public int logprobs() { return logprobs; }

    /**
     * The token ids every generated step scores besides the sampled token (vLLM's {@code logprob_token_ids}), in
     * the request's order; they take precedence over the {@link #logprobs()} count. Empty when unset.
     */
    public int[] logprobTokenIds() { return logprobTokenIds.clone(); }

    /** Whether a request names the logprob token ids explicitly. */
    public boolean namesLogprobTokenIds() { return logprobTokenIds.length > 0; }

    /** Whether generated steps capture scores: a positive {@link #logprobs()} count or {@link #logprobTokenIds()}. */
    public boolean wantsLogprobs() { return logprobs > 0 || logprobTokenIds.length > 0; }

    /**
     * The distribution the generated-token and prompt captures score (vLLM's {@code logprobs_mode}); default
     * {@link LogprobsMode#RAW_LOGPROBS}, the model's own distribution before any sampling stage.
     */
    public LogprobsMode logprobsMode() { return logprobsMode; }

    /** Phrases the output must not spell ({@code bad_words}); empty when none. */
    public List<String> badWords() { return badWords; }

    /** The generated-token minimum before end-of-generation tokens may be sampled (0 = off). */
    public int minTokens() { return minTokens; }

    /**
     * The request's {@code logit_bias} map: token id to the additive bias applied at the sampler's first stage
     * (llama.cpp {@code llama_sampler_init_logit_bias}); empty when none. A bias of {@code -100} or below means
     * ban to OpenAI clients — the caller translates that into the banned ids before the sampler sees it.
     */
    public Map<Integer, Float> logitBias() { return logitBias; }

    /**
     * The thinking-token budget: the tokens generated inside a reasoning block before its end tag is forced
     * (llama.cpp {@code reasoning_budget_tokens}, vLLM {@code thinking_token_budget}), or
     * {@link #REASONING_BUDGET_DISABLED}.
     */
    public int reasoningBudget() { return reasoningBudget; }

    /** Message injected ahead of the forced end tag when the budget runs out, or null for none. */
    public String reasoningBudgetMessage() { return reasoningBudgetMessage; }

    /**
     * Whether the budget stage runs without a budget so a control action can end reasoning mid-stream
     * (llama.cpp {@code reasoning_control}).
     */
    public boolean reasoningControl() { return reasoningControl; }

    /** Explicit reasoning-budget start tags overriding the model's reasoning parser; empty derives them. */
    public List<String> reasoningBudgetStartTags() { return reasoningBudgetStartTags; }

    /** Explicit reasoning-budget end tags overriding the model's reasoning parser; empty derives them. */
    public List<String> reasoningBudgetEndTags() { return reasoningBudgetEndTags; }

    /**
     * The grammar the generated text must conform to ({@code --grammar}, {@code --json-schema}, the OpenAI
     * {@code response_format}), or null for free text. The sampler masks every token the constraint rejects;
     * see {@code RequestStages.constraintStage}.
     */
    public ConstraintSpec constraint() { return constraint; }

    /**
     * Token ids that end generation like an end-of-generation token (vLLM's {@code stop_token_ids}): sampled,
     * then dropped from the output, with {@link StopReason#EOS}. Generation stops on them even under
     * {@link #ignoreEos()}; empty when none.
     */
    public int[] stopTokenIds() { return stopTokenIds.clone(); }

    /**
     * The only token ids the sampler may select (vLLM's {@code allowed_token_ids}); every other id's logit is
     * masked out. Empty when unset, meaning the whole vocabulary stays available.
     */
    public int[] allowedTokenIds() { return allowedTokenIds.clone(); }

    /**
     * The per-prompt-token log probabilities to report (vLLM's {@code prompt_logprobs}): 0 is off (the default),
     * {@code n} reports each prompt token plus its {@code n - 1} top alternatives, and {@link
     * #PROMPT_LOGPROBS_ALL} reports every vocabulary token. The batch engine reads the prompt rows' logits for it.
     */
    public int promptLogprobs() { return promptLogprobs; }

    /** Whether a request asks for per-prompt-token log probabilities. */
    public boolean wantsPromptLogprobs() { return promptLogprobs != 0; }

    /**
     * The first prompt row the prompt captures score (vLLM's {@code prompt_logprob_start}); default 0. Row
     * {@code r} predicts prompt token {@code r + 1}, so positions up to and including the start report no
     * {@link #promptLogprobs()} entry.
     */
    public int promptLogprobStart() { return promptLogprobStart; }

    /**
     * The token ids each scored prompt row scores (vLLM's {@code prompt_logprob_token_ids}): row {@code i} scores
     * its ids as predictions of prompt token {@code promptLogprobStart() + i + 1}, so a prompt of {@code n} tokens
     * needs exactly {@code n - 1 - promptLogprobStart()} rows. Rows are padded to one width with
     * {@link #PROMPT_TOKEN_ID_PAD}. A deep copy; null when unset.
     */
    public int[][] promptLogprobTokenIds() { return deepCopy(promptLogprobTokenIds); }

    /** Whether the prefill reads every prompt row's logits back ({@link #promptLogprobs()} or the token-id rows). */
    public boolean readsPromptRows() { return promptLogprobs != 0 || promptLogprobTokenIds != null; }

    /**
     * Refuses a request that scores its prompt rows on a path whose prefill reads only the last row's logits
     * ({@code path} names it); the batch engine and the generation loop read the prompt rows back.
     *
     * @throws UnsupportedOperationException naming {@code prompt_logprobs} or {@code prompt_logprob_token_ids}
     */
    public void requireNoPromptRows(String path) {
        String field = promptRowsField();
        if (field != null) {
            throw new UnsupportedOperationException(field + ": " + path + " reads no prompt-row logits; submit the"
                    + " request to the batch engine (BatchEngine) or the generation loop (GenerationLoop)");
        }
    }

    /**
     * The request field that makes the prefill read the prompt rows back: {@code prompt_logprobs}, else
     * {@code prompt_logprob_token_ids}; null when neither is set.
     */
    public String promptRowsField() {
        if (promptLogprobs != 0) {
            return "prompt_logprobs";
        }
        return promptLogprobTokenIds != null ? "prompt_logprob_token_ids" : null;
    }

    /**
     * Checks the token ids the request scores against a {@code vocabSize}-token vocabulary, and its prompt
     * token-id rows against a {@code promptTokens}-token prompt: vLLM's {@code prompt_logprob_token_ids} hold one
     * row per scored prompt row, {@code promptTokens - 1 - promptLogprobStart()}.
     *
     * @throws IllegalArgumentException naming {@code logprob_token_ids} or {@code prompt_logprob_token_ids}
     */
    public void requireScoredIds(int promptTokens, int vocabSize) {
        for (int id : logprobTokenIds) {
            if (id >= vocabSize) {
                throw new IllegalArgumentException("logprob_token_ids token id " + id + " is outside the vocabulary"
                        + " [0, " + vocabSize + ")");
            }
        }
        if (promptLogprobTokenIds == null) {
            return;
        }
        int scored = promptTokens - 1 - promptLogprobStart;
        if (promptLogprobTokenIds.length != scored) {
            throw new IllegalArgumentException("prompt_logprob_token_ids has " + promptLogprobTokenIds.length
                    + " rows, but the " + promptTokens + "-token prompt scores " + Math.max(0, scored)
                    + " rows from prompt_logprob_start " + promptLogprobStart);
        }
        for (int[] row : promptLogprobTokenIds) {
            for (int id : row) {
                if (id >= vocabSize) {
                    throw new IllegalArgumentException("prompt_logprob_token_ids token id " + id + " is outside the"
                            + " vocabulary [0, " + vocabSize + ")");
                }
            }
        }
    }

    private static int[][] deepCopy(int[][] rows) {
        if (rows == null) {
            return null;
        }
        int[][] out = new int[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            out[i] = rows[i].clone();
        }
        return out;
    }

    /**
     * The prompt tokens a context shift keeps (llama.cpp server's {@code n_keep}): the first {@code nKeep} prompt
     * positions survive every shift, a leading BOS on top, and {@link #KEEP_PROMPT_ALL} keeps the whole prompt.
     * 0 keeps only a leading BOS (the default). The request's batch engine shifts only when its
     * {@code neuralfree.context.shift} is on.
     */
    public int nKeep() { return nKeep; }

    /**
     * The tokens one context shift discards after {@link #nKeep()} (llama.cpp server's {@code n_discard}); 0
     * discards half of what the context holds beyond the kept tokens (the default).
     */
    public int nDiscard() { return nDiscard; }

    /**
     * The self-extend group-attention factor (llama.cpp {@code --grp-attn-n}): every {@link #grpAttnW()} stored
     * positions are grouped, the group's positions divided by this factor, so the sequence attends at an effective
     * context of {@code nCtx / grpAttnN}. {@link #GRP_ATTN_OFF} (the default) keeps every position as stored.
     */
    public int grpAttnN() { return grpAttnN; }

    /**
     * The self-extend group-attention width (llama.cpp {@code --grp-attn-w}): the positions one group spans before
     * its compression, a multiple of {@link #grpAttnN()}. The default is llama.cpp's {@value #DEFAULT_GRP_ATTN_W}.
     */
    public int grpAttnW() { return grpAttnW; }

    /** Whether the request compresses positions as its sequence grows (self-extend, group attention). */
    public boolean selfExtend() { return grpAttnN > GRP_ATTN_OFF; }

    /**
     * The token run length one KV chunk reuse re-adopts (llama.cpp server's {@code n_cache_reuse}): after the
     * common prefix, cached prompt runs of at least this many tokens skip their prefill when the request's batch
     * engine re-appends their stored keys and values at the prompt's own positions. 0 (the default) turns chunk
     * reuse off.
     */
    public int nCacheReuse() { return nCacheReuse; }

    /** Whether a matched stop string stays in the output text (vLLM's {@code include_stop_str_in_output}). */
    public boolean includeStopString() { return includeStopString; }

    /**
     * Whether control and end-of-generation tokens stay out of the output text (vLLM's
     * {@code skip_special_tokens}, llama.cpp's {@code token_to_piece} with {@code special = false}). The default
     * is true; false renders their texts.
     */
    public boolean skipSpecialTokens() { return skipSpecialTokens; }

    /**
     * Whether consecutive special-token texts are joined with a space (vLLM's
     * {@code spaces_between_special_tokens}); only visible with {@link #skipSpecialTokens()} false.
     */
    public boolean spacesBetweenSpecialTokens() { return spacesBetweenSpecialTokens; }

    /**
     * The n-gram repetition stop (vLLM's {@code repetition_detection}): once at least {@link #minTokens()} tokens
     * were generated, an output ending in a repeated token pattern ends with {@link StopReason#REPETITION}.
     * {@link RepetitionDetection#disabled()} by default.
     */
    public RepetitionDetection repetitionDetection() { return repetitionDetection; }

    /**
     * The scheduling priority (vLLM's {@code priority}): under the batch engine's {@code priority} scheduling policy
     * a lower value is admitted earlier and preempted later, equal values in arrival order; the default
     * {@code fcfs} policy takes 0 only.
     */
    public int priority() { return priority; }

    /** The prefix-cache salt (vLLM's {@code cache_salt}): lookups and cached entries match it exactly. */
    public String cacheSalt() { return cacheSalt; }

    /**
     * The per-request LoRA adapters the request runs with (vLLM's {@code lora_request}, llama.cpp's per-request
     * {@code lora} list), by the names of the model's {@code LoraRegistry}; null runs slot 0, the startup adapters.
     * The batch engine, the generation loop and the speculative decoder serve them on their own passes; a draft
     * model runs without them (vLLM: the drafter does not use the target's LoRA).
     */
    public LoraSelection lora() { return lora; }

    /**
     * Refuses a request that selects per-request adapters on a path whose passes run slot 0 alone ({@code path}
     * names it).
     *
     * @throws UnsupportedOperationException naming the {@code lora} field when {@link #lora()} is set
     */
    public void requireNoLora(String path) {
        if (lora != null) {
            throw new UnsupportedOperationException("lora: " + path + " runs no per-request LoRA adapters ("
                    + lora + "); submit the request to the batch engine (BatchEngine) or the generation loop"
                    + " (GenerationLoop)");
        }
    }

    /**
     * The chat template variables the prompt was rendered with ({@code enable_thinking}, {@code reasoning_effort},
     * the {@code chat_template_kwargs}); the engine's reasoning parser selects its configuration for the generation
     * from them ({@link ReasoningFilter#forRequest}). Empty when the caller sets none.
     */
    public Map<String, Object> templateVariables() { return templateVariables; }

    /**
     * The request's custom logits processor arguments: the members of the request body's {@code vllm_xargs} object
     * (vLLM's {@code SamplingParams.extra_args}), read by every loaded processor factory to decide whether it
     * applies and how; {@link LogitsProcessorArgs#empty()} when the request sets none.
     */
    public LogitsProcessorArgs logitsProcessorArgs() { return logitsProcessorArgs; }

    /**
     * The beam-search width (vLLM's {@code BeamSearchParams.beam_width}, the OpenAI {@code n} under
     * {@code use_beam_search}): 1 searches a single best-first path, values above 1 keep that many beams per
     * step. 1 with beam search off are the same width; the runner a request reaches decides whether
     * {@link #beamSearch()} runs the search.
     */
    public int beamWidth() { return beamWidth; }

    /** Whether the request runs beam search ({@link #beamWidth()} above 1). */
    public boolean beamSearch() { return beamWidth > 1; }

    /**
     * The beam-search length penalty (vLLM's {@code length_penalty}): the exponent on the number of generated
     * tokens (including a final EOS) that normalizes a beam's cumulative log probability when beams rank
     * ({@code ln p / n^penalty}); 1.0, the default, plain averages. Values above 0 favor longer beams, below 0
     * shorter ones.
     */
    public float lengthPenalty() { return lengthPenalty; }

    /** Whether a cancellation hook was supplied. */
    public boolean cancellable() { return cancellation != NEVER; }

    /**
     * A streaming filter that withholds text which may begin a stop string and stops forwarding at the first
     * complete one; see {@link StopStringFilter}.
     *
     * @param downstream receives the released text, or null to only track the stop
     * @param includeStopString whether the matched stop string itself stays in the released text (vLLM's
     *                          {@code include_stop_str_in_output})
     */
    public StopStringFilter newStopFilter(TokenCallback downstream) {
        return new StopStringFilter(stopStrings, downstream, includeStopString);
    }

    /** A builder holding these values; the token budget keeps its explicit or fallback state. */
    public Builder toBuilder() {
        Builder b = builder().sampler(sampler).stopStrings(stopStrings).badWords(badWords).ignoreEos(ignoreEos)
                .cancellation(cancellation).logprobs(logprobs).minTokens(minTokens).logitBias(logitBias)
                .reasoningBudget(reasoningBudget).reasoningBudgetMessage(reasoningBudgetMessage)
                .reasoningControl(reasoningControl).reasoningBudgetStartTags(reasoningBudgetStartTags)
                .reasoningBudgetEndTags(reasoningBudgetEndTags)
                .constraint(constraint)
                .stopTokenIds(stopTokenIds).allowedTokenIds(allowedTokenIds)
                .logprobsMode(logprobsMode).logprobTokenIds(logprobTokenIds)
                .includeStopString(includeStopString).skipSpecialTokens(skipSpecialTokens)
                .spacesBetweenSpecialTokens(spacesBetweenSpecialTokens).repetitionDetection(repetitionDetection)
                .priority(priority).cacheSalt(cacheSalt)
                .lora(lora)
                .nKeep(nKeep).nDiscard(nDiscard)
                .grpAttnN(grpAttnN).grpAttnW(grpAttnW)
                .nCacheReuse(nCacheReuse)
                .templateVariables(templateVariables)
                .logitsProcessorArgs(logitsProcessorArgs)
                .beamWidth(beamWidth).lengthPenalty(lengthPenalty)
                .kvTransfer(kvTransfer);
        if (wantsPromptLogprobs()) {
            b.promptLogprobs(promptLogprobs);
        }
        if (promptLogprobTokenIds != null) {
            b.promptLogprobTokenIds(promptLogprobTokenIds);
        }
        if (promptLogprobStart > 0) {
            b.promptLogprobStart(promptLogprobStart);
        }
        return maxNewTokensExplicit ? b.maxNewTokens(maxNewTokens) : b.fallbackMaxNewTokens(maxNewTokens);
    }

    /**
     * The request a generation runs for a model with {@code defaults}: the sampler takes the model's defaults for
     * every setting it leaves unset ({@link SamplerConfig#withModelDefaults}), and a token budget that is not
     * explicit becomes the model's {@code max_new_tokens} when it has one. Returns this request when
     * {@code defaults} is empty.
     *
     * @throws IllegalArgumentException when a taken sampler value is out of range
     */
    public GenerationRequest withModelDefaults(SamplingDefaults defaults) {
        Objects.requireNonNull(defaults, "defaults");
        if (defaults.isEmpty()) {
            return this;
        }
        Builder b = toBuilder().sampler(sampler.withModelDefaults(defaults));
        if (!maxNewTokensExplicit && defaults.maxNewTokens() != null) {
            b.fallbackMaxNewTokens(defaults.maxNewTokens());
        }
        return b.build();
    }

    @Override
    public String toString() {
        return "GenerationRequest{maxNew=" + (unlimited() ? "unlimited" : String.valueOf(maxNewTokens)) + ", "
                + sampler + ", stops=" + stopStrings.size() + ", ignoreEos=" + ignoreEos + ", cancellable="
                + cancellable() + '}';
    }

    public static final class Builder {
        private int maxNewTokens = DEFAULT_MAX_NEW_TOKENS;
        private boolean maxNewTokensExplicit;
        private SamplerConfig sampler = SamplerConfig.defaults();
        private final List<String> stopStrings = new ArrayList<>();
        private boolean ignoreEos;
        private int logprobs;
        private int minTokens;
        private List<String> badWords = new ArrayList<>();
        private Map<Integer, Float> logitBias = new LinkedHashMap<>();
        private int reasoningBudget = REASONING_BUDGET_DISABLED;
        private String reasoningBudgetMessage;
        private boolean reasoningControl;
        private List<String> reasoningBudgetStartTags = new ArrayList<>();
        private List<String> reasoningBudgetEndTags = new ArrayList<>();
        private ConstraintSpec constraint;
        private int[] stopTokenIds = new int[0];
        private int[] allowedTokenIds = new int[0];
        private int promptLogprobs;
        private LogprobsMode logprobsMode = LogprobsMode.RAW_LOGPROBS;
        private int[] logprobTokenIds = new int[0];
        private int promptLogprobStart;
        private int[][] promptLogprobTokenIds;
        private int nKeep;
        private int nDiscard;
        private int grpAttnN = GRP_ATTN_OFF;
        private int grpAttnW = DEFAULT_GRP_ATTN_W;
        private int nCacheReuse;
        private boolean includeStopString;
        private boolean skipSpecialTokens = true;
        private boolean spacesBetweenSpecialTokens = true;
        private RepetitionDetection repetitionDetection = RepetitionDetection.disabled();
        private int priority;
        private String cacheSalt;
        private LoraSelection lora;
        private Map<String, Object> templateVariables = new LinkedHashMap<>();
        private LogitsProcessorArgs logitsProcessorArgs = LogitsProcessorArgs.empty();
        private int beamWidth = 1;
        private float lengthPenalty = 1.0f;
        private KvTransferParams kvTransfer;

        /** Copies {@code logitBias}; keys must be non-negative ids, values finite. */
        public Builder logitBias(Map<Integer, Float> v) {
            Objects.requireNonNull(v, "logitBias");
            for (Map.Entry<Integer, Float> e : v.entrySet()) {
                logitBias(e.getKey(), e.getValue());
            }
            return this;
        }

        /** Adds one {@code logit_bias} entry; the key must be a non-negative id, the value finite. */
        public Builder logitBias(int tokenId, float bias) {
            Objects.requireNonNull(bias, "bias");
            if (tokenId < 0) {
                throw new IllegalArgumentException("logit_bias token id " + tokenId + " is negative");
            }
            if (!Float.isFinite(bias)) {
                throw new IllegalArgumentException("the bias of token " + tokenId + " is not finite");
            }
            this.logitBias.put(tokenId, bias);
            return this;
        }
        private BooleanSupplier cancellation = NEVER;

        private Builder() {
        }

        /** Sets the token budget explicitly: a model's {@code max_new_tokens} default does not replace it. */
        public Builder maxNewTokens(int v) {
            this.maxNewTokens = v;
            this.maxNewTokensExplicit = true;
            return this;
        }

        /** Sets the token budget as a fallback, which a model's {@code max_new_tokens} default replaces. */
        public Builder fallbackMaxNewTokens(int v) {
            this.maxNewTokens = v;
            this.maxNewTokensExplicit = false;
            return this;
        }

        public Builder sampler(SamplerConfig v) { this.sampler = Objects.requireNonNull(v, "sampler"); return this; }

        /** Replaces the stop strings. */
        public Builder stopStrings(List<String> v) {
            Objects.requireNonNull(v, "stopStrings");
            this.stopStrings.clear();
            for (String s : v) {
                stopString(s);
            }
            return this;
        }

        /** Replaces the stop strings. */
        public Builder stopStrings(String... v) {
            return stopStrings(Arrays.asList(Objects.requireNonNull(v, "stopStrings")));
        }

        /** Adds one stop string. */
        public Builder stopString(String v) {
            Objects.requireNonNull(v, "stop string");
            if (v.isEmpty()) {
                throw new IllegalArgumentException("stop strings must not be empty");
            }
            this.stopStrings.add(v);
            return this;
        }

        public Builder ignoreEos(boolean v) { this.ignoreEos = v; return this; }

        /** Cancellation hook; null removes it. */
        public Builder cancellation(BooleanSupplier v) { this.cancellation = v == null ? NEVER : v; return this; }

        /**
         * Bans end-of-generation tokens until {@code n} tokens were generated (vLLM's {@code min_tokens};
         * 0 = off, the default).
         */
        public Builder minTokens(int n) {
            if (n < 0) {
                throw new IllegalArgumentException("min tokens needs a non-negative count, got " + n);
            }
            this.minTokens = n;
            return this;
        }

        /** Phrases the output must not spell (vLLM's {@code bad_words}); tokenized into sampling bans. */
        public Builder badWords(List<String> v) {
            this.badWords = new ArrayList<>(Objects.requireNonNull(v, "badWords"));
            return this;
        }

        public Builder badWord(String v) {
            this.badWords.add(Objects.requireNonNull(v, "badWord"));
            return this;
        }

        /**
         * Reports {@code n} scores per generated token: the sampled token plus its {@code n - 1} best alternatives
         * in the {@link #logprobsMode} distribution (llama.cpp's {@code n_probs}; 0 = off, the default).
         */
        public Builder logprobs(int n) {
            if (n < 0) {
                throw new IllegalArgumentException("logprobs needs a non-negative alternative count, got " + n);
            }
            this.logprobs = n;
            return this;
        }

        /**
         * Scores exactly these token ids at every generated step, after the sampled token (vLLM's
         * {@code logprob_token_ids}); they take precedence over the {@link #logprobs} count. Order is kept and
         * repeats collapse to their first place; an empty array clears them.
         *
         * @throws IllegalArgumentException naming {@code logprob_token_ids} for a negative id or more than
         *                                  {@link #MAX_LOGPROB_TOKEN_IDS} distinct ids
         */
        public Builder logprobTokenIds(int... ids) {
            Objects.requireNonNull(ids, "logprob_token_ids");
            int[] kept = new int[ids.length];
            int n = 0;
            for (int id : ids) {
                if (id < 0) {
                    throw new IllegalArgumentException("logprob_token_ids token id " + id + " is negative");
                }
                boolean seen = false;
                for (int j = 0; j < n && !seen; j++) {
                    seen = kept[j] == id;
                }
                if (!seen) {
                    kept[n++] = id;
                }
            }
            if (n > MAX_LOGPROB_TOKEN_IDS) {
                throw new IllegalArgumentException("logprob_token_ids holds " + n + " ids, at most "
                        + MAX_LOGPROB_TOKEN_IDS + " are allowed");
            }
            this.logprobTokenIds = Arrays.copyOf(kept, n);
            return this;
        }

        /** The distribution the captures score (default {@link LogprobsMode#RAW_LOGPROBS}). */
        public Builder logprobsMode(LogprobsMode v) {
            this.logprobsMode = Objects.requireNonNull(v, "logprobs_mode");
            return this;
        }

        /**
         * The first prompt row the prompt captures score (vLLM's {@code prompt_logprob_start}; default 0).
         *
         * @throws IllegalArgumentException naming {@code prompt_logprob_start} for a negative row
         */
        public Builder promptLogprobStart(int row) {
            if (row < 0) {
                throw new IllegalArgumentException("prompt_logprob_start must be non-negative, got " + row);
            }
            this.promptLogprobStart = row;
            return this;
        }

        /**
         * The token ids each scored prompt row scores (vLLM's {@code prompt_logprob_token_ids}); shorter rows are
         * padded with {@link #PROMPT_TOKEN_ID_PAD}, which scores {@code -inf}. Null clears them.
         *
         * @throws IllegalArgumentException naming {@code prompt_logprob_token_ids} for no rows, rows of no ids,
         *                                  or an id below {@link #PROMPT_TOKEN_ID_PAD}
         */
        public Builder promptLogprobTokenIds(int[][] rows) {
            if (rows == null) {
                this.promptLogprobTokenIds = null;
                return this;
            }
            int width = 0;
            for (int[] row : rows) {
                width = Math.max(width, Objects.requireNonNull(row, "prompt_logprob_token_ids row").length);
            }
            if (rows.length == 0 || width == 0) {
                throw new IllegalArgumentException("prompt_logprob_token_ids must be a non-empty integer array"
                        + " of shape [num_rows, num_ids]");
            }
            int[][] padded = new int[rows.length][];
            for (int i = 0; i < rows.length; i++) {
                padded[i] = Arrays.copyOf(rows[i], width);
                Arrays.fill(padded[i], rows[i].length, width, PROMPT_TOKEN_ID_PAD);
                for (int id : rows[i]) {
                    if (id < PROMPT_TOKEN_ID_PAD) {
                        throw new IllegalArgumentException("prompt_logprob_token_ids token id " + id
                                + " is negative (-1 pads a row)");
                    }
                }
            }
            this.promptLogprobTokenIds = padded;
            return this;
        }

        /**
         * Caps the tokens generated inside a reasoning block; the end tag is forced when it runs out
         * (llama.cpp {@code reasoning_budget_tokens}, vLLM {@code thinking_token_budget}; {@link #REASONING_BUDGET_DISABLED}
         * turns it off, the default).
         */
        public Builder reasoningBudget(int tokens) {
            if (tokens < REASONING_BUDGET_DISABLED) {
                throw new IllegalArgumentException("reasoningBudget must be " + REASONING_BUDGET_DISABLED
                        + " (disabled) or a non-negative token count, got " + tokens);
            }
            this.reasoningBudget = tokens;
            return this;
        }

        /** Message injected ahead of the forced end tag when the budget runs out; null or empty for none. */
        public Builder reasoningBudgetMessage(String v) {
            this.reasoningBudgetMessage = v;
            return this;
        }

        /** Arms the budget stage without a budget, so reasoning can be ended mid-stream (llama.cpp {@code reasoning_control}). */
        public Builder reasoningControl(boolean v) {
            this.reasoningControl = v;
            return this;
        }

        /** Replaces the explicit budget start tags; empty (the default) derives them from the reasoning parser. */
        public Builder reasoningBudgetStartTags(List<String> v) {
            this.reasoningBudgetStartTags = tagList(v, "reasoningBudgetStartTags");
            return this;
        }

        /** Replaces the explicit budget end tags; empty (the default) derives them from the reasoning parser. */
        public Builder reasoningBudgetEndTags(List<String> v) {
            this.reasoningBudgetEndTags = tagList(v, "reasoningBudgetEndTags");
            return this;
        }

        private static List<String> tagList(List<String> v, String what) {
            Objects.requireNonNull(v, what);
            for (String t : v) {
                if (t == null || t.isEmpty()) {
                    throw new IllegalArgumentException(what + " must be non-empty strings");
                }
            }
            return new ArrayList<>(v);
        }

        /**
         * The grammar the output must conform to; null removes one. The spec compiles when the generation's
         * sampler is built, so an invalid or untranslatable spec throws {@link IllegalArgumentException} then.
         */
        public Builder constraint(ConstraintSpec v) {
            this.constraint = v;
            return this;
        }

        /** Replaces the stop token ids; each must be non-negative, duplicates collapse. */
        public Builder stopTokenIds(List<Integer> v) {
            Objects.requireNonNull(v, "stopTokenIds");
            int[] ids = new int[v.size()];
            for (int i = 0; i < ids.length; i++) {
                ids[i] = v.get(i);
            }
            return stopTokenIds(ids);
        }

        /** Replaces the stop token ids; each must be non-negative, duplicates collapse. */
        public Builder stopTokenIds(int... v) {
            return copyIds("stop_token_ids", v, ids -> this.stopTokenIds = ids);
        }

        /** Replaces the allowed token ids; each must be non-negative, duplicates collapse. */
        public Builder allowedTokenIds(List<Integer> v) {
            Objects.requireNonNull(v, "allowedTokenIds");
            int[] ids = new int[v.size()];
            for (int i = 0; i < ids.length; i++) {
                ids[i] = v.get(i);
            }
            return allowedTokenIds(ids);
        }

        /** Replaces the allowed token ids; each must be non-negative, duplicates collapse. */
        public Builder allowedTokenIds(int... v) {
            return copyIds("allowed_token_ids", v, ids -> this.allowedTokenIds = ids);
        }

        private Builder copyIds(String name, int[] v, Consumer<int[]> set) {
            Objects.requireNonNull(v, name);
            for (int id : v) {
                if (id < 0) {
                    throw new IllegalArgumentException(name + " token id " + id + " is negative");
                }
            }
            int[] sorted = v.clone();
            Arrays.sort(sorted);
            int distinct = 0;
            for (int i = 0; i < sorted.length; i++) {
                if (distinct == 0 || sorted[distinct - 1] != sorted[i]) {
                    sorted[distinct++] = sorted[i];
                }
            }
            set.accept(Arrays.copyOf(sorted, distinct));
            return this;
        }

        /**
         * Replaces the logprob token ids from a list; see {@link #logprobTokenIds(int...)}.
         */
        public Builder logprobTokenIds(List<Integer> v) {
            Objects.requireNonNull(v, "logprobTokenIds");
            int[] ids = new int[v.size()];
            for (int i = 0; i < ids.length; i++) {
                ids[i] = v.get(i);
            }
            return logprobTokenIds(ids);
        }

        /**
         * Reports per-prompt-token log probabilities (vLLM's {@code prompt_logprobs}): {@code n} entries per
         * position — the prompt token plus its {@code n - 1} top alternatives — or {@link #PROMPT_LOGPROBS_ALL}
         * for every vocabulary token; 0 turns it off (the default).
         */
        public Builder promptLogprobs(int n) {
            if (n < 0 && n != PROMPT_LOGPROBS_ALL) {
                throw new IllegalArgumentException("prompt logprobs need a non-negative count or " + PROMPT_LOGPROBS_ALL
                        + " for all tokens, got " + n);
            }
            this.promptLogprobs = n;
            return this;
        }

        /**
         * Sets the prompt tokens a context shift keeps (llama.cpp server's {@code n_keep}); {@link
         * #KEEP_PROMPT_ALL} keeps the whole prompt, 0 a leading BOS only.
         *
         * @throws IllegalArgumentException for a value below {@link #KEEP_PROMPT_ALL}
         */
        public Builder nKeep(int v) {
            if (v < KEEP_PROMPT_ALL) {
                throw new IllegalArgumentException("n_keep must be " + KEEP_PROMPT_ALL
                        + " (the whole prompt) or a non-negative token count, got " + v);
            }
            this.nKeep = v;
            return this;
        }

        /**
         * Sets the tokens one context shift discards after {@link #nKeep(int)} (llama.cpp server's
         * {@code n_discard}); 0 takes the default half.
         *
         * @throws IllegalArgumentException for a negative count
         */
        public Builder nDiscard(int v) {
            if (v < 0) {
                throw new IllegalArgumentException("n_discard must be non-negative, 0 taking the default half, got "
                        + v);
            }
            this.nDiscard = v;
            return this;
        }

        /**
         * Sets the self-extend factor (llama.cpp {@code --grp-attn-n}): the group every {@link #grpAttnW(int)}
         * positions divides its positions by, {@link #GRP_ATTN_OFF} turning it off. The pair is validated together
         * at {@link #build()}.
         *
         * @throws IllegalArgumentException for a value below {@link #GRP_ATTN_OFF}
         */
        public Builder grpAttnN(int v) {
            if (v < GRP_ATTN_OFF) {
                throw new IllegalArgumentException("grp_attn_n must be at least " + GRP_ATTN_OFF
                        + " (1 turns self-extend off), got " + v);
            }
            this.grpAttnN = v;
            return this;
        }

        /**
         * Sets the self-extend width (llama.cpp {@code --grp-attn-w}): the positions one group spans before its
         * compression; {@link #build()} requires a multiple of {@link #grpAttnN(int)}.
         *
         * @throws IllegalArgumentException for a value below 1
         */
        public Builder grpAttnW(int v) {
            if (v < 1) {
                throw new IllegalArgumentException("grp_attn_w must be positive, got " + v);
            }
            this.grpAttnW = v;
            return this;
        }

        /**
         * Sets the KV chunk reuse length (llama.cpp server's {@code n_cache_reuse}): the cached prompt runs of at
         * least this many tokens one prefill skips past the common prefix, their stored keys and values re-appended
         * at the prompt's own positions; 0 turns it off.
         *
         * @throws IllegalArgumentException for a negative count
         */
        public Builder nCacheReuse(int v) {
            if (v < 0) {
                throw new IllegalArgumentException("n_cache_reuse must be non-negative, 0 turning chunk reuse off,"
                        + " got " + v);
            }
            this.nCacheReuse = v;
            return this;
        }

        /** Sets whether a matched stop string stays in the output text. */
        public Builder includeStopString(boolean v) {
            this.includeStopString = v;
            return this;
        }

        /** Sets whether control and end-of-generation tokens stay out of the output text (default true). */
        public Builder skipSpecialTokens(boolean v) {
            this.skipSpecialTokens = v;
            return this;
        }

        /** Sets whether consecutive special-token texts join with a space (default true). */
        public Builder spacesBetweenSpecialTokens(boolean v) {
            this.spacesBetweenSpecialTokens = v;
            return this;
        }

        /** Sets the n-gram repetition stop; {@link RepetitionDetection#disabled()} (the default) turns it off. */
        public Builder repetitionDetection(RepetitionDetection v) {
            this.repetitionDetection = Objects.requireNonNull(v, "repetitionDetection");
            return this;
        }

        /** Sets the scheduling priority: lower is served earlier under the {@code priority} scheduling policy. */
        public Builder priority(int v) {
            this.priority = v;
            return this;
        }

        /**
         * Sets the chat template variables the reasoning parser selects its configuration from (copied; null clears
         * them). The values are the template's: a boolean {@code enable_thinking}, a string
         * {@code reasoning_effort}, any {@code chat_template_kwargs} entry.
         *
         * @throws IllegalArgumentException for a null or empty variable name
         */
        public Builder templateVariables(Map<String, ?> v) {
            Map<String, Object> copy = new LinkedHashMap<>();
            if (v != null) {
                for (Map.Entry<String, ?> e : v.entrySet()) {
                    if (e.getKey() == null || e.getKey().isEmpty()) {
                        throw new IllegalArgumentException("template variable names must be non-empty");
                    }
                    copy.put(e.getKey(), e.getValue());
                }
            }
            this.templateVariables = copy;
            return this;
        }

        /**
         * The custom logits processor arguments of the request body's {@code vllm_xargs} object (vLLM's
         * {@code SamplingParams.extra_args}); {@link LogitsProcessorArgs#empty()} when the request sets none.
         */
        public Builder logitsProcessorArgs(LogitsProcessorArgs v) {
            this.logitsProcessorArgs = Objects.requireNonNull(v, "logitsProcessorArgs");
            return this;
        }

        /** Sets the prefix-cache salt; null removes it. */
        public Builder cacheSalt(String v) {
            this.cacheSalt = v == null || v.isEmpty() ? null : v;
            return this;
        }

        /**
         * Sets the request's disaggregated prefill/decode parameters (vLLM's {@code kv_transfer_params}); null runs
         * a plain request.
         */
        public Builder kvTransfer(KvTransferParams v) {
            this.kvTransfer = v;
            return this;
        }

        /** Sets the per-request LoRA adapters; null runs the base model. */
        public Builder lora(LoraSelection v) {
            this.lora = v;
            return this;
        }

        /**
         * Sets the beam-search width (vLLM's {@code BeamSearchParams.beam_width}); 1 keeps one beam, so
         * {@link #beamSearch()} stays false.
         *
         * @throws IllegalArgumentException for a value below 1
         */
        public Builder beamWidth(int v) {
            if (v < 1) {
                throw new IllegalArgumentException("beam_width must be at least 1, got " + v);
            }
            this.beamWidth = v;
            return this;
        }

        /**
         * Sets the beam-search length penalty (vLLM's {@code length_penalty}); 1.0 (the default) plain averages
         * a beam's cumulative log probability over its generated tokens.
         *
         * @throws IllegalArgumentException for a non-finite value
         */
        public Builder lengthPenalty(float v) {
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                throw new IllegalArgumentException("length_penalty must be finite, got " + v);
            }
            this.lengthPenalty = v;
            return this;
        }

        /** @throws IllegalArgumentException when {@code maxNewTokens} is neither positive nor {@link #UNLIMITED} */
        public GenerationRequest build() {
            if (maxNewTokens <= 0 && maxNewTokens != UNLIMITED) {
                throw new IllegalArgumentException(
                        "maxNewTokens must be positive, or -1 for unlimited, got " + maxNewTokens);
            }
            if (promptLogprobStart > 0 && promptLogprobs == 0 && promptLogprobTokenIds == null) {
                throw new IllegalArgumentException("prompt_logprob_start requires prompt_logprobs or"
                        + " prompt_logprob_token_ids");
            }
            if (grpAttnN > GRP_ATTN_OFF && grpAttnW % grpAttnN != 0) {
                throw new IllegalArgumentException("grp_attn_w must be a multiple of grp_attn_n"
                        + " (llama.cpp: ga_w % ga_n == 0), got " + grpAttnW + " / " + grpAttnN);
            }
            return new GenerationRequest(this);
        }
    }

    /**
     * Streaming stop-string matcher (the hold-back rule of llama.cpp server {@code find_stopping_strings}): text is
     * released as soon as it can no longer be the start of a stop string; the earliest complete stop string (first
     * in list order on ties) ends the stream, and neither it nor anything after it is released. Not thread-safe;
     * one instance per generation.
     */
    public static final class StopStringFilter implements TokenCallback {

        private final List<String> stops;
        private final TokenCallback downstream;
        private final boolean includeStopString;
        private final StringBuilder pending = new StringBuilder();
        private final StringBuilder released = new StringBuilder();
        private String matched;

        /** Copies {@code stops}; {@code includeStopString} keeps the matched stop in the released text. */
        public StopStringFilter(List<String> stops, TokenCallback downstream) {
            this(stops, downstream, false);
        }

        StopStringFilter(List<String> stops, TokenCallback downstream, boolean includeStopString) {
            this.stops = Objects.requireNonNull(stops, "stops");
            this.downstream = downstream;
            this.includeStopString = includeStopString;
        }

        /** Feeds the text of one token; ignored once a stop string matched. */
        @Override
        public void onToken(int tokenId, String text) {
            if (matched != null || text == null || text.isEmpty()) {
                return;
            }
            pending.append(text);
            int hit = -1;
            String hitStop = null;
            for (String s : stops) {
                int at = pending.indexOf(s);
                if (at >= 0 && (hit < 0 || at < hit)) {
                    hit = at;
                    hitStop = s;
                }
            }
            if (hitStop != null) {
                matched = hitStop;
                int keep = includeStopString ? hit + hitStop.length() : hit;
                release(tokenId, keep);
                pending.setLength(0);
                return;
            }
            release(tokenId, pending.length() - heldBack());
        }

        /** Whether text is held back as the possible start of a stop string. */
        public boolean holding() {
            return pending.length() > 0;
        }

        /** Releases the withheld tail when the stream ended without a stop string. */
        public void finish() {
            if (matched == null) {
                release(-1, pending.length());
            }
        }

        /** Whether a stop string matched. */
        public boolean stopped() {
            return matched != null;
        }

        /** The stop string that matched, or null. */
        public String matched() {
            return matched;
        }

        /** All text released so far. */
        public String releasedText() {
            return released.toString();
        }

        /** Length of the longest suffix of the pending text that is a proper prefix of a stop string. */
        private int heldBack() {
            int best = 0;
            int len = pending.length();
            for (String s : stops) {
                int max = Math.min(len, s.length() - 1);
                for (int k = max; k > best; k--) {
                    if (pendingEndsWith(s, k)) {
                        best = k;
                        break;
                    }
                }
            }
            return best;
        }

        /** Whether the pending text ends with the first {@code k} chars of {@code s}. */
        private boolean pendingEndsWith(String s, int k) {
            int base = pending.length() - k;
            for (int i = 0; i < k; i++) {
                if (pending.charAt(base + i) != s.charAt(i)) {
                    return false;
                }
            }
            return true;
        }

        private void release(int tokenId, int count) {
            if (count <= 0) {
                return;
            }
            String out = pending.substring(0, count);
            pending.delete(0, count);
            released.append(out);
            if (downstream != null) {
                downstream.onToken(tokenId, out);
            }
        }
    }
}
