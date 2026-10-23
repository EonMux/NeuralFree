package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.Map;
import java.util.Objects;

/**
 * When an autoregressive generation ends, and how its tokens become output text. Immutable; {@link #start} opens
 * the per-generation {@link Session}, which drivers (the generation loop, the speculative decoder) feed every
 * sampled token.
 *
 * <p>Stop rules, checked per sampled token in this order:
 * <ol>
 *   <li>an end-of-generation token ({@link SpecialTokens#isEog}) ends with {@link StopReason#EOS} and is not part
 *       of the output, unless ignore-eos is in effect;</li>
 *   <li>the token's text completes a stop string: {@link StopReason#STOP_STRING}, the token kept, the text cut
 *       before the stop string;</li>
 *   <li>the token budget is reached: {@link StopReason#MAX_TOKENS}.</li>
 * </ol>
 * Cancellation is polled by the driver through {@link Session#cancelled()}.
 *
 * <p>Ignore-eos follows llama.cpp (llama.cpp common/common.cpp:1342-1360): it is dropped with a warning when the
 * vocabulary has no EOS token; otherwise the end-of-generation tokens are banned from sampling
 * ({@link #bannedTokens()}), so they are never produced.
 *
 * <p>Output text: each token is rendered by a UTF-8 streaming decoder, except that the special tokens of the
 * {@link ReasoningFilter}'s tags ({@link ReasoningFilter#tagTokens()}) are rendered as their tag text, so tags
 * reach the stop-string matcher and the reasoning parser even when the detokenizer hides them. Text then passes
 * the stop-string filter ({@link GenerationRequest#newStopFilter}) and the reasoning filter, which routes it to
 * the session's {@link ReasoningFilter.Sink}. The result text is the stop-filtered text of every channel.
 */
public final class StopCriteria {

    private static final Logger LOGGER = LogManager.getLogger(StopCriteria.class);

    private final GenerationRequest request;
    private final GGUFTokenizer tokenizer;
    private final SpecialTokens special;
    private final boolean ignoreEos;
    private final Map<Integer, String> tagTokens;
    private final ReasoningFilter textFilter;

    private StopCriteria(GenerationRequest request, GGUFTokenizer tokenizer, ReasoningFilter reasoning) {
        this.request = request;
        this.tokenizer = tokenizer;
        this.special = tokenizer.specialTokens();
        boolean noEos = special.eos() == SpecialTokens.NULL;
        if (request.ignoreEos() && noEos) {
            LOGGER.warn("the vocabulary has no EOS token, ignoring ignore-eos");
        }
        this.ignoreEos = request.ignoreEos() && !noEos;
        this.tagTokens = reasoning.tagTokens();
        this.textFilter = textOnly(reasoning);
    }

    /**
     * Stop criteria of {@code request} over {@code tokenizer}'s special tokens.
     *
     * @param reasoning the reasoning parser of the model's chat template ({@link ReasoningFilter#none()} for
     *                  plain text)
     */
    public static StopCriteria of(GenerationRequest request, GGUFTokenizer tokenizer, ReasoningFilter reasoning) {
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(reasoning, "reasoning");
        return new StopCriteria(request, tokenizer, reasoning);
    }

    /** The same parser without token-id tag matching; tag tokens are rendered as tag text before it. */
    private static ReasoningFilter textOnly(ReasoningFilter f) {
        switch (f.mode()) {
            case THINK_TAGS:
                return ReasoningFilter.thinkTags(f.startTags(), f.endTags(), f.interruptTags());
            case HARMONY_CHANNELS:
                return ReasoningFilter.harmony();
            default:
                return ReasoningFilter.none();
        }
    }

    public GenerationRequest request() {
        return request;
    }

    /** Whether end-of-generation tokens are banned and never end generation. */
    public boolean ignoresEos() {
        return ignoreEos;
    }

    /** Whether {@code token} ends generation by itself. */
    public boolean isEndOfGeneration(int token) {
        return !ignoreEos && special.isEog(token);
    }

    /**
     * Token ids the sampler must never select (llama.cpp common/sampling.cpp:329-341): the vocabulary's
     * {@code tokenizer.ggml.suppress_tokens}, plus every end-of-generation token under ignore-eos. Sorted,
     * without duplicates.
     */
    public int[] bannedTokens() {
        int[] suppress = special.suppressTokens();
        int[] out = Arrays.copyOf(suppress, suppress.length + (ignoreEos ? special.eogIds().size() : 0));
        int n = suppress.length;
        if (ignoreEos) {
            for (int id : special.eogIds()) {
                out[n++] = id;
            }
        }
        Arrays.sort(out, 0, n);
        int distinct = 0;
        for (int i = 0; i < n; i++) {
            if (distinct == 0 || out[distinct - 1] != out[i]) {
                out[distinct++] = out[i];
            }
        }
        return Arrays.copyOf(out, distinct);
    }

    /**
     * Renders {@code prompt} as the reasoning parser sees it (tag tokens as tag text, other tokens through a
     * streaming decoder), for {@link ReasoningFilter#newState(ReasoningFilter.Sink, String)}.
     */
    public String renderPrompt(int[] prompt) {
        Objects.requireNonNull(prompt, "prompt");
        GGUFTokenizer.StreamingDecoder decoder = tokenizer.createStreamingDecoder();
        StringBuilder sb = new StringBuilder();
        for (int token : prompt) {
            sb.append(render(decoder, token));
        }
        return sb.append(decoder.flush()).toString();
    }

    private String render(GGUFTokenizer.StreamingDecoder decoder, int token) {
        String tag = tagTokens.get(token);
        if (tag == null) {
            return decoder.decodeToken(token);
        }
        String pending = decoder.flush();
        return pending.isEmpty() ? tag : pending + tag;
    }

    /**
     * Opens the output side of one generation.
     *
     * @param sink           receives the routed output text; {@link ReasoningFilter#contentTo} forwards content only
     * @param renderedPrompt the prompt as text (the rendered chat template, or {@link #renderPrompt}), used to
     *                       detect a generation prefix that opens reasoning; null when unknown
     */
    public Session start(ReasoningFilter.Sink sink, String renderedPrompt) {
        Objects.requireNonNull(sink, "sink");
        return new Session(sink, renderedPrompt);
    }

    /**
     * The output side of one generation: the accepted tokens, the streaming decoder, the stop-string filter and
     * the reasoning state. Not thread-safe.
     */
    public final class Session {

        private final GGUFTokenizer.StreamingDecoder decoder;
        private final ReasoningFilter.State reasoningState;
        private final GenerationRequest.StopStringFilter stopFilter;
        private int[] tokens = new int[64];
        private int count;
        private boolean finished;

        private Session(ReasoningFilter.Sink sink, String renderedPrompt) {
            this.decoder = tokenizer.createStreamingDecoder();
            this.reasoningState = textFilter.newState(sink, renderedPrompt);
            this.stopFilter = request.newStopFilter(reasoningState);
        }

        /**
         * Feeds one sampled token.
         *
         * @return the reason generation ends after this token, or null to continue
         * @throws IllegalStateException after {@link #finish}
         */
        public StopReason offer(int token) {
            if (finished) {
                throw new IllegalStateException("generation session already finished");
            }
            if (isEndOfGeneration(token)) {
                return StopReason.EOS;
            }
            if (count == tokens.length) {
                tokens = Arrays.copyOf(tokens, count * 2);
            }
            tokens[count++] = token;
            stopFilter.onToken(token, render(decoder, token));
            if (stopFilter.stopped()) {
                return StopReason.STOP_STRING;
            }
            if (!request.unlimited() && count >= request.maxNewTokens()) {
                return StopReason.MAX_TOKENS;
            }
            return null;
        }

        /** Polls the request's cancellation hook. */
        public boolean cancelled() {
            return request.cancellation().getAsBoolean();
        }

        /** Tokens accepted into the output so far. */
        public int tokenCount() {
            return count;
        }

        /** Copy of the tokens accepted into the output. */
        public int[] tokens() {
            return Arrays.copyOf(tokens, count);
        }

        /** New tokens the budget still allows, or {@link Integer#MAX_VALUE} when unlimited. */
        public int remaining() {
            return request.unlimited() ? Integer.MAX_VALUE : Math.max(0, request.maxNewTokens() - count);
        }

        /**
         * Releases the withheld text (the decoder's partial character, the stop filter's held-back tail, the
         * reasoning parser's buffer) and builds the result. When the released tail completes a stop string, the
         * reason becomes {@link StopReason#STOP_STRING}.
         *
         * @param reason  why the driver ended generation
         * @param timings measured times
         * @throws IllegalStateException when called twice
         */
        public GenerationResult finish(StopReason reason, GenerationResult.Timings timings) {
            Objects.requireNonNull(reason, "reason");
            Objects.requireNonNull(timings, "timings");
            if (finished) {
                throw new IllegalStateException("generation session already finished");
            }
            finished = true;
            if (!stopFilter.stopped()) {
                stopFilter.onToken(-1, decoder.flush());
                stopFilter.finish();
            }
            reasoningState.finish();
            StopReason r = stopFilter.stopped() ? StopReason.STOP_STRING : reason;
            if (r != StopReason.STOP_STRING && reason == StopReason.STOP_STRING) {
                throw new IllegalStateException("stop reason STOP_STRING without a matched stop string");
            }
            return new GenerationResult(tokens(), stopFilter.releasedText(), r,
                    r == StopReason.STOP_STRING ? stopFilter.matched() : null, timings);
        }
    }
}
