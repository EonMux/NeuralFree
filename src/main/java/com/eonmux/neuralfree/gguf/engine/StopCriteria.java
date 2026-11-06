package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.SpecialTokens;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.nio.charset.StandardCharsets;
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
 *   <li>an end-of-generation token ({@link SpecialTokens#isEog}) or one of the request's stop token ids
 *       ({@link GenerationRequest#stopTokenIds()}) ends with {@link StopReason#EOS} and is not part of the
 *       output, unless ignore-eos is in effect for the vocabulary's own end-of-generation set;</li>
 *   <li>the token's text completes a stop string: {@link StopReason#STOP_STRING}, the token kept, the text cut
 *       before the stop string (the stop string included when {@link GenerationRequest#includeStopString()});</li>
 *   <li>the token budget is reached: {@link StopReason#MAX_TOKENS};</li>
 *   <li>with at least {@link GenerationRequest#minTokens()} tokens generated, the output ends in a pattern the
 *       request's {@link RepetitionDetection} detects: {@link StopReason#REPETITION}, the token kept (vLLM
 *       {@code check_stop}, vllm/v1/core/sched/utils.py).</li>
 * </ol>
 * Cancellation is polled by the driver through {@link Session#cancelled()}.
 *
 * <p>Ignore-eos follows llama.cpp (llama.cpp common/common.cpp:1342-1360): it is dropped with a warning when the
 * vocabulary has no EOS token; otherwise the end-of-generation tokens are banned from sampling
 * ({@link #bannedTokens()}), so they are never produced.
 *
 * <p>Output text: each token is rendered by a UTF-8 streaming decoder, except that the special tokens of the
 * {@link ReasoningFilter}'s tags ({@link ReasoningFilter#tagTokens()}) are rendered as their tag text, so tags
 * reach the stop-string matcher and the reasoning parser even when the detokenizer hides them. With
 * {@code skip_special_tokens} false, control and end-of-generation tokens render as their raw piece texts
 * (consecutive ones joined with a space under {@code spaces_between_special_tokens}). Text then passes
 * the stop-string filter ({@link GenerationRequest#newStopFilter}) and the reasoning filter, which routes it to
 * the session's {@link ReasoningFilter.Sink}. The result text is the stop-filtered text of every channel.
 */
public final class StopCriteria {

    private static final Logger LOGGER = LogManager.getLogger(StopCriteria.class);

    private final GenerationRequest request;
    private final GGUFTokenizer tokenizer;
    private final SpecialTokens special;
    private final boolean ignoreEos;
    private final int[] stopTokenIds;
    private final RepetitionDetection repetition;
    private final Map<Integer, String> tagTokens;
    private final ReasoningFilter textFilter;
    /** Whether special tokens render as their piece text ({@code skip_special_tokens} false). */
    private final boolean showSpecialTokens;
    /** Whether consecutive special-token texts join with a space ({@code spaces_between_special_tokens}). */
    private final boolean spacesBetweenSpecialTokens;

    private StopCriteria(GenerationRequest request, GGUFTokenizer tokenizer, ReasoningFilter reasoning) {
        this.request = request;
        this.tokenizer = tokenizer;
        this.special = tokenizer.specialTokens();
        boolean noEos = special.eos() == SpecialTokens.NULL;
        if (request.ignoreEos() && noEos) {
            LOGGER.warn("the vocabulary has no EOS token, ignoring ignore-eos");
        }
        this.ignoreEos = request.ignoreEos() && !noEos;
        this.stopTokenIds = request.stopTokenIds();
        this.repetition = request.repetitionDetection();
        this.tagTokens = reasoning.tagTokens();
        this.textFilter = reasoning.withoutTagTokens();
        this.showSpecialTokens = !request.skipSpecialTokens();
        this.spacesBetweenSpecialTokens = request.spacesBetweenSpecialTokens();
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

    public GenerationRequest request() {
        return request;
    }

    /** Whether end-of-generation tokens are banned and never end generation. */
    public boolean ignoresEos() {
        return ignoreEos;
    }

    /** Whether {@code token} ends generation by itself. */
    public boolean isEndOfGeneration(int token) {
        if (!ignoreEos && special.isEog(token)) {
            return true;
        }
        for (int id : stopTokenIds) {
            if (id == token) {
                return true;
            }
        }
        return false;
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
        boolean lastSpecialEnded = false;
        for (int token : prompt) {
            String text = render(decoder, token);
            text = spaced(text, token, lastSpecialEnded);
            lastSpecialEnded = revealsSpecial(token) && !text.isEmpty();
            sb.append(text);
        }
        return sb.append(decoder.flush()).toString();
    }

    private String render(GGUFTokenizer.StreamingDecoder decoder, int token) {
        String tag = tagTokens.get(token);
        if (tag != null) {
            String pending = decoder.flush();
            return pending.isEmpty() ? tag : pending + tag;
        }
        if (revealsSpecial(token)) {
            String pending = decoder.flush();
            String piece = new String(tokenizer.pieceWithSpecial(token), StandardCharsets.UTF_8);
            return pending.isEmpty() ? piece : pending + piece;
        }
        return decoder.decodeToken(token);
    }

    /** Whether {@code token} renders as its own special-piece text ({@code skip_special_tokens} false). */
    private boolean revealsSpecial(int token) {
        return showSpecialTokens && tagTokens.get(token) == null && special.isHidden(token);
    }

    /** The text one token renders, the space between consecutive special tokens included. */
    private String spaced(String text, int token, boolean lastSpecialEnded) {
        if (spacesBetweenSpecialTokens && lastSpecialEnded && revealsSpecial(token) && !text.isEmpty()) {
            return " " + text;
        }
        return text;
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
        private final ReasoningFilter.Sink sink;
        private final ReasoningFilter.State reasoningState;
        private final GenerationRequest.StopStringFilter stopFilter;
        private boolean lastSpecialEnded;
        private int[] tokens = new int[64];
        private int count;
        private int reasoningTokens;
        /** Reasoning tokens whose text is still held back: they count once it streams as reasoning. */
        private int heldReasoning;
        /** The request's stop token id that ended the generation, or {@link GenerationResult#NO_STOP_TOKEN}. */
        private int endTokenId = GenerationResult.NO_STOP_TOKEN;
        private boolean finished;

        private Session(ReasoningFilter.Sink sink, String renderedPrompt) {
            this.decoder = tokenizer.createStreamingDecoder();
            this.sink = sink;
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
                if (endedOnStopToken(token)) {
                    endTokenId = token;
                }
                return StopReason.EOS;
            }
            if (count == tokens.length) {
                tokens = Arrays.copyOf(tokens, count * 2);
            }
            tokens[count++] = token;
            sink.stepped(token);
            boolean reasoningBefore = reasoningState.inReasoning();
            String text = spaced(render(decoder, token), token, lastSpecialEnded);
            lastSpecialEnded = revealsSpecial(token) && !text.isEmpty();
            stopFilter.onToken(token, text);
            if (reasoningBefore) {
                countReasoning();
            }
            if (stopFilter.stopped()) {
                return StopReason.STOP_STRING;
            }
            if (!request.unlimited() && count >= request.maxNewTokens()) {
                return StopReason.MAX_TOKENS;
            }
            if (count >= request.minTokens() && repetition.detects(tokens, count)) {
                return StopReason.REPETITION;
            }
            return null;
        }

        /**
         * Counts a token fed in reasoning, vLLM's {@code count_reasoning_tokens} over text: tokens that end reasoning
         * are the closing tag, so a token whose text is held back (a possible tag or stop string) counts only once
         * reasoning goes on past it.
         */
        private void countReasoning() {
            if (!reasoningState.inReasoning()) {
                heldReasoning = 0;
            } else if (stopFilter.holding() || reasoningState.holding()) {
                heldReasoning++;
            } else {
                reasoningTokens += heldReasoning + 1;
                heldReasoning = 0;
            }
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
            if (reasoningState.inReasoning() && !stopFilter.stopped()) {
                // the held tail was reasoning text, not a tag
                reasoningTokens += heldReasoning;
            }
            heldReasoning = 0;
            reasoningState.finish();
            StopReason r = stopFilter.stopped() ? StopReason.STOP_STRING : reason;
            if (r != StopReason.STOP_STRING && reason == StopReason.STOP_STRING) {
                throw new IllegalStateException("stop reason STOP_STRING without a matched stop string");
            }
            // a stop string that surfaces only at the finish owns the stop reason, so the token id drops
            int stopToken = r == StopReason.EOS ? endTokenId : GenerationResult.NO_STOP_TOKEN;
            return new GenerationResult(tokens(), stopFilter.releasedText(), r,
                    r == StopReason.STOP_STRING ? stopFilter.matched() : null, timings)
                    .withReasoningTokens(reasoningTokens)
                    .withStopTokenId(stopToken);
        }

        /**
         * Whether {@code token} is one of the request's stop token ids and not the vocabulary's own EOS token
         * (vLLM's {@code check_stop}: the EOS ends without a stop reason, unless ignore-eos leaves the stop
         * token ids the only way it ends).
         */
        private boolean endedOnStopToken(int token) {
            if (!ignoreEos && token == special.eos()) {
                return false;
            }
            for (int id : stopTokenIds) {
                if (id == token) {
                    return true;
                }
            }
            return false;
        }
    }
}
