package com.eonmux.neuralfree.gguf.engine.batch;

import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A multi-sequence drafter of a {@link BatchEngine}: one proposal stream per running sequence (vLLM's per-request
 * draft workers of {@code scheduled_spec_decode_tokens}, llama.cpp server's per-slot speculative context,
 * {@code common_speculative_init} over {@code n_parallel} slots).
 *
 * <p>Protocol per decode step: the engine asks {@link #draftBatch} once for every sequence that decodes in the
 * step, after each sampled its pending token, verifies the whole batch's proposals in one forward call, and reports
 * each verified round's outcome with {@link #accept} ({@code accept(seq, 0)} when a proposal is not verified:
 * shorter than the draft minimum, or not scheduled). Between a draft and its {@code accept} the engine may read the
 * proposal distribution of each drafted position ({@link #draftProbs}) for rejection sampling. The drafter reads
 * the target's stored tokens through the paged memory it was built with; it holds no lease of the target. Not
 * thread-safe: it runs on the engine's thread.
 */
public interface BatchDrafter extends AutoCloseable {

    /**
     * One sequence's draft request of a step: the sequence slot, the token it just sampled and has not fed back
     * yet, the most tokens to propose, and the sequence's own sampling of a probabilistic drafter (the request's
     * temperature, and the stream its draws come from). Immutable references.
     */
    final class Request {
        private final int seq;
        private final int lastToken;
        private final int limit;
        private final float temperature;
        private final SplittableRandom rng;
        private final com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig watermark;
        private final int promptLength;

        /**
         * @param temperature the request's sampling temperature; 0 or less drafts greedily
         * @param rng         the sequence's draft stream
         * @throws IllegalArgumentException for a limit below 1 or a NaN temperature
         */
        public Request(int seq, int lastToken, int limit, float temperature, SplittableRandom rng) {
            this(seq, lastToken, limit, temperature, rng, null, 0);
        }

        /**
         * A request whose draft tokens are watermarked under {@code watermark}'s draft role, keyed on the
         * sequence's committed tokens and the round's earlier proposals.
         *
         * @param watermark    the request's watermark configuration, or null for an ordinary draft
         * @param promptLength the prompt tokens of the sequence, which the {@code single_turn} scope leaves out of
         *                     its repeated-context search
         * @throws IllegalArgumentException for a limit below 1, a NaN temperature or a negative prompt length
         */
        public Request(int seq, int lastToken, int limit, float temperature, SplittableRandom rng,
                       com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig watermark, int promptLength) {
            if (limit < 1) {
                throw new IllegalArgumentException("a draft request needs a positive limit, got " + limit);
            }
            if (Float.isNaN(temperature)) {
                throw new IllegalArgumentException("the draft temperature is NaN");
            }
            if (promptLength < 0) {
                throw new IllegalArgumentException("prompt length " + promptLength + " is negative");
            }
            this.seq = seq;
            this.lastToken = lastToken;
            this.limit = limit;
            this.temperature = temperature;
            this.rng = Objects.requireNonNull(rng, "rng");
            this.watermark = watermark;
            this.promptLength = promptLength;
        }

        /** The engine's sequence slot the proposal is for. */
        public int seq() {
            return seq;
        }

        /** The token the sequence just sampled. */
        public int lastToken() {
            return lastToken;
        }

        /** The most tokens to propose. */
        public int limit() {
            return limit;
        }

        /** The request's temperature; 0 or less is greedy. */
        public float temperature() {
            return temperature;
        }

        /** The sequence's draft stream. */
        public SplittableRandom rng() {
            return rng;
        }

        /** The request's watermark configuration, or null. */
        public com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig watermark() {
            return watermark;
        }

        /** The prompt tokens of the sequence. */
        public int promptLength() {
            return promptLength;
        }

        @Override
        public String toString() {
            return "Request{seq=" + seq + ", last=" + lastToken + ", limit=" + limit + ", t=" + temperature
                    + (watermark == null ? "" : ", watermark") + '}';
        }
    }

    /**
     * Proposes the tokens following {@code lastToken}, the token sequence {@code seq} just sampled and has not fed
     * back yet.
     *
     * @param seq the engine's sequence slot the proposal is for
     * @param n   the maximum number of tokens to propose, positive
     * @return between 0 and {@code n} proposed token ids in target vocabulary
     * @throws IndexOutOfBoundsException when {@code seq} is outside the memory's sequences
     */
    int[] draft(int seq, int lastToken, int n);

    /**
     * Proposes for every request of one step, in request order; each answer has between 0 and the request's limit
     * tokens in target vocabulary. A drafter that runs a model drafts every sequence in the same forward batches
     * (vLLM's batched proposer). Default: {@link #draft} per request.
     *
     * @throws RuntimeException when drafting fails; the engine then fails every sequence of the call
     */
    default int[][] draftBatch(List<Request> requests) {
        int[][] out = new int[requests.size()][];
        for (int i = 0; i < out.length; i++) {
            Request r = requests.get(i);
            out[i] = draft(r.seq(), r.lastToken(), r.limit());
        }
        return out;
    }

    /**
     * The proposal distribution the last draft of {@code seq} drew its token at {@code position} from, over the
     * target's vocabulary (vLLM's {@code draft_probs}), or null when that token was proposed deterministically
     * (a lookup match, a greedy draft): rejection sampling then treats the proposal as one-hot. Default: null.
     *
     * @return a probability row (no NaN, no negatives, the drafted token's entry positive), or null
     */
    default float[] draftProbs(int seq, int position) {
        return null;
    }

    /**
     * Proposes a draft tree per request of one step, in request order, or null when this drafter proposes chains
     * only (the default): the engine runs the chain protocol of {@link #draftBatch} then. Each answer proposes at
     * most the request's limit nodes, its node list keeping parents ahead of children and its path scores descending
     * along it ({@code TreeProposal}), so the verify budget's cut is a node prefix. The engine verifies every
     * proposal of the step in one masked forward call — the whole point of the tree over the chain — and reports
     * each verified round's outcome with {@link #accept}, {@code nAccepted} counting the accepted path's nodes.
     *
     * @throws RuntimeException when drafting fails; the engine then fails every sequence of the call
     */
    default TreeProposal[] draftTreeBatch(List<Request> requests) {
        return null;
    }

    /**
     * Whether this drafter answers {@link #draftTreeBatch} with tree proposals (default false): the engine refuses
     * the tree verify ({@code neuralfree.draft.tree}) over a drafter that proposes chains only, naming this
     * drafter.
     */
    default boolean proposesTrees() {
        return false;
    }

    /**
     * Reports verification of the last {@link #draft} of {@code seq}: its first {@code nAccepted} tokens were kept
     * and the rest rejected.
     *
     * @throws IllegalArgumentException when {@code nAccepted} is negative
     */
    void accept(int seq, int nAccepted);

    /**
     * Drops the per-sequence state of {@code seq}: the engine calls it when the slot retires, requeues or is
     * preempted, before the slot serves another sequence. Default: nothing (a stateless drafter).
     */
    default void reset(int seq) {
    }

    /**
     * Whether this drafter can draw its proposals under a draft watermark key. A model drafter that samples its
     * logits does; a lookup or suffix drafter that proposes stored tokens does not. A watermarked speculative
     * request is refused by name on a drafter that cannot.
     */
    default boolean supportsWatermarking() {
        return false;
    }

    /** Releases the drafter's resources. Default: nothing. */
    @Override
    default void close() {
    }
}
