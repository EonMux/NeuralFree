package com.eonmux.neuralfree.gguf.engine.watermark;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.DoubleSupplier;

/**
 * The draft leg of speculative-decoding watermarking (vLLM's {@code DraftWatermarker}): while the target samples the
 * tokens it selects under the target role, a drafter draws each proposed token under the draft role, keyed on the
 * last {@link WatermarkConfig#contextWidth()} tokens before the proposal. A confirmed proposal therefore carries the
 * draft key and a replacement or bonus token the target key, which is exactly what the dual-key detector fuses.
 *
 * <p>The context is the target's committed tokens followed by the round's earlier drafts, because a later draft
 * position follows the earlier proposals whether or not the target keeps them (vLLM's {@code prior_contexts}). The
 * repeated-context search of the configuration applies to those contexts: a context already seen is drawn from the
 * ordinary distribution instead, using the caller's uniform stream, as vLLM's seeded draft sampler does.
 *
 * <p>Inactive when the configuration gives the draft role no key (a single-key configuration with
 * {@code allow_target_only_watermarking}), in which case {@link #of} returns null and drafters keep their ordinary
 * pick. Immutable configuration, mutable per-sequence state; not thread-safe.
 */
public final class DraftWatermarker {

    /** The vocabulary the top-k confidence of a draft row is measured over, as the greedy draft pick. */
    private static final int CONFIDENCE_TOP_K = 10;

    private final Watermarker watermarker;
    private final WatermarkState state;
    private final int contextWidth;
    private final Integer maxHistory;
    /** The context ring of the next draw: the last accepted tokens, shifted by the round's drafts. */
    private final int[] context;
    /** The contexts of the round's earlier draws, oldest first. */
    private final List<int[]> roundContexts = new ArrayList<>();
    /** The committed tokens the state was synced to. */
    private int[] synced = new int[0];
    private int promptLength;

    private DraftWatermarker(WatermarkConfig config, Watermarker watermarker) {
        this.watermarker = watermarker;
        this.state = new WatermarkState(config);
        this.contextWidth = config.contextWidth();
        this.maxHistory = config.deduplicateContextsMaxHistory();
        this.context = new int[contextWidth];
        Arrays.fill(context, -1);
    }

    /**
     * The draft watermarker of a configuration, or null when the draft role has no key.
     *
     * @throws IllegalArgumentException when a single-key configuration does not allow target-only watermarking
     */
    public static DraftWatermarker of(WatermarkConfig config) {
        if (config == null) {
            return null;
        }
        Watermarker draft = config.draftWatermarker();
        return draft == null ? null : new DraftWatermarker(config, draft);
    }

    /**
     * Starts a generation whose first {@code promptLength} committed tokens are its prompt.
     *
     * @throws IllegalArgumentException for a negative prompt length
     */
    public void reset(int promptLength) {
        if (promptLength < 0) {
            throw new IllegalArgumentException("prompt length " + promptLength + " is negative");
        }
        state.reset();
        Arrays.fill(context, -1);
        roundContexts.clear();
        synced = new int[0];
        this.promptLength = promptLength;
    }

    /**
     * Brings the state to the target's committed tokens. Tokens since the last {@link #sync} enter the history
     * (incrementally when the stored record was extended); a record rewritten from an earlier position restarts it.
     * The round's drafts are dropped.
     */
    public void sync(int[] committed) {
        Objects.requireNonNull(committed, "committed");
        int common = commonPrefix(synced, committed);
        if (common < synced.length) {
            rebuild(committed);
        } else {
            accept(common, committed);
            synced = committed.clone();
        }
    }

    /**
     * Accepts {@code committed[from..]} into the history, marking the generation's prompt tokens as such at their
     * boundary: the {@code single_turn} scope leaves them out of its repeated-context search.
     */
    private void accept(int from, int[] committed) {
        for (int i = from; i < committed.length; i++) {
            if (i == 0 && promptLength > 0) {
                state.startPrompt();
            } else if (i == promptLength) {
                state.endPrompt();
            }
            state.accept(committed[i]);
        }
        if (promptLength > 0 && committed.length == promptLength) {
            // the record ends at the prompt: close it so the partial-context rule counts no generated tokens
            state.endPrompt();
        }
    }

    /** {@link #sync(int[])} with a token the target sampled and has not stored yet, the draft's last token. */
    public void sync(int[] committed, int pendingToken) {
        Objects.requireNonNull(committed, "committed");
        int[] tokens = Arrays.copyOf(committed, committed.length + 1);
        tokens[committed.length] = pendingToken;
        sync(tokens);
    }

    /** Accepts the whole record as history, marking its first {@link #promptLength} tokens as the prompt. */
    private void rebuild(int[] committed) {
        state.reset();
        state.startPrompt();
        int prompt = Math.min(promptLength, committed.length);
        for (int i = 0; i < prompt; i++) {
            state.accept(committed[i]);
        }
        state.endPrompt();
        for (int i = prompt; i < committed.length; i++) {
            state.accept(committed[i]);
        }
        synced = committed.clone();
    }

    private static int commonPrefix(int[] a, int[] b) {
        int n = Math.min(a.length, b.length);
        int i = 0;
        while (i < n && a[i] == b[i]) {
            i++;
        }
        return i;
    }

    /** Starts a draft round: the context is the committed history's tail, and no draft precedes the first draw. */
    public void beginRound() {
        int[] tail = state.context();
        System.arraycopy(tail, 0, context, 0, contextWidth);
        roundContexts.clear();
    }

    /**
     * Draws one draft token from {@code logits[0..count)} at the request's temperature {@code temperature}: the keyed
     * Gumbel-max selection under the draft role, or, for a context the repeated-context search already saw, an
     * ordinary categorical draw from the same temperature-scaled distribution.
     *
     * @param uniforms one uniform in {@code [0, 1)} per ordinary draw; the keyed draw reads none
     * @throws IllegalArgumentException for a non-positive temperature, a count outside the row, or a negative token
     */
    public Draw draw(float[] logits, int count, float temperature, DoubleSupplier uniforms) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(uniforms, "uniforms");
        if (!(temperature > 0f)) {
            throw new IllegalArgumentException("a watermarked draft draw needs a positive temperature, got "
                    + temperature);
        }
        if (count < 1 || count > logits.length) {
            throw new IllegalArgumentException("cannot draw from " + count + " of " + logits.length + " logits");
        }
        float[] scaled = new float[count];
        for (int i = 0; i < count; i++) {
            scaled[i] = logits[i] / temperature;
        }
        float[] proposal = softmax(scaled, count);
        int[] contextAtDraw = context.clone();
        boolean repeated = state.repeatedContext(contextAtDraw, roundContexts.size())
                || repeatedInRound(contextAtDraw);
        int token;
        if (repeated) {
            token = categorical(proposal, uniforms.getAsDouble());
        } else {
            int[] tokenIds = new int[count];
            for (int i = 0; i < count; i++) {
                tokenIds[i] = i;
            }
            token = watermarker.sample(tokenIds, scaled, count, contextAtDraw, uniforms);
        }
        shift(context, token);
        roundContexts.add(contextAtDraw);
        return new Draw(token, topProbability(logits, count), proposal);
    }

    /** Whether {@code contextAtDraw} occurred at one of the round's earlier drafts inside the history window. */
    private boolean repeatedInRound(int[] contextAtDraw) {
        int searched = maxHistory == null ? roundContexts.size()
                : Math.min(roundContexts.size(), maxHistory);
        for (int i = 1; i <= searched; i++) {
            if (Arrays.equals(contextAtDraw, roundContexts.get(roundContexts.size() - i))) {
                return true;
            }
        }
        return false;
    }

    private static void shift(int[] ring, int token) {
        if (ring.length > 1) {
            System.arraycopy(ring, 1, ring, 0, ring.length - 1);
        }
        ring[ring.length - 1] = token;
    }

    /** The softmax of {@code logits[0..count)}. */
    private static float[] softmax(float[] logits, int count) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) {
            if (logits[i] > max) {
                max = logits[i];
            }
        }
        if (max == Float.NEGATIVE_INFINITY) {
            throw new IllegalStateException("the draft logits hold no finite value");
        }
        double[] e = new double[count];
        double sum = 0.0;
        for (int i = 0; i < count; i++) {
            e[i] = Math.exp(logits[i] - max);
            sum += e[i];
        }
        float[] q = new float[count];
        for (int i = 0; i < count; i++) {
            q[i] = (float) (e[i] / sum);
        }
        return q;
    }

    /** The token of {@code q} whose cumulative probability first reaches {@code u * sum(q)}. */
    private static int categorical(float[] q, double u) {
        double total = 0.0;
        for (float v : q) {
            total += v;
        }
        double target = u * total;
        double run = 0.0;
        int last = -1;
        for (int i = 0; i < q.length; i++) {
            if (q[i] > 0f) {
                last = i;
                run += q[i];
                if (run > target) {
                    return i;
                }
            }
        }
        if (last < 0) {
            throw new IllegalStateException("the draft distribution holds no positive probability");
        }
        return last;
    }

    /**
     * Probability of the row's best candidate among its {@value #CONFIDENCE_TOP_K} largest logits, the confidence
     * the drafters' {@code p_min} compares against.
     */
    private static float topProbability(float[] logits, int count) {
        int k = Math.min(CONFIDENCE_TOP_K, count);
        float[] top = new float[k];
        int filled = 0;
        for (int i = 0; i < count; i++) {
            float v = logits[i];
            if (Float.isNaN(v) || v == Float.NEGATIVE_INFINITY) {
                continue;
            }
            if (filled < k) {
                int j = filled++;
                while (j > 0 && top[j - 1] < v) {
                    top[j] = top[j - 1];
                    j--;
                }
                top[j] = v;
            } else if (v > top[k - 1]) {
                int j = k - 1;
                while (j > 0 && top[j - 1] < v) {
                    top[j] = top[j - 1];
                    j--;
                }
                top[j] = v;
            }
        }
        if (filled == 0) {
            throw new IllegalStateException("the draft logits hold no finite value");
        }
        double sum = 0.0;
        for (int j = 0; j < filled; j++) {
            sum += Math.exp((double) top[j] - top[0]);
        }
        return (float) (1.0 / sum);
    }

    @Override
    public String toString() {
        return "DraftWatermarker{contextWidth=" + contextWidth + ", context=" + Arrays.toString(context)
                + ", synced=" + synced.length + '}';
    }

    /** One draft draw: the selected token, the row's top-candidate confidence and its proposal distribution. */
    public static final class Draw {
        private final int token;
        private final float topProbability;
        private final float[] proposal;

        Draw(int token, float topProbability, float[] proposal) {
            this.token = token;
            this.topProbability = topProbability;
            this.proposal = proposal;
        }

        /** The drawn token, in the row's vocabulary. */
        public int token() {
            return token;
        }

        /** The row's best candidate probability, the drafters' {@code p_min} confidence. */
        public float topProbability() {
            return topProbability;
        }

        /** The distribution the token was drawn from, over the row's vocabulary (copied per access). */
        public float[] proposal() {
            return proposal.clone();
        }
    }
}
