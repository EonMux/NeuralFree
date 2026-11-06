package com.eonmux.neuralfree.gguf.engine.watermark;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * The repeated-context search of generation (vLLM's {@code repeated_context_mask}): it reports whether the context
 * that keys the next token already occurred in the searched history, where a repeated context would reuse one keyed
 * random vector and so is sampled without the watermark.
 *
 * <p>The model context — the last {@code context_width} accepted tokens, padded with {@code -1} before the start —
 * always keys the PRF. The searched scope ({@link WatermarkContextScope}) is the generated tokens alone or the prompt
 * too; the whole scope is searched when the history bound is null, otherwise only the most recent
 * {@code deduplicate_contexts_max_history} positions. The scope {@code all} also skips the first
 * {@code context_width} generated tokens, whose context is incomplete. The search is incremental: each accepted token
 * records the scope window that preceded it, and a lookup answers whether the current window occurred at a recent
 * position. Mutable, owner-confined, not thread-safe.
 */
public final class WatermarkState {

    private final int contextWidth;
    private final WatermarkContextScope scope;
    private final Integer maxHistory;
    private final int[] context;
    private final int[] scopeContext;
    /** The latest scope position at which each context occurred. */
    private final Map<ContextKey, Integer> latestPositions = new HashMap<>();
    private int accepted;
    private int promptLength;
    private int scopedLength;
    private boolean inPrompt;

    /** The state of a configuration: an empty history, every context {@code -1}. */
    public WatermarkState(WatermarkConfig config) {
        if (config == null) {
            throw new NullPointerException("config");
        }
        this.contextWidth = config.contextWidth();
        this.scope = config.deduplicateContexts();
        this.maxHistory = config.deduplicateContextsMaxHistory();
        this.context = new int[contextWidth];
        this.scopeContext = new int[contextWidth];
        Arrays.fill(context, -1);
        Arrays.fill(scopeContext, -1);
    }

    /** The number of prior tokens the PRF keys on. */
    public int contextWidth() {
        return contextWidth;
    }

    /** Which history the search scans. */
    public WatermarkContextScope scope() {
        return scope;
    }

    /** The context of the next sample: the last {@link #contextWidth()} accepted tokens, oldest first. */
    public int[] context() {
        return context.clone();
    }

    /** The number of tokens accepted so far (prompt and generated). */
    public int accepted() {
        return accepted;
    }

    /**
     * Marks the tokens accepted from now on as the prompt: the {@code single_turn} scope leaves them out of its
     * search, and the {@code all} scope searches them but keeps them out of the generated count that
     * {@link #repeatedContext()} tests for an incomplete context. {@link #endPrompt()} closes the prompt.
     */
    public void startPrompt() {
        inPrompt = true;
    }

    /** Ends the prompt started by {@link #startPrompt()}: the tokens accepted so far are the prompt. */
    public void endPrompt() {
        inPrompt = false;
        promptLength = accepted;
    }

    /**
     * Whether the context of the next sample already occurred in the searched history.
     *
     * <p>The scope {@code all} answers true while fewer than {@code contextWidth} tokens were generated, so those
     * partial contexts are sampled without the watermark.
     */
    public boolean repeatedContext() {
        return repeatedContext(0);
    }

    /**
     * {@link #repeatedContext()} for a position that newer, not yet accepted tokens precede: a speculative round's
     * earlier drafts. They count toward the searched window but are not history of their own, as vLLM's
     * {@code history_offsets} of its draft mask.
     *
     * @param newerGenerations how many generated positions follow the searched history and precede this context, at
     *                         least 0
     * @throws IllegalArgumentException for a negative count
     */
    public boolean repeatedContext(int newerGenerations) {
        return repeatedContext(scopeContext, newerGenerations);
    }

    /**
     * Whether {@code window} already occurred in the searched history, for a position that newer, not yet accepted
     * tokens precede: a speculative round's earlier drafts. They count toward the searched window but are not history
     * of their own, as vLLM's {@code history_offsets} of its draft mask. The window may carry the round's draft
     * tokens instead of the current context.
     *
     * @param window            the context of the draw, oldest first
     * @param newerGenerations  how many generated positions follow the searched history and precede the context, at
     *                          least 0
     * @throws IllegalArgumentException for a window of another context width or a negative count
     */
    public boolean repeatedContext(int[] window, int newerGenerations) {
        if (newerGenerations < 0) {
            throw new IllegalArgumentException("newer generation count " + newerGenerations + " is negative");
        }
        if (window.length != contextWidth) {
            throw new IllegalArgumentException("repeated-context window has " + window.length + " tokens, expected "
                    + contextWidth);
        }
        if (scope == WatermarkContextScope.NONE) {
            return false;
        }
        if (scope == WatermarkContextScope.ALL
                && accepted - promptLength + newerGenerations < contextWidth) {
            return true;
        }
        Integer latest = latestPositions.get(new ContextKey(window));
        if (latest == null) {
            return false;
        }
        return maxHistory == null || latest >= scopedLength - maxHistory + newerGenerations;
    }

    /**
     * Records an accepted token: the context window preceding it enters the searched scope (when the token belongs to
     * the scope), then it shifts into the PRF context.
     *
     * @throws IllegalArgumentException for a negative token id
     */
    public void accept(int token) {
        if (token < 0) {
            throw new IllegalArgumentException("token id " + token + " is negative");
        }
        if (scope == WatermarkContextScope.ALL
                || (scope == WatermarkContextScope.SINGLE_TURN && !inPrompt)) {
            latestPositions.put(new ContextKey(scopeContext), scopedLength);
            shift(scopeContext, token);
            scopedLength++;
            prune();
        }
        shift(context, token);
        accepted++;
    }

    /** Keeps the latest-position table proportional to the searched window. */
    private void prune() {
        if (maxHistory == null || latestPositions.size() <= Math.max(1024, 2 * maxHistory)) {
            return;
        }
        int oldest = scopedLength - maxHistory;
        latestPositions.values().removeIf(position -> position < oldest);
    }

    private static void shift(int[] ring, int token) {
        if (ring.length > 1) {
            System.arraycopy(ring, 1, ring, 0, ring.length - 1);
        }
        ring[ring.length - 1] = token;
    }

    /** Returns the state to its construction: no history, every context {@code -1}. */
    public void reset() {
        Arrays.fill(context, -1);
        Arrays.fill(scopeContext, -1);
        latestPositions.clear();
        accepted = 0;
        promptLength = 0;
        scopedLength = 0;
        inPrompt = false;
    }

    /**
     * A copy of the state for {@link #restore(Snapshot)}, which a speculative verifier takes before it accepts draft
     * tokens ahead of their confirmation.
     */
    public Snapshot save() {
        return new Snapshot(this);
    }

    /**
     * Puts back a {@link #save()} snapshot.
     *
     * @throws IllegalArgumentException for a snapshot of another context width
     */
    public void restore(Snapshot snapshot) {
        if (snapshot.context.length != contextWidth) {
            throw new IllegalArgumentException("snapshot has a context of " + snapshot.context.length
                    + " tokens, expected " + contextWidth);
        }
        System.arraycopy(snapshot.context, 0, context, 0, contextWidth);
        System.arraycopy(snapshot.scopeContext, 0, scopeContext, 0, contextWidth);
        latestPositions.clear();
        latestPositions.putAll(snapshot.latestPositions);
        accepted = snapshot.accepted;
        promptLength = snapshot.promptLength;
        scopedLength = snapshot.scopedLength;
        inPrompt = snapshot.inPrompt;
    }

    @Override
    public String toString() {
        return "WatermarkState{scope=" + scope.label() + ", context=" + Arrays.toString(context)
                + ", accepted=" + accepted + ", generated=" + (accepted - promptLength) + '}';
    }

    /** An immutable copy of a {@link WatermarkState}. */
    public static final class Snapshot {
        private final int[] context;
        private final int[] scopeContext;
        private final Map<ContextKey, Integer> latestPositions;
        private final int accepted;
        private final int promptLength;
        private final int scopedLength;
        private final boolean inPrompt;

        private Snapshot(WatermarkState state) {
            this.context = state.context.clone();
            this.scopeContext = state.scopeContext.clone();
            this.latestPositions = new HashMap<>(state.latestPositions);
            this.accepted = state.accepted;
            this.promptLength = state.promptLength;
            this.scopedLength = state.scopedLength;
            this.inPrompt = state.inPrompt;
        }
    }

    /** A token window usable as a map key. */
    private static final class ContextKey {
        private final int[] tokens;
        private final int hash;

        ContextKey(int[] tokens) {
            this.tokens = tokens.clone();
            this.hash = Arrays.hashCode(this.tokens);
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof ContextKey && Arrays.equals(tokens, ((ContextKey) o).tokens);
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }
}
