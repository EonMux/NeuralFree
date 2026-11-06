package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

import java.util.Arrays;

/**
 * llama.cpp's penalties sampler over the last {@code penaltyLastN} accepted tokens (prompt tokens included, as
 * llama.cpp completion does): a seen token's logit is scaled by the repeat penalty (multiplied when the logit is
 * negative, llama.cpp's fix for the old division making negative logits more likely), then reduced by the frequency
 * penalty per occurrence and the presence penalty once. The set is marked unsorted, with no reorder. When the
 * penalties are disabled the stage applies nothing and keeps no history, as llama.cpp's empty sampler does. Mutable
 * (the accepted-token history); owned by the sampler that built it.
 */
public final class PenaltiesStage extends SamplerStage {

    private final float repeatPenalty;
    private final float presencePenalty;
    private final float frequencyPenalty;
    private final History history;

    /**
     * @param penaltyLastN accepted tokens the penalties look back over (0 disables the stage, -1 penalizes over the
     *                     whole context)
     * @param enabled      false for a chain whose configuration turned the penalties off (the stage then keeps no
     *                     history and applies nothing, as llama.cpp's empty penalties sampler does)
     */
    public PenaltiesStage(float repeatPenalty, float presencePenalty, float frequencyPenalty, int penaltyLastN,
                          boolean enabled) {
        this.repeatPenalty = repeatPenalty;
        this.presencePenalty = presencePenalty;
        this.frequencyPenalty = frequencyPenalty;
        this.history = new History(enabled ? penaltyLastN : 0);
    }

    @Override
    public StageKind kind() {
        return StageKind.PENALTIES;
    }

    /** Whether the stage keeps a window ({@code penalty_last_n} != 0); the penalties may still be all neutral. */
    public boolean enabled() {
        return history.window != 0;
    }

    /** llama.cpp's repeat penalty. */
    public float repeatPenalty() {
        return repeatPenalty;
    }

    /** The frequency penalty per occurrence in the window. */
    public float frequencyPenalty() {
        return frequencyPenalty;
    }

    /** The presence penalty once per window token. */
    public float presencePenalty() {
        return presencePenalty;
    }

    /** Distinct token ids with a positive count in the window, in no particular order (a copy). */
    public int[] activeTokens() {
        return Arrays.copyOf(history.active, history.nActive);
    }

    /** The occurrence count of each {@link #activeTokens()} id (a copy). */
    public int[] activeCounts() {
        int[] out = new int[history.nActive];
        for (int a = 0; a < history.nActive; a++) {
            out[a] = history.counts[history.active[a]];
        }
        return out;
    }

    /**
     * Applies the penalties. A vocabulary-ordered set indexes token ids by candidate slot; over a ranked cone (a
     * device top-k readback) the candidates carry their ids explicitly, so their logits are looked up by id.
     */
    @Override
    public void apply(Candidates candidates) {
        if (history.window == 0) {
            return;
        }
        if (!candidates.indexedByTokenId()) {
            for (int i = 0; i < candidates.size(); i++) {
                int id = candidates.id(i);
                if (id < 0 || id >= history.counts.length || history.counts[id] == 0) {
                    continue;
                }
                candidates.setLogit(i, penalize(candidates.logit(i), history.counts[id]));
            }
            candidates.setSorted(false);
            return;
        }
        for (int a = 0; a < history.nActive; a++) {
            int id = history.active[a];
            if (id >= candidates.size()) {
                continue;
            }
            candidates.setLogit(id, penalize(candidates.logit(id), history.counts[id]));
        }
        candidates.setSorted(false);
    }

    /** One token's logit under llama.cpp's repeat, frequency and presence penalties. */
    private float penalize(float logit, int count) {
        float l = logit <= 0f ? logit * repeatPenalty : logit / repeatPenalty;
        return l - ((float) count * frequencyPenalty + presencePenalty);
    }

    @Override
    public void accept(int token) {
        history.accept(token);
    }

    @Override
    public void reset() {
        history.clear();
    }

    /** The window's tokens, oldest first; null when the stage is off. */
    @Override
    public Object saveState() {
        return history.window == 0 ? null : history.tokens();
    }

    /** Rebuilds the window and its counts from the saved tokens. */
    @Override
    public void restoreState(Object state) {
        if (history.window == 0) {
            return;
        }
        if (!(state instanceof int[])) {
            throw new IllegalArgumentException("not a penalties stage state: " + state);
        }
        history.clear();
        for (int token : (int[]) state) {
            history.accept(token);
        }
    }

    /**
     * Accepted tokens of the penalty window (the llama_sampler_penalties ring buffer and {@code token_count}): a ring
     * of the last {@code window} tokens, or every token when the window is {@code -1} (the whole context); nothing is
     * kept for a window of 0. Distinct tokens with a positive count are listed in {@code active}.
     */
    private static final class History {

        private final int window;
        private int[] ring;
        private int head;
        private int stored;
        private int[] counts = new int[0];
        private int[] activePos = new int[0];
        private int[] active = new int[16];
        private int nActive;

        History(int window) {
            this.window = window;
            this.ring = window > 0 ? new int[window] : null;
        }

        void accept(int token) {
            if (window == 0) {
                return;
            }
            if (token < 0) {
                throw new IllegalArgumentException("accepted token id " + token + " is negative");
            }
            increment(token);
            if (window > 0) {
                if (stored == window) {
                    decrement(ring[head]);
                    ring[head] = token;
                    head = (head + 1) % window;
                } else {
                    ring[stored++] = token;
                }
            }
        }

        /** The stored tokens, oldest first. */
        int[] tokens() {
            int[] out = new int[stored];
            int first = stored == window ? head : 0;
            for (int k = 0; k < stored; k++) {
                out[k] = ring[(first + k) % window];
            }
            return out;
        }

        void clear() {
            for (int a = 0; a < nActive; a++) {
                counts[active[a]] = 0;
            }
            nActive = 0;
            head = 0;
            stored = 0;
        }

        private void increment(int token) {
            if (token >= counts.length) {
                int cap = Math.max(token + 1, counts.length * 2);
                counts = Arrays.copyOf(counts, cap);
                activePos = Arrays.copyOf(activePos, cap);
            }
            if (counts[token]++ == 0) {
                if (nActive == active.length) {
                    active = Arrays.copyOf(active, nActive * 2);
                }
                activePos[token] = nActive;
                active[nActive++] = token;
            }
        }

        private void decrement(int token) {
            if (--counts[token] == 0) {
                int pos = activePos[token];
                int last = active[--nActive];
                active[pos] = last;
                activePos[last] = pos;
            }
        }
    }
}
