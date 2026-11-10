package com.eonmux.neuralfree.gguf.engine.sampling;

/**
 * {@code llama_grammar_reject_candidates_for_stack} (llama-grammar.cpp:1039-1131) over candidate slots held in
 * primitive arrays. Every candidate on one walk level sits at the same code point depth, so the walk keeps no
 * per-candidate cursor: it partitions a slot range in place — the candidates the stack rejects first, then those it
 * accepts, then those that read one more code point — and recurses on the last group with the stacks after the
 * consumed character. The candidates a stack rejects end up at the front of the range, so the walk over the next
 * stack reads just that prefix, as llama.cpp feeds one stack's rejects to the next.
 *
 * <p>A slot carries its token id, where its code points start in the shared code point array (a run ending on 0),
 * and the partial UTF-8 tail its piece leaves. The slot arrays are a {@link GrammarWalkBuffers} set the owning
 * grammar stage binds for one mask or one token check ({@link #bind}); not thread-safe.
 */
final class GrammarRejectWalk {

    private static final int REJECT = 0;
    private static final int KEEP = 1;
    private static final int NEXT = 2;

    private final GrammarAutomaton grammar;
    /** The slot order the partitions permute. */
    int[] slots;
    int[] id;
    int[] base;
    int[] remain;
    int[] value;
    /** The code points every slot's {@link #base} indexes. */
    int[] cps;

    GrammarRejectWalk(GrammarAutomaton grammar) {
        this.grammar = grammar;
    }

    /** Walks over the slot arrays of {@code buffers}, sized by the caller; {@link #cps} is the caller's to set. */
    void bind(GrammarWalkBuffers buffers) {
        slots = buffers.slots;
        id = buffers.id;
        base = buffers.base;
        remain = buffers.remain;
        value = buffers.value;
    }

    /** Lets go of the bound arrays, so the walk keeps no scratch between uses. */
    void unbind() {
        slots = null;
        id = null;
        base = null;
        remain = null;
        value = null;
        cps = null;
    }

    /**
     * The candidates in {@code slots[lo, hi)} that no continuation of {@code stack} accepts, at code point
     * {@code depth} of their pieces: the range is permuted so that they come first, and their count is returned.
     */
    int reject(GrammarAutomaton.Stack stack, int lo, int hi, int depth) {
        if (lo == hi) {
            // llama_grammar_reject_candidates returns at once: walking on would follow recursive rules forever
            return 0;
        }
        if (stack.isEmpty()) {
            int r = lo;
            for (int i = lo; i < hi; i++) {
                int c = slots[i];
                if (cps[base[c] + depth] != 0 || remain[c] != 0) {
                    swap(r++, i);
                }
            }
            return r - lo;
        }
        int top = stack.top();
        if (grammar.isToken(top)) {
            int r = lo;
            for (int i = lo; i < hi; i++) {
                int c = slots[i];
                boolean rejected = cps[base[c] + depth] == 0 ? remain[c] != 0 : !grammar.matchToken(top, id[c]);
                if (rejected) {
                    swap(r++, i);
                }
            }
            return r - lo;
        }
        boolean room = GrammarAutomaton.charsLeft(stack);
        int r = lo;
        int i = lo;
        int n = hi;
        while (i < n) {
            int c = slots[i];
            int cls = classify(top, c, depth, room);
            if (cls == REJECT) {
                swap(r++, i++);
            } else if (cls == NEXT) {
                swap(i, --n);
            } else {
                i++;
            }
        }
        if (n == hi) {
            return r - lo;
        }
        int match = grammar.matchChar(top, 0);
        GrammarAutomaton.Stack[] after = grammar.afterChar(stack, match >= 0 ? match : ~match);
        int pending = hi;
        for (GrammarAutomaton.Stack next : after) {
            int k = reject(next, n, pending, depth + 1);
            pending = n + k;
            if (k == 0) {
                return r - lo;
            }
        }
        int nextRejects = pending - n;
        int accepted = n - r;
        int moves = Math.min(accepted, nextRejects);
        for (int j = 0; j < moves; j++) {
            swap(r + j, n + nextRejects - 1 - j);
        }
        return (r - lo) + nextRejects;
    }

    /** Where candidate {@code c} goes on a character stack: rejected, accepted here, or on to the next code point. */
    private int classify(int top, int c, int depth, boolean room) {
        int cp = cps[base[c] + depth];
        if (cp == 0) {
            int nRemain = remain[c];
            return nRemain != 0 && (!room || !grammar.matchPartialChar(top, value[c], nRemain)) ? REJECT : KEEP;
        }
        return room && grammar.matchChar(top, cp) >= 0 ? NEXT : REJECT;
    }

    private void swap(int a, int b) {
        int t = slots[a];
        slots[a] = slots[b];
        slots[b] = t;
    }
}
