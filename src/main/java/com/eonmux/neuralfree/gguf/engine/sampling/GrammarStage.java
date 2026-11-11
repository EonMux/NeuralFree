package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.GrammarElement;

import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

/**
 * The grammar-constrained sampling stage (llama.cpp {@code llama-grammar.cpp}: {@code llama_grammar_apply_impl},
 * {@code llama_grammar_accept_token}): only the tokens the {@link Grammar} can still accept keep their logit, the
 * rest take {@code -inf}, and the accepted token advances the grammar's pushdown stacks. A Java-API custom stage
 * ({@code Sampler.create(config, bannedTokens, extra, vocab)}), so it always applies, including under greedy
 * decoding, and its {@link #modifies() true} keeps an argumentmax chain off the on-device argmax path (the mask
 * must see the logits).
 *
 * <p>The grammar state is the set of partial parses (stacks of flattened-rule indices, top last) plus the partial
 * UTF-8 sequence a multi-byte character split across tokens leaves; a piece ending in a partial sequence survives
 * only on a parse that can complete it, in the mask, {@link #keeps}, {@link #admits} and {@code accept} alike.
 * {@code accept} feeds only tokens sampled after the stage first applied ({@link #apply}): {@code Sampler.accept}
 * also feeds prompt tokens (the penalty stages' history), and llama.cpp's grammar sees generated tokens only — a
 * prompt must not advance the grammar.
 *
 * <p>A budget element ({@link GrammarElement.Type#TOKEN_BUDGET}, {@link GrammarElement.Type#CHAR_BUDGET}) opens a
 * counted region, xgrammar's {@code max_tokens} / {@code max_chars} rule attributes: the stack holds a frame — the
 * remaining count under a negative marker ({@link GrammarAutomaton#TOKEN_FRAME},
 * {@link GrammarAutomaton#CHAR_FRAME}) — below the region's own positions, and the frame pops when the region
 * completes. Every token that starts while a token frame is open spends one of it, and every code point consumed
 * while a character frame is open spends one of that; a parse whose frame is spent cannot go on inside the region,
 * so only the parses that already left it survive. A token that opens a region mid-token does not count against its
 * token budget, as in xgrammar.
 * {@link #reset} returns the initial stacks. {@link #admits} tells whether a token may come next without accepting
 * it, and {@link #saveState} keeps a state to return to, so speculative verification can walk a draft and roll it
 * back (vLLM's {@code GrammarMatcher.validate_tokens} and {@code rollback}).
 *
 * <p>The mask reads every token's code points from the vocabulary's decoded table ({@link GrammarPieces}) unless a
 * character is pending across tokens, walks the candidates in primitive slot arrays ({@link GrammarRejectWalk})
 * borrowed for the one mask ({@link GrammarWalkBuffers}), so a stage between masks holds no row-sized scratch, and
 * remembers the masks of recent grammar states ({@link GrammarMaskMemo}). {@link #keeps} runs the same walk for one
 * token, so a greedy chain can check its argmax before masking the row ({@link TokenGate}).
 *
 * <p>Not thread-safe, owned by its sampler like every stage.
 */
public final class GrammarStage extends SamplerStage implements TokenGate {

    /** A saved grammar state ({@link #saveState}). */
    private static final class State {
        final GrammarAutomaton.Stack[] stacks;
        final int partialValue;
        final int partialRemain;
        final boolean armed;

        State(GrammarAutomaton.Stack[] stacks, int partialValue, int partialRemain, boolean armed) {
            this.stacks = stacks;
            this.partialValue = partialValue;
            this.partialRemain = partialRemain;
            this.armed = armed;
        }
    }

    private final GrammarAutomaton grammar;
    private final SamplerVocab vocab;
    private final GrammarRejectWalk walk;
    /** The slots of a one-token check ({@link #keeps}). */
    private final GrammarWalkBuffers single = new GrammarWalkBuffers(1);
    /** The code points of one piece. */
    private final GrammarPieces.Decoded scratch = new GrammarPieces.Decoded(64);
    private GrammarPieces pieces;
    private GrammarMaskMemo memo;
    /** The row width {@link #memo} holds masks of. */
    private int memoWidth = -1;
    private GrammarAutomaton.Stack[] stacks;
    private int partialValue;
    private int partialRemain;
    private boolean armed;

    /** A stage over {@code grammar}; token pieces and end-of-generation ids come from {@code vocab}. */
    public GrammarStage(Grammar grammar, SamplerVocab vocab) {
        this.grammar = new GrammarAutomaton(Objects.requireNonNull(grammar, "grammar"));
        this.vocab = Objects.requireNonNull(vocab, "vocab");
        this.walk = new GrammarRejectWalk(this.grammar);
        this.stacks = this.grammar.initialStacks();
        this.partialValue = 0;
        this.partialRemain = -1;
    }

    @Override
    public void apply(Sampler.Candidates candidates) {
        armed = true;
        int n = candidates.size();
        GrammarMaskMemo.Key key = null;
        if (candidates.indexedByTokenId()) {
            if (memo == null || memoWidth != n) {
                memo = new GrammarMaskMemo(n);
                memoWidth = n;
            }
            key = new GrammarMaskMemo.Key(stacks, partialValue, partialRemain, n);
            long[] remembered = memo.get(key);
            if (remembered != null) {
                maskBits(candidates, remembered);
                return;
            }
        }
        long[] rejected = new long[(n + 63) >>> 6];
        GrammarWalkBuffers buffers = GrammarWalkBuffers.borrow();
        try {
            buffers.reserve(n);
            walk.bind(buffers);
            int live = loadSlots(candidates, rejected, buffers.rowCodePoints);
            int count = rejectCount(live);
            for (int i = 0; i < count; i++) {
                int slot = walk.slots[i];
                rejected[slot >>> 6] |= 1L << slot;
            }
        } finally {
            walk.unbind();
            GrammarWalkBuffers.release(buffers);
        }
        maskBits(candidates, rejected);
        if (key != null) {
            memo.put(key, rejected);
        }
    }

    /**
     * Fills the walk's slots with the candidates the walk decides, row index as slot, and marks in {@code rejected}
     * the ones decided without it: end-of-generation ids unless a parse is complete, and pieces the mask refuses
     * (empty, or starting on the terminating 0, which llama.cpp rejects alike). A pending partial character decodes
     * the pieces into {@code rowCodePoints}. Returns the slot count.
     */
    private int loadSlots(Sampler.Candidates candidates, long[] rejected, GrammarPieces.Decoded rowCodePoints) {
        int n = candidates.size();
        boolean allowEog = completed();
        GrammarPieces table = pieces();
        boolean fromTable = partialRemain <= 0 && maxId(candidates) < table.size();
        int live = 0;
        if (fromTable) {
            walk.cps = table.codePoints();
            for (int i = 0; i < n; i++) {
                int id = candidates.id(i);
                byte flags = table.flags(id);
                if ((flags & GrammarPieces.EOG) != 0 ? !allowEog : (flags & GrammarPieces.REFUSED) != 0) {
                    rejected[i >>> 6] |= 1L << i;
                } else if ((flags & GrammarPieces.EOG) == 0) {
                    walk.slots[live++] = i;
                    walk.id[i] = id;
                    walk.base[i] = table.base(id);
                    walk.value[i] = table.endValue(id);
                    walk.remain[i] = table.endRemain(id);
                }
            }
            return live;
        }
        rowCodePoints.clear();
        for (int i = 0; i < n; i++) {
            int id = candidates.id(i);
            if (vocab.isEog(id)) {
                if (!allowEog) {
                    rejected[i >>> 6] |= 1L << i;
                }
                continue;
            }
            byte[] piece = vocab.grammarPiece(id);
            if (piece.length == 0 || piece[0] == 0) {
                rejected[i >>> 6] |= 1L << i;
                continue;
            }
            walk.slots[live++] = i;
            walk.id[i] = id;
            walk.base[i] = rowCodePoints.length;
            GrammarPieces.decode(piece, partialValue, partialRemain, rowCodePoints);
            walk.value[i] = rowCodePoints.endValue;
            walk.remain[i] = rowCodePoints.endRemain;
        }
        walk.cps = rowCodePoints.data;
        return live;
    }

    private static int maxId(Sampler.Candidates candidates) {
        if (candidates.indexedByTokenId()) {
            return candidates.size() - 1;
        }
        int max = -1;
        for (int i = 0; i < candidates.size(); i++) {
            max = Math.max(max, candidates.id(i));
        }
        return max;
    }

    /**
     * {@code llama_grammar_reject_candidates}: walks the first {@code live} slots over every partial parse, a
     * candidate staying rejected only while every parse rejects it; returns how many of the leading slots are left
     * rejected.
     */
    private int rejectCount(int live) {
        int count = live;
        for (GrammarAutomaton.Stack stack : stacks) {
            GrammarAutomaton.Stack started = GrammarAutomaton.tokenStart(stack);
            if (started == null) {
                continue; // a spent token budget: this parse takes no token
            }
            count = walk.reject(started, 0, count, 0);
            if (count == 0) {
                break;
            }
        }
        return count;
    }

    private static void maskBits(Sampler.Candidates candidates, long[] rejected) {
        for (int w = 0; w < rejected.length; w++) {
            long bits = rejected[w];
            while (bits != 0) {
                candidates.setLogit((w << 6) + Long.numberOfTrailingZeros(bits), Float.NEGATIVE_INFINITY);
                bits &= bits - 1;
            }
        }
    }

    /**
     * Whether {@link #apply} would keep {@code token}: the same rejection walk for that token alone. Arms the stage,
     * as an apply does.
     */
    @Override
    public boolean keeps(int token) {
        armed = true;
        if (vocab.isEog(token)) {
            return completed();
        }
        byte[] piece = vocab.grammarPiece(token);
        if (piece.length == 0 || piece[0] == 0) {
            return false;
        }
        scratch.clear();
        GrammarPieces.decode(piece, partialValue, partialRemain, scratch);
        walk.bind(single);
        try {
            walk.cps = scratch.data;
            walk.slots[0] = 0;
            walk.id[0] = token;
            walk.base[0] = 0;
            walk.value[0] = scratch.endValue;
            walk.remain[0] = scratch.endRemain;
            return rejectCount(1) == 0;
        } finally {
            walk.unbind();
        }
    }

    @Override
    public void accept(int token) {
        if (!armed) {
            return; // prompt history: the grammar sees generated tokens only (see class doc)
        }
        if (vocab.isEog(token)) {
            if (completed()) {
                return;
            }
            throw new IllegalStateException("grammar cannot accept the end-of-generation token " + token);
        }
        acceptPiece(token, vocab.grammarPiece(token));
    }

    /**
     * {@code llama_grammar_accept_token} over an explicit piece: {@code token} advances the stacks whose top is a
     * token element, {@code piece}'s code points the others. A trigger-gated stage replays the tokens overlapping
     * its trigger with the overlapping part of their piece ({@code llama_grammar_accept_impl}).
     *
     * @throws IllegalStateException when no partial parse accepts the piece (the state is then unchanged)
     */
    void acceptPiece(int token, byte[] piece) {
        if (!advance(token, piece)) {
            throw new IllegalStateException("grammar stacks emptied by token " + token);
        }
    }

    /**
     * {@link #acceptPiece} that reports a rejection instead of throwing: false when no partial parse accepts the
     * piece, which leaves the state unchanged. An empty piece carries no grammar bytes and changes nothing.
     */
    boolean advance(int token, byte[] piece) {
        if (piece.length == 0) {
            return true;
        }
        scratch.clear();
        GrammarPieces.decode(piece, partialValue, partialRemain, scratch);
        int value = scratch.endValue;
        int remain = scratch.endRemain;
        GrammarAutomaton.Stack[] next = nextStacks(token, scratch.data, scratch.length - 1, value, remain);
        if (next.length == 0) {
            return false;
        }
        stacks = next;
        partialValue = value;
        partialRemain = remain;
        return true;
    }

    /**
     * The partial parses after {@code token} whose piece decoded to the code points {@code cps[0, count)} and the
     * trailing partial UTF-8 sequence {@code (value, remain)}; empty on a reject. A parse survives only when it can
     * also take that partial sequence ({@link #takesPartial}), the test the mask applies ({@link GrammarRejectWalk}),
     * so an invalid continuation byte or an impossible partial character is refused here as it is masked there; a
     * token element takes only a piece with a complete code point or no partial sequence.
     */
    private GrammarAutomaton.Stack[] nextStacks(int token, int[] cps, int count, int value, int remain) {
        Set<GrammarAutomaton.Stack> next = new LinkedHashSet<>();
        for (GrammarAutomaton.Stack unstarted : stacks) {
            GrammarAutomaton.Stack stack = unstarted.isEmpty() ? null : GrammarAutomaton.tokenStart(unstarted);
            if (stack == null) {
                continue;
            }
            int pos = stack.top();
            if (grammar.isToken(pos)) {
                if ((count > 0 || remain == 0) && grammar.matchToken(pos, token)) {
                    grammar.acceptToken(stack, pos, next);
                }
            } else {
                for (GrammarAutomaton.Stack surviving : acceptCodePoints(stack, cps, count)) {
                    if (takesPartial(surviving, value, remain)) {
                        next.add(surviving);
                    }
                }
            }
        }
        return next.toArray(new GrammarAutomaton.Stack[0]);
    }

    /** The parses {@code stack} leaves after consuming {@code cps[0, count)} one code point at a time. */
    private Set<GrammarAutomaton.Stack> acceptCodePoints(GrammarAutomaton.Stack stack, int[] cps, int count) {
        Set<GrammarAutomaton.Stack> current = new LinkedHashSet<>();
        current.add(stack);
        for (int i = 0; i < count && !current.isEmpty(); i++) {
            Set<GrammarAutomaton.Stack> step = new LinkedHashSet<>();
            for (GrammarAutomaton.Stack cur : current) {
                grammar.acceptChr(cur, cps[i], step);
            }
            current = step;
        }
        return current;
    }

    /**
     * Whether the grammar can accept {@code token} next, the test {@link #accept} applies, as vLLM's
     * {@code validate_tokens}: before the first {@link #apply} every token passes (prompt history), an
     * end-of-generation token needs a completed parse, a token whose piece the mask rejects (empty, or starting on
     * the terminating 0) is refused, and any other token needs a partial parse that survives its piece.
     */
    @Override
    public boolean admits(int token) {
        if (!armed) {
            return true;
        }
        if (vocab.isEog(token)) {
            return completed();
        }
        byte[] piece = vocab.grammarPiece(token);
        if (piece.length == 0 || piece[0] == 0) {
            return false;
        }
        scratch.clear();
        GrammarPieces.decode(piece, partialValue, partialRemain, scratch);
        return nextStacks(token, scratch.data, scratch.length - 1, scratch.endValue, scratch.endRemain).length > 0;
    }

    /**
     * Whether a parse that read a piece's complete code points can take its trailing partial UTF-8 sequence: none
     * is pending ({@code remain == 0}), or the top is a char element with room left whose ranges hold a completion
     * of it ({@link GrammarAutomaton#matchPartialChar}; an invalid sequence, {@code remain < 0}, has none).
     */
    private boolean takesPartial(GrammarAutomaton.Stack stack, int value, int remain) {
        if (remain == 0) {
            return true;
        }
        if (stack.isEmpty()) {
            return false;
        }
        int top = stack.top();
        return !grammar.isToken(top) && GrammarAutomaton.charsLeft(stack)
                && grammar.matchPartialChar(top, value, remain);
    }

    @Override
    public boolean constrains() {
        return true;
    }

    /** The partial parses, the partial UTF-8 tail and the armed flag; the stacks are immutable. */
    @Override
    public Object saveState() {
        return new State(stacks, partialValue, partialRemain, armed);
    }

    @Override
    public void restoreState(Object state) {
        if (!(state instanceof State)) {
            throw new IllegalArgumentException("not a grammar stage state: " + state);
        }
        State s = (State) state;
        stacks = s.stacks;
        partialValue = s.partialValue;
        partialRemain = s.partialRemain;
        armed = s.armed;
    }

    /** Arms the stage as its first {@link #apply} does: accepted tokens advance the grammar from here on. */
    void arm() {
        armed = true;
    }

    /** Whether a partial parse is complete: the grammar accepts the text so far as a whole sentence. */
    boolean completed() {
        for (GrammarAutomaton.Stack stack : stacks) {
            if (stack.isEmpty()) {
                return true;
            }
        }
        return false;
    }

    @Override
    public boolean modifies() {
        return true;
    }

    @Override
    public void reset() {
        stacks = grammar.initialStacks();
        partialValue = 0;
        partialRemain = -1;
        armed = false;
    }

    /** The decoded vocabulary table, read on first use. */
    private GrammarPieces pieces() {
        if (pieces == null) {
            pieces = GrammarPieces.of(vocab);
        }
        return pieces;
    }
}
