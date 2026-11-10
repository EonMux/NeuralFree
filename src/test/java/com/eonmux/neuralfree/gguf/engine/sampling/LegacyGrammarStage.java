package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.GrammarElement;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * The reference grammar stage of the differential tests: a line-by-line transcription of llama.cpp's grammar runtime
 * ({@code llama-grammar.cpp}: {@code llama_grammar_apply_impl}, {@code llama_grammar_accept_token},
 * {@code llama_grammar_reject_candidates_for_stack}) over list stacks, one candidate object per token and a fresh
 * UTF-8 decode per candidate and step, with the xgrammar budget frames {@link GrammarStage} documents. Accepting a
 * token applies the mask's trailing partial UTF-8 test ({@code llama_grammar_match_partial_char}) to the surviving
 * parses, so a token the mask refuses for its partial character is refused on accept too.
 * {@link GrammarStageDifferentialTest} checks the production stage against it mask for mask.
 */
final class LegacyGrammarStage extends SamplerStage {

    /** One candidate under the stack walk: row index, token id, code points (trailing 0 included) and cursor. */
    private static final class Cand {
        final int index;
        final int id;
        final int[] cps;
        final int off;
        final int partialValue;
        final int partialRemain;

        Cand(int index, int id, int[] cps, int off, int partialValue, int partialRemain) {
            this.index = index;
            this.id = id;
            this.cps = cps;
            this.off = off;
            this.partialValue = partialValue;
            this.partialRemain = partialRemain;
        }
    }

    /** {@code match_char}'s pair: whether the code point matched and the element after the char cluster. */
    private static final class Match {
        final boolean found;
        final int next;

        Match(boolean found, int next) {
            this.found = found;
            this.next = next;
        }
    }

    /** A saved grammar state ({@link #saveState}). */
    private static final class State {
        final List<List<Integer>> stacks;
        final int partialValue;
        final int partialRemain;
        final boolean armed;

        State(List<List<Integer>> stacks, int partialValue, int partialRemain, boolean armed) {
            this.stacks = stacks;
            this.partialValue = partialValue;
            this.partialRemain = partialRemain;
            this.armed = armed;
        }
    }

    private static final int[] LOOKUP = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};

    /** The stack marker of a token budget frame; the remaining token count lies right below it. */
    private static final int TOKEN_FRAME = -1;

    /** The stack marker of a character budget frame; the remaining code point count lies right below it. */
    private static final int CHAR_FRAME = -2;

    private final Grammar grammar;
    private final SamplerVocab vocab;
    private final GrammarElement[] flat;
    private final int[] ruleStart;
    private List<List<Integer>> stacks;
    private int partialValue;
    private int partialRemain;
    private boolean armed;

    /** A stage over {@code grammar}; token pieces and end-of-generation ids come from {@code vocab}. */
    LegacyGrammarStage(Grammar grammar, SamplerVocab vocab) {
        this.grammar = Objects.requireNonNull(grammar, "grammar");
        this.vocab = Objects.requireNonNull(vocab, "vocab");
        this.flat = flatten(grammar);
        this.ruleStart = ruleStarts(grammar);
        this.stacks = initialStacks();
        this.partialValue = 0;
        this.partialRemain = -1;
    }

    @Override
    public void apply(Sampler.Candidates candidates) {
        armed = true;
        int n = candidates.size();
        boolean allowEog = false;
        for (List<Integer> stack : stacks) {
            if (stack.isEmpty()) {
                allowEog = true;
                break;
            }
        }
        Cand[] cands = new Cand[n];
        List<Integer> live = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            int id = candidates.id(i);
            if (vocab.isEog(id)) {
                if (!allowEog) {
                    candidates.setLogit(i, Float.NEGATIVE_INFINITY);
                }
            } else {
                byte[] piece = vocab.grammarPiece(id);
                if (piece.length == 0 || piece[0] == 0) {
                    // llama.cpp rejects an empty piece and a piece starting on the terminating 0 alike
                    candidates.setLogit(i, Float.NEGATIVE_INFINITY);
                } else {
                    cands[i] = decode(piece, i, id);
                    live.add(i);
                }
            }
        }
        List<Cand> rejects = new ArrayList<>(live.size());
        for (int i : live) {
            rejects.add(cands[i]);
        }
        for (List<Integer> stack : stacks) {
            List<Integer> started = tokenStart(stack);
            if (started == null) {
                continue; // a spent token budget: this parse takes no token
            }
            rejects = rejectsForStack(started, rejects);
            if (rejects.isEmpty()) {
                break;
            }
        }
        for (Cand reject : rejects) {
            candidates.setLogit(reject.index, Float.NEGATIVE_INFINITY);
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
        int[][] decoded = decodeUtf8(piece, partialValue, partialRemain);
        List<List<Integer>> next = nextStacks(token, decoded);
        if (next.isEmpty()) {
            return false;
        }
        stacks = next;
        partialValue = decoded[1][0];
        partialRemain = decoded[1][1];
        return true;
    }

    /**
     * The partial parses after {@code token} whose piece decoded to {@code decoded} ({@link #decodeUtf8}: the code
     * points with the trailing 0, then the partial tail); empty on a reject. A parse survives only when it can also
     * take the piece's trailing partial UTF-8 sequence, the test the mask applies ({@link #rejectsForStack}), so an
     * invalid continuation byte or an impossible partial character is refused here as it is masked there.
     */
    private List<List<Integer>> nextStacks(int token, int[][] decoded) {
        int[] cps = decoded[0];
        int value = decoded[1][0];
        int remain = decoded[1][1];
        List<List<Integer>> next = new ArrayList<>();
        for (List<Integer> unstarted : stacks) {
            List<Integer> stack = unstarted.isEmpty() ? null : tokenStart(unstarted);
            if (stack == null) {
                continue;
            }
            int pos = stack.get(stack.size() - 1);
            GrammarElement.Type type = flat[pos].type();
            if (GrammarElement.isToken(type)) {
                if ((cps.length > 1 || remain == 0) && matchToken(pos, token)) {
                    List<Integer> advanced = withoutTop(stack);
                    int after = afterTokenSet(pos);
                    if (!isEndOfSequence(after)) {
                        advanced.add(after);
                    }
                    advanceStack(advanced, next);
                }
            } else {
                List<List<Integer>> current = new ArrayList<>();
                current.add(stack);
                for (int i = 0; i < cps.length - 1 && !current.isEmpty(); i++) {
                    List<List<Integer>> step = new ArrayList<>();
                    for (List<Integer> cur : current) {
                        acceptChr(cur, cps[i], step);
                    }
                    current = step;
                }
                for (List<Integer> surviving : current) {
                    if (takesPartial(surviving, value, remain) && !next.contains(surviving)) {
                        next.add(surviving);
                    }
                }
            }
        }
        return next;
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
        return !nextStacks(token, decodeUtf8(piece, partialValue, partialRemain)).isEmpty();
    }

    /**
     * Whether a parse that read a piece's complete code points can take its trailing partial UTF-8 sequence: none
     * is pending ({@code remain == 0}), or the top is a char element with room left whose ranges hold a completion
     * of it ({@link #matchPartialChar}; an invalid sequence, {@code remain < 0}, has none).
     */
    private boolean takesPartial(List<Integer> stack, int value, int remain) {
        if (remain == 0) {
            return true;
        }
        if (stack.isEmpty()) {
            return false;
        }
        int top = stack.get(stack.size() - 1);
        return !GrammarElement.isToken(flat[top].type()) && charsLeft(stack) && matchPartialChar(top, value, remain);
    }

    @Override
    public boolean constrains() {
        return true;
    }

    /** The partial parses, the partial UTF-8 tail and the armed flag; the stack lists are never changed in place. */
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

    /**
     * Whether {@code piece}, fed after the current state, leaves a partial parse alive; the state is unchanged. A
     * trigger-gated stage uses it to keep a token from completing its trigger with text the grammar rejects.
     */
    boolean acceptsText(byte[] piece) {
        int[][] decoded = decodeUtf8(piece, partialValue, partialRemain);
        int[] cps = decoded[0];
        List<List<Integer>> current = new ArrayList<>(stacks.size());
        for (List<Integer> stack : stacks) {
            List<Integer> started = stack.isEmpty() ? null : tokenStart(stack);
            if (started != null) {
                current.add(started);
            }
        }
        for (int i = 0; i < cps.length - 1 && !current.isEmpty(); i++) {
            List<List<Integer>> step = new ArrayList<>();
            for (List<Integer> cur : current) {
                acceptChr(cur, cps[i], step);
            }
            current = step;
        }
        for (List<Integer> cur : current) {
            if (takesPartial(cur, decoded[1][0], decoded[1][1])) {
                return true;
            }
        }
        return false;
    }

    /** Arms the stage as its first {@link #apply} does: accepted tokens advance the grammar from here on. */
    void arm() {
        armed = true;
    }

    /** Whether a partial parse is complete: the grammar accepts the text so far as a whole sentence. */
    boolean completed() {
        for (List<Integer> stack : stacks) {
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
        stacks = initialStacks();
        partialValue = 0;
        partialRemain = -1;
        armed = false;
    }

    // ------------------------------------------------------------------ rejection walk (llama-grammar.cpp:1039-1131)

    /** Rejects every candidate no continuation of {@code stack} accepts; rejects keep their read cursor. */
    private List<Cand> rejectsForStack(List<Integer> stack, List<Cand> cands) {
        if (cands.isEmpty()) {
            // llama_grammar_reject_candidates returns at once: walking on would follow recursive rules forever
            return cands;
        }
        if (stack.isEmpty()) {
            List<Cand> rejects = new ArrayList<>();
            for (Cand cand : cands) {
                if (cand.cps[cand.off] != 0 || cand.partialRemain != 0) {
                    rejects.add(cand);
                }
            }
            return rejects;
        }
        int top = stack.get(stack.size() - 1);
        GrammarElement.Type type = flat[top].type();
        if (GrammarElement.isToken(type)) {
            List<Cand> rejects = new ArrayList<>();
            for (Cand cand : cands) {
                if (cand.cps[cand.off] == 0) {
                    if (cand.partialRemain != 0) {
                        rejects.add(cand);
                    }
                } else if (!matchToken(top, cand.id)) {
                    rejects.add(cand);
                }
            }
            return rejects;
        }
        List<Cand> next = new ArrayList<>();
        List<Cand> rejects = new ArrayList<>();
        boolean room = charsLeft(stack);
        for (Cand cand : cands) {
            if (cand.cps[cand.off] == 0) {
                if (cand.partialRemain != 0
                        && (!room || !matchPartialChar(top, cand.partialValue, cand.partialRemain))) {
                    rejects.add(cand);
                }
            } else if (room && matchChar(top, cand.cps[cand.off]).found) {
                next.add(new Cand(cand.index, cand.id, cand.cps, cand.off + 1, cand.partialValue,
                        cand.partialRemain));
            } else {
                rejects.add(cand);
            }
        }
        if (next.isEmpty()) {
            return rejects;
        }
        int after = matchChar(top, 0).next;
        List<Integer> stackAfter = consumeChar(withoutTop(stack));
        if (!isEndOfSequence(after)) {
            stackAfter.add(after);
        }
        List<List<Integer>> nextStacks = new ArrayList<>();
        advanceStack(stackAfter, nextStacks);
        List<Cand> nextRejects = next;
        for (List<Integer> ns : nextStacks) {
            nextRejects = rejectsForStack(ns, nextRejects);
            if (nextRejects.isEmpty()) {
                return rejects;
            }
        }
        for (Cand reject : nextRejects) {
            rejects.add(new Cand(reject.index, reject.id, reject.cps, reject.off - 1, reject.partialValue,
                    reject.partialRemain));
        }
        return rejects;
    }

    // ------------------------------------------------------------------ stack mechanics (llama-grammar.cpp:755-940)

    /**
     * {@code llama_grammar_init_impl}'s start state: one partial parse per alternative of the root rule, an empty
     * alternative being the empty (completed) stack.
     */
    private List<List<Integer>> initialStacks() {
        List<List<Integer>> initial = new ArrayList<>();
        int pos = ruleStart[grammar.rootRuleIndex()];
        while (true) {
            List<Integer> stack = new ArrayList<>();
            if (!isEndOfSequence(pos)) {
                stack.add(pos);
            }
            advanceStack(stack, initial);
            while (!isEndOfSequence(pos)) {
                pos++;
            }
            if (flat[pos].type() != GrammarElement.Type.ALT) {
                return initial;
            }
            pos++;
        }
    }

    /**
     * {@code llama_grammar_advance_stack}: expands rule references into every terminal-reaching stack. A budget
     * element pushes its frame under the rest of its sequence; a frame on top (its region complete) pops.
     */
    private void advanceStack(List<Integer> stack, List<List<Integer>> out) {
        Deque<List<Integer>> todo = new ArrayDeque<>();
        todo.push(stack);
        Set<List<Integer>> seen = new HashSet<>();
        while (!todo.isEmpty()) {
            List<Integer> curr = todo.pop();
            if (!seen.add(curr)) {
                continue;
            }
            if (curr.isEmpty()) {
                if (!out.contains(curr)) {
                    out.add(curr);
                }
                continue;
            }
            int pos = curr.get(curr.size() - 1);
            if (pos < 0) {
                todo.push(new ArrayList<>(curr.subList(0, curr.size() - 2)));
                continue;
            }
            GrammarElement.Type type = flat[pos].type();
            if (GrammarElement.isBudget(type)) {
                List<Integer> framed = withoutTop(curr);
                framed.add(flat[pos].value());
                framed.add(type == GrammarElement.Type.TOKEN_BUDGET ? TOKEN_FRAME : CHAR_FRAME);
                if (!isEndOfSequence(pos + 1)) {
                    framed.add(pos + 1);
                }
                todo.push(framed);
            } else if (type == GrammarElement.Type.RULE_REF) {
                int ruleId = flat[pos].value();
                int sub = ruleStart[ruleId];
                while (true) {
                    List<Integer> nextStack = withoutTop(curr);
                    if (!isEndOfSequence(pos + 1)) {
                        nextStack.add(pos + 1);
                    }
                    if (!isEndOfSequence(sub)) {
                        nextStack.add(sub);
                    }
                    todo.push(nextStack);
                    while (!isEndOfSequence(sub)) {
                        sub++;
                    }
                    if (flat[sub].type() == GrammarElement.Type.ALT) {
                        sub++;
                    } else {
                        break;
                    }
                }
            } else if (type == GrammarElement.Type.CHAR || type == GrammarElement.Type.CHAR_NOT
                    || type == GrammarElement.Type.CHAR_ANY || GrammarElement.isToken(type)) {
                if (!out.contains(curr)) {
                    out.add(curr);
                }
            } else {
                throw new IllegalStateException("grammar stack left on " + type
                        + " (a malformed rule ends mid-sequence)");
            }
        }
    }

    /** {@code llama_grammar_accept_chr}: consumes one code point at {@code stack}'s top into {@code out}. */
    private void acceptChr(List<Integer> stack, int chr, List<List<Integer>> out) {
        if (stack.isEmpty()) {
            return;
        }
        int pos = stack.get(stack.size() - 1);
        GrammarElement.Type type = flat[pos].type();
        if (GrammarElement.isToken(type) || !charsLeft(stack)) {
            return;
        }
        Match match = matchChar(pos, chr);
        if (match.found) {
            List<Integer> advanced = consumeChar(withoutTop(stack));
            if (!isEndOfSequence(match.next)) {
                advanced.add(match.next);
            }
            advanceStack(advanced, out);
        }
    }

    /** {@code llama_grammar_match_char}: the code point against the char cluster at {@code pos}. */
    private Match matchChar(int pos, int chr) {
        boolean found = false;
        boolean positive = flat[pos].type() == GrammarElement.Type.CHAR
                || flat[pos].type() == GrammarElement.Type.CHAR_ANY;
        do {
            GrammarElement element = flat[pos];
            if (flat[pos + 1].type() == GrammarElement.Type.CHAR_RNG_UPPER) {
                found = found || (element.value() <= chr && chr <= flat[pos + 1].value());
                pos += 2;
            } else if (element.type() == GrammarElement.Type.CHAR_ANY) {
                found = true;
                pos += 1;
            } else {
                found = found || element.value() == chr;
                pos += 1;
            }
        } while (flat[pos].type() == GrammarElement.Type.CHAR_ALT);
        return new Match(found == positive, pos);
    }

    /** {@code llama_grammar_match_partial_char}: whether the partial UTF-8 tail could satisfy {@code pos}. */
    private boolean matchPartialChar(int pos, int partialValue, int nRemain) {
        boolean positive = flat[pos].type() == GrammarElement.Type.CHAR
                || flat[pos].type() == GrammarElement.Type.CHAR_ANY;
        if (nRemain < 0 || (nRemain == 1 && partialValue < 2)) {
            return false;
        }
        int low = partialValue << (nRemain * 6);
        int high = low | ((1 << (nRemain * 6)) - 1);
        if (low == 0) {
            if (nRemain == 2) {
                low = 1 << 11;
            } else if (nRemain == 3) {
                low = 1 << 16;
            }
        }
        do {
            GrammarElement element = flat[pos];
            if (flat[pos + 1].type() == GrammarElement.Type.CHAR_RNG_UPPER) {
                if (element.value() <= high && low <= flat[pos + 1].value()) {
                    return positive;
                }
                pos += 2;
            } else if (element.type() == GrammarElement.Type.CHAR_ANY) {
                return true;
            } else {
                if (low <= element.value() && element.value() <= high) {
                    return positive;
                }
                pos += 1;
            }
        } while (flat[pos].type() == GrammarElement.Type.CHAR_ALT);
        return !positive;
    }

    /**
     * {@code llama_grammar_match_token}: the token id against a TOKEN / TOKEN_NOT element and the TOKEN_ALT ids
     * after it: a TOKEN set matches any of its ids, a TOKEN_NOT set any token outside them, TOKEN_ANY any token.
     */
    private boolean matchToken(int pos, int token) {
        GrammarElement.Type type = flat[pos].type();
        if (type == GrammarElement.Type.TOKEN_ANY) {
            return true;
        }
        if (type != GrammarElement.Type.TOKEN && type != GrammarElement.Type.TOKEN_NOT) {
            throw new IllegalStateException("token match outside a token element: " + type);
        }
        boolean found = flat[pos].value() == token;
        for (int i = pos + 1; !found && flat[i].type() == GrammarElement.Type.TOKEN_ALT; i++) {
            found = flat[i].value() == token;
        }
        return found == (type == GrammarElement.Type.TOKEN);
    }

    /** The position after the token element at {@code pos} and its TOKEN_ALT ids. */
    private int afterTokenSet(int pos) {
        int i = pos + 1;
        while (flat[i].type() == GrammarElement.Type.TOKEN_ALT) {
            i++;
        }
        return i;
    }

    private boolean isEndOfSequence(int pos) {
        GrammarElement.Type type = flat[pos].type();
        return type == GrammarElement.Type.END || type == GrammarElement.Type.ALT;
    }

    // ------------------------------------------------------------------ rule flattening and UTF-8

    private static GrammarElement[] flatten(Grammar grammar) {
        int total = 0;
        for (int r = 0; r < grammar.ruleCount(); r++) {
            total += grammar.rule(r).size();
        }
        GrammarElement[] flat = new GrammarElement[total];
        int at = 0;
        for (int r = 0; r < grammar.ruleCount(); r++) {
            List<GrammarElement> rule = grammar.rule(r);
            for (GrammarElement element : rule) {
                flat[at++] = element;
            }
        }
        return flat;
    }

    private static int[] ruleStarts(Grammar grammar) {
        int[] starts = new int[grammar.ruleCount()];
        int at = 0;
        for (int r = 0; r < grammar.ruleCount(); r++) {
            starts[r] = at;
            at += grammar.rule(r).size();
        }
        return starts;
    }

    private static List<Integer> withoutTop(List<Integer> stack) {
        return new ArrayList<>(stack.subList(0, stack.size() - 1));
    }

    // ------------------------------------------------------------------ budget frames

    /**
     * {@code stack} as a new token starts on it: every open token frame spends one, or null when one is already
     * spent (the parse cannot take the token inside its region). The same list when no token frame is open.
     */
    private static List<Integer> tokenStart(List<Integer> stack) {
        return spend(stack, TOKEN_FRAME);
    }

    /** Whether {@code stack} may consume one more code point: none of its character frames is spent. */
    private static boolean charsLeft(List<Integer> stack) {
        for (int i = stack.size() - 1; i > 0; i--) {
            if (stack.get(i) == CHAR_FRAME) {
                if (stack.get(i - 1) == 0) {
                    return false;
                }
                i--;
            } else if (stack.get(i) == TOKEN_FRAME) {
                i--;
            }
        }
        return true;
    }

    /** {@code stack} after one code point: every open character frame spends one ({@link #charsLeft} holds). */
    private static List<Integer> consumeChar(List<Integer> stack) {
        List<Integer> spent = spend(stack, CHAR_FRAME);
        if (spent == null) {
            throw new IllegalStateException("a code point consumed past a spent character budget");
        }
        return spent;
    }

    /**
     * {@code stack} with every {@code marker} frame's count spent by one, or null when one of them is already 0.
     * Frames are read from the top: a marker's count lies right below it, so a count is never read as a marker.
     */
    private static List<Integer> spend(List<Integer> stack, int marker) {
        List<Integer> out = null;
        for (int i = stack.size() - 1; i > 0; i--) {
            int entry = stack.get(i);
            if (entry >= 0) {
                continue;
            }
            if (entry == marker) {
                int left = stack.get(i - 1);
                if (left == 0) {
                    return null;
                }
                if (out == null) {
                    out = new ArrayList<>(stack);
                }
                out.set(i - 1, left - 1);
            }
            i--;
        }
        return out == null ? stack : out;
    }

    /**
     * {@code decode_utf8}: the piece's code points with a trailing 0, plus the partial tail a multi-byte character
     * split across tokens leaves ({@code {value, nRemain}}).
     */
    private static int[][] decodeUtf8(byte[] src, int startValue, int startRemain) {
        List<Integer> codePoints = new ArrayList<>(src.length + 1);
        int value = startValue;
        int nRemain = startRemain;
        int pos = 0;
        while (pos < src.length && nRemain > 0) {
            int nextByte = src[pos] & 0xFF;
            if ((nextByte >> 6) != 2) {
                codePoints.add(0);
                return new int[][]{{0}, {0, -1}};
            }
            value = (value << 6) + (nextByte & 0x3F);
            pos++;
            nRemain--;
        }
        if (startRemain > 0 && nRemain == 0) {
            codePoints.add(value);
        }
        while (pos < src.length) {
            int firstByte = src[pos] & 0xFF;
            nRemain = LOOKUP[firstByte >> 4] - 1;
            if (nRemain < 0) {
                return new int[][]{{0}, {0, nRemain}};
            }
            int mask = (1 << (7 - nRemain)) - 1;
            value = firstByte & mask;
            pos++;
            while (pos < src.length && nRemain > 0) {
                value = (value << 6) + (src[pos] & 0x3F);
                pos++;
                nRemain--;
            }
            if (nRemain == 0) {
                codePoints.add(value);
            }
        }
        codePoints.add(0);
        int[] cps = new int[codePoints.size()];
        for (int i = 0; i < cps.length; i++) {
            cps[i] = codePoints.get(i);
        }
        return new int[][]{cps, {value, nRemain}};
    }

    /** The candidate's own decode, from the grammar's current partial tail. */
    private Cand decode(byte[] piece, int index, int id) {
        int[][] decoded = decodeUtf8(piece, partialValue, partialRemain);
        return new Cand(index, id, decoded[0], 0, decoded[1][0], decoded[1][1]);
    }
}
