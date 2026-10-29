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
 * The grammar-constrained sampling stage (llama.cpp {@code llama-grammar.cpp}: {@code llama_grammar_apply_impl},
 * {@code llama_grammar_accept_token}): only the tokens the {@link Grammar} can still accept keep their logit, the
 * rest take {@code -inf}, and the accepted token advances the grammar's pushdown stacks. A Java-API custom stage
 * ({@code Sampler.create(config, bannedTokens, extra, vocab)}), so it always applies, including under greedy
 * decoding, and its {@link #modifies() true} keeps an argumentmax chain off the on-device argmax path (the mask
 * must see the logits).
 *
 * <p>The grammar state is the set of partial parses (stacks of flattened-rule indices, top last) plus the partial
 * UTF-8 sequence a multi-byte character split across tokens leaves. {@code accept} feeds only tokens sampled after
 * the stage first applied ({@link #apply}): {@code Sampler.accept} also feeds prompt tokens (the penalty stages'
 * history), and llama.cpp's grammar sees generated tokens only — a prompt must not advance the grammar.
 * {@link #reset} returns the initial stacks.
 *
 * <p>Not thread-safe, owned by its sampler like every stage.
 */
public final class GrammarStage extends SamplerStage {

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

    private static final int[] LOOKUP = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};

    private final Grammar grammar;
    private final SamplerVocab vocab;
    private final GrammarElement[] flat;
    private final int[] ruleStart;
    private List<List<Integer>> stacks;
    private int partialValue;
    private int partialRemain;
    private boolean armed;

    /** A stage over {@code grammar}; token pieces and end-of-generation ids come from {@code vocab}. */
    public GrammarStage(Grammar grammar, SamplerVocab vocab) {
        this.grammar = Objects.requireNonNull(grammar, "grammar");
        this.vocab = Objects.requireNonNull(vocab, "vocab");
        this.flat = flatten(grammar);
        this.ruleStart = ruleStarts(grammar);
        this.stacks = new ArrayList<>();
        advanceStack(newList(ruleStart[grammar.rootRuleIndex()]), this.stacks);
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
                byte[] piece = vocab.piece(id);
                if (piece.length == 0 || piece[0] == 0) {
                    // llama.cpp rejects an empty piece and a piece starting on the terminating 0 alike
                    candidates.setLogit(i, Float.NEGATIVE_INFINITY);
                } else {
                    cands[i] = decode(piece, i);
                    live.add(i);
                }
            }
        }
        List<Cand> rejects = new ArrayList<>(live.size());
        for (int i : live) {
            rejects.add(cands[i]);
        }
        for (List<Integer> stack : stacks) {
            rejects = rejectsForStack(stack, rejects);
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
            for (List<Integer> stack : stacks) {
                if (stack.isEmpty()) {
                    return;
                }
            }
            throw new IllegalStateException("grammar cannot accept the end-of-generation token " + token);
        }
        byte[] piece = vocab.piece(token);
        if (piece.length == 0) {
            return; // a control piece carries no grammar bytes (the mask kept such tokens out)
        }
        int[][] decoded = decodeUtf8(piece, partialValue, partialRemain);
        int[] cps = decoded[0];
        List<List<Integer>> next = new ArrayList<>();
        for (List<Integer> stack : stacks) {
            if (stack.isEmpty()) {
                continue;
            }
            int pos = stack.get(stack.size() - 1);
            GrammarElement.Type type = flat[pos].type();
            if (type == GrammarElement.Type.TOKEN || type == GrammarElement.Type.TOKEN_NOT) {
                if (matchToken(pos, token)) {
                    List<Integer> advanced = withoutTop(stack);
                    if (!isEndOfSequence(pos + 1)) {
                        advanced.add(pos + 1);
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
                    if (!next.contains(surviving)) {
                        next.add(surviving);
                    }
                }
            }
        }
        stacks = next;
        partialValue = decoded[1][0];
        partialRemain = decoded[1][1];
        if (stacks.isEmpty()) {
            throw new IllegalStateException("grammar stacks emptied by token " + token);
        }
    }

    @Override
    public boolean modifies() {
        return true;
    }

    @Override
    public void reset() {
        stacks = new ArrayList<>();
        advanceStack(newList(ruleStart[grammar.rootRuleIndex()]), stacks);
        partialValue = 0;
        partialRemain = -1;
        armed = false;
    }

    // ------------------------------------------------------------------ rejection walk (llama-grammar.cpp:1039-1131)

    /** Rejects every candidate no continuation of {@code stack} accepts; rejects keep their read cursor. */
    private List<Cand> rejectsForStack(List<Integer> stack, List<Cand> cands) {
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
        if (type == GrammarElement.Type.TOKEN || type == GrammarElement.Type.TOKEN_NOT) {
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
        for (Cand cand : cands) {
            if (cand.cps[cand.off] == 0) {
                if (cand.partialRemain != 0 && !matchPartialChar(top, cand.partialValue, cand.partialRemain)) {
                    rejects.add(cand);
                }
            } else if (matchChar(top, cand.cps[cand.off]).found) {
                next.add(new Cand(cand.index, cand.id, cand.cps, cand.off + 1, cand.partialValue,
                        cand.partialRemain));
            } else {
                rejects.add(cand);
            }
        }
        int after = matchChar(top, 0).next;
        List<Integer> stackAfter = withoutTop(stack);
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

    /** {@code llama_grammar_advance_stack}: expands rule references into every terminal-reaching stack. */
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
            GrammarElement.Type type = flat[pos].type();
            if (type == GrammarElement.Type.RULE_REF) {
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
                    || type == GrammarElement.Type.CHAR_ANY || type == GrammarElement.Type.TOKEN
                    || type == GrammarElement.Type.TOKEN_NOT) {
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
        if (type == GrammarElement.Type.TOKEN || type == GrammarElement.Type.TOKEN_NOT) {
            return;
        }
        Match match = matchChar(pos, chr);
        if (match.found) {
            List<Integer> advanced = withoutTop(stack);
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

    /** {@code llama_grammar_match_token}: the token id against a TOKEN / TOKEN_NOT element. */
    private boolean matchToken(int pos, int token) {
        GrammarElement.Type type = flat[pos].type();
        if (type == GrammarElement.Type.TOKEN) {
            return flat[pos].value() == token;
        }
        if (type == GrammarElement.Type.TOKEN_NOT) {
            return flat[pos].value() != token;
        }
        throw new IllegalStateException("token match outside a token element: " + type);
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

    private static List<Integer> newList(int value) {
        List<Integer> list = new ArrayList<>(1);
        list.add(value);
        return list;
    }

    private static List<Integer> withoutTop(List<Integer> stack) {
        return new ArrayList<>(stack.subList(0, stack.size() - 1));
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
    private Cand decode(byte[] piece, int index) {
        int[][] decoded = decodeUtf8(piece, partialValue, partialRemain);
        return new Cand(index, -1, decoded[0], 0, decoded[1][0], decoded[1][1]);
    }
}
