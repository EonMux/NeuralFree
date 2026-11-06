package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.GrammarTriggers;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * A grammar that waits for a trigger (llama.cpp's lazy grammar, {@code llama_grammar_accept_impl} with
 * {@code awaiting_trigger}; xgrammar's tag dispatch under {@link GrammarTriggers.Mode#DISPATCH}). While it waits the
 * stage leaves the candidates alone and buffers the accepted pieces; when a {@link GrammarTriggers} trigger fires
 * — a trigger token is sampled, or a word or pattern matches the buffered text — a fresh {@link GrammarStage} starts
 * and the tokens overlapping the trigger replay into it, each with the part of its piece from the trigger start on,
 * as llama.cpp replays {@code trigger_buffer_positions}. From then on the grammar masks the candidates.
 *
 * <p>{@link GrammarTriggers.Mode#LAZY} keeps the grammar to the end of the generation. Under
 * {@link GrammarTriggers.Mode#DISPATCH} a completed grammar sentence returns the stage to free text and the next
 * trigger starts a fresh grammar.
 *
 * <p>While waiting, a candidate that would complete a trigger word with text the grammar rejects is masked: the
 * sampler cannot pick a token whose trailing bytes would empty the grammar the trigger starts (llama.cpp aborts the
 * generation there instead). Pattern and token triggers fire on what the model wrote and are not pre-masked. The
 * same pass masks the compiled constraint's free-text exclusions
 * ({@link com.eonmux.neuralfree.gguf.grammar.CompiledConstraint#freeExcludes}, the {@code excludes} and
 * {@code exclude_tokens} of a dispatching structural-tag format): a candidate whose piece writes an excluded string,
 * or whose id is an excluded token, is masked while the stage waits.
 *
 * <p>Prompt tokens never advance the stage: like {@link GrammarStage} it counts accepted tokens only after its
 * first {@link #apply}. Not thread-safe, owned by its sampler.
 */
public final class TriggeredGrammarStage extends SamplerStage {

    /** One buffered token and its byte span in the trigger buffer. */
    private static final class Span {
        final int token;
        final int start;
        final int end;

        Span(int token, int start, int end) {
            this.token = token;
            this.start = start;
            this.end = end;
        }
    }

    /** A saved stage state ({@link #saveState}); spans are immutable and the lists are copies. */
    private static final class State {
        final GrammarStage active;
        final Object activeState;
        final byte[] buffer;
        final List<Span> spans;
        final boolean armed;

        State(GrammarStage active, Object activeState, byte[] buffer, List<Span> spans, boolean armed) {
            this.active = active;
            this.activeState = activeState;
            this.buffer = buffer;
            this.spans = spans;
            this.armed = armed;
        }
    }

    private final Grammar grammar;
    private final SamplerVocab vocab;
    private final GrammarTriggers triggers;
    private final byte[][] words;
    private final boolean[] wordLastByte;
    private final int longestWord;
    private final byte[][] excludeWords;
    private final boolean[] excludeLastByte;
    private final int longestExclude;
    private final int[] excludeTokens;

    private final ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    private final List<Span> spans = new ArrayList<>();
    private GrammarStage active;
    private GrammarStage probe;
    private boolean armed;

    /** A stage over {@code grammar} gated by {@code triggers}; pieces and end-of-generation ids from {@code vocab}. */
    public TriggeredGrammarStage(Grammar grammar, GrammarTriggers triggers, SamplerVocab vocab) {
        this(grammar, triggers, vocab, null, null);
    }

    /**
     * A stage whose free text also refuses {@code freeExcludes} strings and {@code freeExcludeTokens} ids while it
     * waits for a trigger ({@link com.eonmux.neuralfree.gguf.grammar.CompiledConstraint#freeExcludes}); either may be
     * null.
     *
     * @throws IllegalArgumentException when an excluded string is empty
     */
    public TriggeredGrammarStage(Grammar grammar, GrammarTriggers triggers, SamplerVocab vocab,
            List<String> freeExcludes, int[] freeExcludeTokens) {
        this.grammar = Objects.requireNonNull(grammar, "grammar");
        this.triggers = Objects.requireNonNull(triggers, "triggers");
        this.vocab = Objects.requireNonNull(vocab, "vocab");
        List<String> w = triggers.words();
        this.words = new byte[w.size()][];
        this.wordLastByte = new boolean[256];
        int longest = 0;
        for (int i = 0; i < words.length; i++) {
            words[i] = w.get(i).getBytes(StandardCharsets.UTF_8);
            wordLastByte[words[i][words[i].length - 1] & 0xFF] = true;
            longest = Math.max(longest, words[i].length);
        }
        this.longestWord = longest;
        List<String> excludes = freeExcludes == null ? Collections.emptyList() : freeExcludes;
        this.excludeWords = new byte[excludes.size()][];
        this.excludeLastByte = new boolean[256];
        int longestExcluded = 0;
        for (int i = 0; i < excludeWords.length; i++) {
            if (excludes.get(i).isEmpty()) {
                throw new IllegalArgumentException("free-text exclusion " + i + " is empty");
            }
            excludeWords[i] = excludes.get(i).getBytes(StandardCharsets.UTF_8);
            excludeLastByte[excludeWords[i][excludeWords[i].length - 1] & 0xFF] = true;
            longestExcluded = Math.max(longestExcluded, excludeWords[i].length);
        }
        this.longestExclude = longestExcluded;
        this.excludeTokens = freeExcludeTokens == null ? new int[0] : freeExcludeTokens.clone();
    }

    /** Whether the stage still waits for a trigger (the output is free text). */
    public boolean awaitingTrigger() {
        return active == null;
    }

    @Override
    public void apply(Sampler.Candidates candidates) {
        armed = true;
        if (active != null) {
            active.apply(candidates);
            return;
        }
        if (words.length > 0 || excludeWords.length > 0 || excludeTokens.length > 0) {
            maskAwaiting(candidates);
        }
    }

    @Override
    public void accept(int token) {
        if (!consume(token)) {
            throw new IllegalStateException(vocab.isEog(token)
                    ? "grammar cannot accept the end-of-generation token " + token
                    : "grammar stacks emptied by token " + token);
        }
    }

    /**
     * The step of {@link #accept}: false when the active grammar (or the one a trigger in this token starts)
     * refuses the token, in which case the stage may be left part-way changed.
     */
    private boolean consume(int token) {
        if (!armed) {
            return true; // prompt history (see class doc)
        }
        if (active != null) {
            if (vocab.isEog(token) ? !active.completed() : !active.advance(token, vocab.grammarPiece(token))) {
                return false;
            }
            if (triggers.mode() == GrammarTriggers.Mode.DISPATCH && active.completed()) {
                active = null; // the structure closed: free text until the next trigger
            }
            return true;
        }
        if (vocab.isEog(token)) {
            return true;
        }
        byte[] piece = vocab.grammarPiece(token);
        if (triggers.firesOn(token)) {
            start();
            return active.advance(token, piece);
        }
        int from = buffer.size();
        buffer.writeBytes(piece);
        spans.add(new Span(token, from, buffer.size()));
        byte[] bytes = buffer.toByteArray();
        String text = new String(bytes, StandardCharsets.UTF_8);
        int at = triggers.firstStart(text);
        if (at < 0) {
            return true;
        }
        int byteStart = text.substring(0, at).getBytes(StandardCharsets.UTF_8).length;
        List<Span> replay = new ArrayList<>(spans);
        start();
        for (Span s : replay) {
            if (s.end <= byteStart) {
                continue;
            }
            int pieceStart = Math.max(s.start, byteStart);
            if (!active.advance(s.token, Arrays.copyOfRange(bytes, pieceStart, s.end))) {
                return false;
            }
        }
        return true;
    }

    /**
     * Whether {@link #accept} takes {@code token} next: free text passes unless the token fires a trigger whose
     * grammar refuses the replayed text, and an active grammar must take the token's piece (an end-of-generation
     * token, a completed sentence).
     */
    @Override
    public boolean admits(int token) {
        if (!armed) {
            return true;
        }
        Object saved = saveState();
        try {
            return consume(token);
        } finally {
            restoreState(saved);
        }
    }

    @Override
    public boolean constrains() {
        return true;
    }

    /** The active grammar with its own state, the trigger buffer with its spans and the armed flag. */
    @Override
    public Object saveState() {
        return new State(active, active == null ? null : active.saveState(), buffer.toByteArray(),
                new ArrayList<>(spans), armed);
    }

    @Override
    public void restoreState(Object state) {
        if (!(state instanceof State)) {
            throw new IllegalArgumentException("not a triggered grammar stage state: " + state);
        }
        State s = (State) state;
        active = s.active;
        if (active != null) {
            active.restoreState(s.activeState);
        }
        buffer.reset();
        buffer.writeBytes(s.buffer);
        spans.clear();
        spans.addAll(s.spans);
        armed = s.armed;
    }

    @Override
    public boolean modifies() {
        return true;
    }

    @Override
    public void reset() {
        active = null;
        armed = false;
        buffer.reset();
        spans.clear();
    }

    /** Opens a fresh grammar for the trigger that just fired and clears the trigger buffer. */
    private void start() {
        active = new GrammarStage(grammar, vocab);
        active.arm();
        buffer.reset();
        spans.clear();
    }

    /** Masks what the waiting phase refuses: trigger-completing tokens with rejected text, and the exclusions. */
    private void maskAwaiting(Sampler.Candidates candidates) {
        byte[] all = buffer.toByteArray();
        byte[] tail = awaitingTail(all, Math.max(longestWord, longestExclude));
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates.logit(i) == Float.NEGATIVE_INFINITY) {
                continue;
            }
            int id = candidates.id(i);
            if (vocab.isEog(id)) {
                continue;
            }
            if (completesTriggerOrExclusion(candidates, i, id, tail)) {
                candidates.setLogit(i, Float.NEGATIVE_INFINITY);
            }
        }
    }

    /** The buffer tail a next piece can still extend into a trigger word or an excluded string. */
    private static byte[] awaitingTail(byte[] all, int longest) {
        int keep = longest <= 0 ? 0 : Math.min(all.length, longest - 1);
        return Arrays.copyOfRange(all, all.length - keep, all.length);
    }

    /** Whether candidate {@code i} completes a trigger word with rejected text, or writes an exclusion. */
    private boolean completesTriggerOrExclusion(Sampler.Candidates candidates, int i, int id, byte[] tail) {
        for (int token : excludeTokens) {
            if (token == id) {
                return true;
            }
        }
        byte[] piece = vocab.grammarPiece(id);
        if (piece.length == 0 || (!mayEndWord(piece, wordLastByte) && !mayEndWord(piece, excludeLastByte))) {
            return false;
        }
        byte[] joined = new byte[tail.length + piece.length];
        System.arraycopy(tail, 0, joined, 0, tail.length);
        System.arraycopy(piece, 0, joined, tail.length, piece.length);
        if (excludeWords.length > 0 && completesExclusion(joined, tail.length)) {
            return true;
        }
        if (words.length == 0) {
            return false;
        }
        int start = earliestWordEndingInPiece(joined, tail.length);
        if (start < 0) {
            return false;
        }
        if (probe == null) {
            probe = new GrammarStage(grammar, vocab);
        }
        return !probe.acceptsText(Arrays.copyOfRange(joined, start, joined.length));
    }

    /** Whether {@code joined} gains an occurrence of an excluded string ending inside the candidate's piece. */
    private boolean completesExclusion(byte[] joined, int pieceFrom) {
        for (byte[] w : excludeWords) {
            for (int s = Math.max(0, pieceFrom - w.length + 1); s + w.length <= joined.length; s++) {
                if (regionEquals(joined, s, w)) {
                    return true;
                }
            }
        }
        return false;
    }

    /** Whether the piece holds the last byte of any word or excluded string, worth a full join check. */
    private static boolean mayEndWord(byte[] piece, boolean[] lastByte) {
        for (byte b : piece) {
            if (lastByte[b & 0xFF]) {
                return true;
            }
        }
        return false;
    }

    /** The start of the earliest word occurrence in {@code joined} that ends past {@code pieceFrom}, or -1. */
    private int earliestWordEndingInPiece(byte[] joined, int pieceFrom) {
        int best = -1;
        for (byte[] w : words) {
            int first = Math.max(0, pieceFrom - w.length + 1);
            for (int s = first; s + w.length <= joined.length; s++) {
                if (best >= 0 && s >= best) {
                    break;
                }
                if (regionEquals(joined, s, w)) {
                    best = s;
                    break;
                }
            }
        }
        return best;
    }

    private static boolean regionEquals(byte[] a, int at, byte[] w) {
        for (int k = 0; k < w.length; k++) {
            if (a[at + k] != w[k]) {
                return false;
            }
        }
        return true;
    }
}
