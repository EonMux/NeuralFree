package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.grammar.Grammar;
import com.eonmux.neuralfree.gguf.grammar.GrammarTriggers;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
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
 * generation there instead). Pattern and token triggers fire on what the model wrote and are not pre-masked.
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

    private final Grammar grammar;
    private final SamplerVocab vocab;
    private final GrammarTriggers triggers;
    private final byte[][] words;
    private final boolean[] wordLastByte;
    private final int longestWord;

    private final ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    private final List<Span> spans = new ArrayList<>();
    private GrammarStage active;
    private GrammarStage probe;
    private boolean armed;

    /** A stage over {@code grammar} gated by {@code triggers}; pieces and end-of-generation ids from {@code vocab}. */
    public TriggeredGrammarStage(Grammar grammar, GrammarTriggers triggers, SamplerVocab vocab) {
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
        if (words.length > 0) {
            maskRejectedTriggers(candidates);
        }
    }

    @Override
    public void accept(int token) {
        if (!armed) {
            return; // prompt history (see class doc)
        }
        if (active != null) {
            active.accept(token);
            if (triggers.mode() == GrammarTriggers.Mode.DISPATCH && active.completed()) {
                active = null; // the structure closed: free text until the next trigger
            }
            return;
        }
        if (vocab.isEog(token)) {
            return;
        }
        byte[] piece = vocab.grammarPiece(token);
        if (triggers.firesOn(token)) {
            start();
            active.acceptPiece(token, piece);
            return;
        }
        int from = buffer.size();
        buffer.writeBytes(piece);
        spans.add(new Span(token, from, buffer.size()));
        byte[] bytes = buffer.toByteArray();
        String text = new String(bytes, StandardCharsets.UTF_8);
        int at = triggers.firstStart(text);
        if (at < 0) {
            return;
        }
        int byteStart = text.substring(0, at).getBytes(StandardCharsets.UTF_8).length;
        List<Span> replay = new ArrayList<>(spans);
        start();
        for (Span s : replay) {
            if (s.end <= byteStart) {
                continue;
            }
            int pieceStart = Math.max(s.start, byteStart);
            active.acceptPiece(s.token, Arrays.copyOfRange(bytes, pieceStart, s.end));
        }
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

    /**
     * Masks every candidate whose piece completes a trigger word followed by bytes the grammar rejects: the text
     * from the earliest word the piece completes must leave the fresh grammar a live parse.
     */
    private void maskRejectedTriggers(Sampler.Candidates candidates) {
        byte[] all = buffer.toByteArray();
        int keep = Math.min(all.length, longestWord - 1);
        byte[] tail = Arrays.copyOfRange(all, all.length - keep, all.length);
        for (int i = 0; i < candidates.size(); i++) {
            if (candidates.logit(i) == Float.NEGATIVE_INFINITY) {
                continue;
            }
            int id = candidates.id(i);
            if (vocab.isEog(id)) {
                continue;
            }
            byte[] piece = vocab.grammarPiece(id);
            if (!mayEndWord(piece)) {
                continue;
            }
            byte[] joined = new byte[tail.length + piece.length];
            System.arraycopy(tail, 0, joined, 0, tail.length);
            System.arraycopy(piece, 0, joined, tail.length, piece.length);
            int start = earliestWordEndingInPiece(joined, tail.length);
            if (start < 0) {
                continue;
            }
            if (probe == null) {
                probe = new GrammarStage(grammar, vocab);
            }
            if (!probe.acceptsText(Arrays.copyOfRange(joined, start, joined.length))) {
                candidates.setLogit(i, Float.NEGATIVE_INFINITY);
            }
        }
    }

    private boolean mayEndWord(byte[] piece) {
        for (byte b : piece) {
            if (wordLastByte[b & 0xFF]) {
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
