package com.eonmux.neuralfree.gguf.tokenizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * SentencePiece BPE tokenizer ({@code tokenizer.ggml.model = "llama"}), transcribed from
 * llama.cpp {@code llm_tokenizer_spm_session} (src/llama-vocab.cpp:115-239) and the SPM branch of
 * {@code llama_vocab::impl::tokenize} (src/llama-vocab.cpp:3726-3776).
 *
 * <p>Each raw fragment is prefixed with a space when {@code add_space_prefix} is set and the
 * previous fragment was special (or absent), every ASCII space is escaped to U+2581, the text is
 * seeded as one symbol per code point, adjacent pairs whose concatenation is a vocabulary token
 * merge highest score first (ties: leftmost), and symbols that are not tokens fall back to
 * {@code <0xXX>} byte tokens.</p>
 */
public final class SpmTokenizer implements TokenizerAlgorithm {

    private static final String[] BYTE_TOKENS = new String[256];

    static {
        for (int b = 0; b < 256; b++) BYTE_TOKENS[b] = String.format("<0x%02X>", b);
    }

    private final TokenizerVocab vocab;
    private final SpecialTokens special;

    public SpmTokenizer(TokenizerVocab vocab, SpecialTokens special) {
        if (vocab == null || special == null) throw new IllegalArgumentException("vocabulary and special tokens are required");
        if (vocab.type() != TokenizerVocab.Type.SPM) {
            throw new IllegalArgumentException("SpmTokenizer requires an SPM vocabulary, got " + vocab.type()
                    + " (model '" + vocab.model() + "')");
        }
        this.vocab = vocab;
        this.special = special;
    }

    @Override
    public int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        IdBuffer out = new IdBuffer(Math.max(16, text.length() / 3));
        boolean isPrevSpecial = true;
        if (addSpecial && vocab.addBos()) {
            out.add(special.requireBos());
        }
        for (SpecialTokens.Fragment f : special.partition(text, parseSpecial)) {
            if (f.isToken()) {
                out.add(f.token());
                isPrevSpecial = true;
            } else {
                String raw = vocab.addSpacePrefix() && isPrevSpecial ? " " + f.text() : f.text();
                tokenizeFragment(TokenizerVocab.escapeWhitespace(raw), out);
                isPrevSpecial = false;
            }
        }
        if (addSpecial && vocab.addEos()) {
            out.add(special.requireEos());
        }
        return out.toArray();
    }

    @Override
    public String decode(int[] ids) {
        return TokenizerAlgorithm.detokenize(this, vocab, special, ids);
    }

    @Override
    public byte[] piece(int id) {
        return TokenizerAlgorithm.sentencePiece(vocab, id);
    }

    /** {@code llm_tokenizer_spm_session::tokenize} for one escaped fragment. */
    private void tokenizeFragment(String text, IdBuffer out) {
        if (text.isEmpty()) return;
        new Session(text).merge().emit(out);
    }

    /** Per-fragment merge state: a doubly linked list of symbols over the fragment text. */
    private final class Session {
        private final String text;
        private final int[] start;
        private final int[] len;
        private final int[] prev;
        private final int[] next;
        private final int count;
        private final PriorityQueue<Bigram> queue = new PriorityQueue<>();
        private final Map<String, int[]> revMerge = new HashMap<>();

        Session(String text) {
            this.text = text;
            int cps = text.codePointCount(0, text.length());
            start = new int[cps];
            len = new int[cps];
            prev = new int[cps];
            next = new int[cps];
            int offs = 0;
            for (int i = 0; i < cps; i++) {
                int n = Character.charCount(text.codePointAt(offs));
                start[i] = offs;
                len[i] = n;
                offs += n;
                prev[i] = i - 1;
                next[i] = offs == text.length() ? -1 : i + 1;
            }
            count = cps;
        }

        /** Seeds all adjacent pairs, then merges the best pair until none is a token. */
        Session merge() {
            for (int i = 1; i < count; i++) tryAddBigram(i - 1, i);
            while (!queue.isEmpty()) {
                Bigram b = queue.poll();
                if (len[b.left] == 0 || len[b.right] == 0 || len[b.left] + len[b.right] != b.size) continue;
                len[b.left] += len[b.right];
                len[b.right] = 0;
                next[b.left] = next[b.right];
                if (next[b.right] >= 0) prev[next[b.right]] = b.left;
                tryAddBigram(prev[b.left], b.left);
                tryAddBigram(b.left, next[b.left]);
            }
            return this;
        }

        void emit(IdBuffer out) {
            for (int i = 0; i != -1; i = next[i]) resegment(i, out);
        }

        List<String> symbols() {
            List<String> result = new ArrayList<>();
            for (int i = 0; i != -1; i = next[i]) result.add(symbolText(i));
            return result;
        }

        private String symbolText(int i) {
            return text.substring(start[i], start[i] + len[i]);
        }

        private void tryAddBigram(int left, int right) {
            if (left == -1 || right == -1) return;
            String merged = text.substring(start[left], start[left] + len[left] + len[right]);
            int id = vocab.tokenId(merged);
            if (id < 0 || id >= vocab.size()) return;
            queue.add(new Bigram(left, right, vocab.score(id), merged.length()));
            revMerge.put(merged, new int[]{left, right});
        }

        private void resegment(int i, IdBuffer out) {
            String s = symbolText(i);
            int id = vocab.tokenId(s);
            if (id >= 0) {
                out.add(id);
                return;
            }
            int[] p = revMerge.get(s);
            if (p == null) {
                for (byte b : TokenizerAlgorithm.utf8(s)) out.add(byteToToken(b & 0xFF));
                return;
            }
            resegment(p[0], out);
            resegment(p[1], out);
        }
    }

    /**
     * {@code llama_vocab::byte_to_token} for SPM (llama-vocab.cpp:4316-4326): the {@code <0xXX>}
     * token, else the single-character token for an ASCII byte. A vocabulary without either
     * falls back to its unknown token, as SentencePiece does without byte fallback.
     */
    private int byteToToken(int b) {
        int id = vocab.tokenId(BYTE_TOKENS[b]);
        if (id >= 0) return id;
        if (b < 0x80) {
            id = vocab.tokenId(String.valueOf((char) b));
            if (id >= 0) return id;
        }
        if (special.unk() != SpecialTokens.NULL) return special.unk();
        throw new IllegalStateException(String.format(
                "SPM vocabulary has no byte token for 0x%02X and no unknown token", b));
    }

    /**
     * {@code llm_bigram_spm::comparator}: highest score first, then smallest left index. Scores
     * compare as primitives, as in C++, so {@code -0.0f} and {@code 0.0f} tie.
     */
    private static final class Bigram implements Comparable<Bigram> {
        final int left;
        final int right;
        final float score;
        final int size;

        Bigram(int left, int right, float score, int size) {
            this.left = left;
            this.right = right;
            this.score = score;
            this.size = size;
        }

        @Override
        public int compareTo(Bigram o) {
            if (score > o.score) return -1;
            if (score < o.score) return 1;
            return Integer.compare(left, o.left);
        }
    }

    /** The symbols left after merging one escaped fragment, before byte fallback. */
    List<String> mergedSymbols(String escapedFragment) {
        if (escapedFragment.isEmpty()) return new ArrayList<>();
        return new Session(escapedFragment).merge().symbols();
    }
}
