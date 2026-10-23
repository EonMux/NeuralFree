package com.eonmux.neuralfree.gguf.tokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Merge-rank BPE tokenizer ({@code tokenizer.ggml.model} = gpt2, hybriddna, whitespace, gemma4),
 * transcribed from llama.cpp {@code llm_tokenizer_bpe_session} (src/llama-vocab.cpp:596-816),
 * {@code llm_tokenizer_whitespace_session} (1986-2012), {@code llm_tokenizer_hybriddna_session}
 * (1788-1984) and the BPE branch of {@code llama_vocab::impl::tokenize} (3774-3809).
 *
 * <p>Each raw fragment is optionally whitespace-escaped, split by the pre-tokenizer regex
 * sequence, optionally GPT-2 byte-encoded, and every word is merged lowest rank first (ties:
 * leftmost). Symbols that are not tokens fall back to single-byte tokens.</p>
 */
public final class BpeTokenizer implements TokenizerAlgorithm {

    private static final byte[] EMPTY = new byte[0];
    private static final int RAW_TEXT_ATTRS =
            TokenizerVocab.ATTR_UNKNOWN | TokenizerVocab.ATTR_CONTROL | TokenizerVocab.ATTR_USER_DEFINED;

    /** GPT-2 {@code bytes_to_unicode}: printable bytes map to themselves, the rest to U+0100+. */
    private static final char[] BYTE_TO_UNICODE = new char[256];
    private static final int[] UNICODE_TO_BYTE = new int[0x144];
    private static final String[] HEX_BYTE_TOKENS = new String[256];

    static {
        Arrays.fill(UNICODE_TO_BYTE, -1);
        int nextCode = 256;
        for (int b = 0; b < 256; b++) {
            boolean printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            BYTE_TO_UNICODE[b] = (char) (printable ? b : nextCode++);
            UNICODE_TO_BYTE[BYTE_TO_UNICODE[b]] = b;
            HEX_BYTE_TOKENS[b] = String.format("<0x%02X>", b);
        }
    }

    private enum Mode { STANDARD, WHITESPACE, HYBRIDDNA }

    private final TokenizerVocab vocab;
    private final SpecialTokens special;
    private final PreTokenizerRegistry.PreTokenizer pre;
    private final Mode mode;

    /** Uses the vocabulary's own pre-tokenizer. */
    public BpeTokenizer(TokenizerVocab vocab, SpecialTokens special) {
        this(vocab, special, vocab != null ? vocab.preTokenizer() : null);
    }

    /** Uses an explicit pre-tokenizer in place of the vocabulary's {@code tokenizer.ggml.pre}. */
    public BpeTokenizer(TokenizerVocab vocab, SpecialTokens special, PreTokenizerRegistry.PreTokenizer pre) {
        if (vocab == null || special == null) throw new IllegalArgumentException("vocabulary and special tokens are required");
        if (vocab.type() != TokenizerVocab.Type.BPE) {
            throw new IllegalArgumentException("BpeTokenizer requires a BPE vocabulary, got " + vocab.type()
                    + " (model '" + vocab.model() + "')");
        }
        if (pre == null) throw new IllegalArgumentException("BPE vocabulary '" + vocab.model() + "' has no pre-tokenizer");
        this.vocab = vocab;
        this.special = special;
        this.pre = pre;
        this.mode = "whitespace".equals(vocab.model()) ? Mode.WHITESPACE
                : "hybriddna".equals(vocab.model()) ? Mode.HYBRIDDNA : Mode.STANDARD;
    }

    @Override
    public int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        IdBuffer out = new IdBuffer(Math.max(16, text.length() / 3));
        if (addSpecial && vocab.addBos()) out.add(special.requireBos());
        for (SpecialTokens.Fragment f : special.partition(text, parseSpecial)) {
            if (f.isToken()) {
                out.add(f.token());
            } else {
                String raw = vocab.escapeWhitespaces() ? TokenizerVocab.escapeWhitespace(f.text()) : f.text();
                tokenizeFragment(raw, out);
            }
        }
        if (addSpecial && vocab.addEos()) out.add(special.requireEos());
        return out.toArray();
    }

    @Override
    public String decode(int[] ids) {
        return TokenizerAlgorithm.detokenize(this, vocab, special, ids);
    }

    /** BPE branch of {@code token_to_piece} (llama-vocab.cpp:4028-4049). */
    @Override
    public byte[] piece(int id) {
        if (id < 0 || id >= vocab.size()) return EMPTY;
        String text = vocab.text(id);
        if (vocab.hasAttr(id, RAW_TEXT_ATTRS)) return TokenizerAlgorithm.utf8(text);
        if (vocab.isNormal(id)) {
            return vocab.escapeWhitespaces()
                    ? TokenizerAlgorithm.utf8(TokenizerVocab.unescapeWhitespace(text))
                    : decodeText(text);
        }
        if (vocab.isByte(id)) {
            int b = vocab.byteValue(id);
            return b >= 0 ? new byte[]{(byte) b} : TokenizerAlgorithm.utf8(text);
        }
        return EMPTY;
    }

    private void tokenizeFragment(String text, IdBuffer out) {
        switch (mode) {
            case WHITESPACE: tokenizeWhitespace(text, out); break;
            case HYBRIDDNA: new HybridDna(this).tokenize(text, out); break;
            default: tokenizeBpe(text, out); break;
        }
    }

    /** {@code llm_tokenizer_bpe_session::tokenize}: split, merge per word, resolve with byte fallback. */
    void tokenizeBpe(String text, IdBuffer out) {
        for (String word : words(text)) {
            for (String symbol : mergeWord(word)) emitSymbol(symbol, out);
        }
    }

    /** The pre-tokenized (and, when configured, byte-encoded and metaspace-split) words. */
    List<String> words(String text) {
        List<String> words = pre.split(text);
        if (pre.byteEncode()) {
            for (int i = 0; i < words.size(); i++) words.set(i, byteEncode(words.get(i)));
        }
        return pre.metaspaceWords() ? metaspaceSplit(words) : words;
    }

    /** mmBERT metaspace: each word starts with U+2581 and every U+2581 starts a new word. */
    private static List<String> metaspaceSplit(List<String> words) {
        List<String> out = new ArrayList<>(words.size() * 2);
        for (String w : words) {
            if (isAllNewlines(w)) {
                out.add(w);
                continue;
            }
            String word = w.startsWith("\u2581") ? w : "\u2581" + w;
            int start = 0;
            while (start < word.length()) {
                int end = word.indexOf('\u2581', start + 1);
                if (end < 0) end = word.length();
                out.add(word.substring(start, end));
                start = end;
            }
        }
        return out;
    }

    /** Merges one word lowest-rank first and returns the surviving symbols in order. */
    List<String> mergeWord(String word) {
        List<String> result = new ArrayList<>();
        boolean whole = (pre.ignoreMerges() && vocab.tokenId(word) >= 0)
                || (pre.wholeNewlineWords() && isAllNewlines(word) && vocab.tokenId(word) >= 0);
        if (whole) {
            result.add(word);
            return result;
        }
        int n = word.codePointCount(0, word.length());
        if (n == 0) return result;
        int[] start = new int[n];
        int[] len = new int[n];
        int[] prev = new int[n];
        int[] next = new int[n];
        int offs = 0;
        for (int i = 0; i < n; i++) {
            int cc = Character.charCount(word.codePointAt(offs));
            start[i] = offs;
            len[i] = cc;
            offs += cc;
            prev[i] = i - 1;
            next[i] = offs == word.length() ? -1 : i + 1;
        }
        PriorityQueue<Bigram> queue = new PriorityQueue<>();
        for (int i = 1; i < n; i++) addBigram(queue, word, start, len, i - 1, i);
        while (!queue.isEmpty()) {
            Bigram b = queue.poll();
            if (len[b.left] == 0 || len[b.right] == 0) continue;
            String l = word.substring(start[b.left], start[b.left] + len[b.left]);
            String r = word.substring(start[b.right], start[b.right] + len[b.right]);
            if (!(l + r).equals(b.text)) continue;
            len[b.left] += len[b.right];
            len[b.right] = 0;
            next[b.left] = next[b.right];
            if (next[b.right] >= 0) prev[next[b.right]] = b.left;
            addBigram(queue, word, start, len, prev[b.left], b.left);
            addBigram(queue, word, start, len, b.left, next[b.left]);
        }
        for (int i = 0; i != -1; i = next[i]) {
            if (len[i] > 0) result.add(word.substring(start[i], start[i] + len[i]));
        }
        return result;
    }

    private void addBigram(PriorityQueue<Bigram> queue, String word, int[] start, int[] len, int left, int right) {
        if (left == -1 || right == -1) return;
        String l = word.substring(start[left], start[left] + len[left]);
        String r = word.substring(start[right], start[right] + len[right]);
        int rank = vocab.mergeRank(l, r);
        if (rank < 0) return;
        queue.add(new Bigram(left, right, l + r, rank));
    }

    /** Resolves one merged symbol; a non-token symbol emits the single-byte token of each byte. */
    private void emitSymbol(String symbol, IdBuffer out) {
        int id = vocab.tokenId(symbol);
        if (id >= 0) {
            out.add(id);
            return;
        }
        for (byte b : TokenizerAlgorithm.utf8(symbol)) {
            int ub = b & 0xFF;
            int t = pre.byteEncode()
                    ? (ub < 0x80 ? vocab.tokenId(String.valueOf((char) ub)) : -1)
                    : vocab.tokenId(HEX_BYTE_TOKENS[ub]);
            if (t >= 0) out.add(t);
        }
    }

    /** {@code llm_tokenizer_whitespace_session::tokenize}: whitespace-free segments, optionally lowercased. */
    private void tokenizeWhitespace(String text, IdBuffer out) {
        boolean lowercase = vocab.normalizerLowercase();
        StringBuilder segment = new StringBuilder();
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            i += Character.charCount(cp);
            if (TokenizerVocab.isWhitespace(cp)) {
                flush(segment, out);
            } else {
                segment.appendCodePoint(lowercase ? Character.toLowerCase(cp) : cp);
            }
        }
        flush(segment, out);
    }

    private void flush(StringBuilder segment, IdBuffer out) {
        if (segment.length() > 0) {
            tokenizeBpe(segment.toString(), out);
            segment.setLength(0);
        }
    }

    TokenizerVocab vocab() { return vocab; }

    private static boolean isAllNewlines(String s) {
        for (int i = 0; i < s.length(); i++) if (s.charAt(i) != '\n') return false;
        return true;
    }

    /** GPT-2 byte encoding of a word's UTF-8 bytes ({@code unicode_byte_to_utf8}). */
    static String byteEncode(String word) {
        byte[] bytes = TokenizerAlgorithm.utf8(word);
        char[] chars = new char[bytes.length];
        for (int i = 0; i < bytes.length; i++) chars[i] = BYTE_TO_UNICODE[bytes[i] & 0xFF];
        return new String(chars);
    }

    /** {@code llama_decode_text}: GPT-2 byte decoding; code points outside the table are marked. */
    static byte[] decodeText(String text) {
        ByteSink out = new ByteSink(text.length());
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            i += Character.charCount(cp);
            int b = cp < UNICODE_TO_BYTE.length ? UNICODE_TO_BYTE[cp] : -1;
            if (b >= 0) {
                out.write(b);
            } else {
                StringBuilder marker = new StringBuilder("[UNK_BYTE_0x");
                for (byte u : TokenizerAlgorithm.utf8(new String(Character.toChars(cp)))) {
                    marker.append(String.format("%02x", u & 0xFF));
                }
                marker.append(text).append(']');
                byte[] m = TokenizerAlgorithm.utf8(marker.toString());
                out.write(m, 0, m.length);
            }
        }
        return out.toArray();
    }

    /** {@code llm_bigram_bpe}: lowest rank first, then smallest left index. */
    private static final class Bigram implements Comparable<Bigram> {
        final int left;
        final int right;
        final String text;
        final int rank;

        Bigram(int left, int right, String text, int rank) {
            this.left = left;
            this.right = right;
            this.text = text;
            this.rank = rank;
        }

        @Override
        public int compareTo(Bigram o) {
            return rank != o.rank ? Integer.compare(rank, o.rank) : Integer.compare(left, o.left);
        }
    }

    /**
     * {@code llm_tokenizer_hybriddna_session}: text inside {@code <dna>...</dna>} (and, with
     * {@code dna_auto_tags}, detected DNA runs) becomes 6-mer tokens; everything else is BPE.
     */
    private static final class HybridDna {
        private static final String OPEN_TAG = "<dna>";
        private static final String CLOSE_TAG = "</dna>";
        private static final int K = 6;

        private final BpeTokenizer bpe;
        private final TokenizerVocab vocab;
        private int beginId;
        private int endId;
        private int oovId;
        private char padChar = 'A';

        HybridDna(BpeTokenizer bpe) {
            this.bpe = bpe;
            this.vocab = bpe.vocab();
        }

        void tokenize(String text, IdBuffer out) {
            beginId = vocab.tokenId(OPEN_TAG);
            endId = vocab.tokenId(CLOSE_TAG);
            oovId = vocab.tokenId("<oov>");
            if (beginId < 0 || endId < 0 || oovId < 0) {
                bpe.tokenizeBpe(text, out);
                return;
            }
            padChar = vocab.tokenId("ANNNNN" + TokenizerVocab.DNA_KMER_MARKER) >= 0 ? 'N' : 'A';
            int pos = 0;
            while (pos < text.length()) {
                int start = text.indexOf(OPEN_TAG, pos);
                int end = text.indexOf(CLOSE_TAG, pos);
                if (start < 0 && end < 0) {
                    tokenizeText(text.substring(pos), out);
                    break;
                }
                if (start < 0 || (end >= 0 && end < start)) {
                    tokenizeDna(text.substring(pos, end), false, true, out);
                    pos = end + CLOSE_TAG.length();
                    continue;
                }
                tokenizeText(text.substring(pos, start), out);
                int contentStart = start + OPEN_TAG.length();
                if (end < 0) {
                    tokenizeDna(text.substring(contentStart), true, false, out);
                    break;
                }
                tokenizeDna(text.substring(contentStart, end), true, true, out);
                pos = end + CLOSE_TAG.length();
            }
        }

        private void bpe(String piece, IdBuffer out) {
            if (!piece.isEmpty()) bpe.tokenizeBpe(piece, out);
        }

        /** Auto-tags pure ACGT runs of at least 6 and IUPAC runs of at least 12 with at most 20% ambiguity. */
        private void tokenizeText(String raw, IdBuffer out) {
            if (!vocab.dnaAutoTags()) {
                bpe(raw, out);
                return;
            }
            String text = normalizeSpacedDna(raw);
            int prev = 0;
            int i = 0;
            while (i < text.length()) {
                if (!isIupac(text.charAt(i))) {
                    i++;
                    continue;
                }
                int j = i;
                int ambiguous = 0;
                for (; j < text.length() && isIupac(text.charAt(j)); j++) {
                    if (!isAcgt(toUpper(text.charAt(j)))) ambiguous++;
                }
                int len = j - i;
                if ((ambiguous == 0 && len >= 6) || (ambiguous > 0 && len >= 12 && 5 * ambiguous <= len)) {
                    bpe(text.substring(prev, i), out);
                    tokenizeDna(text.substring(i, j), true, true, out);
                    prev = j;
                }
                i = j;
            }
            bpe(text.substring(prev), out);
        }

        /** {@code _normalize_spaced_dna}: joins at least six blank-separated ACGT chunks totalling twelve bases. */
        private static String normalizeSpacedDna(String text) {
            StringBuilder out = new StringBuilder(text.length());
            int p = 0;
            while (p < text.length()) {
                if (isAcgt(text.charAt(p)) && (p == 0 || !isLetter(text.charAt(p - 1)))) {
                    int matchEnd = -1;
                    int chunk = p;
                    for (int nChunks = 1; ; nChunks++) {
                        int chunkEnd = chunk;
                        while (chunkEnd < text.length() && chunkEnd - chunk < 5 && isAcgt(text.charAt(chunkEnd))) chunkEnd++;
                        if (chunkEnd == chunk) break;
                        if (nChunks >= 6 && (chunkEnd == text.length() || !isLetter(text.charAt(chunkEnd)))) {
                            matchEnd = chunkEnd;
                        }
                        int next = chunkEnd;
                        while (next < text.length() && isBlank(text.charAt(next))) next++;
                        if (next == chunkEnd) break;
                        chunk = next;
                    }
                    if (matchEnd >= 0) {
                        StringBuilder seq = new StringBuilder();
                        for (int i = p; i < matchEnd; i++) if (!isBlank(text.charAt(i))) seq.append(text.charAt(i));
                        out.append(seq.length() >= 12 ? seq : text.substring(p, matchEnd));
                        p = matchEnd;
                        continue;
                    }
                }
                out.append(text.charAt(p++));
            }
            return out.toString();
        }

        /** {@code _process_dna_sequence}: trims, uppercases and emits one token per k bases. */
        private void tokenizeDna(String content, boolean hasStart, boolean hasEnd, IdBuffer out) {
            if (hasStart) out.add(beginId);
            int[] cpts = content.codePoints().toArray();
            int first = 0;
            int last = cpts.length;
            while (first < last && TokenizerVocab.isWhitespace(cpts[first])) first++;
            while (last > first && TokenizerVocab.isWhitespace(cpts[last - 1])) last--;
            int i = first;
            for (; i + K <= last; i += K) out.add(kmerToken(cpts, i, i + K));
            if (i < last) out.add(kmerToken(cpts, i, last));
            if (hasEnd) out.add(endId);
        }

        private int kmerToken(int[] cpts, int from, int to) {
            StringBuilder kmer = new StringBuilder(K + 1);
            for (int i = from; i < to; i++) {
                int c = cpts[i] < 128 ? toUpper((char) cpts[i]) : cpts[i];
                if (c > 127 || !isAcgt((char) c)) return oovId;
                kmer.append((char) c);
            }
            while (kmer.length() < K) kmer.append(padChar);
            int tok = vocab.tokenId(kmer + TokenizerVocab.DNA_KMER_MARKER);
            return tok >= 0 ? tok : oovId;
        }

        private static boolean isLetter(char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
        private static boolean isAcgt(char c) { return c == 'A' || c == 'C' || c == 'G' || c == 'T'; }
        private static boolean isBlank(char c) { return c == ' ' || c == '\t'; }
        private static char toUpper(char c) { return c >= 'a' && c <= 'z' ? (char) (c - 32) : c; }
        private static boolean isIupac(char c) { return "ACGTRYSWKMBDHVN".indexOf(toUpper(c)) >= 0; }
    }
}
