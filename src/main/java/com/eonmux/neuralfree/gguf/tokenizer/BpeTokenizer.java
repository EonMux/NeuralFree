package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;

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
 *
 * <p>The Hugging Face overlays of {@link TokenizerVocab} extend this: fragments are normalized
 * ({@link TokenizerVocab#normalizer()}) and prefixed with a space
 * ({@link TokenizerVocab#bpeAddPrefixSpace()}) before escaping, and a vocabulary with
 * {@link TokenizerVocab#bpeModelOverlay()} merges each word as the Hugging Face BPE model does
 * (subword prefix, end-of-word suffix, unknown token, {@code <0xXX>} byte fallback as
 * {@link TokenizerVocab#bpeByteFallback()} sets it). Without overlays every branch is the
 * llama.cpp one.</p>
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
    private final PreTokenizerType vocabPreType;
    private final Mode mode;
    private final boolean byteFallback;

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
        PreTokenizerRegistry.PreTokenizer named = vocab.namedPreTokenizer();
        this.vocabPreType = named != null && named.type() != null ? named.type() : PreTokenizerType.DEFAULT;
        this.mode = "whitespace".equals(vocab.model()) ? Mode.WHITESPACE
                : "hybriddna".equals(vocab.model()) ? Mode.HYBRIDDNA : Mode.STANDARD;
        Boolean fallback = vocab.bpeByteFallback();
        this.byteFallback = fallback != null ? fallback : !pre.byteEncode();
    }

    /**
     * {@inheritDoc}
     *
     * <p>Text fragments arrive normalized from {@link SpecialTokens#partition(String, boolean, boolean)};
     * only the first fragment begins the input, which a Metaspace {@code first} prepend scheme of
     * the pre-tokenizer or of the vocabulary ({@link TokenizerVocab#metaspacePrependScheme()}) reads.</p>
     */
    @Override
    public int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        IdBuffer out = new IdBuffer(Math.max(16, text.length() / 3));
        if (addSpecial) addAll(special.leadingIds(), out);
        List<SpecialTokens.Fragment> fragments = special.partition(text, parseSpecial, true);
        for (int i = 0; i < fragments.size(); i++) {
            SpecialTokens.Fragment f = fragments.get(i);
            if (f.isToken()) {
                out.add(f.token());
            } else {
                tokenizeFragment(prepareFragment(f.text(), i == 0), i == 0, out);
            }
        }
        if (addSpecial) addAll(special.trailingIds(), out);
        return out.toArray();
    }

    private static void addAll(int[] ids, IdBuffer out) {
        for (int id : ids) out.add(id);
    }

    /**
     * Hugging Face ByteLevel {@code add_prefix_space} (a space before a fragment that does not
     * start with one), then llama.cpp whitespace escaping, then the vocabulary's Metaspace
     * {@code first} scheme: U+2581 before the first fragment of the input unless it starts with
     * one (llama-vocab.cpp:3902-3907).
     */
    private String prepareFragment(String text, boolean inputStart) {
        String raw = vocab.bpeAddPrefixSpace() && !text.startsWith(" ") ? " " + text : text;
        String escaped = vocab.escapeWhitespaces() ? TokenizerVocab.escapeWhitespace(raw) : raw;
        boolean prepend = inputStart && vocab.metaspacePrependScheme() == PreTokenizerSpec.PrependScheme.FIRST
                && !escaped.startsWith("\u2581");
        return prepend ? "\u2581" + escaped : escaped;
    }

    @Override
    public String decode(int[] ids) {
        return TokenizerAlgorithm.detokenize(this, vocab, special, ids);
    }

    /**
     * BPE branch of {@code token_to_piece} (llama-vocab.cpp:4135-4157); NORMAL tokens of a
     * {@link PreTokenizerType#rawNormalPieces()} vocabulary render their raw text.
     */
    @Override
    public byte[] piece(int id) {
        if (id < 0 || id >= vocab.size()) return EMPTY;
        String text = vocab.text(id);
        if (vocab.hasAttr(id, RAW_TEXT_ATTRS)) return TokenizerAlgorithm.utf8(text);
        if (vocab.isNormal(id)) {
            if (vocabPreType.rawNormalPieces()) return TokenizerAlgorithm.utf8(text);
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

    private void tokenizeFragment(String text, boolean inputStart, IdBuffer out) {
        switch (mode) {
            case WHITESPACE: tokenizeWhitespace(text, out); break;
            case HYBRIDDNA: new HybridDna(this).tokenize(text, out); break;
            default: tokenizeBpe(text, inputStart, out); break;
        }
    }

    /** {@link #tokenizeBpe(String, boolean, IdBuffer)} for a text that begins the input. */
    void tokenizeBpe(String text, IdBuffer out) {
        tokenizeBpe(text, true, out);
    }

    /**
     * {@code llm_tokenizer_bpe_session::tokenize}: the vocabulary's pre-tokenizer normalization
     * ({@link PreTokenizerType#normalizeFragment}), split, merge per word, resolve with byte
     * fallback; with {@link TokenizerVocab#bpeModelOverlay()}, each word goes through
     * {@link #tokenizeModelWord} instead.
     */
    private void tokenizeBpe(String text, boolean inputStart, IdBuffer out) {
        boolean model = vocab.bpeModelOverlay();
        for (String word : words(vocabPreType.normalizeFragment(text), inputStart)) {
            if (model) {
                tokenizeModelWord(word, out);
            } else {
                for (String symbol : mergeWord(word)) emitSymbol(symbol, out);
            }
        }
    }

    /** The words of a text that begins the input; see {@link #words(String, boolean)}. */
    List<String> words(String text) {
        return words(text, true);
    }

    /**
     * The pre-tokenized words ({@link PreTokenizerRegistry.PreTokenizer#words}: split and, when
     * configured, byte-encoded), metaspace-split for mmBERT. {@code inputStart} tells whether the
     * text begins the whole input.
     */
    List<String> words(String text, boolean inputStart) {
        List<String> words = pre.words(text, inputStart);
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

    /**
     * Merges one word lowest-rank first and returns the surviving symbols in order. The
     * pre-tokenizer decides only the splitting; {@code ignore_merges} is the vocabulary's
     * resolved flag ({@link TokenizerVocab#ignoreMerges()}).
     */
    List<String> mergeWord(String word) {
        List<String> result = new ArrayList<>();
        boolean whole = (vocab.ignoreMerges() && vocab.tokenId(word) >= 0)
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

    /**
     * Hugging Face {@code BPE::merge_word} and {@code Word::merge_all} for one word: with
     * {@code ignore_merges} a word that is a token is emitted whole; otherwise each character
     * becomes a symbol (with {@link TokenizerVocab#bpeContinuingSubwordPrefix()} unless first,
     * with {@link TokenizerVocab#bpeEndOfWordSuffix()} when last). A symbol that is not a token
     * becomes its {@code <0xXX>} byte tokens when the pre-tokenizer does not byte-encode and the
     * vocabulary has all of them; else the unknown token (consecutive ones fused with
     * {@link TokenizerVocab#bpeUnkFuse()}); else it is dropped. Pairs merge lowest rank first
     * (ties: leftmost) into {@code left + right} minus the right symbol's subword prefix.
     */
    private void tokenizeModelWord(String word, IdBuffer out) {
        if (vocab.ignoreMerges()) {
            int id = vocab.tokenId(word);
            if (id >= 0) {
                out.add(id);
                return;
            }
        }
        List<String> texts = new ArrayList<>();
        IdBuffer ids = new IdBuffer(word.length());
        seedModelSymbols(word, texts, ids);
        int n = texts.size();
        if (n == 0) return;
        String[] text = texts.toArray(new String[0]);
        int[] id = ids.toArray();
        int[] prev = new int[n];
        int[] next = new int[n];
        for (int i = 0; i < n; i++) {
            prev[i] = i - 1;
            next[i] = i + 1 < n ? i + 1 : -1;
        }
        PriorityQueue<Bigram> queue = new PriorityQueue<>();
        for (int i = 1; i < n; i++) addModelBigram(queue, text, i - 1, i);
        while (!queue.isEmpty()) {
            Bigram b = queue.poll();
            if (text[b.left] == null || text[b.right] == null || next[b.left] != b.right
                    || !b.text.equals(modelMerge(text[b.left], text[b.right]))) {
                continue;
            }
            text[b.left] = b.text;
            id[b.left] = vocab.tokenId(b.text);
            text[b.right] = null;
            next[b.left] = next[b.right];
            if (next[b.right] >= 0) prev[next[b.right]] = b.left;
            addModelBigram(queue, text, prev[b.left], b.left);
            addModelBigram(queue, text, b.left, next[b.left]);
        }
        for (int i = 0; i != -1; i = next[i]) out.add(id[i]);
    }

    /**
     * The initial symbols of {@link #tokenizeModelWord}: the id of each symbol and its token text.
     * Unknown-token and byte-fallback symbols carry their token's text, so they merge exactly when
     * a merge names their token, as the id pairs of Hugging Face's merge map do.
     */
    private void seedModelSymbols(String word, List<String> texts, IdBuffer ids) {
        String prefix = vocab.bpeContinuingSubwordPrefix();
        String suffix = vocab.bpeEndOfWordSuffix();
        int unk = special.unk();
        boolean unkPending = false;
        for (int i = 0; i < word.length(); ) {
            int cp = word.codePointAt(i);
            int end = i + Character.charCount(cp);
            String s = (i > 0 ? prefix : "") + word.substring(i, end) + (end == word.length() ? suffix : "");
            i = end;
            int id = vocab.tokenId(s);
            if (id >= 0) {
                if (unkPending) {
                    addSymbol(unk, texts, ids);
                    unkPending = false;
                }
                texts.add(s);
                ids.add(id);
                continue;
            }
            int[] bytes = byteFallback ? byteTokens(s) : null;
            if (bytes != null) {
                for (int b : bytes) addSymbol(b, texts, ids);
                continue;
            }
            if (unk == SpecialTokens.NULL) continue;
            if (unkPending && !vocab.bpeUnkFuse()) addSymbol(unk, texts, ids);
            unkPending = true;
        }
        if (unkPending) addSymbol(unk, texts, ids);
    }

    /** A symbol of a token found by id: an unknown-token or byte-fallback symbol. */
    private void addSymbol(int id, List<String> texts, IdBuffer ids) {
        texts.add(vocab.text(id));
        ids.add(id);
    }

    /** The {@code <0xXX>} token of each UTF-8 byte of {@code s}, or null when any is missing. */
    private int[] byteTokens(String s) {
        byte[] bytes = TokenizerAlgorithm.utf8(s);
        int[] out = new int[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            out[i] = vocab.tokenId(HEX_BYTE_TOKENS[bytes[i] & 0xFF]);
            if (out[i] < 0) return null;
        }
        return out;
    }

    private void addModelBigram(PriorityQueue<Bigram> queue, String[] text, int left, int right) {
        if (left == -1 || right == -1 || text[left] == null || text[right] == null) return;
        int rank = vocab.mergeRank(text[left], text[right]);
        if (rank < 0) return;
        String merged = modelMerge(text[left], text[right]);
        if (vocab.tokenId(merged) < 0) return;
        queue.add(new Bigram(left, right, merged, rank));
    }

    /** Hugging Face's merged token text: {@code left + right} without the right's subword prefix. */
    private String modelMerge(String left, String right) {
        String prefix = vocab.bpeContinuingSubwordPrefix();
        return left + (right.startsWith(prefix) ? right.substring(prefix.length()) : right);
    }

    /**
     * Resolves one merged symbol; a non-token symbol becomes the unknown token for a
     * {@link PreTokenizerType#unknownSymbolsAsUnk()} vocabulary, else the single-byte token of
     * each byte, or nothing when the pre-tokenizer does not byte-encode and byte fallback is off.
     *
     * @throws IllegalStateException when the symbol must become the unknown token and the
     *         vocabulary has none
     */
    private void emitSymbol(String symbol, IdBuffer out) {
        int id = vocab.tokenId(symbol);
        if (id >= 0) {
            out.add(id);
            return;
        }
        if (vocabPreType.unknownSymbolsAsUnk()) {
            if (special.unk() == SpecialTokens.NULL) {
                throw new IllegalStateException("pre-tokenizer '" + vocab.pre() + "' encodes the symbol '" + symbol
                        + "', which is not a token, as the unknown token, but the vocabulary has no "
                        + GgufKeys.TOKENIZER_UNK_ID);
            }
            out.add(special.unk());
            return;
        }
        if (!pre.byteEncode() && !byteFallback) return;
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

    /** The byte GPT-2 {@code bytes_to_unicode} maps to code point {@code cp}, or -1 outside the table. */
    static int gpt2Byte(int cp) {
        return cp >= 0 && cp < UNICODE_TO_BYTE.length ? UNICODE_TO_BYTE[cp] : -1;
    }

    /** {@code llama_decode_text}: GPT-2 byte decoding; code points outside the table are marked. */
    static byte[] decodeText(String text) {
        ByteSink out = new ByteSink(text.length());
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            i += Character.charCount(cp);
            int b = gpt2Byte(cp);
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
