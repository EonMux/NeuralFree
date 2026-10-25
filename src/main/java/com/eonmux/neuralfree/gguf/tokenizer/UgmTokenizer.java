package com.eonmux.neuralfree.gguf.tokenizer;

import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.HashMap;
import java.util.Map;

/**
 * SentencePiece Unigram tokenizer ({@code tokenizer.ggml.model = "t5"}: T5, mT5, UMT5),
 * transcribed from llama.cpp {@code llm_tokenizer_ugm} / {@code llm_tokenizer_ugm_session}
 * (src/llama-vocab.cpp:945-1307) and the UGM branch of {@code llama_vocab::impl::tokenize}
 * (src/llama-vocab.cpp:3844-3875).
 *
 * <p>The input is normalized on UTF-8 bytes (user-defined prefixes kept verbatim, the
 * precompiled XCDA charsmap applied, spaces escaped, prepended and merged per the vocabulary
 * flags), then a Viterbi pass over code points picks the segmentation with the highest total
 * score. NORMAL, USER_DEFINED and UNUSED tokens are matchable; user-defined tokens score 0 and an
 * unmatched code point costs {@code min_normal_score - 10}. Runs of unknown tokens collapse.</p>
 */
public final class UgmTokenizer implements TokenizerAlgorithm {

    private static final float UNKNOWN_TOKEN_SCORE_PENALTY = 10.0f;
    private static final byte[] ESCAPED_SPACE = {(byte) 0xE2, (byte) 0x96, (byte) 0x81};
    private static final byte[] PLAIN_SPACE = {' '};
    private static final byte[] REPLACEMENT_CHAR = {(byte) 0xEF, (byte) 0xBF, (byte) 0xBD};

    private final TokenizerVocab vocab;
    private final SpecialTokens special;
    /** Matchable tokens keyed by their UTF-8 bytes as an ISO-8859-1 string. */
    private final Map<String, Integer> tokenMatcher;
    private final int maxTokenBytes;
    private final ByteTrie userDefinedMatcher;
    private final float unknownTokenScore;
    private final PrecompiledCharsmap charsmap;

    public UgmTokenizer(TokenizerVocab vocab, SpecialTokens special) {
        if (vocab == null || special == null) throw new IllegalArgumentException("vocabulary and special tokens are required");
        if (vocab.type() != TokenizerVocab.Type.UGM) {
            throw new IllegalArgumentException("UgmTokenizer requires a UGM vocabulary, got " + vocab.type()
                    + " (model '" + vocab.model() + "')");
        }
        this.vocab = vocab;
        this.special = special;
        byte[] blob = vocab.precompiledCharsmap();
        this.charsmap = blob != null ? new PrecompiledCharsmap(blob) : null;

        Map<String, Integer> matcher = new HashMap<>();
        ByteTrie userDefined = new ByteTrie();
        float minScore = Float.MAX_VALUE;
        int maxBytes = 0;
        for (int id = 0; id < vocab.size(); id++) {
            if (vocab.isNormal(id)) minScore = Math.min(minScore, vocab.score(id));
            if (vocab.hasAttr(id, TokenizerVocab.ATTR_NORMAL | TokenizerVocab.ATTR_USER_DEFINED | TokenizerVocab.ATTR_UNUSED)) {
                byte[] b = TokenizerAlgorithm.utf8(vocab.text(id));
                matcher.put(new String(b, StandardCharsets.ISO_8859_1), id);
                maxBytes = Math.max(maxBytes, b.length);
            }
            if (vocab.isUserDefined(id)) userDefined.insert(TokenizerAlgorithm.utf8(vocab.text(id)));
        }
        this.tokenMatcher = matcher;
        this.maxTokenBytes = maxBytes;
        this.userDefinedMatcher = userDefined;
        this.unknownTokenScore = minScore - UNKNOWN_TOKEN_SCORE_PENALTY;
    }

    @Override
    public int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        IdBuffer out = new IdBuffer(Math.max(16, text.length() / 3));
        if (addSpecial && vocab.addBos()) out.add(special.requireBos());
        for (SpecialTokens.Fragment f : special.partition(text, parseSpecial)) {
            if (f.isToken()) out.add(f.token());
            else tokenizeFragment(f.text(), out);
        }
        if (addSpecial && vocab.addEos()) out.add(special.requireEos());
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

    /** {@code llm_tokenizer_ugm_session::tokenize}: normalization, Viterbi and backtracking. */
    private void tokenizeFragment(String text, IdBuffer out) {
        String input = vocab.normalizerNfc() ? Normalizer.normalize(text, Normalizer.Form.NFC) : text;
        byte[] normalized = normalize(TokenizerAlgorithm.utf8(input));
        int n = normalized.length;
        if (n == 0) return;

        int unk = special.unk();
        int[] tokenAt = new int[n + 1];
        int[] fromOffset = new int[n + 1];
        double[] scoreSum = new double[n + 1];
        java.util.Arrays.fill(tokenAt, unk);
        java.util.Arrays.fill(scoreSum, -Double.MAX_VALUE);
        scoreSum[0] = 0;

        for (int off = 0; off < n; ) {
            int units = Math.min(utf8LeadLength(normalized[off]), n - off);
            boolean singleCodepointFound = false;
            double current = scoreSum[off];
            int limit = Math.min(maxTokenBytes, n - off);
            for (int len = 1; len <= limit; len++) {
                Integer id = tokenMatcher.get(new String(normalized, off, len, StandardCharsets.ISO_8859_1));
                if (id == null) continue;
                if (len == units) singleCodepointFound = true;
                double tokenScore = vocab.isUserDefined(id) ? 0.0 : vocab.score(id);
                double challenger = current + tokenScore;
                if (challenger > scoreSum[off + len]) {
                    tokenAt[off + len] = id;
                    fromOffset[off + len] = off;
                    scoreSum[off + len] = challenger;
                }
            }
            if (!singleCodepointFound) {
                double challenger = current + unknownTokenScore;
                int end = off + units;
                if (challenger > scoreSum[end]) {
                    tokenAt[end] = unk;
                    fromOffset[end] = off;
                    scoreSum[end] = challenger;
                }
            }
            off += units;
        }

        IdBuffer reversed = new IdBuffer(16);
        boolean prevUnknown = false;
        for (int pos = n; ; pos = fromOffset[pos]) {
            int id = tokenAt[pos];
            boolean isUnknown = id == unk;
            if (!(prevUnknown && isUnknown)) {
                if (id == SpecialTokens.NULL) {
                    throw new IllegalStateException("UGM tokenization needs an unknown token but the vocabulary defines none");
                }
                reversed.add(id);
            }
            if (fromOffset[pos] == 0) break;
            prevUnknown = isUnknown;
        }
        for (int i = reversed.length() - 1; i >= 0; i--) out.add(reversed.get(i));
    }

    /** {@code normalize}: whitespace escaping, prefix and merging over normalized prefixes. */
    private byte[] normalize(byte[] input) {
        ByteSink out = new ByteSink(input.length * 3);
        byte[] space = vocab.escapeWhitespaces() ? ESCAPED_SPACE : PLAIN_SPACE;
        boolean prependSpace = vocab.addSpacePrefix();
        boolean mergeSpaces = vocab.removeExtraWhitespaces();
        boolean spacePrepended = false;
        boolean processingNonWs = false;
        int[] res = new int[3];
        for (int off = 0; off < input.length; ) {
            byte[] src = normalizePrefix(input, off, res);
            for (int i = res[0]; i < res[0] + res[1]; i++) {
                byte c = src[i];
                if (c != ' ') {
                    if (!processingNonWs) {
                        processingNonWs = true;
                        if ((prependSpace && !spacePrepended) || mergeSpaces) {
                            out.write(space, 0, space.length);
                            spacePrepended = true;
                        }
                    }
                    out.write(c);
                } else {
                    processingNonWs = false;
                    if (!mergeSpaces) out.write(space, 0, space.length);
                }
            }
            off += res[2];
        }
        return out.toArray();
    }

    /**
     * {@code normalize_prefix}: returns the source array and fills {@code res} with
     * {offset, length, consumed input bytes}.
     */
    private byte[] normalizePrefix(byte[] input, int off, int[] res) {
        int userDefined = userDefinedMatcher.longestPathPrefix(input, off);
        if (userDefined > 0) {
            res[0] = off;
            res[1] = userDefined;
            res[2] = userDefined;
            return input;
        }
        if (charsmap != null) {
            byte[] repl = charsmap.replace(input, off, res);
            if (repl != null) return repl;
        }
        int len = codepointLength(input, off);
        if (len > 0) {
            res[0] = off;
            res[1] = len;
            res[2] = len;
            return input;
        }
        res[0] = 0;
        res[1] = REPLACEMENT_CHAR.length;
        res[2] = 1;
        return REPLACEMENT_CHAR;
    }

    /** {@code unicode_len_utf8}: sequence length implied by a lead byte. */
    private static int utf8LeadLength(byte lead) {
        int hi = (lead & 0xFF) >> 4;
        return hi < 12 ? 1 : hi < 14 ? 2 : hi == 14 ? 3 : 4;
    }

    /** {@code unicode_cpt_from_utf8} validation: the sequence length, or 0 when invalid. */
    private static int codepointLength(byte[] s, int off) {
        int b0 = s[off] & 0xFF;
        if ((b0 & 0x80) == 0) return 1;
        if ((b0 & 0x40) == 0) return 0;
        if ((b0 & 0x20) == 0) {
            if (off + 1 >= s.length || !cont(s[off + 1])) return 0;
            int cp = ((b0 & 0x1F) << 6) | (s[off + 1] & 0x3F);
            return cp < 0x80 ? 0 : 2;
        }
        if ((b0 & 0x10) == 0) {
            if (off + 2 >= s.length || !cont(s[off + 1]) || !cont(s[off + 2])) return 0;
            int cp = ((b0 & 0x0F) << 12) | ((s[off + 1] & 0x3F) << 6) | (s[off + 2] & 0x3F);
            return cp < 0x800 ? 0 : 3;
        }
        if ((b0 & 0x08) == 0) {
            if (off + 3 >= s.length || !cont(s[off + 1]) || !cont(s[off + 2]) || !cont(s[off + 3])) return 0;
            int cp = ((b0 & 0x07) << 18) | ((s[off + 1] & 0x3F) << 12) | ((s[off + 2] & 0x3F) << 6) | (s[off + 3] & 0x3F);
            return cp < 0x10000 || cp > 0x10FFFF ? 0 : 4;
        }
        return 0;
    }

    private static boolean cont(byte b) {
        return (b & 0xC0) == 0x80;
    }

    /** {@code naive_trie} over UTF-8 bytes; {@link #longestPathPrefix} follows the trie path only. */
    private static final class ByteTrie {
        private final Map<Integer, ByteTrie> children = new HashMap<>();

        void insert(byte[] key) {
            ByteTrie node = this;
            for (byte b : key) node = node.children.computeIfAbsent(b & 0xFF, k -> new ByteTrie());
        }

        /** {@code get_longest_prefix}: the depth reached along the trie, whether or not a key ends there. */
        int longestPathPrefix(byte[] s, int off) {
            ByteTrie node = this;
            int depth = 0;
            while (off + depth < s.length) {
                ByteTrie child = node.children.get(s[off + depth] & 0xFF);
                if (child == null) break;
                node = child;
                depth++;
            }
            return depth;
        }
    }
}
