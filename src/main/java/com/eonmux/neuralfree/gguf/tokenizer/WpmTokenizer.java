package com.eonmux.neuralfree.gguf.tokenizer;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;

/**
 * WordPiece tokenizer ({@code tokenizer.ggml.model = "bert"}), transcribed from llama.cpp
 * {@code llm_tokenizer_wpm_session} (src/llama-vocab.cpp:822-939) and the WPM branch of
 * {@code llama_vocab::impl::tokenize} (src/llama-vocab.cpp:3817-3843): BOS, NFD/lowercase/accent-stripping
 * pre-processing, greedy longest-match word pieces with the unknown token per unmatched word, then SEP.
 *
 * <p>Pre-processing splits on whitespace, drops NUL, U+FFFD and control/format code points, drops
 * accent marks when {@code strip_accents} is on, lowercases when {@code lowercase} is on, and turns
 * every punctuation character, ASCII symbol and CJK ideograph into a word of its own. Each word gets
 * the U+2581 phantom-space prefix and is matched greedily, longest token first; a word with any
 * unmatched position becomes a single unknown token. Unicode categories and case mappings come
 * from the JDK's Unicode data, which llama.cpp's generated tables mirror.</p>
 */
public final class WpmTokenizer implements TokenizerAlgorithm {

    private static final String PHANTOM_SPACE = "▁";
    /** First code point whose NFD differs from itself (llama.cpp {@code unicode_ranges_nfd}). */
    private static final int FIRST_DECOMPOSABLE = 0xC0;

    private final TokenizerVocab vocab;
    private final SpecialTokens special;
    /** llama.cpp {@code max_token_len}: the longest token text in UTF-8 bytes. */
    private final int maxTokenLen;

    public WpmTokenizer(TokenizerVocab vocab, SpecialTokens special) {
        if (vocab == null || special == null) throw new IllegalArgumentException("vocabulary and special tokens are required");
        if (vocab.type() != TokenizerVocab.Type.WPM) {
            throw new IllegalArgumentException("WpmTokenizer requires a WPM vocabulary, got " + vocab.type()
                    + " (model '" + vocab.model() + "')");
        }
        this.vocab = vocab;
        this.special = special;
        int max = 0;
        for (int id = 0; id < vocab.size(); id++) max = Math.max(max, utf8Length(vocab.text(id)));
        this.maxTokenLen = max;
    }

    /**
     * BOS (when {@code addSpecial}), the WordPiece tokens of each raw fragment with special tokens
     * passed through, then SEP (when {@code addSpecial}). As in llama.cpp, the WPM branch applies
     * BOS and SEP unconditionally for {@code addSpecial}, independent of the add_bos/add_sep flags.
     */
    @Override
    public int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        IdBuffer out = new IdBuffer(Math.max(16, text.length() / 3));
        if (addSpecial) out.add(special.requireBos());
        for (SpecialTokens.Fragment f : special.partition(text, parseSpecial)) {
            if (f.isToken()) out.add(f.token());
            else tokenizeFragment(f.text(), out);
        }
        if (addSpecial) out.add(special.requireSep());
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

    /** {@code llm_tokenizer_wpm_session::tokenize} for one raw fragment. */
    private void tokenizeFragment(String text, IdBuffer out) {
        for (String word : preprocess(text, vocab.normalizerLowercase(), vocab.normalizerStripAccents())) {
            if (word.isEmpty()) continue;
            tokenizeWord(PHANTOM_SPACE + word, out);
        }
    }

    /**
     * Greedy longest-match over one prefixed word. Positions advance in UTF-8 bytes as in llama.cpp;
     * a vocabulary text is valid UTF-8, so only candidate ends on code point boundaries can match and
     * the scan visits exactly those, longest first, within {@code max_token_len + 1} bytes. The
     * byte limit only grows as the start advances, so the furthest admissible end is tracked
     * incrementally.
     */
    private void tokenizeWord(String word, IdBuffer out) {
        int[] charAt = new int[word.length() + 1];
        int[] byteAt = new int[word.length() + 1];
        int boundaries = 0;
        int bytes = 0;
        for (int c = 0; c < word.length(); ) {
            charAt[boundaries] = c;
            byteAt[boundaries++] = bytes;
            int cp = word.codePointAt(c);
            bytes += utf8Length(cp);
            c += Character.charCount(cp);
        }
        charAt[boundaries] = word.length();
        byteAt[boundaries] = bytes;

        int[] matched = new int[boundaries];
        int count = 0;
        boolean complete = true;
        int longestEnd = 0;
        for (int i = 0; i < boundaries; ) {
            int limit = byteAt[i] + maxTokenLen + 1;
            longestEnd = Math.max(longestEnd, i);
            while (longestEnd < boundaries && byteAt[longestEnd + 1] <= limit) longestEnd++;
            int found = -1;
            int foundEnd = -1;
            for (int j = longestEnd; j > i; j--) {
                int id = vocab.tokenId(word.substring(charAt[i], charAt[j]));
                if (id >= 0) {
                    found = id;
                    foundEnd = j;
                    break;
                }
            }
            if (found < 0) {
                complete = false;
                break;
            }
            matched[count++] = found;
            i = foundEnd;
        }
        if (complete && count > 0) {
            for (int k = 0; k < count; k++) out.add(matched[k]);
        } else {
            out.add(requireUnk());
        }
    }

    private int requireUnk() {
        int unk = special.unk();
        if (unk == SpecialTokens.NULL) {
            throw new IllegalStateException("WPM tokenization of an unmatched word needs an unknown token,"
                    + " but the vocabulary defines none (tokenizer.ggml.unknown_token_id)");
        }
        return unk;
    }

    /**
     * {@code llm_tokenizer_wpm_session::preprocess}: normalizes and splits text into the non-empty
     * words that llama.cpp tokenizes (it skips the empty ones its splitting leaves behind).
     */
    static List<String> preprocess(String text, boolean lowercase, boolean stripAccents) {
        List<String> words = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (int c = 0; c < text.length(); ) {
            int cpt = text.codePointAt(c);
            c += Character.charCount(cpt);
            if (stripAccents) cpt = nfdBase(cpt);

            if (TokenizerVocab.isWhitespace(cpt)) {
                finishWord(words, current);
                continue;
            }
            int type = Character.getType(cpt);
            if (cpt == 0 || cpt == 0xFFFD || isControl(type)) continue;
            if (stripAccents && isAccentMark(type)) continue;

            int mapped = lowercase ? Character.toLowerCase(cpt) : cpt;
            if (isPunctuation(type) || (cpt < 0x7F && isSymbol(type)) || isChineseChar(cpt)) {
                finishWord(words, current);
                words.add(new String(Character.toChars(mapped)));
            } else {
                current.appendCodePoint(mapped);
            }
        }
        finishWord(words, current);
        return words;
    }

    private static void finishWord(List<String> words, StringBuilder current) {
        if (current.length() > 0) {
            words.add(current.toString());
            current.setLength(0);
        }
    }

    /**
     * {@code unicode_cpts_normalize_nfd}: llama.cpp maps each code point to the first code point
     * of its canonical decomposition (scripts/gen-unicode-data.py:158-161).
     */
    static int nfdBase(int cpt) {
        if (cpt < FIRST_DECOMPOSABLE || (cpt >= Character.MIN_SURROGATE && cpt <= Character.MAX_SURROGATE)) return cpt;
        String s = new String(Character.toChars(cpt));
        if (Normalizer.isNormalized(s, Normalizer.Form.NFD)) return cpt;
        return Normalizer.normalize(s, Normalizer.Form.NFD).codePointAt(0);
    }

    /** {@code \p{C}} without {@code Cn}: Cc, Cf, Co and Cs (gen-unicode-data.py:102-107). */
    private static boolean isControl(int type) {
        return type == Character.CONTROL || type == Character.FORMAT
                || type == Character.PRIVATE_USE || type == Character.SURROGATE;
    }

    /** {@code \p{M}}: Mn, Mc and Me. */
    private static boolean isAccentMark(int type) {
        return type == Character.NON_SPACING_MARK || type == Character.ENCLOSING_MARK
                || type == Character.COMBINING_SPACING_MARK;
    }

    /** {@code \p{P}}: Pc, Pd, Ps, Pe, Pi, Pf and Po. */
    private static boolean isPunctuation(int type) {
        switch (type) {
            case Character.CONNECTOR_PUNCTUATION:
            case Character.DASH_PUNCTUATION:
            case Character.START_PUNCTUATION:
            case Character.END_PUNCTUATION:
            case Character.INITIAL_QUOTE_PUNCTUATION:
            case Character.FINAL_QUOTE_PUNCTUATION:
            case Character.OTHER_PUNCTUATION:
                return true;
            default:
                return false;
        }
    }

    /** {@code \p{S}}: Sm, Sc, Sk and So. */
    private static boolean isSymbol(int type) {
        return type == Character.MATH_SYMBOL || type == Character.CURRENCY_SYMBOL
                || type == Character.MODIFIER_SYMBOL || type == Character.OTHER_SYMBOL;
    }

    /** {@code llm_tokenizer_wpm_session::is_chinese_char}, including llama.cpp's 0x2B920 lower bound. */
    static boolean isChineseChar(int cpt) {
        return (cpt >= 0x04E00 && cpt <= 0x09FFF)
                || (cpt >= 0x03400 && cpt <= 0x04DBF)
                || (cpt >= 0x20000 && cpt <= 0x2A6DF)
                || (cpt >= 0x2A700 && cpt <= 0x2B73F)
                || (cpt >= 0x2B740 && cpt <= 0x2B81F)
                || (cpt >= 0x2B920 && cpt <= 0x2CEAF)
                || (cpt >= 0x0F900 && cpt <= 0x0FAFF)
                || (cpt >= 0x2F800 && cpt <= 0x2FA1F);
    }

    private static int utf8Length(int cp) {
        return cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
    }

    private static int utf8Length(String s) {
        int bytes = 0;
        for (int i = 0; i < s.length(); ) {
            int cp = s.codePointAt(i);
            bytes += utf8Length(cp);
            i += Character.charCount(cp);
        }
        return bytes;
    }
}
