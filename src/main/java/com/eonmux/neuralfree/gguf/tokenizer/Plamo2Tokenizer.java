package com.eonmux.neuralfree.gguf.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * PLaMo-2 tokenizer ({@code tokenizer.ggml.model = "plamo2"}), transcribed from llama.cpp
 * {@code llm_tokenizer_plamo2} with {@code pre_segment = false} (src/llama-vocab.cpp:1422-1779) and
 * the PLaMo branch of {@code llama_vocab::impl::tokenize} (src/llama-vocab.cpp:3893-3926).
 *
 * <p>Each raw fragment, minus a leading U+FEFF, is segmented by {@link Lattice}: an Aho-Corasick
 * style suffix table over code points and a right-to-left dynamic program that maximizes the sum of
 * token scores, with unmatched code points falling back to {@code <0xXX>} byte tokens. BOS and EOS
 * follow the vocabulary's add_bos/add_eos flags.</p>
 */
public final class Plamo2Tokenizer implements TokenizerAlgorithm {

    private final TokenizerVocab vocab;
    private final SpecialTokens special;
    private final Lattice lattice;

    public Plamo2Tokenizer(TokenizerVocab vocab, SpecialTokens special) {
        if (vocab == null || special == null) throw new IllegalArgumentException("vocabulary and special tokens are required");
        if (vocab.type() != TokenizerVocab.Type.PLAMO2) {
            throw new IllegalArgumentException("Plamo2Tokenizer requires a PLAMO2 vocabulary, got " + vocab.type()
                    + " (model '" + vocab.model() + "')");
        }
        this.vocab = vocab;
        this.special = special;
        this.lattice = Lattice.build(vocab);
    }

    @Override
    public int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        return encode(vocab, special, text, addSpecial, parseSpecial, this::tokenizeFragment);
    }

    @Override
    public String decode(int[] ids) {
        return TokenizerAlgorithm.detokenize(this, vocab, special, ids);
    }

    @Override
    public byte[] piece(int id) {
        return plamoPiece(vocab, id);
    }

    /** {@code llm_tokenizer_plamo2::encode} without pre-segmentation: drops one leading U+FEFF. */
    private void tokenizeFragment(String text, IdBuffer out) {
        int[] cps = text.codePoints().toArray();
        int from = cps.length > 0 && cps[0] == 0xFEFF ? 1 : 0;
        lattice.encode(cps, from, cps.length, out);
    }

    /** Encodes one raw fragment into the output buffer. */
    interface FragmentEncoder {
        void encode(String text, IdBuffer out);
    }

    /**
     * The PLaMo branch of {@code llama_vocab::impl::tokenize}: BOS when {@code addSpecial} and
     * add_bos, every raw fragment of the {@code parseSpecial} partition through {@code fragments},
     * special tokens passed through, then EOS when {@code addSpecial} and add_eos.
     */
    static int[] encode(TokenizerVocab vocab, SpecialTokens special, String text, boolean addSpecial,
                        boolean parseSpecial, FragmentEncoder fragments) {
        IdBuffer out = new IdBuffer(Math.max(16, text.length() / 2));
        if (addSpecial && vocab.addBos()) out.add(special.requireBos());
        for (SpecialTokens.Fragment f : special.partition(text, parseSpecial)) {
            if (f.isToken()) out.add(f.token());
            else fragments.encode(f.text(), out);
        }
        if (addSpecial && vocab.addEos()) out.add(special.requireEos());
        return out.toArray();
    }

    /**
     * The PLaMo branch of {@code token_to_piece} (llama-vocab.cpp:4066-4086): a byte token spelled
     * {@code <0xXX>} renders its byte, every other token its raw text.
     */
    static byte[] plamoPiece(TokenizerVocab vocab, int id) {
        if (id < 0 || id >= vocab.size()) return new byte[0];
        if (vocab.isByte(id)) {
            int b = hexByte(vocab.text(id));
            if (b >= 0) return new byte[]{(byte) b};
        }
        return TokenizerAlgorithm.utf8(vocab.text(id));
    }

    /** The byte of a {@code <0xXX>} text with ASCII hex digits (either case), or -1. */
    static int hexByte(String text) {
        if (text.length() != 6 || !text.startsWith("<0x") || text.charAt(5) != '>') return -1;
        int hi = asciiHex(text.charAt(3));
        int lo = asciiHex(text.charAt(4));
        return hi < 0 || lo < 0 ? -1 : (hi << 4) | lo;
    }

    private static int asciiHex(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    /**
     * The immutable {@code llm_tokenizer_plamo2} tables built by {@code build} and the dynamic
     * program of {@code encode_cpts}. Shared by the PLaMo-2 and PLaMo-3 tokenizers.
     *
     * <p>Every token text and every proper code-point suffix of one forms the suffix set (tokens
     * carry their score, other suffixes none), plus the empty suffix. Suffixes are ordered by their
     * byte-reversed UTF-8 encoding. Each suffix owns a block of table rows: its prefixes that are in
     * the set, longest first, then a sentinel row for an unknown single code point. A suffix's id is
     * the index of its first row, and {@code toSuffixId} maps (first code point, id of the rest) to
     * the suffix id, which lets the right-to-left scan extend the current suffix one code point at
     * a time.</p>
     */
    static final class Lattice {
        static final int INVALID_SCORE = -20000000;
        static final int UNKNOWN_SCORE = -10000000;
        private static final long INITIAL_COST = 1L << 60;

        private final int[] byteTokens;
        private final PieceCodeMap toSuffixId;
        private final int[] pieceLength;
        private final int[] tokenId;
        private final int[] score;
        private final int[] pieceId;

        private Lattice(int[] byteTokens, PieceCodeMap toSuffixId,
                        int[] pieceLength, int[] tokenId, int[] score, int[] pieceId) {
            this.byteTokens = byteTokens;
            this.toSuffixId = toSuffixId;
            this.pieceLength = pieceLength;
            this.tokenId = tokenId;
            this.score = score;
            this.pieceId = pieceId;
        }

        /** {@code llm_tokenizer_plamo2::build}. */
        static Lattice build(TokenizerVocab vocab) {
            int[] byteTokens = new int[256];
            Arrays.fill(byteTokens, -1);
            Map<String, Float> suffixToScore = new HashMap<>();
            for (int id = 0; id < vocab.size(); id++) {
                String text = vocab.text(id);
                if (vocab.isByte(id)) {
                    int b = hexByte(text);
                    if (b >= 0) byteTokens[b] = id;
                    continue;
                }
                suffixToScore.put(text, vocab.score(id));
                for (int i = text.offsetByCodePoints(0, 1); i < text.length(); i = text.offsetByCodePoints(i, 1)) {
                    suffixToScore.putIfAbsent(text.substring(i), Float.NaN);
                }
            }
            for (int b = 0; b < 256; b++) {
                if (byteTokens[b] < 0) {
                    throw new IllegalArgumentException(String.format(
                            "PLaMo vocabulary has no byte token <0x%02X> (BYTE-typed token required for byte fallback)", b));
                }
            }

            List<String> suffixes = sortedSuffixes(suffixToScore);
            Map<String, Integer> suffixToId = new HashMap<>(suffixes.size() * 2);
            PieceCodeMap toSuffixId = new PieceCodeMap(suffixes.size());
            int numPieces = 0;
            for (String suffix : suffixes) {
                suffixToId.put(suffix, numPieces);
                if (suffix.isEmpty()) {
                    numPieces++;
                    continue;
                }
                int first = suffix.codePointAt(0);
                String remaining = suffix.substring(Character.charCount(first));
                long code = ((long) first << 32) | suffixToId.getOrDefault(remaining, 0);
                toSuffixId.put(code, numPieces);
                int pieces = 1;
                for (String piece : prefixesLongestFirst(suffix)) {
                    if (suffixToScore.containsKey(piece)) pieces++;
                }
                numPieces += pieces;
            }

            int[] pieceLength = new int[numPieces];
            int[] tokenId = new int[numPieces];
            int[] score = new int[numPieces];
            int[] pieceId = new int[numPieces];
            int row = 0;
            for (String suffix : suffixes) {
                int length = suffix.codePointCount(0, suffix.length());
                for (String piece : prefixesLongestFirst(suffix)) {
                    Float s = suffixToScore.get(piece);
                    if (s == null) {
                        length--;
                        continue;
                    }
                    pieceLength[row] = length--;
                    tokenId[row] = vocab.tokenId(piece);
                    score[row] = Float.isFinite(s) ? roundHalfAway((double) s * 1e4) : INVALID_SCORE;
                    pieceId[row] = suffixToId.get(piece);
                    row++;
                }
                pieceLength[row] = 1;
                tokenId[row] = -1;
                score[row] = UNKNOWN_SCORE;
                pieceId[row] = 0;
                row++;
            }
            return new Lattice(byteTokens, toSuffixId, pieceLength, tokenId, score, pieceId);
        }

        /** Suffix strings plus the empty suffix, ordered as {@code std::string} compares their reversed bytes. */
        private static List<String> sortedSuffixes(Map<String, Float> suffixToScore) {
            List<Object[]> keyed = new ArrayList<>(suffixToScore.size() + 1);
            for (String s : suffixToScore.keySet()) keyed.add(new Object[]{s, reversedUtf8(s)});
            keyed.add(new Object[]{"", new byte[0]});
            keyed.sort((a, b) -> Arrays.compareUnsigned((byte[]) a[1], (byte[]) b[1]));
            List<String> out = new ArrayList<>(keyed.size());
            for (Object[] k : keyed) out.add((String) k[0]);
            return out;
        }

        private static byte[] reversedUtf8(String s) {
            byte[] b = s.getBytes(StandardCharsets.UTF_8);
            for (int i = 0, j = b.length - 1; i < j; i++, j--) {
                byte t = b[i];
                b[i] = b[j];
                b[j] = t;
            }
            return b;
        }

        /** The non-empty code-point prefixes of {@code s}, longest first. */
        private static List<String> prefixesLongestFirst(String s) {
            List<String> out = new ArrayList<>();
            for (int end = s.length(); end > 0; end = s.offsetByCodePoints(end, -1)) out.add(s.substring(0, end));
            return out;
        }

        /** {@code std::round} to int: halves round away from zero. */
        static int roundHalfAway(double x) {
            double a = Math.abs(x);
            double r = Math.floor(a);
            if (a - r >= 0.5) r += 1.0;
            return (int) (x < 0 ? -r : r);
        }

        /**
         * {@code encode_cpts} over {@code cps[from, to)}: the right-to-left dynamic program, then the
         * left-to-right path walk with UTF-8 byte fallback for unknown code points.
         */
        void encode(int[] cps, int from, int to, IdBuffer out) {
            int n = to - from;
            if (n <= 0) return;
            long[] cost = new long[n + 1];
            Arrays.fill(cost, INITIAL_COST);
            cost[n] = 0;
            int[] pathLength = new int[n + 1];
            int[] pathToken = new int[n + 1];
            int rows = score.length;

            int suffixId = 0;
            for (int i = n - 1; i >= 0; i--) {
                int c = cps[from + i];
                for (int p = suffixId; p < rows; p++) {
                    suffixId = toSuffixId.get(((long) c << 32) | pieceId[p]);
                    if (suffixId > 0 || score[p] == UNKNOWN_SCORE) break;
                }
                for (int p = suffixId; p < rows; p++) {
                    int s = score[p];
                    if (s > INVALID_SCORE) {
                        int len = pieceLength[p];
                        long candidate = cost[i + len] - s;
                        if (candidate < cost[i]) {
                            cost[i] = candidate;
                            pathLength[i] = len;
                            pathToken[i] = tokenId[p];
                        }
                    }
                    if (s == UNKNOWN_SCORE) break;
                }
            }

            for (int pos = 0; pos < n; pos += pathLength[pos]) {
                if (pathLength[pos] <= 0) {
                    throw new IllegalStateException("PLaMo lattice left position " + pos + " without a path");
                }
                if (pathToken[pos] >= 0) {
                    out.add(pathToken[pos]);
                } else {
                    emitBytes(cps[from + pos], out);
                }
            }
        }

        /** Byte fallback: the UTF-8 encoding of one code point as {@code <0xXX>} tokens. */
        private void emitBytes(int c, IdBuffer out) {
            int s = 1 + (c >= 0x80 ? 1 : 0) + (c >= 0x800 ? 1 : 0) + (c >= 0x10000 ? 1 : 0);
            for (int i = 0; i < s; i++) {
                int b;
                if (s == 1) b = c & 0xFF;
                else b = i == 0 ? (0xF00 >> s) & 0xFF : 0x80;
                out.add(byteTokens[b | ((c >> ((s - i - 1) * 6)) & 0x3F)]);
            }
        }
    }

    /**
     * Open-addressing map from non-negative piece codes ({@code codepoint << 32 | suffix id}) to
     * suffix ids; {@link #get} returns 0, the empty suffix's id, for an absent code, as llama.cpp's
     * lookup does. Written only while the lattice is built.
     */
    private static final class PieceCodeMap {
        private static final long EMPTY = -1L;

        private final long[] keys;
        private final int[] values;
        private final int mask;

        PieceCodeMap(int expected) {
            int capacity = Integer.highestOneBit(Math.max(4, expected) * 2 - 1) << 1;
            this.keys = new long[capacity];
            this.values = new int[capacity];
            this.mask = capacity - 1;
            Arrays.fill(keys, EMPTY);
        }

        void put(long key, int value) {
            if (key < 0) throw new IllegalArgumentException("negative PLaMo piece code " + key);
            int slot = slot(key);
            while (keys[slot] != EMPTY && keys[slot] != key) slot = (slot + 1) & mask;
            keys[slot] = key;
            values[slot] = value;
        }

        int get(long key) {
            int slot = slot(key);
            while (true) {
                long k = keys[slot];
                if (k == key) return values[slot];
                if (k == EMPTY) return 0;
                slot = (slot + 1) & mask;
            }
        }

        private int slot(long key) {
            long h = key * 0x9E3779B97F4A7C15L;
            return (int) (h ^ (h >>> 32)) & mask;
        }
    }
}
