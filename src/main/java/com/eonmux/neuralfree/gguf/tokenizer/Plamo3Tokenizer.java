package com.eonmux.neuralfree.gguf.tokenizer;

/**
 * PLaMo-3 tokenizer ({@code tokenizer.ggml.model = "plamo3"}), transcribed from llama.cpp
 * {@code llm_tokenizer_plamo2} with {@code pre_segment = true} (src/llama-vocab.cpp:1422-1779, selected at
 * src/llama-vocab.cpp:2358 and 3530-3531): each raw fragment is cut into segments, and each segment
 * is encoded by the PLaMo dynamic program ({@link Plamo2Tokenizer.Lattice}).
 *
 * <p>Cuts fall around {@code <|plamo:NAME|>} markers (NAME at most 64 code points without '|',
 * whitespace or U+001C..U+001F), around runs of four or more identical code points other than '\n', and around runs
 * of two or more spaces; a run never crosses a marker cut. U+EE00 ends a segment and is not
 * emitted. A leading U+FEFF is kept. BOS and EOS follow the vocabulary's add_bos/add_eos flags.</p>
 */
public final class Plamo3Tokenizer implements TokenizerAlgorithm {

    private static final int[] MARKER_PREFIX = {'<', '|', 'p', 'l', 'a', 'm', 'o', ':'};
    private static final int MAX_MARKER_BODY = 64;
    private static final int SEGMENT_BOUNDARY = 0xEE00;

    private final TokenizerVocab vocab;
    private final SpecialTokens special;
    private final Plamo2Tokenizer.Lattice lattice;

    public Plamo3Tokenizer(TokenizerVocab vocab, SpecialTokens special) {
        if (vocab == null || special == null) throw new IllegalArgumentException("vocabulary and special tokens are required");
        if (vocab.type() != TokenizerVocab.Type.PLAMO3) {
            throw new IllegalArgumentException("Plamo3Tokenizer requires a PLAMO3 vocabulary, got " + vocab.type()
                    + " (model '" + vocab.model() + "')");
        }
        this.vocab = vocab;
        this.special = special;
        this.lattice = Plamo2Tokenizer.Lattice.build(vocab);
    }

    @Override
    public int[] encode(String text, boolean addSpecial, boolean parseSpecial) {
        return Plamo2Tokenizer.encode(vocab, special, text, addSpecial, parseSpecial, this::tokenizeFragment);
    }

    @Override
    public String decode(int[] ids) {
        return TokenizerAlgorithm.detokenize(this, vocab, special, ids);
    }

    @Override
    public byte[] piece(int id) {
        return Plamo2Tokenizer.plamoPiece(vocab, id);
    }

    /** {@code llm_tokenizer_plamo2::encode} with pre-segmentation. */
    private void tokenizeFragment(String text, IdBuffer out) {
        int[] cps = text.codePoints().toArray();
        boolean[] cut = segmentCuts(cps);
        int n = cps.length;
        int segStart = 0;
        for (int segEnd = 0; segEnd <= n; segEnd++) {
            boolean boundary = segEnd < n && cps[segEnd] == SEGMENT_BOUNDARY;
            if (segEnd == n || cut[segEnd] || boundary) {
                if (segStart < segEnd) lattice.encode(cps, segStart, segEnd, out);
                segStart = segEnd + (boundary ? 1 : 0);
            }
        }
    }

    /** Pass 1 (marker cuts) and pass 2 (run cuts); {@code cut[i]} starts a new segment at {@code i}. */
    static boolean[] segmentCuts(int[] cps) {
        int n = cps.length;
        boolean[] cut = new boolean[n + 1];
        markMarkers(cps, cut);
        markRuns(cps, cut);
        return cut;
    }

    private static void markMarkers(int[] cps, boolean[] cut) {
        int n = cps.length;
        int prefixLen = MARKER_PREFIX.length;
        int i = 0;
        while (i + prefixLen <= n) {
            if (!startsWithPrefix(cps, i)) {
                i++;
                continue;
            }
            int bodyStart = i + prefixLen;
            int j = bodyStart;
            while (j < n && j - bodyStart < MAX_MARKER_BODY && cps[j] != '|'
                    && (cps[j] < 0x1C || cps[j] > 0x1F) && !TokenizerVocab.isWhitespace(cps[j])) {
                j++;
            }
            if (j + 1 < n && cps[j] == '|' && cps[j + 1] == '>') {
                cut[i] = true;
                cut[j + 2] = true;
                i = j + 2;
            } else {
                i++;
            }
        }
    }

    private static boolean startsWithPrefix(int[] cps, int at) {
        for (int k = 0; k < MARKER_PREFIX.length; k++) {
            if (cps[at + k] != MARKER_PREFIX[k]) return false;
        }
        return true;
    }

    private static void markRuns(int[] cps, boolean[] cut) {
        int n = cps.length;
        int i = 0;
        while (i < n) {
            int c = cps[i];
            int run = 1;
            while (i + run < n && cps[i + run] == c && !cut[i + run]) run++;
            boolean repeatedChars = c != '\n' && run >= 4;
            boolean spaces = c == ' ' && run >= 2;
            if (repeatedChars || spaces) {
                cut[i] = true;
                cut[i + run] = true;
            }
            i += run;
        }
    }
}
