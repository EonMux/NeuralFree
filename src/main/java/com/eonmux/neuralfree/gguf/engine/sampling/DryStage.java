package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler.Candidates;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * llama.cpp's DRY sampler (llama_sampler_dry_apply, llama.cpp src/llama-sampler.cpp:3392): over the last
 * {@code penaltyLastN} accepted tokens, every token that would extend a repeated suffix that already appears earlier
 * in the window is penalized by {@code multiplier * base^(repeatLength - allowedLength)} - a port of the KoboldCpp
 * Z-algorithm DRY (hence the reverse Z-box scan). Restart sequences cap the repeat length: a window position that ends
 * one (a "head" token followed by its "tails") makes everything before it a fresh start. Finally the candidate whose
 * token heads a single-token sequence breaker is never penalized, and the set is marked unsorted.
 *
 * <p>Construction is llama.cpp's: the string breakers go through
 * {@code get_overlapping_token_sequences} (llama-sampler.cpp:3336, capped at 40 breaker characters and 20 tail
 * tokens), so a head token feeds both the full-in-token breakers (empty tail) and the suffix tokenizations; or the
 * breakers arrive already tokenized as head + tail id lists like llama.cpp's {@code init_dry_testing} wrapper.
 * {@code multiplier == 0}, {@code base < 1} or a clamped {@code penaltyLastN} of 0 disable the stage with no history,
 * as llama.cpp's empty sampler does; {@code penaltyLastN} below 0 clamps to 0 exacty as llama.cpp's
 * {@code max(dry_penalty_last_n, 0)} does.
 */
public final class DryStage extends SamplerStage {

    /** Breaker strings longer than this are truncated (llama.cpp's MAX_CHAR_LEN). */
    static final int MAX_CHAR_LEN = 40;
    /** Tail tokenizations longer than this are truncated (llama.cpp's MAX_SEQ_LEN). */
    static final int MAX_SEQ_LEN = 20;

    private final float multiplier;
    private final float base;
    private final int allowedLength;
    private final int penaltyLastN;
    /** head token -> its tail suffixes (any of which completes a restart sequence at that head). */
    private final Map<Integer, List<List<Integer>>> breakers;
    /** The same breakers as one array per restart sequence: the head token first, its tail ids after it. */
    private final int[][] breakerEntries;

    private int[] ring;
    private int ringHead;
    private int ringSize;

    /** Breakers tokenized with the model's vocabulary, as llama.cpp's chain construction does it. */
    public DryStage(float multiplier, float base, int allowedLength, int penaltyLastN, List<String> sequenceBreakers,
                    SamplerVocab vocab) {
        this(multiplier, base, allowedLength, penaltyLastN,
                processedBreakers(multiplier, base, penaltyLastN, sequenceBreakers, vocab));
    }

    /**
     * Already-tokenized breakers (one per restart sequence: token 0 is the head, the rest the tail), llama.cpp's
     * {@code llama_sampler_init_dry_testing} input for tests and embedders.
     */
    public DryStage(float multiplier, float base, int allowedLength, int penaltyLastN, List<int[]> tokenBreakers) {
        Objects.requireNonNull(tokenBreakers, "tokenBreakers");
        int lastN = Math.max(penaltyLastN, 0);
        this.multiplier = multiplier;
        this.base = base;
        this.allowedLength = allowedLength;
        this.penaltyLastN = lastN;
        boolean enabled = multiplier != 0.0f && base >= 1.0f && lastN != 0;
        Map<Integer, List<List<Integer>>> processed = new HashMap<>();
        if (enabled && !tokenBreakers.isEmpty()) {
            for (int[] breaker : tokenBreakers) {
                if (breaker == null || breaker.length == 0) {
                    continue;
                }
                List<List<Integer>> tails = processed.computeIfAbsent(breaker[0], k -> new ArrayList<>());
                List<Integer> tail = new ArrayList<>(breaker.length - 1);
                for (int j = 1; j < breaker.length; j++) {
                    tail.add(breaker[j]);
                }
                boolean duplicate = false;
                for (List<Integer> existing : tails) {
                    if (existing.equals(tail)) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    tails.add(tail);
                }
            }
        }
        this.breakers = processed;
        this.breakerEntries = flatten(processed);
        this.ring = enabled ? new int[lastN] : new int[0];
    }

    /** The head-and-tail breaker map as one id array per restart sequence. */
    private static int[][] flatten(Map<Integer, List<List<Integer>>> breakers) {
        List<int[]> entries = new ArrayList<>();
        for (Map.Entry<Integer, List<List<Integer>>> e : breakers.entrySet()) {
            for (List<Integer> tail : e.getValue()) {
                int[] entry = new int[1 + tail.size()];
                entry[0] = e.getKey();
                for (int i = 0; i < tail.size(); i++) {
                    entry[1 + i] = tail.get(i);
                }
                entries.add(entry);
            }
        }
        return entries.toArray(new int[0][]);
    }

    /** llama.cpp's breaker processing: every breaker string goes through the token scans once. */
    private static List<int[]> processedBreakers(float multiplier, float base, int penaltyLastN,
                                                 List<String> sequenceBreakers, SamplerVocab vocab) {
        Objects.requireNonNull(sequenceBreakers, "sequenceBreakers");
        List<int[]> processed = new ArrayList<>();
        boolean enabled = multiplier != 0.0f && base >= 1.0f && Math.max(penaltyLastN, 0) != 0;
        if (enabled && !sequenceBreakers.isEmpty()) {
            if (vocab == null) {
                throw new IllegalArgumentException(
                        "the DRY stage needs the model's vocabulary (SamplerVocab) to tokenize its sequence breakers");
            }
            for (String breaker : sequenceBreakers) {
                if (breaker == null || breaker.isEmpty()) {
                    continue;
                }
                String sequenceBreak = breaker.length() > MAX_CHAR_LEN ? breaker.substring(0, MAX_CHAR_LEN) : breaker;
                for (int[] seq : getOverlappingTokenSequences(vocab, sequenceBreak)) {
                    processed.add(seq);
                }
            }
        }
        return processed;
    }

    /**
     * llama.cpp {@code get_overlapping_token_sequences} (src/llama-sampler.cpp:3336): every vocabulary token whose
     * piece contains the breaker string is a head with an empty tail (the breaker ends inside the token), and every
     * token whose piece starts a partial match at the end of the piece is a head with the rest of the breaker,
     * tokenized, as its tail (at most {@link #MAX_SEQ_LEN} tokens). One entry per distinct tail per token.
     */
    static List<int[]> getOverlappingTokenSequences(SamplerVocab vocab, String breaker) {
        List<int[]> out = new ArrayList<>();
        for (int id = 0; id < vocab.vocabSize(); id++) {
            String word = vocab.pieceText(id);
            if (word.contains(breaker)) {
                out.add(new int[]{id});
                continue;
            }
            if (breaker.isEmpty()) {
                continue;
            }
            char head = breaker.charAt(0);
            List<List<Integer>> tails = null;
            int pos = -1;
            while ((pos = word.indexOf(head, pos + 1)) != -1) {
                boolean match = true;
                int i;
                for (i = 1; i < breaker.length() && i + pos < word.length(); i++) {
                    if (word.charAt(pos + i) != breaker.charAt(i)) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    String rest = breaker.substring(Math.min(i, breaker.length()));
                    List<Integer> tail = new ArrayList<>();
                    if (!rest.isEmpty()) {
                        for (int t : vocab.tokenize(rest)) {
                            tail.add(t);
                        }
                        if (tail.size() > MAX_SEQ_LEN) {
                            tail = tail.subList(0, MAX_SEQ_LEN);
                        }
                    }
                    if (tails == null) {
                        tails = new ArrayList<>();
                    }
                    if (!containsList(tails, tail)) {
                        tails.add(tail);
                    }
                }
            }
            if (tails != null) {
                for (List<Integer> tail : tails) {
                    int[] entry = new int[1 + tail.size()];
                    entry[0] = id;
                    for (int t = 0; t < tail.size(); t++) {
                        entry[1 + t] = tail.get(t);
                    }
                    out.add(entry);
                }
            }
        }
        return out;
    }

    private static boolean containsList(List<List<Integer>> existing, List<Integer> tail) {
        for (List<Integer> e : existing) {
            if (e.equals(tail)) {
                return true;
            }
        }
        return false;
    }

    /** Whether the stage applies ({@code multiplier != 0}, {@code base >= 1} and a non-zero window). */
    public boolean enabled() {
        return multiplier != 0.0f && base >= 1.0f && penaltyLastN != 0;
    }

    /** DRY's repetition strength. */
    public float multiplier() {
        return multiplier;
    }

    /** DRY's penalty base. */
    public float base() {
        return base;
    }

    /** The repeated suffix length the penalty starts at. */
    public int allowedLength() {
        return allowedLength;
    }

    /** Accepted tokens currently in the window. */
    public int windowSize() {
        return ringSize;
    }

    /** The window's accepted tokens, oldest first (a copy); empty when the stage is off. */
    public int[] windowTokens() {
        if (!enabled()) {
            return new int[0];
        }
        int[] tokens = new int[ringSize];
        for (int k = 0; k < ringSize; k++) {
            tokens[k] = ring[(ringHead + k) % ring.length];
        }
        return tokens;
    }

    /** The restart sequences as the head token followed by its tail ids; each entry is a copy. */
    public int[][] breakerEntries() {
        int[][] copy = new int[breakerEntries.length][];
        for (int i = 0; i < breakerEntries.length; i++) {
            copy[i] = breakerEntries[i].clone();
        }
        return copy;
    }

    /** Number of restart sequences the stage carries. */
    public int breakerCount() {
        return breakerEntries.length;
    }

    /** Total tail token ids across the restart sequences. */
    public int breakerTailCount() {
        int tails = 0;
        for (int[] entry : breakerEntries) {
            tails += entry.length - 1;
        }
        return tails;
    }

    @Override
    public StageKind kind() {
        return StageKind.DRY;
    }

    @Override
    public boolean modifies() {
        return enabled();
    }

    @Override
    public void apply(Candidates candidates) {
        if (!enabled()) {
            return;
        }
        int lastNRepeat = Math.min(ringSize, penaltyLastN);
        if (lastNRepeat <= allowedLength) {
            return;
        }

        int[] repeat = new int[lastNRepeat];

        // Step 1: restart sequences limit the maximum repetition length: the first position whose head token
        // completes a tail sets rep_limit (llama.cpp llama-sampler.cpp:3408-3464).
        int repLimit = lastNRepeat;
        for (int i = 0; i < lastNRepeat; i++) {
            List<List<Integer>> heads = breakers.get(rat(i));
            if (heads == null) {
                continue;
            }
            int longestMatch = -1;
            for (List<Integer> tail : heads) {
                int seqLen = tail.size();
                if (seqLen > longestMatch && seqLen <= i) {
                    boolean match = true;
                    for (int offset = 0; offset < seqLen; offset++) {
                        if (tail.get(offset) != rat(i - offset - 1)) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        longestMatch = seqLen;
                    }
                }
            }
            if (longestMatch >= 0) {
                repLimit = i - longestMatch;
                break;
            }
        }
        if (repLimit < allowedLength) {
            return;
        }

        // Step 2: the reverse Z-algorithm over the window (llama-sampler.cpp:3469-3531).
        int last = lastNRepeat - 1;
        int rt = 0;
        int lt = 0;
        for (int k = 1; k < lastNRepeat; k++) {
            if (k > rt) {
                int n = 0;
                while (n + k < lastNRepeat && rat(n) == rat(n + k)) {
                    n++;
                }
                repeat[last - k] = Math.min(n, repLimit);
                if (n > 0) {
                    lt = k;
                    rt = k + n - 1;
                }
            } else {
                int p = k - lt;
                int rightPartLen = rt - k + 1;
                if (repeat[last - p] < rightPartLen) {
                    repeat[last - k] = Math.min(repeat[last - p], repLimit);
                } else {
                    int i = rt + 1;
                    while (i < lastNRepeat && rat(i) == rat(i - k)) {
                        i++;
                    }
                    repeat[last - k] = Math.min(i - k, repLimit);
                    lt = k;
                    rt = i - 1;
                }
            }
        }

        // Step 3: the token that extends each repeated suffix, and how far that repetition runs
        // (llama-sampler.cpp:3533-3558); the largest repeat wins per token.
        Map<Integer, Integer> maxTokenRepeat = new HashMap<>();
        for (int i = 0; i < lastNRepeat - 1; i++) {
            int repeatLen = repeat[i];
            if (repeatLen >= allowedLength) {
                int token = rat(lastNRepeat - 2 - i);
                Integer previous = maxTokenRepeat.get(token);
                if (previous == null || previous < repeatLen) {
                    maxTokenRepeat.put(token, repeatLen);
                }
            }
        }

        // Step 4: penalties (llama-sampler.cpp:3560-3594): the exponent clamps so base^repeat cannot overflow float.
        float maxExponentFloat = 0;
        if (base > 1.000001f) {
            maxExponentFloat = 88.7228391f / (float) Math.log(base);
        }
        int maxExponent = (int) maxExponentFloat > 0 ? (int) maxExponentFloat : 0;
        for (int i = 0; i < candidates.size(); i++) {
            Integer repeatLen = maxTokenRepeat.get(candidates.id(i));
            if (repeatLen != null && !isSingleTokenBreaker(candidates.id(i))) {
                int repeatExp = repeatLen - allowedLength;
                if (maxExponent > 0 && repeatExp > maxExponent) {
                    repeatExp = maxExponent;
                }
                candidates.setLogit(i, candidates.logit(i) - multiplier * (float) Math.pow(base, repeatExp));
            }
        }
        candidates.setSorted(false);
    }

    /** True when one of the token's breakers has an empty tail (the breaker ends inside the token itself). */
    private boolean isSingleTokenBreaker(int token) {
        List<List<Integer>> heads = breakers.get(token);
        if (heads == null) {
            return false;
        }
        for (List<Integer> tail : heads) {
            if (tail.isEmpty()) {
                return true;
            }
        }
        return false;
    }

    /** The {@code i}-th most recently accepted token, llama.cpp ring_buffer::rat. */
    private int rat(int i) {
        return ring[(ringHead + ringSize - i - 1 + ring.length) % ring.length];
    }

    @Override
    public void accept(int token) {
        if (!enabled()) {
            return;
        }
        if (ringSize == penaltyLastN) {
            ring[ringHead] = token;
            ringHead = (ringHead + 1) % penaltyLastN;
        } else {
            ring[(ringHead + ringSize) % penaltyLastN] = token;
            ringSize++;
        }
    }

    @Override
    public void reset() {
        ringSize = 0;
        ringHead = 0;
    }

    /** The accepted tokens in the window, oldest first; null when the stage is off. */
    @Override
    public Object saveState() {
        if (!enabled()) {
            return null;
        }
        int[] tokens = new int[ringSize];
        for (int k = 0; k < ringSize; k++) {
            tokens[k] = ring[(ringHead + k) % ring.length];
        }
        return tokens;
    }

    @Override
    public void restoreState(Object state) {
        if (!enabled()) {
            return;
        }
        if (!(state instanceof int[])) {
            throw new IllegalArgumentException("not a DRY stage state: " + state);
        }
        int[] tokens = (int[]) state;
        System.arraycopy(tokens, 0, ring, 0, tokens.length);
        ringHead = 0;
        ringSize = tokens.length;
    }
}
