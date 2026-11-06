package com.eonmux.neuralfree.gguf.engine.draft;

import java.util.Arrays;
import java.util.Objects;

/**
 * The prompt-lookup searches of {@code ngram-simple} drafting, shared by {@link NgramDrafter} and the batch engine's
 * drafter. Both read the token history followed by the just-sampled token ({@code inp}, the history's length
 * {@code len}, so {@code inp[len]} is the sampled token):
 * <ul>
 *   <li>One lookup size ({@code minN == maxN}): llama.cpp's {@code common_ngram_simple_draft}
 *       (common/ngram-map.cpp). The n-gram is the last {@code maxN - 1} history tokens plus the sampled token. Its
 *       most recent earlier occurrence proposes the up to {@code sizeM} tokens that followed it, at least
 *       {@code maxN} of them or none at all.</li>
 *   <li>A size range ({@code minN < maxN}): vLLM's {@code NgramProposer}
 *       ({@code _find_longest_matched_ngram_and_propose_tokens}, v1/spec_decode/ngram_proposer.py). The longest
 *       suffix of {@code inp} between {@code minN} and {@code maxN} tokens that occurs earlier wins, and of its
 *       occurrences the earliest. The up to {@code sizeM} tokens that follow it are the proposal.</li>
 * </ul>
 * Stateless.
 */
public final class PromptLookup {

    private static final int[] NONE = new int[0];

    private PromptLookup() {
    }

    /**
     * The proposal for the history {@code inp[0 .. len)} followed by the sampled token {@code inp[len]}.
     *
     * @param minN  the smallest lookup n-gram (vLLM {@code prompt_lookup_min}), at least 1
     * @param maxN  the largest lookup n-gram (vLLM {@code prompt_lookup_max}, llama.cpp {@code size_n}); with
     *              {@code minN == maxN} it is llama.cpp's single size and must be at least 2
     * @param sizeM the most tokens proposed (llama.cpp {@code size_m})
     * @return the proposed tokens, possibly none
     * @throws IllegalArgumentException for sizes out of order or a {@code len} outside {@code inp}
     */
    public static int[] draft(int[] inp, int len, int minN, int maxN, int sizeM) {
        Objects.requireNonNull(inp, "inp");
        if (len < 0 || len >= inp.length) {
            throw new IllegalArgumentException("len " + len + " outside the " + inp.length + " lookup tokens");
        }
        if (minN < 1 || minN > maxN || sizeM < 1) {
            throw new IllegalArgumentException("lookup sizes need 1 <= minN <= maxN and sizeM >= 1, got minN="
                    + minN + ", maxN=" + maxN + ", sizeM=" + sizeM);
        }
        return minN == maxN ? simple(inp, len, maxN, sizeM) : longest(inp, len + 1, minN, maxN, sizeM);
    }

    /** llama.cpp {@code common_ngram_simple_draft} over {@code inp} ({@code cur_len} = {@code len}, sampled last). */
    private static int[] simple(int[] inp, int len, int sizeN, int sizeM) {
        if (sizeN < 2) {
            throw new IllegalArgumentException("a single lookup size needs at least 2 tokens (a history token plus"
                    + " the sampled token), got " + sizeN);
        }
        if (len <= sizeN + sizeM + 1) {
            return NONE;
        }
        int patternStart = len - sizeN + 1;
        // search backwards, skipping the current (rightmost) match: position 0 means no match
        int matchPos = 0;
        for (int j = len - sizeN - 1; j > 0; j--) {
            if (Arrays.equals(inp, j, j + sizeN, inp, patternStart, len + 1)) {
                matchPos = j;
                break;
            }
        }
        if (matchPos == 0) {
            return NONE;
        }
        int copyMax = Math.min(sizeM, len - (matchPos + sizeN));
        if (copyMax < sizeN) {
            return NONE;
        }
        return Arrays.copyOfRange(inp, matchPos + sizeN, matchPos + sizeN + copyMax);
    }

    /**
     * vLLM's longest-match search over {@code tokens[0 .. total)}: a KMP prefix function over the reversed tokens
     * finds the longest suffix (capped at {@code maxN}) that occurs earlier, preferring the earliest occurrence, and
     * proposes the up to {@code k} tokens after it.
     */
    private static int[] longest(int[] tokens, int total, int minN, int maxN, int k) {
        if (total < minN) {
            return NONE;
        }
        // lps[i]: the longest proper prefix of the reversed tokens that ends at reversed index i, for i < maxN
        int[] lps = new int[maxN];
        int longest = 0;
        int position = 0;
        int prevLps = 0;
        int i = 1;
        while (i < total) {
            if (reversed(tokens, total, prevLps) == reversed(tokens, total, i)) {
                prevLps++;
                // ">=": an equally long match further back in the original order replaces the later one
                if (prevLps >= longest) {
                    longest = prevLps;
                    position = i;
                }
                if (i < maxN) {
                    lps[i] = prevLps;
                }
                if (prevLps == maxN) {
                    prevLps = lps[maxN - 1];
                }
                i++;
            } else if (prevLps != 0) {
                prevLps = lps[prevLps - 1];
            } else {
                i++;
            }
        }
        if (longest < minN) {
            return NONE;
        }
        int start = total - 1 - position + longest;
        int count = Math.min(k, total - start);
        return Arrays.copyOfRange(tokens, start, start + count);
    }

    private static int reversed(int[] tokens, int total, int index) {
        return tokens[total - 1 - index];
    }
}
