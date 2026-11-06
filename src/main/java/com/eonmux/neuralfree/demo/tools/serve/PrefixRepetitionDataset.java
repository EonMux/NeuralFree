package com.eonmux.neuralfree.demo.tools.serve;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * The {@code prefix_repetition} dataset, vLLM's {@code PrefixRepetitionRandomDataset}: it draws
 * {@code numPrefixes} shared prefixes of {@code prefixLen} allowed tokens, gives every request a
 * fresh suffix of {@code suffixLen} tokens, and shuffles the requests so prefixes and suffixes
 * interleave. The workload stresses the prefix cache: requests sharing a prefix reuse it only when
 * the server's cache keeps it. Prompts are token ids, so lengths are exact.
 */
final class PrefixRepetitionDataset implements ServeBenchDataset {

    /** The dataset's defaults, the reference's. */
    static final int DEFAULT_PREFIX_LEN = 256;
    static final int DEFAULT_SUFFIX_LEN = 256;
    static final int DEFAULT_NUM_PREFIXES = 10;
    static final int DEFAULT_OUTPUT_LEN = 128;

    private final ServeBenchOptions opts;
    private final int vocabSize;
    private final int[] skipIds;

    PrefixRepetitionDataset(ServeBenchOptions opts, int vocabSize, int[] skipIds) {
        this.opts = opts;
        this.vocabSize = vocabSize;
        this.skipIds = skipIds;
    }

    @Override
    public List<BenchSample> sample() {
        int numRequests = opts.numPrompts();
        int numPrefixes = opts.prefixRepetitionNumPrefixes();
        int promptsPerPrefix = numRequests / numPrefixes;
        if (promptsPerPrefix == 0) {
            throw new IllegalArgumentException("num prompts (" + numRequests
                    + ") must be greater than or equal to num prefixes (" + numPrefixes + ")");
        }
        Random random = new Random(opts.seed());
        int[] allowed = allowedTokens();
        int outputLen = opts.outputLen() != null ? opts.outputLen() : opts.prefixRepetitionOutputLen();
        List<BenchSample> out = new ArrayList<>(numRequests);
        for (int p = 0; p < numPrefixes; p++) {
            int[] prefix = uniformTokens(random, allowed, opts.prefixRepetitionPrefixLen());
            for (int i = 0; i < promptsPerPrefix; i++) {
                int[] suffix = uniformTokens(random, allowed, opts.prefixRepetitionSuffixLen());
                int[] tokens = new int[prefix.length + suffix.length];
                System.arraycopy(prefix, 0, tokens, 0, prefix.length);
                System.arraycopy(suffix, 0, tokens, prefix.length, suffix.length);
                out.add(BenchSample.ofTokens(tokens, outputLen, opts.requestIdPrefix() + out.size()));
            }
        }
        if (!opts.disableShuffle()) {
            Collections.shuffle(out, new Random(opts.seed()));
        }
        return out;
    }

    /** The vocabulary the prompts draw from: every id but the skipped special and stop ids. */
    private int[] allowedTokens() {
        List<Integer> allowed = new ArrayList<>(vocabSize);
        for (int id = 0; id < vocabSize; id++) {
            if (!ServeBenchDataset.skipped(id, skipIds)) {
                allowed.add(id);
            }
        }
        if (allowed.isEmpty()) {
            throw new IllegalArgumentException("the vocabulary of " + vocabSize
                    + " tokens has no allowed token left: the tokenizer reports only special ids");
        }
        int[] out = new int[allowed.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = allowed.get(i);
        }
        return out;
    }

    /** {@code n} uniform draws over the allowed ids. */
    private int[] uniformTokens(Random random, int[] allowed, int n) {
        int[] tokens = new int[n];
        for (int i = 0; i < n; i++) {
            tokens[i] = allowed[random.nextInt(allowed.length)];
        }
        return tokens;
    }
}
