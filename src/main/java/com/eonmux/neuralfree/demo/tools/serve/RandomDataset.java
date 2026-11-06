package com.eonmux.neuralfree.demo.tools.serve;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The {@code random} dataset, vLLM's {@code RandomDataset} over token-id prompts: every request
 * draws its input and output lengths uniformly from the integer ranges the {@code rangeRatio} spans
 * around {@code randomInputLen} and {@code randomOutputLen} (clamped to at least one output token),
 * a fixed prefix of {@code randomPrefixLen} allowed tokens precedes the body, and the body walks the
 * allowed token ids cyclically from a per-request offset — {@code (offset + index + arange(n)) %
 * allowed} — so the sequence is reproducible and never repeats a full window. The prompt is sent as
 * token ids, so the counted prompt length is exact and no decode/re-encode step runs.
 */
final class RandomDataset implements ServeBenchDataset {

    /** The dataset's defaults, the reference's. */
    static final int DEFAULT_INPUT_LEN = 1024;
    static final int DEFAULT_OUTPUT_LEN = 128;
    static final double DEFAULT_RANGE_RATIO = 0.0;
    static final int DEFAULT_PREFIX_LEN = 0;

    private final ServeBenchOptions opts;
    private final int vocabSize;
    private final int[] skipIds;

    RandomDataset(ServeBenchOptions opts, int vocabSize, int[] skipIds) {
        this.opts = opts;
        this.vocabSize = vocabSize;
        this.skipIds = skipIds;
    }

    @Override
    public List<BenchSample> sample() {
        int numRequests = opts.numPrompts();
        long seed = opts.seed();
        double inputRatio = opts.randomRangeRatio().input();
        double outputRatio = opts.randomRangeRatio().output();
        if (!(inputRatio >= 0.0 && inputRatio < 1.0)) {
            throw new IllegalArgumentException("the input range ratio must be in [0, 1), got " + inputRatio);
        }
        if (!(outputRatio >= 0.0 && outputRatio < 1.0)) {
            throw new IllegalArgumentException("the output range ratio must be in [0, 1), got " + outputRatio);
        }
        int inputLen = opts.inputLen() != null ? opts.inputLen() : opts.randomInputLen();
        int outputLen = opts.outputLen() != null ? opts.outputLen() : opts.randomOutputLen();
        int inputLow = (int) Math.floor(inputLen * (1.0 - inputRatio));
        int inputHigh = (int) Math.ceil(inputLen * (1.0 + inputRatio));
        int outputLow = Math.max((int) Math.floor(outputLen * (1.0 - outputRatio)), 1);
        int outputHigh = Math.max((int) Math.ceil(outputLen * (1.0 + outputRatio)), 1);
        if (inputLow > inputHigh) {
            throw new IllegalArgumentException("invalid input sampling interval: low=" + inputLow
                    + " > high=" + inputHigh);
        }
        if (outputLow > outputHigh) {
            throw new IllegalArgumentException("invalid output sampling interval: low=" + outputLow
                    + " > high=" + outputHigh);
        }
        Random random = new Random(seed);
        int[] allowed = allowedTokens();
        int prefixLen = opts.randomPrefixLen();
        int[] prefix = prefixTokens(random, allowed, prefixLen);
        List<BenchSample> out = new ArrayList<>(numRequests);
        for (int i = 0; i < numRequests; i++) {
            int n = inputLow + random.nextInt(inputHigh - inputLow + 1);
            int gen = outputLow + random.nextInt(outputHigh - outputLow + 1);
            int offset = random.nextInt(Math.max(allowed.length, 1));
            int[] tokens = new int[prefixLen + n];
            System.arraycopy(prefix, 0, tokens, 0, prefixLen);
            for (int j = 0; j < n; j++) {
                tokens[prefixLen + j] = allowed[(offset + i + j) % allowed.length];
            }
            out.add(BenchSample.ofTokens(tokens, gen, opts.requestIdPrefix() + i));
        }
        return ServeBenchDataset.oversample(out, numRequests, opts.requestIdPrefix(), opts.noOversample(), seed);
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

    /** The shared prefix: {@code prefixLen} uniform draws over the allowed ids, empty at length zero. */
    private int[] prefixTokens(Random random, int[] allowed, int prefixLen) {
        int[] prefix = new int[prefixLen];
        for (int i = 0; i < prefixLen; i++) {
            prefix[i] = allowed[random.nextInt(allowed.length)];
        }
        return prefix;
    }
}
