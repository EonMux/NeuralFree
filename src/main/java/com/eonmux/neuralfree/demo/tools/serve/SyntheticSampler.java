package com.eonmux.neuralfree.demo.tools.serve;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The length and token draws the synthetic datasets share, the reference's
 * {@code get_sampling_params} and reproducible token walk: every request's input and output lengths
 * are uniform over the integer ranges the range ratio spans around the configured means, each
 * request draws an offset into the allowed vocabulary, and its body is
 * {@code (offset + index + j) % |allowed|}. The datasets that must send text where the reference
 * sends decoded tokens render the walk through the tokenizer's {@code detokenize}.
 */
final class SyntheticSampler {

    /** One request's drawn lengths and vocabulary offset. */
    static final class Draw {

        final int inputLen;
        final int outputLen;
        final int offset;

        Draw(int inputLen, int outputLen, int offset) {
            this.inputLen = inputLen;
            this.outputLen = outputLen;
            this.offset = offset;
        }
    }

    private SyntheticSampler() {
    }

    /** The vocabulary a synthetic prompt draws from: every id but the skipped special and stop ids. */
    static int[] allowed(int vocabSize, int[] skipIds) {
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

    /** One length draw's interval, the reference's clamp to at least one output token. */
    static int[] interval(int mean, double ratio, boolean output) {
        if (!(ratio >= 0.0 && ratio < 1.0)) {
            throw new IllegalArgumentException("the range ratio must be in [0, 1), got " + ratio);
        }
        int low = (int) Math.floor(mean * (1.0 - ratio));
        int high = (int) Math.ceil(mean * (1.0 + ratio));
        if (output) {
            low = Math.max(low, 1);
            high = Math.max(high, 1);
        }
        if (low > high) {
            throw new IllegalArgumentException("invalid sampling interval: low=" + low + " > high=" + high);
        }
        return new int[]{low, high};
    }

    /** {@code count} draws whose lengths and offsets each come from their own mean. */
    static Draw[] draws(ServeBenchOptions opts, Random random, int count, int inputMean, int outputMean) {
        double inputRatio = opts.randomRangeRatio().input();
        double outputRatio = opts.randomRangeRatio().output();
        int[] input = interval(inputMean, inputRatio, false);
        int[] output = interval(outputMean, outputRatio, true);
        int[] inputs = new int[count];
        int[] outputs = new int[count];
        int[] offsets = new int[count];
        for (int i = 0; i < count; i++) {
            inputs[i] = input[0] + random.nextInt(input[1] - input[0] + 1);
        }
        for (int i = 0; i < count; i++) {
            outputs[i] = output[0] + random.nextInt(output[1] - output[0] + 1);
        }
        return rebuild(inputs, outputs, offsets);
    }

    private static Draw[] rebuild(int[] inputs, int[] outputs, int[] offsets) {
        Draw[] draws = new Draw[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            draws[i] = new Draw(inputs[i], outputs[i], offsets[i]);
        }
        return draws;
    }

    /** Draws the per-request vocabulary offsets in the reference's order. */
    static void drawOffsets(Draw[] draws, Random random, int vocabSize) {
        for (int i = 0; i < draws.length; i++) {
            draws[i] = new Draw(draws[i].inputLen, draws[i].outputLen, random.nextInt(Math.max(vocabSize, 1)));
        }
    }

    /** The token walk of one request: {@code (offset + index + j) % |allowed|}. */
    static int[] walk(int[] allowed, int offset, int index, int length) {
        int[] tokens = new int[length];
        for (int j = 0; j < length; j++) {
            tokens[j] = allowed[Math.floorMod(offset + index + j, allowed.length)];
        }
        return tokens;
    }

    /** The text of a token sequence through the tokenizer, refused by name when it cannot decode. */
    static String text(BenchTokenizer tokenizer, int[] tokens) {
        try {
            return tokenizer.detokenize(tokens);
        } catch (UnsupportedOperationException e) {
            throw new IllegalArgumentException("this dataset needs the tokenizer to decode synthetic"
                    + " token ids into text: pass --tokenizer or use a server with /detokenize", e);
        }
    }
}
