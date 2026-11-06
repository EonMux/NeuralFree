package com.eonmux.neuralfree.demo.tools;

/**
 * The vocabulary math of llama.cpp's perplexity tool (tools/perplexity/perplexity.cpp): the softmax and
 * log-softmax of one logits row, the 16-bit log-prob rows its {@code --save-all-logits} file stores, the
 * per-row KL-divergence comparison of a model's logits against a stored base row, and the statistics the
 * KL report prints. Float operations keep llama.cpp's precision so the two tools' numbers and files agree.
 */
final class PerplexityMath {

    private PerplexityMath() {
    }

    /** Softmax over the whole row, evaluated at one token (perplexity.cpp {@code softmax}). */
    static float softmaxAt(float[] logits, int token) {
        float maxLogit = logits[0];
        for (int i = 1; i < logits.length; i++) {
            maxLogit = Math.max(maxLogit, logits[i]);
        }
        double sumExp = 0.0;
        for (int i = 0; i < logits.length; i++) {
            sumExp += (float) Math.exp(logits[i] - maxLogit);
        }
        return (float) ((float) Math.exp(logits[token] - maxLogit) / sumExp);
    }

    /**
     * The natural log of the row's probability at {@code token} (perplexity.cpp {@code log_softmax}); the
     * negative log-likelihood of the token given the context that produced the row.
     */
    static double logSoftmax(float[] logits, int token) {
        float maxLogit = logits[0];
        for (int i = 1; i < logits.length; i++) {
            maxLogit = Math.max(maxLogit, logits[i]);
        }
        double sumExp = 0.0;
        for (int i = 0; i < logits.length; i++) {
            sumExp += (float) Math.exp(logits[i] - maxLogit);
        }
        return (logits[token] - maxLogit) - Math.log(sumExp);
    }

    /**
     * Rounds to the nearest integer, halves to even, the bit trick of perplexity.cpp's {@code nearest_int}
     * (the quantized rows store rounded values, so the rounding must match llama.cpp's exactly).
     */
    static int nearestInt(float value) {
        int bits = Float.floatToIntBits(value + 12582912.0f);
        return (bits & 0x007fffff) - 0x00400000;
    }

    /**
     * Entries of one stored row (perplexity.cpp {@code nv}): two floats, then the quantized vocabulary
     * entries padded to an even count.
     */
    static int rowLength(int nVocab) {
        return 2 * ((nVocab + 1) / 2) + 4;
    }

    /**
     * Encodes one row into the file's 16-bit form and returns the row's negative log-likelihood at
     * {@code token} (perplexity.cpp's writing {@code log_softmax}): the first two floats are the scale and
     * the minimum log-probability, then {@code nearest_int((logit - min_logit) / scale)} per entry, zero
     * below the cutoff.
     */
    static double quantLogSoftmaxRow(float[] logits, int token, short[] row, int off) {
        float maxLogit = logits[0];
        float minLogit = logits[0];
        for (int i = 1; i < logits.length; i++) {
            maxLogit = Math.max(maxLogit, logits[i]);
            minLogit = Math.min(minLogit, logits[i]);
        }
        minLogit = Math.max(minLogit, maxLogit - 16.0f);
        double sumExp = 0.0;
        for (int i = 0; i < logits.length; i++) {
            sumExp += (float) Math.exp(logits[i] - maxLogit);
        }
        float logSumExp = (float) Math.log(sumExp);
        float minLogProb = minLogit - maxLogit - logSumExp;
        float scale = (maxLogit - minLogit) / 65535.0f;
        putFloat(row, off, scale);
        putFloat(row, off + 2, minLogProb);
        int entries = off + 4;
        if (scale != 0.0f) {
            float invScale = 1.0f / scale;
            for (int i = 0; i < logits.length; i++) {
                row[entries + i] = logits[i] > minLogit
                        ? (short) nearestInt(invScale * (logits[i] - minLogit))
                        : 0;
            }
        } else {
            for (int i = 0; i < logits.length; i++) {
                row[entries + i] = 0;
            }
        }
        return maxLogit + logSumExp - logits[token];
    }

    /** The scale float of a stored row. */
    static float rowScale(short[] row, int off) {
        return getFloat(row, off);
    }

    /** The minimum-log-probability float of a stored row. */
    static float rowMinLogProb(short[] row, int off) {
        return getFloat(row, off + 2);
    }

    /** The unsigned entry of a stored row at vocabulary index {@code i}. */
    static int rowEntry(short[] row, int off, int i) {
        return row[off + 4 + i] & 0xFFFF;
    }

    /**
     * The KL-divergence sums of perplexity.cpp's {@code kl_divergence_result}, accumulated per compared row
     * and reduced by the report's statistics.
     */
    static final class Kld {
        double sumNll;
        double sumNll2;
        double sumNllBase;
        double sumNllBase2;
        double sumNllNllBase;
        double sumKld;
        double sumKld2;
        double sumPDiff;
        double sumPDiff2;
        double sumPDiff4;
        float maxPDiff;
        long nSameTop;
        long count;

        /** The mean and the standard error of a sum and its square sum ({@code mean_and_uncertainty}). */
        double[] meanAndUncertainty(double sum, double sum2) {
            if (count < 1) {
                return new double[] {0.0, 0.0};
            }
            double f = sum / count;
            double df = sum2 / count - f * f;
            df = df > 0 && count > 10 ? Math.sqrt(df / (count - 1)) : 0.0;
            return new double[] {f, df};
        }

        /** The covariance of two means ({@code covariance}), zero below ten paired values. */
        double covariance(double sumA, double sumB, double sumAB) {
            if (count < 10) {
                return 0.0;
            }
            double var = sumAB / count - (sumA / count) * (sumB / count);
            return var / (count - 1);
        }
    }

    /**
     * Compares one row of the evaluated model against one stored base row and accumulates into
     * {@code acc} (perplexity.cpp's comparing {@code log_softmax}); {@code out} receives the row's KL
     * divergence at index 0 and its probability difference at index 1.
     */
    static void klRow(float[] logits, short[] base, int baseOff, int token, Kld acc, float[] out) {
        float maxLogit = logits[0];
        int iMax = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > maxLogit) {
                maxLogit = logits[i];
                iMax = i;
            }
        }
        double sumExp = 0.0;
        for (int i = 0; i < logits.length; i++) {
            sumExp += (float) Math.exp(logits[i] - maxLogit);
        }
        float logSumExp = (float) Math.log(sumExp);
        float scale = getFloat(base, baseOff);
        float minLogProb = getFloat(base, baseOff + 2);
        int entries = baseOff + 4;

        float nll = maxLogit + logSumExp - logits[token];
        acc.sumNll += nll;
        acc.sumNll2 += (double) (nll * nll);

        float nllBase = -(scale * rowEntry(base, baseOff, token) + minLogProb);
        acc.sumNllBase += nllBase;
        acc.sumNllBase2 += (double) (nllBase * nllBase);

        acc.sumNllNllBase += (double) (nll * nllBase);

        maxLogit += logSumExp;
        double sum = 0;
        int iMaxBase = -1;
        float pLogBaseMax = 0;
        for (int i = 0; i < logits.length; i++) {
            float pLogBase = scale * rowEntry(base, baseOff, i) + minLogProb;
            if (i == 0 || pLogBase > pLogBaseMax) {
                pLogBaseMax = pLogBase;
                iMaxBase = i;
            }
            if (pLogBase > -16.0f) {
                float pBase = (float) Math.exp(pLogBase);
                sum += pBase * (pLogBase - logits[i] + maxLogit);
            }
        }
        acc.sumKld += sum;
        acc.sumKld2 += (double) (sum * sum);
        acc.count++;
        if (iMax == iMaxBase) {
            acc.nSameTop++;
        }

        float pBase = (float) Math.exp(-nllBase);
        float p = (float) Math.exp(-nll);
        float pDiff = p - pBase;
        acc.sumPDiff += pDiff;
        double pDiff2 = (double) (pDiff * pDiff);
        acc.sumPDiff2 += pDiff2;
        acc.sumPDiff4 += pDiff2 * pDiff2;
        acc.maxPDiff = Math.max(acc.maxPDiff, Math.abs(pDiff));

        out[0] = (float) sum;
        out[1] = pDiff;
    }

    /**
     * A percentile of the sorted values, the interpolation of perplexity.cpp's {@code percentile}.
     *
     * @param fraction in {@code [0, 1]}
     */
    static float percentile(float[] sorted, float fraction) {
        if (sorted.length == 0) {
            throw new IllegalArgumentException("a percentile of no values");
        }
        if (fraction <= 0.0f) {
            return sorted[0];
        }
        if (fraction >= 1.0f) {
            return sorted[sorted.length - 1];
        }
        float p = fraction * (sorted.length - 1);
        int ip = (int) p;
        p -= ip;
        return (1.0f - p) * sorted[ip] + p * sorted[Math.min(ip + 1, sorted.length - 1)];
    }

    /** A float stored as two little-endian 16-bit halves of the row. */
    private static void putFloat(short[] row, int off, float value) {
        int bits = Float.floatToIntBits(value);
        row[off] = (short) (bits & 0xFFFF);
        row[off + 1] = (short) (bits >>> 16);
    }

    private static float getFloat(short[] row, int off) {
        int bits = (row[off] & 0xFFFF) | ((row[off + 1] & 0xFFFF) << 16);
        return Float.intBitsToFloat(bits);
    }
}
