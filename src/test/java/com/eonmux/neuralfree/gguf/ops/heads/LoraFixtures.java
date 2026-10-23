package com.eonmux.neuralfree.gguf.ops.heads;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * In-memory LoRA adapters and a scalar reference of their deltas for the adapter tests: pairs filled with seeded
 * values in {@code [-0.5, 0.5)}, and {@code y += s * B[e] (A[e] x)} / {@code y += s * B A[id]} evaluated in double.
 */
final class LoraFixtures {

    private LoraFixtures() {
    }

    static float[] random(long seed, int n) {
        Random r = new Random(seed);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = r.nextFloat() - 0.5f;
        }
        return out;
    }

    static LoraAdapter adapter(String path, float alpha, LoraAdapter.Pair... pairs) {
        Map<String, LoraAdapter.Pair> map = new LinkedHashMap<>();
        for (LoraAdapter.Pair p : pairs) {
            map.put(p.target(), p);
        }
        return new LoraAdapter(path, "llama", alpha, null, null, null, map, Collections.<String, String>emptyMap());
    }

    static LoraAdapter.Pair projection(String target, int in, int out, int rank, long seed) {
        return new LoraAdapter.Pair(target, LoraAdapter.Target.PROJECTION, in, out, rank, 1,
                random(seed, rank * in), random(seed + 1, out * rank));
    }

    static LoraAdapter.Pair experts(String target, int in, int out, int rank, int nExpert, long seed) {
        return new LoraAdapter.Pair(target, LoraAdapter.Target.EXPERTS, in, out, rank, nExpert,
                random(seed, nExpert * rank * in), random(seed + 1, nExpert * out * rank));
    }

    static LoraAdapter.Pair embedding(String target, int nVocab, int width, int rank, long seed) {
        return new LoraAdapter.Pair(target, LoraAdapter.Target.EMBEDDING, nVocab, width, rank, 1,
                random(seed, nVocab * rank), random(seed + 1, width * rank));
    }

    /** {@code y[yOff + o] += s * sum_r B[e][o][r] * sum_i A[e][r][i] * x[xOff + i]} of a projection or expert pair. */
    static void project(LoraAdapter.Pair p, float s, int e, float[] x, int xOff, float[] y, int yOff) {
        double[] low = new double[p.rank];
        for (int r = 0; r < p.rank; r++) {
            double acc = 0.0;
            for (int i = 0; i < p.inDim; i++) {
                acc += (double) p.a[(e * p.rank + r) * p.inDim + i] * x[xOff + i];
            }
            low[r] = acc;
        }
        for (int o = 0; o < p.outDim; o++) {
            double acc = 0.0;
            for (int r = 0; r < p.rank; r++) {
                acc += (double) p.b[(e * p.outDim + o) * p.rank + r] * low[r];
            }
            y[yOff + o] += (float) (s * acc);
        }
    }

    /** {@code y[yOff + c] += s * sum_r B[c][r] * A[id][r]} of an embedding pair. */
    static void embed(LoraAdapter.Pair p, float s, int id, float[] y, int yOff) {
        for (int c = 0; c < p.outDim; c++) {
            double acc = 0.0;
            for (int r = 0; r < p.rank; r++) {
                acc += (double) p.b[c * p.rank + r] * p.a[id * p.rank + r];
            }
            y[yOff + c] += (float) (s * acc);
        }
    }
}
