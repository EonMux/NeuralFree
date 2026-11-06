package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuKvLayer;
import org.junit.Test;

import java.util.Random;
import java.util.function.IntUnaryOperator;

import static org.junit.Assert.assertEquals;

/**
 * The Step5 sparse and windowed CPU attention against a direct transcription of the vLLM arithmetic: selected
 * regions (or the sliding window) up to the query position, an F32 softmax in base 2, BF16-rounded weights, BF16
 * output rounding and the BF16 sigmoid gate.
 */
public class CpuStep5AttnTest {

    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int HEAD_DIM = 8;
    private static final int REGION = 8;

    @Test
    public void sparseAttentionMatchesTheDirectEvaluation() {
        int tokens = 20;
        Holder kv = cache(tokens);
        Random rnd = new Random(11);
        float[] q = random(rnd, tokens * HEADS * HEAD_DIM);
        float[] gate = random(rnd, tokens * HEADS);
        int[] qPos = new int[tokens];
        int slots = 3;
        int[] sel = new int[tokens * KV_HEADS * slots];
        for (int t = 0; t < tokens; t++) {
            qPos[t] = t;
            int closed = t / REGION;
            for (int g = 0; g < KV_HEADS; g++) {
                for (int s = 0; s < slots; s++) {
                    // the last slot is the open region; the others pick a closed region when one exists
                    int region = s == slots - 1 ? closed : (closed == 0 ? -1 : (t + g + s) % closed);
                    sel[(t * KV_HEADS + g) * slots + s] = region;
                }
            }
        }
        float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
        float[] out = new float[tokens * HEADS * HEAD_DIM];
        IntUnaryOperator rowAt = p -> p;
        CpuStep5Attn.sparse(tokens, HEADS, KV_HEADS, HEAD_DIM, REGION, slots, q, 0, HEADS * HEAD_DIM, scale, kv.kv,
                rowAt, qPos, sel, 0, KV_HEADS * slots, slots, gate, 0, HEADS, out, 0, HEADS * HEAD_DIM);
        float[] expected = direct(kv, q, gate, qPos, sel, slots, scale, tokens, true, 0);
        for (int i = 0; i < out.length; i++) {
            assertEquals("element " + i, expected[i], out[i], 1e-6f);
        }
    }

    @Test
    public void windowedAttentionMatchesTheDirectEvaluation() {
        int tokens = 12;
        int window = 5;
        Holder kv = cache(tokens);
        Random rnd = new Random(13);
        float[] q = random(rnd, tokens * HEADS * HEAD_DIM);
        float[] gate = random(rnd, tokens * HEADS);
        int[] qPos = new int[tokens];
        for (int t = 0; t < tokens; t++) {
            qPos[t] = t;
        }
        float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
        float[] out = new float[tokens * HEADS * HEAD_DIM];
        CpuStep5Attn.windowed(tokens, HEADS, KV_HEADS, HEAD_DIM, window, q, 0, HEADS * HEAD_DIM, scale, kv.kv,
                p -> p, qPos, gate, 0, HEADS, out, 0, HEADS * HEAD_DIM);
        float[] expected = direct(kv, q, gate, qPos, null, 0, scale, tokens, false, window);
        for (int i = 0; i < out.length; i++) {
            assertEquals("element " + i, expected[i], out[i], 1e-6f);
        }
    }

    /** A F32 cache plus the plain arrays the direct evaluation reads. */
    private static final class Holder {
        final CpuKvLayer kv;
        final float[] k;
        final float[] v;

        Holder(CpuKvLayer kv, float[] k, float[] v) {
            this.kv = kv;
            this.k = k;
            this.v = v;
        }

        float[] k(int row, int group) {
            float[] out = new float[HEAD_DIM];
            System.arraycopy(k, (row * KV_HEADS + group) * HEAD_DIM, out, 0, HEAD_DIM);
            return out;
        }

        float[] v(int row, int group) {
            float[] out = new float[HEAD_DIM];
            System.arraycopy(v, (row * KV_HEADS + group) * HEAD_DIM, out, 0, HEAD_DIM);
            return out;
        }
    }

    private static Holder cache(int tokens) {
        KvLayout layout = KvLayout.of(tokens, false, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KvDType.F32,
                KvDType.F32, KvWindow.FULL_CAUSAL);
        CpuKvLayer kv = CpuKvLayer.allocate(0, layout, 0, KV_HEADS, HEAD_DIM, HEAD_DIM);
        float[] k = new float[tokens * KV_HEADS * HEAD_DIM];
        float[] v = new float[tokens * KV_HEADS * HEAD_DIM];
        Random rnd = new Random(5);
        for (int i = 0; i < k.length; i++) {
            k[i] = (float) rnd.nextGaussian() * 0.3f;
            v[i] = (float) rnd.nextGaussian() * 0.3f;
        }
        for (int p = 0; p < tokens; p++) {
            kv.writeK(p, k, p * KV_HEADS * HEAD_DIM);
            kv.writeV(p, v, p * KV_HEADS * HEAD_DIM);
        }
        return new Holder(kv, k, v);
    }

    private static float[] direct(Holder kv, float[] q, float[] gate, int[] qPos, int[] sel, int slots,
                                  float scale, int tokens, boolean sparse, int window) {
        float[] out = new float[tokens * HEADS * HEAD_DIM];
        float[] kRow = new float[HEAD_DIM];
        float[] vRow = new float[HEAD_DIM];
        for (int t = 0; t < tokens; t++) {
            int pos = qPos[t];
            for (int h = 0; h < HEADS; h++) {
                int group = h / (HEADS / KV_HEADS);
                int[] cells = new int[Math.max(tokens, slots * REGION + 1)];
                float[] logits = new float[cells.length];
                int count = 0;
                if (sparse) {
                    for (int s = 0; s < slots; s++) {
                        int region = sel[(t * KV_HEADS + group) * slots + s];
                        if (region < 0) {
                            continue;
                        }
                        for (int p = region * REGION; p <= Math.min(region * REGION + REGION - 1, pos); p++) {
                            cells[count] = p;
                            logits[count] = dot(q, t * HEADS * HEAD_DIM + h * HEAD_DIM, kv.k(p, group), scale);
                            count++;
                        }
                    }
                } else {
                    int first = window > 0 ? Math.max(0, pos - window + 1) : 0;
                    for (int p = first; p <= pos; p++) {
                        cells[count] = p;
                        logits[count] = dot(q, t * HEADS * HEAD_DIM + h * HEAD_DIM, kv.k(p, group), scale);
                        count++;
                    }
                }
                float max = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < count; i++) {
                    max = Math.max(max, logits[i] * 1.4426950408889634f);
                }
                double total = 0.0;
                for (int i = 0; i < count; i++) {
                    logits[i] = Step5Dtypes.exp2(logits[i] * 1.4426950408889634f - max);
                    total += logits[i];
                }
                float[] acc = new float[HEAD_DIM];
                for (int i = 0; i < count; i++) {
                    float w = Step5Dtypes.bf16(logits[i]);
                    for (int d = 0; d < HEAD_DIM; d++) {
                        acc[d] += w * kv.v(cells[i], group)[d];
                    }
                }
                float inv = total > 0.0 ? (float) (1.0 / Math.max(total, 1e-30)) : 0f;
                float sig = Step5Dtypes.gateSigmoid(gate[t * HEADS + h]);
                for (int d = 0; d < HEAD_DIM; d++) {
                    out[(t * HEADS + h) * HEAD_DIM + d] =
                            Step5Dtypes.bf16(Step5Dtypes.bf16(acc[d] * inv) * sig);
                }
            }
        }
        return out;
    }

    private static float dot(float[] a, int aOff, float[] b, float scale) {
        float sum = 0f;
        for (int i = 0; i < b.length; i++) {
            sum += a[aOff + i] * b[i];
        }
        return sum * scale;
    }

    private static float[] random(Random rnd, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) rnd.nextGaussian() * 0.4f;
        }
        return x;
    }
}
