package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link CpuLayerExecutor#forwardKV} over a real one-layer llama with a KV memory and the registry's real blocks:
 * the attention part only — the FFN never runs — and the cache rows it writes are the ones a whole {@link
 * CpuLayerExecutor#forward} writes, so the step that follows attends over the same rows and produces the same
 * stream.
 */
public class CpuLayerExecutorKvOnlyTest {

    private static final int ROWS = 3;

    /**
     * The single-token step after the prompt: whether the prompt ran whole ({@code forward}) or as its cache writes
     * only ({@code forwardKV}), the next token reads the same attention and the streams agree.
     */
    @Test
    public void forwardKVWritesTheCachesAForwardWrites() throws IOException {
        CpuLayerFixture f = new CpuLayerFixture();
        try {
            float[] prompt = stream(ROWS, f.width);
            float[] token = stream(1, f.width);

            // the whole layer over the prompt, then the next token over the prompt's output
            float[] x = prompt.clone();
            f.forward(0, ROWS, x);
            float[] afterPrompt = x.clone();
            System.arraycopy(token, 0, x, 0, f.width);
            f.forward(ROWS, 1, x);
            float[] wholeNext = Arrays.copyOf(x, f.width);

            // the same prompt as its cache writes only, then the same next token
            f.memory.reset();
            f.ffnRuns.set(0);
            f.forwardKV(0, ROWS, prompt.clone());
            assertEquals("forwardKV runs no FFN", 0, f.ffnRuns.get());
            float[] y = new float[prompt.length];
            System.arraycopy(afterPrompt, (ROWS - 1) * f.width, y, 0, f.width);
            System.arraycopy(token, 0, y, 0, f.width);
            f.forward(ROWS, 1, y);
            float[] kvNext = Arrays.copyOf(y, f.width);
            assertEquals("the following whole step runs its FFN", 1, f.ffnRuns.get());

            assertArrayEquals(wholeNext, kvNext, 1e-9f);
            boolean nonZero = false;
            for (float v : wholeNext) {
                nonZero |= v != 0f;
            }
            assertTrue("the streams are not all zero", nonZero);
        } finally {
            f.close();
        }
    }

    /**
     * The prompt's cache rows after its attention part alone are bit-for-bit the rows a whole {@code forward}
     * writes: the KV-only step runs the same K/V projections, norms, RoPE and store, and nothing else.
     */
    @Test
    public void forwardKVWritesTheSameCacheBits() throws IOException {
        CpuLayerFixture f = new CpuLayerFixture();
        try {
            float[] prompt = stream(ROWS, f.width);
            f.forward(0, ROWS, prompt.clone());
            int[] whole = cacheBits(f);

            f.memory.reset();
            f.forwardKV(0, ROWS, prompt.clone());
            int[] kvOnly = cacheBits(f);

            assertArrayEquals(whole, kvOnly);
            boolean nonZero = false;
            for (int bits : whole) {
                nonZero |= bits != 0;
            }
            assertTrue("the cache rows are not all zero", nonZero);
        } finally {
            f.close();
        }
    }

    /** The raw bits of every stored K and V head over the first {@link #ROWS} positions. */
    private static int[] cacheBits(CpuLayerFixture f) {
        KvAccessor cache = f.memory.kv(0);
        int heads = cache.nHeadKv();
        int headK = cache.headDimK();
        int headV = cache.headDimV();
        int[] out = new int[ROWS * heads * (headK + headV)];
        float[] k = new float[headK];
        float[] v = new float[headV];
        int at = 0;
        for (int row = 0; row < ROWS; row++) {
            for (int h = 0; h < heads; h++) {
                cache.readK(row, h, k, 0);
                for (float x : k) {
                    out[at++] = Float.floatToRawIntBits(x);
                }
                cache.readV(row, h, v, 0);
                for (float x : v) {
                    out[at++] = Float.floatToRawIntBits(x);
                }
            }
        }
        return out;
    }

    private static float[] stream(int n, int width) {
        Random rnd = new Random(42L * n);
        float[] x = new float[n * width];
        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() * 2f - 1f;
        }
        return x;
    }
}
