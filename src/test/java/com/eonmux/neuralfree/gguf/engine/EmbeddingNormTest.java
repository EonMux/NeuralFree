package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertThrows;

/** llama.cpp's embedding normalization modes. */
public class EmbeddingNormTest {

    private static final float[] V = {3f, -4f};

    @Test
    public void eachModeDividesByItsNorm() {
        assertArrayEquals(new float[]{3f, -4f}, EmbeddingNorm.normalize(V, EmbeddingNorm.NONE), 1e-6f);
        assertArrayEquals(new float[]{3f * 32760f / 4f, -32760f}, EmbeddingNorm.normalize(V, 0), 1e-2f);
        assertArrayEquals(new float[]{3f / 7f, -4f / 7f}, EmbeddingNorm.normalize(V, 1), 1e-6f);
        assertArrayEquals(new float[]{0.6f, -0.8f}, EmbeddingNorm.normalize(V, EmbeddingNorm.EUCLIDEAN), 1e-6f);
        double p3 = Math.cbrt(27 + 64);
        assertArrayEquals(new float[]{(float) (3 / p3), (float) (-4 / p3)}, EmbeddingNorm.normalize(V, 3), 1e-6f);
    }

    @Test
    public void aZeroVectorStaysZeroAndTheInputIsUntouched() {
        assertArrayEquals(new float[]{0f, 0f}, EmbeddingNorm.normalize(new float[]{0f, 0f}, 2), 0f);
        EmbeddingNorm.normalize(V, 2);
        assertArrayEquals(new float[]{3f, -4f}, V, 0f);
        assertThrows(IllegalArgumentException.class, () -> EmbeddingNorm.normalize(V, -2));
    }
}
