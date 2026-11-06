package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * {@link EmbeddingSimilarity} against llama.cpp's {@code common_embd_similarity_cos}: the dot product over the norms,
 * the zero-vector cases, and the float products the C function accumulates in double.
 */
public class EmbeddingSimilarityTest {

    @Test
    public void cosineIsTheDotProductOverTheNorms() {
        assertEquals(0f, EmbeddingSimilarity.cosine(new float[]{1, 0}, new float[]{0, 1}), 0f);
        assertEquals(1f, EmbeddingSimilarity.cosine(new float[]{1, 2}, new float[]{2, 4}), 1e-6f);
        assertEquals(-1f, EmbeddingSimilarity.cosine(new float[]{1, 0}, new float[]{-3, 0}), 1e-6f);
        assertEquals(0.6f, EmbeddingSimilarity.cosine(new float[]{1, 0}, new float[]{0.6f, 0.8f}), 1e-6f);
        assertEquals(0f, EmbeddingSimilarity.cosine(new float[]{1, 1, 0, 0}, new float[]{0, 0, 1, 1}), 0f);
    }

    @Test
    public void zeroVectorsFollowLlamaCpp() {
        assertEquals(1f, EmbeddingSimilarity.cosine(new float[3], new float[3]), 0f);
        assertEquals(0f, EmbeddingSimilarity.cosine(new float[3], new float[]{1, 2, 3}), 0f);
        assertEquals(0f, EmbeddingSimilarity.cosine(new float[]{1, 2, 3}, new float[3]), 0f);
    }

    /**
     * The products round to float before the double sum, as {@code embd1[i] * embd2[i]} does in C: a square whose
     * float product underflows to zero makes the vector a zero vector, where a double product would not.
     */
    @Test
    public void productsRoundToFloatLikeTheCFunction() {
        float tiny = 1e-23f; // tiny * tiny = 1e-46 underflows float (below Float.MIN_VALUE), not double
        assertEquals(0f, tiny * tiny, 0f);
        assertEquals(1f, EmbeddingSimilarity.cosine(new float[]{tiny}, new float[]{tiny}), 0f);
        assertEquals(0f, EmbeddingSimilarity.cosine(new float[]{tiny}, new float[]{1f}), 0f);
    }

    @Test
    public void cosineValidatesItsInputs() {
        assertThrows(IllegalArgumentException.class, () -> EmbeddingSimilarity.cosine(new float[2], new float[3]));
        assertThrows(IllegalArgumentException.class, () -> EmbeddingSimilarity.cosine(new float[0], new float[0]));
    }
}
