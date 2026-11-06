package com.eonmux.neuralfree.gguf.engine.watermark;

/**
 * A keyed pseudorandom function mapping a token's context to a uniform draw: {@code uniform(context, tokenId)}
 * answers one value in {@code (0, 1)} for every token the context can produce. The function is stateless, so the
 * watermarker and the detector derive the same draws from the same key and context. Implementations return values
 * that are exactly representable as a {@code float}, matching vLLM's {@code float32} draws.
 */
public interface WatermarkPrf {

    /** A version tag of the function and its construction, as vLLM records {@code prf_version}. */
    String version();

    /**
     * The uniform draw of {@code tokenId} under {@code context}.
     *
     * @param context    the preceding token ids, oldest first, each within {@code [-1, 2^32 - 1]}
     * @param tokenId    the candidate token id
     * @return a value in {@code (0, 1)}
     */
    double uniform(int[] context, int tokenId);

    /**
     * {@link #uniform(int[], int)} for {@code count} token ids at once; {@code out[i]} is the draw of
     * {@code tokenIds[i]}. Implementations with per-context setup override this to compute it once.
     *
     * @throws IllegalArgumentException when {@code out} is shorter than {@code count}
     */
    default void uniforms(int[] context, int[] tokenIds, int count, double[] out) {
        if (out.length < count) {
            throw new IllegalArgumentException("output array of " + out.length + " values cannot hold " + count);
        }
        for (int i = 0; i < count; i++) {
            out[i] = uniform(context, tokenIds[i]);
        }
    }
}
