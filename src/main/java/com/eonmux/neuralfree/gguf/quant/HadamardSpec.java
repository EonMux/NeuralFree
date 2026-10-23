package com.eonmux.neuralfree.gguf.quant;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable description of one normalized Sylvester Walsh-Hadamard transform applied to activation rows,
 * and the Java contract of the {@value #KERNEL} kernel ({@code kernels/llm/hadamard.cl}) and of {@link CpuHadamard}.
 *
 * <p>Each row of {@code width} elements is split into blocks of {@link #block()} elements. Output element
 * {@code o = base + i} of the block starting at {@code base} is
 * <pre>
 *   y[o] = after(o) * (1/sqrt(block)) * sum_j (-1)^popcount(i &amp; j) * before(base + j) * x[src(base + j)]
 * </pre>
 * with the orthonormal Hadamard matrix of llama.cpp (llama.cpp src/llama-model.cpp:2300-2312,
 * src/llama-kv-cache.cpp:23 {@code ggml_gen_hadamard}; {@code H*H = I}):
 * <ul>
 *   <li>{@link #sourceIndex(int) src} is the identity, or with {@link #hasPermutation()} the gather from
 *       tiled head order {@code [permHd, permNk, permRep]} to grouped order {@code [permHd, permRep, permNk]}
 *       (llama.cpp src/llama-graph.cpp:1610-1618);</li>
 *   <li>{@code before}/{@code after} are the {@code +/-1} signs per {@link #signOrder()}: before the
 *       rotation for a PRISM folded weight's matmul input {@code H*(D*x)} (llama.cpp src/llama-graph.cpp:1619-1624),
 *       after it for the inverse transform of a latent embedding row {@code D*(H*z)}
 *       (llama.cpp src/llama-graph.cpp:2685-2690), otherwise 1.</li>
 * </ul>
 * Plain rotations ({@link #rotation(int)}) serve the quantized-KV and DSA-indexer rotations
 * (llama.cpp src/llama-kv-cache.cpp:315-361, src/llama-graph.cpp:3225-3271).
 */
public final class HadamardSpec {

    /** Kernel name of the GPU transform. */
    public static final String KERNEL = "LLM_HadamardRotate";

    /** Argument indices of {@value #KERNEL}, in declaration order. */
    public static final int ARG_N_ROWS = 0;
    public static final int ARG_WIDTH = 1;
    public static final int ARG_BLOCK = 2;
    public static final int ARG_HAS_SIGNS = 3;
    public static final int ARG_PERM_HD = 4;
    public static final int ARG_PERM_NK = 5;
    public static final int ARG_PERM_REP = 6;
    public static final int ARG_X_OFF = 7;
    public static final int ARG_Y_OFF = 8;
    public static final int ARG_SIGNS = 9;
    public static final int ARG_X = 10;
    public static final int ARG_Y = 11;

    /** Largest block the kernel transforms ({@code NF_HD_MAX_BLOCK}, its local-memory buffer). */
    public static final int MAX_KERNEL_BLOCK = 1024;

    /** Prefix of the sign-vector names, {@code prism.hadamard.signs.<width>} (llama.cpp src/llama-model.cpp:2345). */
    public static final String SIGNS_NAME_PREFIX = "prism.hadamard.signs.";

    /** Where the {@code +/-1} signs are applied relative to the rotation; codes are the kernel's {@code has_signs}. */
    public enum SignOrder {
        /** No signs. */
        NONE(0),
        /** {@code y = H * (D * x)}: matmul input of a PRISM folded weight. */
        BEFORE_ROTATION(1),
        /** {@code y = D * (H * z)}: inverse transform after a latent-table row lookup. */
        AFTER_ROTATION(2);

        private final int kernelCode;

        SignOrder(int kernelCode) {
            this.kernelCode = kernelCode;
        }

        /** Value of the kernel's {@code has_signs} argument. */
        public int kernelCode() {
            return kernelCode;
        }
    }

    private final int block;
    private final SignOrder signOrder;
    private final float[] signs;
    private final int permHd;
    private final int permNk;
    private final int permRep;

    private HadamardSpec(int block, SignOrder signOrder, float[] signs, int permHd, int permNk, int permRep) {
        this.block = block;
        this.signOrder = signOrder;
        this.signs = signs;
        this.permHd = permHd;
        this.permNk = permNk;
        this.permRep = permRep;
    }

    /**
     * Plain block rotation without signs or permutation (quantized KV, DSA indexer).
     *
     * @throws IllegalArgumentException when {@code block} is not a positive power of two
     */
    public static HadamardSpec rotation(int block) {
        checkBlock(block);
        return new HadamardSpec(block, SignOrder.NONE, null, 0, 0, 0);
    }

    /**
     * General transform.
     *
     * @param block     block size, a positive power of two
     * @param signOrder where {@code signs} apply; {@link SignOrder#NONE} requires {@code signs == null}
     * @param signs     {@code +/-1} values, one per row element (copied); null for {@link SignOrder#NONE}
     * @param permHd    head size of the tiled-to-grouped permutation (ignored when {@code permRep <= 1})
     * @param permNk    group count of the permutation (ignored when {@code permRep <= 1})
     * @param permRep   heads per group; a value above 1 enables the permutation
     * @throws IllegalArgumentException when a value is out of range or the sign length, the permutation width
     *                                  and the block size disagree
     */
    public static HadamardSpec of(int block, SignOrder signOrder, float[] signs, int permHd, int permNk, int permRep) {
        checkBlock(block);
        Objects.requireNonNull(signOrder, "signOrder");
        float[] s = null;
        if (signOrder == SignOrder.NONE) {
            if (signs != null) {
                throw new IllegalArgumentException("Hadamard signs given with sign order NONE");
            }
        } else {
            if (signs == null || signs.length == 0) {
                throw new IllegalArgumentException("Hadamard sign order " + signOrder + " requires a non-empty sign vector");
            }
            if (signs.length % block != 0) {
                throw new IllegalArgumentException("Hadamard sign vector length " + signs.length
                        + " is not a multiple of the block size " + block);
            }
            s = signs.clone();
            for (int i = 0; i < s.length; i++) {
                if (s[i] != 1.0f && s[i] != -1.0f) {
                    throw new IllegalArgumentException("Hadamard sign " + i + " is " + s[i] + ", expected +1 or -1");
                }
            }
        }
        int hd = 0;
        int nk = 0;
        int rep = 0;
        if (permRep > 1) {
            if (permHd < 1 || permNk < 1) {
                throw new IllegalArgumentException("Hadamard permutation needs positive head size and group count, got hd="
                        + permHd + " nk=" + permNk);
            }
            long w = (long) permHd * permNk * permRep;
            if (w > Integer.MAX_VALUE || w % block != 0) {
                throw new IllegalArgumentException("Hadamard permutation width " + w
                        + " is not an int multiple of the block size " + block);
            }
            if (s != null && s.length != w) {
                throw new IllegalArgumentException("Hadamard sign vector length " + s.length
                        + " differs from the permutation width " + w);
            }
            hd = permHd;
            nk = permNk;
            rep = permRep;
        } else if (permRep < 0) {
            throw new IllegalArgumentException("Hadamard permutation repeat must be non-negative, got " + permRep);
        }
        return new HadamardSpec(block, signOrder, s, hd, nk, rep);
    }

    private static void checkBlock(int block) {
        if (block <= 0 || (block & (block - 1)) != 0) {
            throw new IllegalArgumentException("Hadamard block size must be a positive power of two, got " + block);
        }
    }

    /** Block size (a power of two). */
    public int block() {
        return block;
    }

    /** Where the signs apply. */
    public SignOrder signOrder() {
        return signOrder;
    }

    /** True when a sign vector applies. */
    public boolean hasSigns() {
        return signs != null;
    }

    /** Copy of the sign vector, or null without signs. */
    public float[] signs() {
        return signs == null ? null : signs.clone();
    }

    /** Sign of row element {@code i}; 1 without signs. */
    public float sign(int i) {
        return signs == null ? 1.0f : signs[i];
    }

    /** Device name of the shared sign vector, {@code prism.hadamard.signs.<width>}, or null without signs. */
    public String signsName() {
        return signs == null ? null : SIGNS_NAME_PREFIX + signs.length;
    }

    /** Head size of the permutation (0 without one). */
    public int permHd() {
        return permHd;
    }

    /** Group count of the permutation (0 without one). */
    public int permNk() {
        return permNk;
    }

    /** Heads per group of the permutation (0 without one). */
    public int permRep() {
        return permRep;
    }

    /** True when the input is gathered from tiled to grouped head order first. */
    public boolean hasPermutation() {
        return permRep > 1;
    }

    /**
     * Row width the transform is bound to: the sign length or the permutation width, or 0 when any multiple of
     * {@link #block()} is accepted.
     */
    public int requiredWidth() {
        if (signs != null) {
            return signs.length;
        }
        return hasPermutation() ? permHd * permNk * permRep : 0;
    }

    /**
     * Validates a row width against this transform.
     *
     * @throws IllegalArgumentException when {@code width} is not a positive multiple of the block or differs
     *                                  from {@link #requiredWidth()}
     */
    public void checkWidth(int width) {
        if (width <= 0 || width % block != 0) {
            throw new IllegalArgumentException("Hadamard row width " + width + " is not a positive multiple of the block size "
                    + block);
        }
        int required = requiredWidth();
        if (required != 0 && required != width) {
            throw new IllegalArgumentException("Hadamard transform is bound to row width " + required + ", got " + width);
        }
    }

    /** True when {@value #KERNEL} supports the block size. */
    public boolean kernelSupported() {
        return block <= MAX_KERNEL_BLOCK;
    }

    /**
     * True when the GPU kernel may write its output over its input (same buffer and offset), which requires that
     * no permutation moves elements across blocks.
     */
    public boolean inPlaceAllowed() {
        return !hasPermutation();
    }

    /** Value of the kernel's {@code has_signs} argument. */
    public int kernelSignMode() {
        return signOrder.kernelCode();
    }

    /**
     * Input element of grouped row position {@code o}: {@code o = h + hd*(r + rep*k)} reads
     * {@code h + hd*(k + nk*r)} (ggml_permute(x, 0, 2, 1, 3) of the [hd, nk, rep] view); identity without a
     * permutation.
     */
    public int sourceIndex(int o) {
        if (permRep <= 1) {
            return o;
        }
        int h = o % permHd;
        int t = o / permHd;
        int r = t % permRep;
        int k = t / permRep;
        return h + permHd * (k + permNk * r);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof HadamardSpec)) {
            return false;
        }
        HadamardSpec that = (HadamardSpec) o;
        return block == that.block && signOrder == that.signOrder && permHd == that.permHd && permNk == that.permNk
                && permRep == that.permRep && Arrays.equals(signs, that.signs);
    }

    @Override
    public int hashCode() {
        int h = Objects.hash(block, signOrder, permHd, permNk, permRep);
        return 31 * h + Arrays.hashCode(signs);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("HadamardSpec{block=").append(block).append(", signs=").append(signOrder);
        if (signs != null) {
            sb.append('[').append(signs.length).append(']');
        }
        if (hasPermutation()) {
            sb.append(", perm=[").append(permHd).append(',').append(permNk).append(',').append(permRep).append(']');
        }
        return sb.append('}').toString();
    }
}
