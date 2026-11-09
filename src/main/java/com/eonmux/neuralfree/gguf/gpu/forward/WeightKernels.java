package com.eonmux.neuralfree.gguf.gpu.forward;

/** Kernel names of one weight type. */
final class WeightKernels {
    final String matvec;
    final String matvecAdd;
    final String batch;
    final String batch2d;
    final String tiled;
    final String embed;
    final String batchEmbed;
    /** Integer-dot (Q8_1 activation) single-token matvec, or null when the type has none. */
    final String matvecQ8;
    /** Integer-dot (Q8_1 activation) accumulating matvec, or null when the type has none. */
    final String matvecAddQ8;
    /** Integer-dot (Q8_1 activation) multi-column matvec of up to {@code NF_MVQN_MAX} token rows,
     * or null when the type has none. */
    final String matvecNQ8;
    /** Float-input multi-column matvec of up to {@code NF_MVQN_MAX} token rows, or null when the
     * type has none. */
    final String matvecN;

    WeightKernels(String matvec, String matvecAdd, String batch, String batch2d, String tiled, String embed,
                  String batchEmbed) {
        this(matvec, matvecAdd, batch, batch2d, tiled, embed, batchEmbed, null, null, null, null);
    }

    WeightKernels(String matvec, String matvecAdd, String batch, String batch2d, String tiled, String embed,
                  String batchEmbed, String matvecQ8, String matvecAddQ8) {
        this(matvec, matvecAdd, batch, batch2d, tiled, embed, batchEmbed, matvecQ8, matvecAddQ8, null, null);
    }

    WeightKernels(String matvec, String matvecAdd, String batch, String batch2d, String tiled, String embed,
                  String batchEmbed, String matvecQ8, String matvecAddQ8, String matvecNQ8) {
        this(matvec, matvecAdd, batch, batch2d, tiled, embed, batchEmbed, matvecQ8, matvecAddQ8, matvecNQ8, null);
    }

    WeightKernels(String matvec, String matvecAdd, String batch, String batch2d, String tiled, String embed,
                  String batchEmbed, String matvecQ8, String matvecAddQ8, String matvecNQ8, String matvecN) {
        this.matvec = matvec;
        this.matvecAdd = matvecAdd;
        this.batch = batch;
        this.batch2d = batch2d;
        this.tiled = tiled;
        this.embed = embed;
        this.batchEmbed = batchEmbed;
        this.matvecQ8 = matvecQ8;
        this.matvecAddQ8 = matvecAddQ8;
        this.matvecNQ8 = matvecNQ8;
        this.matvecN = matvecN;
    }

    static WeightKernels standard(String t) {
        return new WeightKernels(t + "_MatVec", t + "_MatVecAdd", t + "_BatchMatMul", t + "_BatchMatMul_2D",
                t + "_MatMul_Tiled", t + "_EmbeddingLookup", t + "_BatchEmbeddingLookup");
    }

    /** {@link #standard} plus the {@code {T}_MatVec_Q8} integer-dot forms. */
    static WeightKernels int8(String t) {
        return int8(standard(t));
    }

    /** {@code f} with its {@code _MatVec_Q8}, {@code _MatVecAdd_Q8} and {@code _MatVecN_Q8} integer-dot
     * forms added; every other kernel name is {@code f}'s, so legacy-named families (Q8_0, MXFP4) keep their
     * batch kernels. */
    static WeightKernels int8(WeightKernels f) {
        return new WeightKernels(f.matvec, f.matvecAdd, f.batch, f.batch2d, f.tiled, f.embed, f.batchEmbed,
                f.matvec + "_Q8", f.matvecAdd + "_Q8", f.matvec + "N_Q8", f.matvecN);
    }

    /** {@code f} with its float-input multi-column matvec {@code {T}_MatVecN} added. */
    static WeightKernels multiColumn(WeightKernels f) {
        return new WeightKernels(f.matvec, f.matvecAdd, f.batch, f.batch2d, f.tiled, f.embed, f.batchEmbed,
                f.matvecQ8, f.matvecAddQ8, f.matvecNQ8, f.matvec + "N");
    }
}
