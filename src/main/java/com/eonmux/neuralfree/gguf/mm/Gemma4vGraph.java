package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

/**
 * The Gemma 4 vision graphs. models/gemma4v.cpp {@code clip_graph_gemma4v}: the image scaled to {@code 2x - 1},
 * the bias-free patch convolution, the learned column and row position tables (the two halves of
 * {@code v.position_embd}, gathered by each patch's column and row), and the RMS-norm ViT whose attention rotates
 * Q and K with the NEOX-mode 2D RoPE (columns on the first half of each head, rows on the second, base 100),
 * normalizes every value head, attends at scale 1 and runs every block matrix through its clippable-linear bounds;
 * then the projector: the {@code n_merge x n_merge} average pool, the {@code sqrt(n_embd)} scale, the optional
 * standardization {@code (x - v.std_bias) * v.std_scale}, a weightless RMS norm and the clippable
 * {@code mm.input_projection}. models/gemma4uv.cpp {@code clip_graph_gemma4uv}: no blocks; every
 * {@code patch_size * n_merge} patch vector is layer-normed, projected with its bias and layer-normed, the
 * position tables are added and layer-normed, then a weightless RMS norm and {@code mm.input_projection}. Written
 * once against {@link VisionOps}. Package-private executor support.
 */
final class Gemma4vGraph {

    /** RoPE base of the vision attention (clip.cpp load_hparams). */
    static final double ROPE_THETA = 100.0;
    /** Epsilon of the unified embedder's layer norms, PyTorch's {@code nn.LayerNorm} default. */
    static final float UNIFIED_EPS = 1e-5f;
    /** The projection into the text model's embedding space. */
    static final String INPUT_PROJECTION = "mm.input_projection.weight";
    /** The standardization pair of the projector input. */
    static final String STD_BIAS = "v.std_bias";
    static final String STD_SCALE = "v.std_scale";
    /** The clippable-linear bounds of {@link #INPUT_PROJECTION}: input min / max, output min / max. */
    static final String[] PROJECTION_BOUNDS = {"mm.input_projection.input_min", "mm.input_projection.input_max",
            "mm.input_projection.output_min", "mm.input_projection.output_max"};

    private static final float[] UNBOUNDED = {-Float.MAX_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE,
            Float.MAX_VALUE};

    private Gemma4vGraph() {
    }

    // ------------------------------------------------------------------ gemma4v

    /**
     * The gemma4v encoder rows of {@code image}, raster order.
     *
     * @throws IllegalArgumentException when the patch grid is not a multiple of the pool kernel, exceeds the
     *                                  position tables, or the head width is not a multiple of 4
     */
    static Encoded encode(VisionOps ops, ClipImage image) {
        int[] grid = ClipVit.grid(ops.hp(), image);
        int pw = grid[0];
        int ph = grid[1];
        requirePoolGrid(pw, ph, ops.hp().nMerge());
        PixtralGraph.requireRopeHeadDim(ops.model().dHead(), "gemma4v");
        VisionOps.Block x = ops.patchRows(signedUnit(image), pw, ph);
        int n = pw * ph;
        int[] cols = new int[n];
        int[] rows = new int[n];
        addAxisTables(ops, x, pw, ph, cols, rows);
        VisionOps.TowerOptions opts = VisionOps.TowerOptions.NONE
                .withRope(VisionOps.RopePlan.neoxHalves(cols, rows, ROPE_THETA))
                .withAttnScale(1f)
                .withValueNorm();
        VisionOps.Block out = ops.tower(x, n, NormKind.RMS, opts);
        return new Encoded(ops.download(out), n, pw, ph);
    }

    /**
     * The gemma4v projector: the average pool over {@code n_merge x n_merge} patches, the {@code sqrt(n_embd)}
     * scale, the optional standardization, the weightless RMS norm, then the clippable input projection.
     */
    static float[][] project(VisionOps ops, Encoded enc) {
        enc.requireGrid("the gemma4v projector");
        ClipHparams hp = ops.hp();
        int k = hp.nMerge();
        int nEmbd = hp.nEmbd();
        requirePoolGrid(enc.pw, enc.ph, k);
        float[] pooled = avgPool(enc.rows, enc.pw, enc.ph, nEmbd, k);
        int n = (enc.pw / k) * (enc.ph / k);
        float scale = (float) Math.sqrt(nEmbd);
        for (int i = 0; i < pooled.length; i++) {
            pooled[i] *= scale;
        }
        standardize(ops, pooled, nEmbd);
        CpuVitTower.normRows(pooled, pooled, n, nEmbd, null, null, NormKind.RMS, hp.eps());
        float[] bounds = projectionBounds(ops);
        clamp(pooled, bounds[0], bounds[1]);
        float[] out = ops.download(ops.linear(INPUT_PROJECTION, ops.upload(pooled, nEmbd), null));
        clamp(out, bounds[2], bounds[3]);
        return VisionRows.tokenRows(out, ops.model().outputDim());
    }

    /** {@code ggml_scale_bias(x, 2, -1)}: the normalized image mapped as {@code 2x - 1}. */
    private static ClipImage signedUnit(ClipImage image) {
        float[] px = image.pixels().clone();
        for (int i = 0; i < px.length; i++) {
            px[i] = px[i] * 2f - 1f;
        }
        return ClipImage.of(image.width(), image.height(), px);
    }

    /** {@code (x - v.std_bias) * v.std_scale} in place over rows of {@code width}, when the file carries the pair. */
    private static void standardize(VisionOps ops, float[] x, int width) {
        float[] bias = ops.hostVectorOrNull(STD_BIAS);
        float[] scale = ops.hostVectorOrNull(STD_SCALE);
        if (bias == null || scale == null) {
            return;
        }
        for (int i = 0; i < x.length; i++) {
            int c = i % width;
            x[i] = (x[i] - bias[c]) * scale[c];
        }
    }

    /** The input projection's bounds {@code {inMin, inMax, outMin, outMax}}, unbounded where the file has none. */
    private static float[] projectionBounds(VisionOps ops) {
        float[] b = UNBOUNDED.clone();
        for (int i = 0; i < PROJECTION_BOUNDS.length; i++) {
            float[] v = ops.hostVectorOrNull(PROJECTION_BOUNDS[i]);
            if (v != null) {
                b[i] = v[0];
            }
        }
        return b;
    }

    private static void clamp(float[] x, float lo, float hi) {
        for (int i = 0; i < x.length; i++) {
            x[i] = Math.max(Math.min(x[i], hi), lo);
        }
    }

    /**
     * {@code ggml_pool_2d(AVG, k, k, k, k)} over the {@code pw x ph} grid of {@code width}-wide rows: the
     * {@code (pw / k) x (ph / k)} raster of window means.
     */
    static float[] avgPool(float[] rows, int pw, int ph, int width, int k) {
        int ow = pw / k;
        int oh = ph / k;
        float[] out = new float[ow * oh * width];
        float area = k * k;
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                int dst = (oy * ow + ox) * width;
                for (int c = 0; c < width; c++) {
                    float sum = 0f;
                    for (int dy = 0; dy < k; dy++) {
                        for (int dx = 0; dx < k; dx++) {
                            sum += rows[((oy * k + dy) * pw + ox * k + dx) * width + c];
                        }
                    }
                    out[dst + c] = sum / area;
                }
            }
        }
        return out;
    }

    private static void requirePoolGrid(int pw, int ph, int k) {
        if (k <= 0 || pw % k != 0 || ph % k != 0) {
            throw new IllegalArgumentException("the gemma4v pool needs a patch grid whose sides are multiples of"
                    + " the pool size " + k + ", got " + pw + "x" + ph);
        }
    }

    // ------------------------------------------------------------------ gemma4uv

    /**
     * The gemma4uv embedder rows of {@code image}, raster order over the {@code patch_size * n_merge} patches.
     *
     * @throws IllegalArgumentException when the image is not a multiple of the merged patch or its grid exceeds the
     *                                  position tables
     */
    static Encoded encodeUnified(VisionOps ops, ClipImage image) {
        int k = ops.model().patchKernelSize();
        if (image.width() % k != 0 || image.height() % k != 0) {
            throw new IllegalArgumentException("image size " + image.width() + "x" + image.height() + " is not a"
                    + " multiple of the gemma4uv patch size " + k);
        }
        int pw = image.width() / k;
        int ph = image.height() / k;
        int n = pw * ph;
        int nEmbd = ops.hp().nEmbd();
        float[] patches = ClipPatches.raster(image, k, pw, ph);
        CpuVitTower.normRows(patches, patches, n, ClipPatches.rowWidth(k), ops.hostVector("v.patch_norm.1.weight"),
                ops.hostVector("v.patch_norm.1.bias"), NormKind.LAYER, UNIFIED_EPS);
        VisionOps.Block x = ops.patchEmbed(patches, n);
        ops.norm(x, NormKind.LAYER, "v.patch_norm.2.weight", "v.patch_norm.2.bias", UNIFIED_EPS, n, nEmbd);
        addAxisTables(ops, x, pw, ph, new int[n], new int[n]);
        ops.norm(x, NormKind.LAYER, "v.patch_norm.3.weight", "v.patch_norm.3.bias", UNIFIED_EPS, n, nEmbd);
        return new Encoded(ops.download(x), n, pw, ph);
    }

    /** The gemma4uv projector: the weightless RMS norm, then the input projection. */
    static float[][] projectUnified(VisionOps ops, Encoded enc) {
        int nEmbd = ops.hp().nEmbd();
        VisionOps.Block x = ops.upload(enc.rows, nEmbd);
        ops.normByRef(x, NormKind.RMS, null, null, ops.hp().eps(), enc.nTokens, nEmbd);
        VisionOps.Block out = ops.linear(INPUT_PROJECTION, x, null);
        return VisionRows.tokenRows(ops.download(out), ops.model().outputDim());
    }

    // ------------------------------------------------------------------ shared

    /**
     * Adds the column table row of every patch's column and the row table row of its row (the two
     * {@code [n_embd, P]} halves of {@code v.position_embd}), filling {@code cols} and {@code rows} with them.
     *
     * @throws IllegalArgumentException when the grid has more columns or rows than the tables hold
     */
    private static void addAxisTables(VisionOps ops, VisionOps.Block x, int pw, int ph, int[] cols, int[] rows) {
        TensorRef pos = ops.model().posEmbd();
        int tableRows = (int) pos.ne(1);
        if (pw > tableRows || ph > tableRows) {
            throw new IllegalArgumentException("the " + pw + "x" + ph + " patch grid exceeds the " + tableRows
                    + " column and row positions of " + pos.name());
        }
        int n = pw * ph;
        int[] rowIdx = new int[n];
        for (int i = 0; i < n; i++) {
            cols[i] = i % pw;
            rows[i] = i / pw;
            rowIdx[i] = tableRows + rows[i];
        }
        int nEmbd = ops.hp().nEmbd();
        ops.addGathered(x, pos, cols, nEmbd);
        ops.addGathered(x, pos, rowIdx, nEmbd);
    }
}
