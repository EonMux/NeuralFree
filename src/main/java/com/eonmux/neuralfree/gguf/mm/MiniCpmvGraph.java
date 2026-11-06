package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

/**
 * The MiniCPM-V graph of every resampler version (models/minicpmv.cpp {@code clip_graph_minicpmv}): a SigLIP-style
 * ViT over any patch grid whose learned position rows are gathered from a 70x70 bucket table, then the resampler,
 * one cross-attention of {@code clip.minicpmv_query_num} learned queries over the encoder rows with 2D sin-cos
 * positions added to the keys, a post norm and the output projection. The versions differ only in the query count
 * (and the text side); every tile of the llava-uhd slicing encodes to the same query count. Written once against
 * {@link VisionOps}. Package-private executor support.
 */
final class MiniCpmvGraph {

    /** Bucket count per side of the learned position table. */
    static final int POS_BUCKETS = 70;
    private static final double OMEGA_BASE = 10000.0;

    private MiniCpmvGraph() {
    }

    /** The encoder rows of {@code image}, raster order. */
    static Encoded encode(VisionOps ops, ClipImage image) {
        int[] grid = ClipVit.grid(ops.hp(), image);
        int pw = grid[0];
        int ph = grid[1];
        VisionOps.Block x = ops.patchRows(image, pw, ph);
        addBucketPositions(ops, x, pw, ph);
        x = ops.tower(x, pw * ph, NormKind.LAYER, VisionOps.TowerOptions.NONE);
        return new Encoded(ops.download(x), pw * ph, pw, ph);
    }

    /**
     * Adds the learned position row of every patch of the {@code pw x ph} grid: the bucket
     * {@code floor(70 * y / ph) * 70 + floor(70 * x / pw)} of {@code v.position_embd.weight}.
     *
     * @throws IllegalArgumentException when a bucket is beyond the table's rows
     */
    static void addBucketPositions(VisionOps ops, VisionOps.Block x, int pw, int ph) {
        int nEmbd = ops.hp().nEmbd();
        TensorRef posEmbd = ops.model().posEmbd();
        int posRows = positionRows(ops, posEmbd, nEmbd);
        int[] idx = new int[pw * ph];
        for (int y = 0; y < ph; y++) {
            int bucketY = (int) Math.floor(POS_BUCKETS * (double) y / ph);
            for (int x0 = 0; x0 < pw; x0++) {
                int bucketX = (int) Math.floor(POS_BUCKETS * (double) x0 / pw);
                int bucket = bucketY * POS_BUCKETS + bucketX;
                if (bucket >= posRows) {
                    throw new IllegalArgumentException("position bucket " + bucket + " is outside the " + posRows
                            + " rows of v.position_embd.weight (a " + POS_BUCKETS + "x" + POS_BUCKETS
                            + " table is expected)");
                }
                idx[y * pw + x0] = bucket;
            }
        }
        ops.addGathered(x, posEmbd, idx, nEmbd);
    }

    /** The row count of the learned position table. */
    private static int positionRows(VisionOps ops, TensorRef posEmbd, int nEmbd) {
        float[] host = ops.hostWeight(posEmbd);
        if (host == null || host.length % nEmbd != 0) {
            throw new IllegalArgumentException("v.position_embd.weight is absent or does not split into rows of "
                    + nEmbd);
        }
        return host.length / nEmbd;
    }

    /** The resampler over the encoder rows: {@code [query_num][d]} rows, {@code d} the resampler width. */
    static float[][] resample(VisionOps ops, Encoded enc) {
        enc.requireGrid("the minicpmv resampler");
        float eps = ops.hp().eps();
        int d = ops.model().outputDim();
        int nq = ops.hp().minicpmvQueryNum();
        int n = enc.nTokens;

        VisionOps.Block q = ops.upload(ops.hostVector("resampler.query").clone(), d);
        ops.norm(q, NormKind.LAYER, "resampler.ln_q.weight", "resampler.ln_q.bias", eps, nq, d);
        VisionOps.Block v = ops.linear("resampler.kv.weight", ops.upload(enc.rows, enc.width), null);
        ops.norm(v, NormKind.LAYER, "resampler.ln_kv.weight", "resampler.ln_kv.bias", eps, n, d);
        VisionOps.Block k = ops.upload(sinCosPositions(n, enc.pw, d), d);
        ops.addBlock(k, v);

        VisionOps.Block qp = ops.linear("resampler.attn.q.weight", q, "resampler.attn.q.bias");
        VisionOps.Block kp = ops.linear("resampler.attn.k.weight", k, "resampler.attn.k.bias");
        VisionOps.Block vp = ops.linear("resampler.attn.v.weight", v, "resampler.attn.v.bias");
        int dh = ClipProjectorTensors.RESAMPLER_HEAD_DIM;
        float scale = (float) (1.0 / Math.sqrt(dh));
        VisionOps.Block attn = ops.attend(qp, 0, nq, kp, vp, 0, n, d, d / dh, 1, dh, scale);
        attn = ops.linear("resampler.attn.out.weight", attn, "resampler.attn.out.bias");
        ops.norm(attn, NormKind.LAYER, "resampler.ln_post.weight", "resampler.ln_post.bias", eps, nq, d);
        VisionOps.Block out = ops.linear("resampler.proj.weight", attn, null);
        return VisionRows.tokenRows(ops.download(out), d);
    }

    /**
     * The 2D sin-cos key positions of the resampler: for patch {@code t} at column {@code t % pw} and row
     * {@code t / pw}, the row {@code [sin(col * omega), cos(col * omega), sin(row * omega), cos(row * omega)]},
     * {@code omega[i] = 10000^(-i / (d/4))}.
     */
    static float[] sinCosPositions(int n, int pw, int d) {
        int quarter = d / 4;
        float[] omega = new float[quarter];
        for (int i = 0; i < quarter; i++) {
            omega[i] = (float) (1.0 / Math.pow(OMEGA_BASE, (double) i / quarter));
        }
        float[] pe = new float[n * d];
        for (int t = 0; t < n; t++) {
            float col = t % pw;
            float row = t / pw;
            int base = t * d;
            for (int i = 0; i < quarter; i++) {
                float tx = omega[i] * col;
                float ty = omega[i] * row;
                pe[base + i] = (float) Math.sin(tx);
                pe[base + quarter + i] = (float) Math.cos(tx);
                pe[base + 2 * quarter + i] = (float) Math.sin(ty);
                pe[base + 3 * quarter + i] = (float) Math.cos(ty);
            }
        }
        return pe;
    }
}
