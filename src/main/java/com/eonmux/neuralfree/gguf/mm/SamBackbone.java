package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;

/**
 * The SAM ViT-B backbone of the DeepSeek-OCR encoders on the CPU (models/deepseekocr.cpp
 * {@code clip_graph_deepseekocr::build_sam}): the biased patch conv, the absolute position grid (bicubic-resized to
 * the patch grid when the sizes differ), the blocks (layer norm at 1e-6, attention over {@code window x window}
 * windows of the zero-padded grid or, at blocks 2, 5, 8 and 11, over the whole grid, each with the decomposed
 * relative-position bias of its row and column tables, linearly resized when their length is not
 * {@code 2 * side - 1}; then the MLP of the file's activation), then the neck (1x1 conv, channel layer norm, 3x3
 * conv, channel layer norm) and two stride-2 3x3 convs. Rows are raster order ({@code y * side + x}). Weights
 * resolve from the projector tensors ({@link SamTensors}). Not thread-safe across calls.
 */
final class SamBackbone {

    /** The layer-norm epsilon of every SAM norm. */
    static final float EPS = 1e-6f;

    /** The blocks that attend over the whole grid ({@code global_attn_indices}). */
    private static final int[] GLOBAL_BLOCKS = {2, 5, 8, 11};

    private final ClipVisionWeights w;
    private final VisionFamilyHparams.Sam sam;
    private final int patch;
    private final int dHead;
    private final float scale;
    private final ActOp act;

    /** The backbone over {@code hp}'s SAM keys and the resolved weights {@code w}. */
    SamBackbone(ClipHparams hp, ClipVisionWeights w) {
        this.w = w;
        this.sam = hp.visionFamily().sam();
        if (sam == null) {
            throw new IllegalStateException("projector type " + hp.projectorType() + " has no SAM backbone keys");
        }
        CpuTensorRef kernel = w.matrix(SamTensors.PATCH);
        this.patch = (int) Math.round(Math.sqrt((double) (kernel.rows() * kernel.cols()) / (3L * sam.nEmbd())));
        this.dHead = sam.nEmbd() / sam.nHead();
        this.scale = (float) (1.0 / Math.sqrt(dHead));
        this.act = sam.ffnOp().unary();
    }

    /** The output of one image: {@code side * side} rows of {@code channels} values, raster order. */
    static final class Output {
        final float[] rows;
        final int side;
        final int channels;

        Output(float[] rows, int side, int channels) {
            this.rows = rows;
            this.side = side;
            this.channels = channels;
        }
    }

    /**
     * Runs the backbone over a square image whose side is a multiple of the patch kernel.
     *
     * @throws IllegalArgumentException when the image is not square or its side is not a patch multiple (named)
     */
    Output forward(ClipImage image) {
        int g = patchSide(image);
        int e = sam.nEmbd();
        int n = g * g;
        float[] x = linear(kernel(SamTensors.PATCH, e), ClipPatches.raster(image, patch, g, g), n,
                w.vector(SamTensors.PATCH_BIAS));
        float[] pos = w.vector(SamTensors.POS);
        int s = (int) Math.round(Math.sqrt((double) pos.length / e));
        CpuOps.add(s == g ? pos : Glm4vGraph.resizeBicubic(pos, e, s, g, g), 0, x, 0, x, 0, n * e);
        for (int il = 0; il < sam.nLayer(); il++) {
            block(x, g, il);
        }
        return neck(x, g);
    }

    private int patchSide(ClipImage image) {
        if (image.width() != image.height() || image.width() % patch != 0) {
            throw new IllegalArgumentException("the SAM backbone needs a square image whose side is a multiple of"
                    + " the patch size " + patch + ", got " + image.width() + "x" + image.height());
        }
        return image.width() / patch;
    }

    /** True for the blocks that attend over the whole grid. */
    static boolean isGlobal(int il) {
        for (int g : GLOBAL_BLOCKS) {
            if (g == il) {
                return true;
            }
        }
        return false;
    }

    /** One block in place over the {@code g x g} rows of {@code x}. */
    private void block(float[] x, int g, int il) {
        int e = sam.nEmbd();
        int n = g * g * e;
        float[] cur = new float[n];
        CpuOps.layerNorm(x, 0, cur, 0, g * g, e, w.vector(SamTensors.block(il, "pre_ln.weight")),
                w.vector(SamTensors.block(il, "pre_ln.bias")), EPS);
        float[] attn;
        if (isGlobal(il)) {
            attn = attention(cur, 1, g, il);
        } else {
            int win = sam.window();
            int nw = (g + win - 1) / win;
            float[] parts = attention(SamWindows.partition(cur, e, g, win), nw * nw, win, il);
            attn = SamWindows.unpartition(parts, e, g, win);
        }
        CpuOps.add(attn, 0, x, 0, x, 0, n);
        CpuOps.layerNorm(x, 0, cur, 0, g * g, e, w.vector(SamTensors.block(il, "post_ln.weight")),
                w.vector(SamTensors.block(il, "post_ln.bias")), EPS);
        float[] up = linear(w.matrix(SamTensors.block(il, "mlp.lin1.weight")), cur, g * g,
                w.vector(SamTensors.block(il, "mlp.lin1.bias")));
        VisionRows.activate(act, up);
        float[] down = linear(w.matrix(SamTensors.block(il, "mlp.lin2.weight")), up, g * g,
                w.vector(SamTensors.block(il, "mlp.lin2.bias")));
        CpuOps.add(down, 0, x, 0, x, 0, n);
    }

    /**
     * Multi-head attention within each of {@code nBatch} {@code side x side} groups of rows, scores
     * {@code q.k * scale} plus the relative-position bias {@code q . Rw[qx - kx + side - 1] +
     * q . Rh[qy - ky + side - 1]}, then the output projection.
     */
    private float[] attention(float[] x, int nBatch, int side, int il) {
        int e = sam.nEmbd();
        int t = side * side;
        float[] qkv = linear(w.matrix(SamTensors.block(il, "attn.qkv.weight")), x, nBatch * t,
                w.vector(SamTensors.block(il, "attn.qkv.bias")));
        float[] relW = SamWindows.relTable(w.vector(SamTensors.block(il, "attn.pos_w.weight")), dHead, side);
        float[] relH = SamWindows.relTable(w.vector(SamTensors.block(il, "attn.pos_h.weight")), dHead, side);
        float[] out = new float[nBatch * t * e];
        int nHead = sam.nHead();
        CpuParallel.parallelFor(nBatch * nHead * t, Math.max(1, 4096 / t), (from, to) -> {
            float[] scores = new float[t];
            float[] rw = new float[side];
            float[] rh = new float[side];
            for (int idx = from; idx < to; idx++) {
                int b = idx / (nHead * t);
                int h = idx / t % nHead;
                int tq = idx % t;
                int rowBase = b * t;
                int qOff = (rowBase + tq) * 3 * e + h * dHead;
                for (int k = 0; k < side; k++) {
                    rw[k] = VisionRows.dot(relW, (tq % side - k + side - 1) * dHead, qkv, qOff, dHead);
                    rh[k] = VisionRows.dot(relH, (tq / side - k + side - 1) * dHead, qkv, qOff, dHead);
                }
                attendRow(qkv, qOff, rowBase, h, t, side, rw, rh, scores, out, (rowBase + tq) * e + h * dHead);
            }
        });
        return linear(w.matrix(SamTensors.block(il, "attn.out.weight")), out, nBatch * t,
                w.vector(SamTensors.block(il, "attn.out.bias")));
    }

    /** The softmax row of one (head, query) pair over its group's keys into {@code out}. */
    private void attendRow(float[] qkv, int qOff, int rowBase, int h, int t, int side, float[] rw, float[] rh,
                           float[] scores, float[] out, int dst) {
        int e = sam.nEmbd();
        float max = Float.NEGATIVE_INFINITY;
        for (int k = 0; k < t; k++) {
            float kq = VisionRows.dot(qkv, qOff, qkv, (rowBase + k) * 3 * e + e + h * dHead, dHead);
            float s = kq * scale + (rw[k % side] + rh[k / side]);
            scores[k] = s;
            max = Math.max(max, s);
        }
        double sum = 0;
        for (int k = 0; k < t; k++) {
            double ex = Math.exp(scores[k] - max);
            scores[k] = (float) ex;
            sum += ex;
        }
        float inv = (float) (1.0 / sum);
        for (int k = 0; k < t; k++) {
            float p = scores[k] * inv;
            int vOff = (rowBase + k) * 3 * e + 2 * e + h * dHead;
            for (int c = 0; c < dHead; c++) {
                out[dst + c] += p * qkv[vOff + c];
            }
        }
    }

    /** The neck and the two stride-2 convs over the {@code g x g} rows of {@code x}. */
    private Output neck(float[] x, int g) {
        int e = sam.nEmbd();
        int n = g * g;
        CpuTensorRef neck0 = w.matrix(SamTensors.NECK_0);
        int c = Math.toIntExact(neck0.rows() * neck0.cols() / e);
        float[] cur = linear(CpuVitTower.matrix(neck0, c, e), x, n, null);
        CpuOps.layerNorm(cur, 0, cur, 0, n, c, w.vector(SamTensors.NECK_1 + ".weight"),
                w.vector(SamTensors.NECK_1 + ".bias"), EPS);
        int k = SamTensors.CONV_KERNEL;
        cur = linear(kernel(SamTensors.NECK_2, c), PixelShuffle.im2col(cur, c, g, g, k, 1, 1), n, null);
        CpuOps.layerNorm(cur, 0, cur, 0, n, c, w.vector(SamTensors.NECK_3 + ".weight"),
                w.vector(SamTensors.NECK_3 + ".bias"), EPS);
        int g2 = PixelShuffle.convOut(g, k, 2, 1);
        int c2 = outChannels(SamTensors.NET_2, c);
        cur = linear(kernel(SamTensors.NET_2, c2), PixelShuffle.im2col(cur, c, g, g, k, 2, 1), g2 * g2, null);
        int g3 = PixelShuffle.convOut(g2, k, 2, 1);
        int c3 = outChannels(SamTensors.NET_3, c2);
        cur = linear(kernel(SamTensors.NET_3, c3), PixelShuffle.im2col(cur, c2, g2, g2, k, 2, 1), g3 * g3, null);
        return new Output(cur, g3, c3);
    }

    /** The output channels of a 3x3 kernel over {@code in} channels. */
    private int outChannels(String name, int in) {
        CpuTensorRef r = w.matrix(name);
        return Math.toIntExact(r.rows() * r.cols() / ((long) SamTensors.CONV_KERNEL * SamTensors.CONV_KERNEL * in));
    }

    /** Kernel {@code name} read as its {@code [out][values / out]} im2col matrix. */
    private CpuTensorRef kernel(String name, int out) {
        CpuTensorRef r = w.matrix(name);
        return CpuVitTower.matrix(r, out, r.rows() * r.cols() / out);
    }

    /** {@code n} rows of {@code m}'s output width: {@code m . x}, plus {@code bias} when given. */
    private float[] linear(CpuTensorRef m, float[] x, int n, float[] bias) {
        float[] y = new float[Math.multiplyExact(n, Math.toIntExact(m.rows()))];
        w.matVec.matmul(m, x, 0, y, 0, n);
        VisionRows.addBias(y, bias);
        return y;
    }
}
