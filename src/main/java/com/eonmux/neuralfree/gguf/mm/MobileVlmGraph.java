package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.spec.TensorRef;


/**
 * The MobileVLM projectors of the llava graph (models/llava.cpp): LDP (a GELU MLP, then two blocks of depthwise
 * 3x3 conv, layer norm, hardswish, squeeze-excite and a pointwise conv with layer norm, the first with a residual
 * at stride 1, the second at stride 2) and LDPv2 (a GELU MLP, a 2x2 average pool and a depthwise 3x3 position
 * encoding added back). The reference fixes the grid side at 24 (a 336-pixel tile of 14-pixel patches); this
 * executor runs the square grid of the file's {@code image_size}, which is 24 for the MobileVLM checkpoints.
 * Feature maps are pixel-major rows {@code [y * side + x][channel]}. Written once against {@link VisionOps}.
 * Package-private executor support.
 */
final class MobileVlmGraph {

    private MobileVlmGraph() {
    }

    /** LDP over the gathered patch rows of a square grid. */
    static float[][] ldp(VisionOps ops, Encoded enc) {
        int side = squareSide(enc, "ldp");
        int n = enc.nTokens;
        VisionOps.Block h = ops.linear("mm.model.mlp.1.weight", ops.upload(enc.rows, enc.width),
                "mm.model.mlp.1.bias");
        ops.activate(h, ActOp.GELU_TANH);
        h = ops.linear("mm.model.mlp.3.weight", h, "mm.model.mlp.3.bias");
        int d = h.width();
        VisionOps.Block block1 = block(ops, "mm.model.mb_block.1.block.", h, side, d, 1);
        ops.addBlock(block1, h);
        VisionOps.Block block2 = block(ops, "mm.model.mb_block.2.block.", block1, side, d, 2);
        return VisionRows.tokenRows(ops.download(block2), d);
    }

    /**
     * One LDP block: depthwise 3x3 conv at {@code stride} (zero padding 1), layer norm, hardswish, the
     * squeeze-excite gate (global average pool, {@code fc1} + ReLU, {@code fc2} + hardsigmoid) scaling the
     * channels, then the pointwise conv and its layer norm.
     */
    private static VisionOps.Block block(VisionOps ops, String p, VisionOps.Block x, int side, int d, int stride) {
        float eps = ops.hp().eps();
        int out = (side - 1) / stride + 1;
        int n = out * out;
        VisionOps.Block cur = ops.depthwise3x3(x, side, side, vectorRef(ops, p + "0.0.weight"), stride);
        ops.norm(cur, NormKind.LAYER, p + "0.1.weight", p + "0.1.bias", eps, n, d);
        ops.hardswish(cur);
        float[] values = ops.download(cur);
        float[] pooled = new float[d];
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < d; c++) {
                pooled[c] += values[t * d + c];
            }
        }
        for (int c = 0; c < d; c++) {
            pooled[c] /= n;
        }
        VisionOps.Block se = ops.linear(p + "1.fc1.weight", ops.upload(pooled, d), p + "1.fc1.bias");
        ops.activate(se, ActOp.RELU);
        se = ops.linear(p + "1.fc2.weight", se, p + "1.fc2.bias");
        ops.hardsigmoid(se);
        ops.mulRowScale(cur, ops.download(se));
        VisionOps.Block pw = ops.linear(p + "2.0.weight", cur, null);
        ops.norm(pw, NormKind.LAYER, p + "2.1.weight", p + "2.1.bias", eps, n, pw.width());
        return pw;
    }

    /** The tensor reference of projector vector {@code name}. */
    private static TensorRef vectorRef(VisionOps ops, String name) {
        TensorRef ref = ops.model().projectorVectors().get(name);
        if (ref == null) {
            throw new IllegalStateException("the projector resolved no vector " + name);
        }
        return ref;
    }

    /** LDPv2 over the gathered patch rows of a square grid. */
    static float[][] ldpv2(VisionOps ops, Encoded enc) {
        int side = squareSide(enc, "ldpv2");
        int n = enc.nTokens;
        VisionOps.Block h = ops.linear("mm.model.mlp.0.weight", ops.upload(enc.rows, enc.width),
                "mm.model.mlp.0.bias");
        ops.activate(h, ActOp.GELU_TANH);
        h = ops.linear("mm.model.mlp.2.weight", h, "mm.model.mlp.2.bias");
        int d = h.width();
        int half = (side - 2) / 2 + 1;
        float[] values = ops.download(h);
        float[] pooled = new float[half * half * d];
        for (int oy = 0; oy < half; oy++) {
            for (int ox = 0; ox < half; ox++) {
                int dst = (oy * half + ox) * d;
                for (int ky = 0; ky < 2; ky++) {
                    for (int kx = 0; kx < 2; kx++) {
                        int src = ((oy * 2 + ky) * side + ox * 2 + kx) * d;
                        for (int c = 0; c < d; c++) {
                            pooled[dst + c] += 0.25f * values[src + c];
                        }
                    }
                }
            }
        }
        VisionOps.Block pooledBlock = ops.upload(pooled, d);
        VisionOps.Block peg = ops.depthwise3x3(pooledBlock, half, half, vectorRef(ops, "mm.model.peg.0.weight"), 1);
        ops.addHost(peg, repeated(ops.hostVector("mm.model.peg.0.bias"), peg.n() * peg.width()));
        ops.addBlock(peg, pooledBlock);
        return VisionRows.tokenRows(ops.download(peg), d);
    }

    /** {@code bias} tiled to {@code total} values (a row bias over the whole block). */
    private static float[] repeated(float[] bias, int total) {
        if (bias.length == 0 || total % bias.length != 0) {
            throw new IllegalArgumentException("a bias of " + bias.length + " values does not tile to " + total);
        }
        if (total == bias.length) {
            return bias;
        }
        float[] out = new float[total];
        for (int at = 0; at < total; at += bias.length) {
            System.arraycopy(bias, 0, out, at, bias.length);
        }
        return out;
    }

    /**
     * {@code ggml_conv_2d_dw} with a 3x3 kernel and zero padding 1 over a {@code width x height} map of raster
     * rows: weight slot {@code (c * 3 + ky) * 3 + kx} multiplies pixel {@code (x * stride + kx - 1,
     * y * stride + ky - 1)} of channel {@code c}.
     */
    static float[] depthwise3x3(float[] x, int width, int height, int d, float[] weight, int stride) {
        int outW = (width - 1) / stride + 1;
        int outH = (height - 1) / stride + 1;
        float[] y = new float[outW * outH * d];
        for (int oy = 0; oy < outH; oy++) {
            for (int ox = 0; ox < outW; ox++) {
                int dst = (oy * outW + ox) * d;
                for (int ky = 0; ky < 3; ky++) {
                    int iy = oy * stride + ky - 1;
                    if (iy < 0 || iy >= height) {
                        continue;
                    }
                    for (int kx = 0; kx < 3; kx++) {
                        int ix = ox * stride + kx - 1;
                        if (ix < 0 || ix >= width) {
                            continue;
                        }
                        int src = (iy * width + ix) * d;
                        for (int c = 0; c < d; c++) {
                            y[dst + c] += weight[(c * 3 + ky) * 3 + kx] * x[src + c];
                        }
                    }
                }
            }
        }
        return y;
    }

    private static int squareSide(Encoded enc, String family) {
        enc.requireGrid("the " + family + " projector");
        if (enc.pw != enc.ph || enc.pw < 2) {
            throw new IllegalArgumentException("the " + family + " projector needs a square patch grid of side"
                    + " 2 or more, got " + enc.pw + "x" + enc.ph);
        }
        return enc.pw;
    }
}
