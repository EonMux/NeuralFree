package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.ATTN_NORM;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.BN1;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.BN2;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.CONV_EXP;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.CONV_PWL;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.DW_MID;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.DW_MID_NORM;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.DW_START;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.DW_START_NORM;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.KEY;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.KEY_DOWN;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.KEY_NORM;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.LAYER_SCALE;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.OUTPUT;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.PW_EXP;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.PW_EXP_NORM;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.PW_PROJ;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.PW_PROJ_NORM;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.QUERY;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.VALUE;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.VALUE_DOWN;
import static com.eonmux.neuralfree.gguf.mm.MobileNetV5Layout.VALUE_NORM;

/**
 * The Gemma 3n vision graph (models/mobilenetv5.cpp {@code clip_graph_mobilenetv5}), written once against
 * {@link VisionOps} so the CPU and GPU executors run the same graph: the SAME-padded stride-2
 * stem conv with its bias, norm and GELU; the {@link MobileNetV5Layout} blocks, the first of every stage
 * downsampling by 2; the multi-scale fusion adapter (the fusion features upscaled to the first one's size, concatenated
 * along the channels, the pointwise expansion with norm and GELU and the projection with norm, the average pool
 * down to {@value #OUTPUT_SIDE} pixels a side, the closing norm); then the embedder: the pixels in raster order
 * scaled by {@code sqrt(n_embd)}, the RMS norm times {@code mm.soft_emb_norm}, {@code mm.input_projection} and a
 * weightless RMS norm. Every norm is the per-pixel RMS norm over the channels at epsilon {@value #EPS}, every
 * activation GELU. Feature maps are pixel-row blocks; the convolutions, depthwise convolutions and pools run on
 * the executor ({@link VisionOps#im2colPadded}, {@link VisionOps#depthwise}, {@link VisionOps#avgPool}), the
 * 1x1 convolutions as matmuls over the map in place, the fusion adapter's upscale and channel concat on the host.
 * Package-private executor support.
 */
final class MobileNetV5Graph {

    /** The side the fusion adapter pools its output down to. */
    static final int OUTPUT_SIDE = 16;

    /** Epsilon of every norm of the graph. */
    static final float EPS = 1e-6f;

    private MobileNetV5Graph() {
    }

    /** A feature map: {@code w x h} pixel rows of {@code c} channels. */
    private static final class Features {
        final VisionOps.Block x;
        final int w;
        final int h;
        final int c;

        Features(VisionOps.Block x, int w, int h, int c) {
            this.x = x;
            this.w = w;
            this.h = h;
            this.c = c;
        }

        int pixels() {
            return w * h;
        }
    }

    /** The embedding rows of {@code image}, one per pixel of the adapter's output map in raster order. */
    static float[][] encode(VisionOps ops, ClipImage image) {
        MobileNetV5Layout layout = MobileNetV5Layout.discover(name -> ops.hasMatrix(name) || ops.hasVector(name));
        Features x = new Features(ops.upload(image.pixels(), 3), image.width(), image.height(), 3);
        x = conv(ops, x, MobileNetV5Layout.STEM_CONV, 3, 2, true,
                ops.hasVector(MobileNetV5Layout.STEM_BIAS) ? MobileNetV5Layout.STEM_BIAS : null);
        norm(ops, x, MobileNetV5Layout.STEM_NORM);
        ops.activate(x.x, ActOp.GELU_TANH);
        List<MobileNetV5Layout.Block> blocks = layout.blocks();
        List<Features> fusion = new ArrayList<>();
        for (int i = 0; i < blocks.size(); i++) {
            MobileNetV5Layout.Block b = blocks.get(i);
            int stride = layout.stageStart(i) ? 2 : 1;
            switch (b.kind) {
                case EDGE:
                    x = edge(ops, x, b, stride);
                    break;
                case ATTENTION:
                    x = attention(ops, x, b);
                    break;
                default:
                    x = inverted(ops, x, b, stride);
                    break;
            }
            if (layout.fusionPoint(i)) {
                fusion.add(x);
            }
        }
        if (!fusion.isEmpty()) {
            x = fuse(ops, fusion);
        }
        return embed(ops, x);
    }

    /** {@code build_edge_residual}: 3x3 expansion (SAME-padded when striding), GELU, pointwise projection. */
    private static Features edge(VisionOps ops, Features in, MobileNetV5Layout.Block b, int stride) {
        Features cur = conv(ops, in, b.name(CONV_EXP), 3, stride, stride == 2, null);
        norm(ops, cur, b.name(BN1));
        ops.activate(cur.x, ActOp.GELU_TANH);
        cur = conv(ops, cur, b.name(CONV_PWL), 1, 1, false, null);
        norm(ops, cur, b.name(BN2));
        if (stride == 1 && in.c == cur.c && in.w == cur.w) {
            ops.addBlock(cur.x, in.x);
        }
        return cur;
    }

    /**
     * {@code build_inverted_residual}: the depthwise start, the pointwise expansion with GELU, the depthwise mid
     * (which strides, SAME-padded) with GELU and the pointwise projection, each when present and followed by its
     * norm; the layer scale and the residual when the map keeps its shape.
     */
    private static Features inverted(VisionOps ops, Features in, MobileNetV5Layout.Block b, int stride) {
        Features cur = in;
        if (ops.hasVector(b.name(DW_START))) {
            cur = depthwise(ops, cur, b.name(DW_START), 1, false);
            norm(ops, cur, b.name(DW_START_NORM));
        }
        if (ops.hasMatrix(b.name(PW_EXP))) {
            cur = conv(ops, cur, b.name(PW_EXP), 1, 1, false, null);
            norm(ops, cur, b.name(PW_EXP_NORM));
            ops.activate(cur.x, ActOp.GELU_TANH);
        }
        if (ops.hasVector(b.name(DW_MID))) {
            cur = depthwise(ops, cur, b.name(DW_MID), stride, stride > 1);
            norm(ops, cur, b.name(DW_MID_NORM));
            ops.activate(cur.x, ActOp.GELU_TANH);
        }
        if (ops.hasMatrix(b.name(PW_PROJ))) {
            cur = conv(ops, cur, b.name(PW_PROJ), 1, 1, false, null);
            norm(ops, cur, b.name(PW_PROJ_NORM));
        }
        if (cur == in) {
            cur = copy(ops, in);
        }
        if (ops.hasVector(b.name(LAYER_SCALE))) {
            ops.mulRowScale(cur.x, ops.hostVector(b.name(LAYER_SCALE)));
        }
        if (in.w == cur.w && in.h == cur.h && in.c == cur.c) {
            ops.addBlock(cur.x, in.x);
        }
        return cur;
    }

    /**
     * {@code build_mobilenet_attn}: the block norm, multi-query attention of the query heads over one key/value
     * head (both downsampled by a SAME-padded stride-2 depthwise conv and normed when the block carries the
     * convs), the output projection, then the layer scale and residual when the map keeps its width and channels.
     */
    private static Features attention(VisionOps ops, Features in, MobileNetV5Layout.Block b) {
        Features cur = copy(ops, in);
        norm(ops, cur, b.name(ATTN_NORM));
        Features q = conv(ops, cur, b.name(QUERY), 1, 1, false, null);
        Features k = conv(ops, downsample(ops, cur, b.name(KEY_DOWN), b.name(KEY_NORM)), b.name(KEY), 1, 1, false,
                null);
        Features v = conv(ops, downsample(ops, cur, b.name(VALUE_DOWN), b.name(VALUE_NORM)), b.name(VALUE), 1, 1,
                false, null);
        int d = k.c;
        int heads = q.c / d;
        VisionOps.Block kqv = ops.attend(q.x, 0, q.pixels(), k.x, v.x, 0, k.pixels(), d, heads, heads, d,
                (float) (1.0 / Math.sqrt(d)));
        Features out = conv(ops, new Features(kqv, q.w, q.h, q.c), b.name(OUTPUT), 1, 1, false, null);
        if (in.w == out.w && in.c == out.c) {
            if (ops.hasVector(b.name(LAYER_SCALE))) {
                ops.mulRowScale(out.x, ops.hostVector(b.name(LAYER_SCALE)));
            }
            ops.addBlock(out.x, in.x);
        }
        return out;
    }

    /** The key or value input: the stride-2 depthwise conv and its norm when the block carries the conv. */
    private static Features downsample(VisionOps ops, Features x, String conv, String normName) {
        if (!ops.hasVector(conv)) {
            return x;
        }
        Features out = depthwise(ops, x, conv, 2, true);
        norm(ops, out, normName);
        return out;
    }

    /**
     * The fusion adapter: every feature map upscaled (nearest) to the first one's size, the channel concat, the
     * pointwise expansion with norm and GELU and projection with norm, the average pool down to
     * {@value #OUTPUT_SIDE} pixels a side, and the closing norm.
     */
    private static Features fuse(VisionOps ops, List<Features> feats) {
        Features target = feats.get(0);
        float[] rows = null;
        int c = 0;
        for (Features f : feats) {
            float[] fr = ops.download(f.x);
            if (f.w < target.w || f.h < target.h) {
                if (target.w % f.w != 0) {
                    throw new IllegalArgumentException("the MobileNetV5 fusion adapter upscales a " + f.w + "x" + f.h
                            + " feature map to " + target.w + "x" + target.h + " by a whole factor; got none");
                }
                int factor = target.w / f.w;
                fr = MobileNetV5Maps.upscaleNearest(fr, f.w, f.h, f.c, factor);
                f = new Features(f.x, f.w * factor, f.h * factor, f.c);
            }
            if (f.w != target.w || f.h != target.h) {
                throw new IllegalArgumentException("the MobileNetV5 fusion features of " + f.w + "x" + f.h + " and "
                        + target.w + "x" + target.h + " pixels do not meet on one grid");
            }
            rows = rows == null ? fr.clone() : MobileNetV5Maps.concat(rows, c, fr, f.c, target.pixels());
            c += f.c;
        }
        Features x = new Features(ops.upload(rows, c), target.w, target.h, c);
        if (ops.hasMatrix(MobileNetV5Layout.MSFA_EXPAND)) {
            x = conv(ops, x, MobileNetV5Layout.MSFA_EXPAND, 1, 1, false, null);
            norm(ops, x, MobileNetV5Layout.MSFA_EXPAND_NORM);
            ops.activate(x.x, ActOp.GELU_TANH);
        }
        if (ops.hasMatrix(MobileNetV5Layout.MSFA_PROJECT)) {
            x = conv(ops, x, MobileNetV5Layout.MSFA_PROJECT, 1, 1, false, null);
            norm(ops, x, MobileNetV5Layout.MSFA_PROJECT_NORM);
        }
        if (x.w > OUTPUT_SIDE) {
            if (x.w % OUTPUT_SIDE != 0) {
                throw new IllegalArgumentException("the MobileNetV5 fusion adapter pools its " + x.w + "-pixel-wide"
                        + " map down to " + OUTPUT_SIDE + " by a whole factor; got none");
            }
            int s = x.w / OUTPUT_SIDE;
            int ow = MobileNetV5Maps.convOut(x.w, s, s, 0);
            int oh = MobileNetV5Maps.convOut(x.h, s, s, 0);
            x = new Features(ops.avgPool(x.x, x.w, x.h, x.c, s, s, ow, oh), ow, oh, x.c);
        }
        norm(ops, x, MobileNetV5Layout.MSFA_NORM);
        return x;
    }

    /**
     * The Gemma 3n embedder over the pixels in raster order: the {@code sqrt(n_embd)} scale, the RMS norm times
     * {@code mm.soft_emb_norm}, {@code mm.input_projection} and the weightless RMS norm.
     */
    private static float[][] embed(VisionOps ops, Features x) {
        float[] scale = new float[x.c];
        Arrays.fill(scale, (float) Math.sqrt(x.c));
        ops.mulRowScale(x.x, scale);
        ops.norm(x.x, NormKind.RMS, MobileNetV5Layout.SOFT_EMB_NORM, null, EPS, x.pixels(), x.c);
        VisionOps.Block out = ops.linear(MobileNetV5Layout.INPUT_PROJECTION, x.x, null);
        int d = ops.model().outputDim();
        ops.normByRef(out, NormKind.RMS, null, null, EPS, x.pixels(), d);
        return VisionRows.tokenRows(ops.download(out), d);
    }

    /**
     * A {@code k x k} stride-{@code s} conv ({@code ggml_conv_2d_direct}) with the optional per-channel bias: SAME
     * padding when {@code same}, else {@code k / 2} on every side. The im2col and the matmul run on the executor,
     * so the map never leaves it; a 1x1 stride-1 conv reads the map in place.
     */
    private static Features conv(VisionOps ops, Features x, String kernel, int k, int s, boolean same,
                                 String bias) {
        TensorRef ref = ops.model().projectorMatrices().get(kernel);
        int out = Math.toIntExact(ref.ne(3));
        int[] padY = same ? MobileNetV5Maps.samePad(x.h, k, s) : new int[]{k / 2, 2 * (k / 2)};
        int[] padX = same ? MobileNetV5Maps.samePad(x.w, k, s) : new int[]{k / 2, 2 * (k / 2)};
        int ow = MobileNetV5Maps.convOut(x.w, k, s, padX[1]);
        int oh = MobileNetV5Maps.convOut(x.h, k, s, padY[1]);
        int width = x.c * k * k;
        VisionOps.Block cols = k == 1 && s == 1 ? x.x
                : ops.im2colPadded(x.x, x.c, x.w, x.h, k, s, padY[0], padX[0], ow, oh);
        VisionOps.Block y = ops.linearKernel(kernel, out, width, cols, bias);
        return new Features(y, ow, oh, out);
    }

    /** A depthwise conv ({@code ggml_conv_2d_dw}) on the executor: SAME padding when {@code same}, else
     * {@code k / 2}. */
    private static Features depthwise(VisionOps ops, Features x, String kernel, int s, boolean same) {
        int k = Math.toIntExact(ops.model().projectorVectors().get(kernel).ne(0));
        int[] padY = same ? MobileNetV5Maps.samePad(x.h, k, s) : new int[]{k / 2, 2 * (k / 2)};
        int[] padX = same ? MobileNetV5Maps.samePad(x.w, k, s) : new int[]{k / 2, 2 * (k / 2)};
        int ow = MobileNetV5Maps.convOut(x.w, k, s, padX[1]);
        int oh = MobileNetV5Maps.convOut(x.h, k, s, padY[1]);
        VisionOps.Block y = ops.depthwise(x.x, x.w, x.h, ops.model().projectorVectors().get(kernel), k, s, padY[0],
                padX[0], ow, oh, null);
        return new Features(y, ow, oh, x.c);
    }

    /** The per-pixel RMS norm over the channels times {@code weight}, in place, when the file carries it. */
    private static void norm(VisionOps ops, Features x, String weight) {
        if (ops.hasVector(weight)) {
            ops.norm(x.x, NormKind.RMS, weight, null, EPS, x.pixels(), x.c);
        }
    }

    private static Features copy(VisionOps ops, Features x) {
        return new Features(ops.upload(ops.download(x.x).clone(), x.c), x.w, x.h, x.c);
    }
}
