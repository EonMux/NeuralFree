package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

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
 * The validated tensors of the Gemma 3n MobileNetV5 encoder and its embedder (clip.cpp {@code load_tensors} for
 * {@code PROJECTOR_TYPE_GEMMA3NV}), shape-checked block by block along the channel count they carry: the stem
 * conv, every block of the {@link MobileNetV5Layout}, the fusion adapter, the soft embedding norm and
 * {@code mm.input_projection}. Conv kernels are filed as matrices the executor reads as im2col matmuls,
 * depthwise kernels, norms and channel scales as vectors. The convolutions take F32 or F16 kernels
 * ({@code ggml_conv_2d_direct} and {@code ggml_conv_2d_dw}); any other kernel type is refused by name. Static
 * methods only.
 */
final class MobileNetV5Tensors {

    private MobileNetV5Tensors() {
    }

    /**
     * The tensors into {@code c}; the output width is {@code mm.input_projection}'s.
     *
     * @throws MissingTensorException   when a tensor the graph reads is absent
     * @throws IllegalArgumentException when a shape or kernel type differs from what the graph reads, or the
     *                                  features do not end with {@code n_embd} channels (named)
     */
    static ClipProjectorTensors resolve(ClipProjectorTensors.Collector c, TensorCatalog tc, ClipHparams hp) {
        MobileNetV5Layout layout = MobileNetV5Layout.discover(tc::has);
        int ch = conv(c, tc, MobileNetV5Layout.STEM_CONV, 3, 3);
        if (tc.has(MobileNetV5Layout.STEM_BIAS)) {
            c.vector(MobileNetV5Layout.STEM_BIAS, 1, 1, ch);
        }
        c.optionalVector(MobileNetV5Layout.STEM_NORM, ch);
        List<MobileNetV5Layout.Block> blocks = layout.blocks();
        int fused = 0;
        for (int i = 0; i < blocks.size(); i++) {
            MobileNetV5Layout.Block b = blocks.get(i);
            switch (b.kind) {
                case EDGE:
                    ch = edge(c, tc, b, ch);
                    break;
                case ATTENTION:
                    ch = attention(c, tc, b, ch);
                    break;
                default:
                    ch = inverted(c, tc, b, ch);
                    break;
            }
            if (layout.fusionPoint(i)) {
                fused += ch;
            }
        }
        if (!blocks.isEmpty()) {
            ch = fusion(c, tc, fused);
        }
        if (ch != hp.nEmbd()) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': the MobileNetV5 features end"
                    + " with " + ch + " channels, clip.vision.embedding_length is " + hp.nEmbd());
        }
        c.vector(MobileNetV5Layout.SOFT_EMB_NORM, ch);
        int out = c.outDim(MobileNetV5Layout.INPUT_PROJECTION, ch);
        c.matrix(MobileNetV5Layout.INPUT_PROJECTION, ch, out);
        return c.build(out);
    }

    /** {@code build_edge_residual}: the 3x3 expansion, its norm, the pointwise projection and its norm. */
    private static int edge(ClipProjectorTensors.Collector c, TensorCatalog tc, MobileNetV5Layout.Block b, int in) {
        int mid = conv(c, tc, b.name(CONV_EXP), 3, in);
        c.optionalVector(b.name(BN1), mid);
        int out = conv(c, tc, b.name(CONV_PWL), 1, mid);
        c.optionalVector(b.name(BN2), out);
        return out;
    }

    /** {@code build_inverted_residual}: every present stage with its norm, then the layer scale. */
    private static int inverted(ClipProjectorTensors.Collector c, TensorCatalog tc, MobileNetV5Layout.Block b,
                                int in) {
        int ch = in;
        if (tc.has(b.name(DW_START))) {
            depthwise(c, tc, b.name(DW_START), ch);
            c.optionalVector(b.name(DW_START_NORM), ch);
        }
        if (tc.has(b.name(PW_EXP))) {
            ch = conv(c, tc, b.name(PW_EXP), 1, ch);
            c.optionalVector(b.name(PW_EXP_NORM), ch);
        }
        if (tc.has(b.name(DW_MID))) {
            depthwise(c, tc, b.name(DW_MID), ch);
            c.optionalVector(b.name(DW_MID_NORM), ch);
        }
        if (tc.has(b.name(PW_PROJ))) {
            ch = conv(c, tc, b.name(PW_PROJ), 1, ch);
            c.optionalVector(b.name(PW_PROJ_NORM), ch);
        }
        if (tc.has(b.name(LAYER_SCALE))) {
            c.vector(b.name(LAYER_SCALE), 1, 1, ch);
        }
        return ch;
    }

    /**
     * {@code build_mobilenet_attn}: the block norm, the query, the key and value projections of {@code D} channels
     * each (one shared head), downsampled together or not at all, the output projection and the layer scale.
     */
    private static int attention(ClipProjectorTensors.Collector c, TensorCatalog tc, MobileNetV5Layout.Block b,
                                 int in) {
        c.optionalVector(b.name(ATTN_NORM), in);
        int q = conv(c, tc, b.name(QUERY), 1, in);
        if (tc.has(b.name(KEY_DOWN)) != tc.has(b.name(VALUE_DOWN))) {
            String missing = tc.has(b.name(KEY_DOWN)) ? b.name(VALUE_DOWN) : b.name(KEY_DOWN);
            throw new MissingTensorException(tc.archName(), -1, missing);
        }
        if (tc.has(b.name(KEY_DOWN))) {
            depthwise(c, tc, b.name(KEY_DOWN), in);
            c.optionalVector(b.name(KEY_NORM), in);
            depthwise(c, tc, b.name(VALUE_DOWN), in);
            c.optionalVector(b.name(VALUE_NORM), in);
        }
        int d = conv(c, tc, b.name(KEY), 1, in);
        int dv = conv(c, tc, b.name(VALUE), 1, in);
        if (dv != d || q % d != 0) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': MobileNetV5 block "
                    + b.name("") + " projects keys to " + d + ", values to " + dv + " and queries to " + q
                    + " channels; the multi-query attention needs equal key and value heads that divide the queries");
        }
        int out = conv(c, tc, b.name(OUTPUT), 1, q);
        if (tc.has(b.name(LAYER_SCALE))) {
            c.vector(b.name(LAYER_SCALE), 1, 1, out);
        }
        return out;
    }

    /** The fusion adapter over the {@code in} concatenated channels: expansion, projection, closing norm. */
    private static int fusion(ClipProjectorTensors.Collector c, TensorCatalog tc, int in) {
        int ch = in;
        if (tc.has(MobileNetV5Layout.MSFA_EXPAND)) {
            ch = conv(c, tc, MobileNetV5Layout.MSFA_EXPAND, 1, ch);
            c.optionalVector(MobileNetV5Layout.MSFA_EXPAND_NORM, ch);
        }
        if (tc.has(MobileNetV5Layout.MSFA_PROJECT)) {
            ch = conv(c, tc, MobileNetV5Layout.MSFA_PROJECT, 1, ch);
            c.optionalVector(MobileNetV5Layout.MSFA_PROJECT_NORM, ch);
        }
        c.optionalVector(MobileNetV5Layout.MSFA_NORM, ch);
        return ch;
    }

    /** A {@code [k, k, in, out]} conv kernel; returns {@code out}. */
    private static int conv(ClipProjectorTensors.Collector c, TensorCatalog tc, String name, int k, int in) {
        TensorRef r = tc.require(name);
        requireConvType(tc, r);
        if (r.rank() > 4 || r.ne(3) <= 0 || r.ne(3) > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + name + " has shape "
                    + r + ", the loader expects a {" + k + ", " + k + ", " + in + ", out} conv kernel");
        }
        int out = (int) r.ne(3);
        c.kernel(name, k, k, in, out);
        return out;
    }

    /** A {@code [k, k, 1, channels]} depthwise kernel of any side {@code k}. */
    private static void depthwise(ClipProjectorTensors.Collector c, TensorCatalog tc, String name, int channels) {
        TensorRef r = tc.require(name);
        requireConvType(tc, r);
        if (r.ne(0) <= 0 || r.ne(0) > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': tensor " + name + " has shape "
                    + r + ", the loader expects a {k, k, 1, " + channels + "} depthwise kernel");
        }
        c.depthwise(name, (int) r.ne(0), channels);
    }

    private static void requireConvType(TensorCatalog tc, TensorRef r) {
        if (r.type() != GGMLType.F32 && r.type() != GGMLType.F16) {
            throw new IllegalArgumentException("architecture '" + tc.archName() + "': the MobileNetV5 convolutions"
                    + " take F32 or F16 kernels, tensor " + r.name() + " is " + r.type());
        }
    }
}
