package com.eonmux.neuralfree.gguf.ops.audio;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.ops.attn.CpuAttentionCore;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * CPU executor of {@link WavTokenizerDecoder}, the reference twin of {@link GpuWavTokenizer}: the same step
 * order, layouts and math (PK CPU twins {@link CpuConv1d}, {@link NormMath#groupNorm}, {@link CpuAttentionCore},
 * {@link CpuOps}, {@link CpuMatVec}).
 *
 * <p>Layouts: the residual signal is channel-major {@code [C][T]} as in the ggml graph (wavtokenizer-dec.cpp:124);
 * LayerNorms, projections and attention run token-major {@code [T][C]} between explicit transposes, exactly where
 * the graph has {@code ggml_cont(ggml_transpose(..))}. The posnet 1x1 q/k/v/output projections are the graph's
 * {@code ggml_conv_1d_ph} with {@code K = 1}; attention is single-head, non-causal, scale {@code 1/sqrt(C)}, with
 * the online softmax of the flash kernels, which equals {@code soft_max_ext} followed by {@code kq * v}
 * (wavtokenizer-dec.cpp:185-189).
 *
 * <p>Conv, norm, bias and gamma tensors are read as F32 vectors ({@link CpuWeights#floats}); {@code pw1},
 * {@code pw2} and {@code output} stay in their stored type and run through {@link CpuMatVec}. Work arrays are
 * reserved from {@link CpuScratch} at bind for the capacity; a call allocates only its result rows. Not
 * thread-safe.
 */
public final class CpuWavTokenizer implements WavTokenizerDecoder.Backend {

    /** Row lookup of the code embedding table into {@code dst[off ..]}. */
    @FunctionalInterface
    interface RowLookup {
        void row(int code, float[] dst, int off);
    }

    /** Resolved tensors of one posnet layer; unused fields are null for the layer's kind. */
    static final class PosnetLayer {
        final WavTokenizerDecoder.PosnetKind kind;
        final float[] normW;
        final float[] normB;
        final float[] conv1W;
        final float[] conv1B;
        final float[] norm2W;
        final float[] norm2B;
        final float[] conv2W;
        final float[] conv2B;
        final float[][] attnW;
        final float[][] attnB;

        private PosnetLayer(WavTokenizerDecoder.PosnetKind kind, float[] normW, float[] normB, float[] conv1W,
                            float[] conv1B, float[] norm2W, float[] norm2B, float[] conv2W, float[] conv2B,
                            float[][] attnW, float[][] attnB) {
            this.kind = kind;
            this.normW = normW;
            this.normB = normB;
            this.conv1W = conv1W;
            this.conv1B = conv1B;
            this.norm2W = norm2W;
            this.norm2B = norm2B;
            this.conv2W = conv2W;
            this.conv2B = conv2B;
            this.attnW = attnW;
            this.attnB = attnB;
        }
    }

    /** Resolved tensors of one ConvNeXt layer. */
    static final class ConvnextLayer {
        final float[] dwW;
        final float[] dwB;
        final float[] normW;
        final float[] normB;
        final CpuTensorRef pw1;
        final float[] pw1B;
        final CpuTensorRef pw2;
        final float[] pw2B;
        final float[] gamma;

        ConvnextLayer(float[] dwW, float[] dwB, float[] normW, float[] normB, CpuTensorRef pw1, float[] pw1B,
                      CpuTensorRef pw2, float[] pw2B, float[] gamma) {
            this.dwW = dwW;
            this.dwB = dwB;
            this.normW = normW;
            this.normB = normB;
            this.pw1 = pw1;
            this.pw1B = pw1B;
            this.pw2 = pw2;
            this.pw2B = pw2B;
            this.gamma = gamma;
        }
    }

    /** Every tensor of the decoder, resolved once at bind. */
    static final class Weights {
        final RowLookup tokEmbd;
        final float[] convW;
        final float[] convB;
        final List<PosnetLayer> posnet;
        final float[] tokNormW;
        final float[] tokNormB;
        final List<ConvnextLayer> convnext;
        final float[] outNormW;
        final float[] outNormB;
        final CpuTensorRef output;
        final float[] outputB;

        private Weights(RowLookup tokEmbd, float[] convW, float[] convB, List<PosnetLayer> posnet, float[] tokNormW,
                        float[] tokNormB, List<ConvnextLayer> convnext, float[] outNormW, float[] outNormB,
                        CpuTensorRef output, float[] outputB) {
            this.tokEmbd = tokEmbd;
            this.convW = convW;
            this.convB = convB;
            this.posnet = posnet;
            this.tokNormW = tokNormW;
            this.tokNormB = tokNormB;
            this.convnext = convnext;
            this.outNormW = outNormW;
            this.outNormB = outNormB;
            this.output = output;
            this.outputB = outputB;
        }

        /**
         * Resolves every tensor of {@code dec} by name.
         *
         * @param floats  name to F32 values (required tensors; read only)
         * @param refs    name to stored tensor (required tensors)
         * @param tokEmbd code embedding row lookup
         */
        static Weights resolve(WavTokenizerDecoder dec, Function<String, float[]> floats,
                               Function<String, CpuTensorRef> refs, RowLookup tokEmbd) {
            List<PosnetLayer> posnet = new ArrayList<>();
            for (int il = 0; il < dec.posnetLayers(); il++) {
                posnet.add(posnetLayer(il, floats));
            }
            List<ConvnextLayer> convnext = new ArrayList<>();
            for (int il = 0; il < dec.convnextLayers(); il++) {
                convnext.add(new ConvnextLayer(
                        floats.apply(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_DW, false)),
                        floats.apply(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_DW, true)),
                        floats.apply(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_NORM, false)),
                        floats.apply(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_NORM, true)),
                        refs.apply(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_PW1, false)),
                        floats.apply(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_PW1, true)),
                        refs.apply(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_PW2, false)),
                        floats.apply(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_PW2, true)),
                        floats.apply(WavTokenizerDecoder.convnext(il, TensorNames.CONVNEXT_GAMMA, false))));
            }
            return new Weights(Objects.requireNonNull(tokEmbd, "tokEmbd"),
                    floats.apply(TensorNames.weight(TensorNames.CONV1D)),
                    floats.apply(TensorNames.bias(TensorNames.CONV1D)),
                    Collections.unmodifiableList(posnet),
                    floats.apply(TensorNames.weight(TensorNames.TOKEN_EMBD_NORM)),
                    floats.apply(TensorNames.bias(TensorNames.TOKEN_EMBD_NORM)),
                    Collections.unmodifiableList(convnext),
                    floats.apply(TensorNames.weight(TensorNames.OUTPUT_NORM)),
                    floats.apply(TensorNames.bias(TensorNames.OUTPUT_NORM)),
                    refs.apply(TensorNames.weight(TensorNames.OUTPUT)),
                    floats.apply(TensorNames.bias(TensorNames.OUTPUT)));
        }

        private static PosnetLayer posnetLayer(int il, Function<String, float[]> f) {
            WavTokenizerDecoder.PosnetKind kind = WavTokenizerDecoder.posnetKind(il);
            switch (kind) {
                case RESNET:
                    return new PosnetLayer(kind,
                            f.apply(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_NORM1, false)),
                            f.apply(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_NORM1, true)),
                            f.apply(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_CONV1, false)),
                            f.apply(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_CONV1, true)),
                            f.apply(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_NORM2, false)),
                            f.apply(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_NORM2, true)),
                            f.apply(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_CONV2, false)),
                            f.apply(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_CONV2, true)),
                            null, null);
                case ATTENTION: {
                    String[] proj = ATTENTION_PROJECTIONS;
                    float[][] w = new float[proj.length][];
                    float[][] b = new float[proj.length][];
                    for (int i = 0; i < proj.length; i++) {
                        w[i] = f.apply(WavTokenizerDecoder.posnet(il, proj[i], false));
                        b[i] = f.apply(WavTokenizerDecoder.posnet(il, proj[i], true));
                    }
                    return new PosnetLayer(kind,
                            f.apply(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_ATTN_NORM, false)),
                            f.apply(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_ATTN_NORM, true)),
                            null, null, null, null, null, null, w, b);
                }
                case NORM:
                    return new PosnetLayer(kind,
                            f.apply(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_ATTN_NORM, false)),
                            f.apply(WavTokenizerDecoder.posnet(il, TensorNames.POS_NET_ATTN_NORM, true)),
                            null, null, null, null, null, null, null, null);
                default:
                    throw new IllegalStateException("unhandled posnet kind " + kind);
            }
        }
    }

    /** Posnet attention projections in q, k, v, output order (wavtokenizer-dec.cpp:56-66). */
    static final String[] ATTENTION_PROJECTIONS = {TensorNames.POS_NET_ATTN_Q, TensorNames.POS_NET_ATTN_K,
            TensorNames.POS_NET_ATTN_V, TensorNames.POS_NET_ATTN_OUT};
    static final int PROJ_Q = 0;
    static final int PROJ_K = 1;
    static final int PROJ_V = 2;
    static final int PROJ_OUT = 3;

    /** Transposed elements per parallel task. */
    private static final int TRANSPOSE_GRAIN = 4096;

    private final WavTokenizerDecoder dec;
    private final Weights w;
    private final CpuMatVec matVec;
    private final int capacity;
    private final CpuAttentionCore.Params attnParams;
    private final float[] emb;
    private final float[] x;
    private final float[] a;
    private final float[] b;
    private final float[] tok;
    private final float[] q;
    private final float[] k;
    private final float[] v;
    private final float[] ff;
    private final float[] out;
    private final int[] pos;
    private boolean closed;

    CpuWavTokenizer(WavTokenizerDecoder dec, Weights w, CpuMatVec matVec, CpuScratch scratch, int capacity) {
        this.dec = Objects.requireNonNull(dec, "decoder");
        this.w = Objects.requireNonNull(w, "weights");
        this.matVec = Objects.requireNonNull(matVec, "matVec");
        Objects.requireNonNull(scratch, "scratch");
        dec.checkCapacity(capacity);
        this.capacity = capacity;
        int c = dec.width();
        int signal = Math.max(dec.nEmbd(), c) * capacity;
        this.emb = scratch.temp("wavtok.emb", dec.nEmbd() * capacity);
        this.x = scratch.temp("wavtok.x", c * capacity);
        this.a = scratch.temp("wavtok.a", signal);
        this.b = scratch.temp("wavtok.b", c * capacity);
        this.tok = scratch.temp("wavtok.tok", c * capacity);
        boolean attention = dec.posnetLayers() > WavTokenizerDecoder.POSNET_ATTENTION_LAYER;
        this.q = attention ? scratch.temp("wavtok.q", c * capacity) : null;
        this.k = attention ? scratch.temp("wavtok.k", c * capacity) : null;
        this.v = attention ? scratch.temp("wavtok.v", c * capacity) : null;
        this.ff = dec.convnextLayers() > 0 ? scratch.temp("wavtok.ff", dec.nFf() * capacity) : null;
        this.out = scratch.temp("wavtok.out", dec.nOut() * capacity);
        this.pos = scratch.intTemp("wavtok.pos", capacity);
        for (int t = 0; t < capacity; t++) {
            pos[t] = t;
        }
        this.attnParams = CpuAttentionCore.Params.builder(1).scale(dec.attentionScale())
                .window(KvWindow.FULL_BIDIRECTIONAL).build();
    }

    /** Binds to a model's CPU context with the scratch ubatch as capacity. */
    public static CpuWavTokenizer bind(CpuForwardContext ctx) {
        Objects.requireNonNull(ctx, "ctx");
        return bind(WavTokenizerDecoder.of(ctx.spec()), ctx.weights(), ctx.matVec(), ctx.scratch(),
                ctx.scratch().ubatch());
    }

    /**
     * Binds to explicit weights and scratch.
     *
     * @param capacity largest code sequence per call
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a tensor is missing
     */
    public static CpuWavTokenizer bind(WavTokenizerDecoder dec, CpuWeights weights, CpuMatVec matVec,
                                       CpuScratch scratch, int capacity) {
        Objects.requireNonNull(dec, "decoder");
        Objects.requireNonNull(weights, "weights");
        CpuTensorRef table = weights.get(TensorNames.weight(TensorNames.TOKEN_EMBD));
        Weights w = Weights.resolve(dec, weights::floats, weights::get,
                (code, dst, off) -> weights.embedRow(table, code, dst, off));
        return new CpuWavTokenizer(dec, w, matVec, scratch, capacity);
    }

    @Override
    public float[][] decode(int[] codes) {
        ensureOpen();
        dec.checkCodes(codes, capacity);
        int t = codes.length;
        int e = dec.nEmbd();
        int c = dec.width();
        for (int i = 0; i < t; i++) {
            w.tokEmbd.row(codes[i], emb, i * e);
        }
        transpose(emb, t, e, a);
        CpuConv1d.same(t, e, c, WavTokenizerDecoder.INPUT_CONV_K, 1, a, 0, w.convW, w.convB, x, 0);

        for (PosnetLayer layer : w.posnet) {
            posnet(layer, t);
        }

        transpose(x, c, t, tok);
        CpuOps.layerNorm(tok, 0, tok, 0, t, c, w.tokNormW, w.tokNormB, dec.normEps());
        transpose(tok, t, c, x);

        for (ConvnextLayer layer : w.convnext) {
            convnext(layer, t);
        }

        transpose(x, c, t, tok);
        CpuOps.layerNorm(tok, 0, tok, 0, t, c, w.outNormW, w.outNormB, dec.normEps());
        int o = dec.nOut();
        matVec.matmul(w.output, tok, 0, out, 0, t);
        CpuOps.addRowBias(out, 0, t, o, w.outputB, 0);
        float[][] rows = new float[t][o];
        for (int i = 0; i < t; i++) {
            System.arraycopy(out, i * o, rows[i], 0, o);
        }
        return rows;
    }

    /** One posnet layer over the channel-major residual {@code x} (wavtokenizer-dec.cpp:130-205). */
    private void posnet(PosnetLayer l, int t) {
        int c = dec.width();
        int n = c * t;
        int groups = dec.nGroups();
        float eps = dec.groupEps();
        switch (l.kind) {
            case RESNET:
                NormMath.groupNorm(c, t, groups, eps, x, 0, l.normW, l.normB, a, 0);
                CpuOps.unary(ActOp.SILU, ActParams.NONE, a, 0, a, 0, n);
                CpuConv1d.same(t, c, c, WavTokenizerDecoder.POSNET_CONV_K, 1, a, 0, l.conv1W, l.conv1B, b, 0);
                NormMath.groupNorm(c, t, groups, eps, b, 0, l.norm2W, l.norm2B, a, 0);
                CpuOps.unary(ActOp.SILU, ActParams.NONE, a, 0, a, 0, n);
                CpuConv1d.same(t, c, c, WavTokenizerDecoder.POSNET_CONV_K, 1, a, 0, l.conv2W, l.conv2B, b, 0);
                CpuOps.add(b, 0, x, 0, x, 0, n);
                return;
            case ATTENTION:
                NormMath.groupNorm(c, t, groups, eps, x, 0, l.normW, l.normB, a, 0);
                projectTokenMajor(l, PROJ_Q, t, q);
                projectTokenMajor(l, PROJ_K, t, k);
                projectTokenMajor(l, PROJ_V, t, v);
                CpuAttentionCore.attendRows(attnParams, t, q, 0, c, pos, 0,
                        CpuAttentionCore.DenseKv.of(k, 0, c, c, v, 0, c, c, 1, c, c, pos, 0), 0, t, tok, 0, c);
                transpose(tok, t, c, a);
                CpuConv1d.same(t, c, c, 1, 1, a, 0, l.attnW[PROJ_OUT], l.attnB[PROJ_OUT], b, 0);
                CpuOps.add(b, 0, x, 0, x, 0, n);
                return;
            case NORM:
                NormMath.groupNorm(c, t, groups, eps, x, 0, l.normW, l.normB, a, 0);
                System.arraycopy(a, 0, x, 0, n);
                return;
            default:
                throw new IllegalStateException("unhandled posnet kind " + l.kind);
        }
    }

    /** 1x1 projection {@code p} of the normed signal in {@code a}, written token-major into {@code dst}. */
    private void projectTokenMajor(PosnetLayer l, int p, int t, float[] dst) {
        int c = dec.width();
        CpuConv1d.same(t, c, c, 1, 1, a, 0, l.attnW[p], l.attnB[p], b, 0);
        transpose(b, c, t, dst);
    }

    /** One ConvNeXt layer over the channel-major residual {@code x} (wavtokenizer-dec.cpp:218-245). */
    private void convnext(ConvnextLayer l, int t) {
        int c = dec.width();
        int f = dec.nFf();
        CpuConv1d.depthwise(t, c, WavTokenizerDecoder.CONVNEXT_DW_K, x, 0, l.dwW, l.dwB, a, 0);
        transpose(a, c, t, tok);
        CpuOps.layerNorm(tok, 0, tok, 0, t, c, l.normW, l.normB, dec.normEps());
        matVec.matmul(l.pw1, tok, 0, ff, 0, t);
        CpuOps.addRowBias(ff, 0, t, f, l.pw1B, 0);
        CpuOps.unary(ActOp.GELU_TANH, ActParams.NONE, ff, 0, ff, 0, f * t);
        matVec.matmul(l.pw2, ff, 0, tok, 0, t);
        CpuOps.addRowBias(tok, 0, t, c, l.pw2B, 0);
        NormMath.mulRowScale(t, c, l.gamma, 0, c, tok, 0);
        transpose(tok, t, c, a);
        CpuOps.add(a, 0, x, 0, x, 0, c * t);
    }

    /**
     * {@code dst[c*rows + r] = src[r*cols + c]}: the dense {@code [rows][cols]} tensor as {@code [cols][rows]}
     * ({@code ggml_cont(ggml_transpose(..))}); the arrays must differ.
     */
    static void transpose(float[] src, int rows, int cols, float[] dst) {
        if (src == dst) {
            throw new IllegalArgumentException("transpose needs distinct source and destination arrays");
        }
        long n = (long) rows * cols;
        if (rows <= 0 || cols <= 0 || n > src.length || n > dst.length) {
            throw new IllegalArgumentException("transpose of " + rows + " x " + cols + " does not fit arrays of "
                    + src.length + " and " + dst.length);
        }
        CpuParallel.parallelFor(cols, Math.max(1, TRANSPOSE_GRAIN / rows), (from, to) -> {
            for (int col = from; col < to; col++) {
                int base = col * rows;
                for (int r = 0; r < rows; r++) {
                    dst[base + r] = src[r * cols + col];
                }
            }
        });
    }

    @Override
    public int capacity() {
        return capacity;
    }

    @Override
    public WavTokenizerDecoder decoder() {
        return dec;
    }

    /** Releases nothing (the scratch and weights belong to the model); further decodes fail. Idempotent. */
    @Override
    public void close() {
        closed = true;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("wavtokenizer CPU decoder is closed");
        }
    }

    @Override
    public String toString() {
        return "CpuWavTokenizer{" + dec + ", capacity=" + capacity + '}';
    }
}
