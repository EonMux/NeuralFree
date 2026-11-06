package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.audio.MelFrontend;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;
import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkVector;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * The Voxtral Realtime encoder and its projector (llama.cpp tools/mtmd/models/voxtral-rt.cpp) over one mel chunk
 * of {@value #CONTEXT_FRAMES} context frames and {@code 2 n} new frames: the causal conv stem (kernel-3 stride-1
 * conv, GELU-erf, then the kernel-3 stride-2 conv with GELU-erf over the stem tail and those {@code 2 n} frames),
 * transformer layers with RMS norms, Q and V biases, interleaved RoPE over the frame positions, attention to the
 * frame itself and the {@code clip.audio.window_size - 1} frames before it and the gated SiLU FFN, the RMS trunk
 * norm, then the frame stack of {@code clip.audio.projector.stack_factor} and the GELU-erf MLP.
 * <p>
 * The chunk is encoded as the first call of a stream (zero stem tail, positions from 0), which equals a stream fed
 * the same frames over several calls: the stem tail, the positions and the key window carry over between calls.
 * The attention width is {@code head_count * head_dim}, independent of the embedding width. Package-private
 * executor support.
 */
final class VoxtralRtTower implements FamilyEncoder {

    /** Mel frames before the new frames: the left context of the stride-1 conv. */
    static final int CONTEXT_FRAMES = MelFrontend.VOXTRAL_RT_CONTEXT_FRAMES;
    private static final int K = 3;

    private final int nMel;
    private final int width;
    private final int stack;
    private final float eps;
    private final CpuTensorRef conv1;
    private final float[] conv1B;
    private final CpuTensorRef conv2;
    private final float[] conv2B;
    private final Layer[] layers;
    private final float[] postLn;
    private final CpuTensorRef mm1;
    private final CpuTensorRef mm2;
    private final CpuMatVec matVec;

    private VoxtralRtTower(ClipHparams hp, TensorCatalog tc, CpuMatVec matVec, Function<String, float[]> floats,
                           Function<String, CpuTensorRef> refs) {
        this.nMel = hp.audio().nMelBins();
        this.width = hp.nEmbd();
        this.stack = hp.audio().stackFactor();
        this.eps = hp.eps();
        this.conv1 = CpuVitTower.matrix(refs.apply("a.conv1d.1.weight"), width, (long) K * nMel);
        this.conv1B = floats.apply("a.conv1d.1.bias");
        this.conv2 = CpuVitTower.matrix(refs.apply("a.conv1d.2.weight"), width, (long) K * width);
        this.conv2B = floats.apply("a.conv1d.2.bias");
        this.layers = new Layer[hp.nLayer()];
        for (int il = 0; il < layers.length; il++) {
            layers[il] = new Layer(tc, hp, il, floats, refs);
        }
        this.postLn = floats.apply("a.post_ln.weight");
        this.mm1 = refs.apply("mm.a.mlp.1.weight");
        this.mm2 = refs.apply("mm.a.mlp.2.weight");
        this.matVec = matVec;
    }

    /** Encoder rows of a chunk of {@code nFrames} mel frames: the context frames, the stride-2 stem, the stack. */
    static int tokens(int nFrames, int stack) {
        return (nFrames - CONTEXT_FRAMES) / 2 / stack;
    }

    /** Shape-checks the stem, every layer, the trunk norm and the projector against the widths. */
    static void validate(ClipHparams hp, TensorCatalog tc) {
        int e = hp.nEmbd();
        shape(tc, tc.require("a.conv1d.1.weight"), K, hp.audio().nMelBins(), e);
        shape(tc, tc.require("a.conv1d.1.bias"), 1, e);
        shape(tc, tc.require("a.conv1d.2.weight"), K, e, e);
        shape(tc, tc.require("a.conv1d.2.bias"), 1, e);
        shape(tc, tc.require("a.post_ln.weight"), e);
        for (int il = 0; il < hp.nLayer(); il++) {
            Layer.validate(tc, hp, il);
        }
        TensorRef mm1 = tc.require("mm.a.mlp.1.weight");
        long hidden = mm1.ne(1);
        shape(tc, mm1, (long) e * hp.audio().stackFactor(), hidden);
        shape(tc, tc.require("mm.a.mlp.2.weight"), hidden, hp.projectionDim());
    }

    /** Binds the executor over {@code model} with the weights of {@code floats} / {@code refs}. */
    static VoxtralRtTower bind(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                               Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(model, "model");
        return new VoxtralRtTower(model.hparams(), model.catalog(), Objects.requireNonNull(matVec, "matVec"),
                floats, refs);
    }

    /** The stem, the layers, the trunk norm, the stack and the projector MLP over the chunk. */
    @Override
    public float[][] encode(MelChunk chunk) {
        if (chunk.nMel() != nMel) {
            throw new IllegalArgumentException("mel chunk has " + chunk.nMel() + " bins, the encoder needs " + nMel);
        }
        int n = (chunk.nLen() - CONTEXT_FRAMES) / 2;
        if (n <= 0 || n % stack != 0) {
            throw new IllegalArgumentException("mel chunk of " + chunk.nLen() + " frames gives " + n
                    + " encoder frames; voxtral_rt needs " + CONTEXT_FRAMES + " context frames and a positive"
                    + " multiple of 2 * stack_factor (" + 2 * stack + ") new frames");
        }
        float[] x = stem(chunk.data(), chunk.nLen(), n);
        int[] pos = AudioOps.iota(n);
        for (Layer l : layers) {
            l.forward(matVec, x, n, pos, eps);
        }
        AudioOps.rmsNorm(x, n, width, postLn, null, eps);
        // build_stack: n is a multiple of the stack factor, so the rows regroup without padding
        int tokens = n / stack;
        float[] h = AudioOps.linear(matVec, mm1, null, x, tokens);
        AudioOps.activate(ActOp.GELU_ERF, h);
        float[] out = AudioOps.linear(matVec, mm2, null, h, tokens);
        return VisionRows.tokenRows(out, (int) mm2.rows());
    }

    /**
     * The causal conv stem over the mel-major chunk: conv1 (valid, stride 1) to {@code 2 n} frames, then conv2
     * (valid, stride 2) over the zero stem tail and those frames; {@code [n][width]} rows.
     */
    private float[] stem(float[] mel, int nLen, int n) {
        float[] c1 = conv(conv1, conv1B, mel, nLen, nMel, 1, 0);
        float[] tailed = new float[width * (2 * n + 1)];
        for (int c = 0; c < width; c++) {
            System.arraycopy(c1, c * 2 * n, tailed, c * (2 * n + 1) + 1, 2 * n);
        }
        return AudioOps.transpose(conv(conv2, conv2B, tailed, 2 * n + 1, width, 2, 0), width, n);
    }

    /**
     * {@code ggml_conv_1d(w, in, stride, 0, 1)} plus the bias and the GELU-erf over a channel-major
     * {@code [nCh][len]} input; the output is channel-major {@code [width][outLen]}.
     */
    private float[] conv(CpuTensorRef w, float[] bias, float[] in, int len, int nCh, int stride, int pad) {
        float[] out = AudioOps.conv2d(matVec, in, len, 1, nCh, w, K, 1, stride, pad, bias);
        AudioOps.activate(ActOp.GELU_ERF, out);
        return out;
    }

    /** One encoder layer {@code a.blk.N}: RMS norm, windowed RoPE attention, RMS norm, gated SiLU FFN. */
    private static final class Layer {
        private final int width;
        private final int heads;
        private final int kvHeads;
        private final int dh;
        private final int window;
        private final float theta;
        private final float[] ln1;
        private final float[] ln2;
        private final CpuTensorRef q;
        private final float[] qB;
        private final CpuTensorRef k;
        private final CpuTensorRef v;
        private final float[] vB;
        private final CpuTensorRef o;
        private final float[] oB;
        private final CpuTensorRef up;
        private final CpuTensorRef gate;
        private final CpuTensorRef down;
        private final float[] downB;

        Layer(TensorCatalog tc, ClipHparams hp, int il, Function<String, float[]> floats,
              Function<String, CpuTensorRef> refs) {
            String p = prefix(il);
            this.width = hp.nEmbd();
            this.heads = hp.nHead();
            this.kvHeads = hp.nHeadKv();
            this.dh = hp.dHead();
            this.window = hp.audio().attnWindow();
            this.theta = hp.audio().ropeTheta();
            this.ln1 = floats.apply(p + "ln1.weight");
            this.ln2 = floats.apply(p + "ln2.weight");
            this.q = refs.apply(p + "attn_q.weight");
            this.qB = floats.apply(p + "attn_q.bias");
            this.k = refs.apply(p + "attn_k.weight");
            this.v = refs.apply(p + "attn_v.weight");
            this.vB = floats.apply(p + "attn_v.bias");
            this.o = refs.apply(p + "attn_out.weight");
            this.oB = tc.has(p + "attn_out.bias") ? floats.apply(p + "attn_out.bias") : null;
            this.up = refs.apply(p + "ffn_up.weight");
            this.gate = tc.has(p + "ffn_gate.weight") ? refs.apply(p + "ffn_gate.weight") : null;
            this.down = refs.apply(p + "ffn_down.weight");
            this.downB = tc.has(p + "ffn_down.bias") ? floats.apply(p + "ffn_down.bias") : null;
        }

        private static String prefix(int il) {
            return "a.blk." + il + ".";
        }

        /** Q is {@code [e, heads * dh]}, K and V {@code [e, kvHeads * dh]}; the graph adds the Q and V biases. */
        static void validate(TensorCatalog tc, ClipHparams hp, int il) {
            String p = prefix(il);
            int e = hp.nEmbd();
            long qWidth = (long) hp.nHead() * hp.dHead();
            long kvWidth = (long) hp.nHeadKv() * hp.dHead();
            shape(tc, tc.require(p + "ln1.weight"), e);
            shape(tc, tc.require(p + "ln2.weight"), e);
            shape(tc, tc.require(p + "attn_q.weight"), e, qWidth);
            shape(tc, tc.require(p + "attn_q.bias"), qWidth);
            shape(tc, tc.require(p + "attn_k.weight"), e, kvWidth);
            shape(tc, tc.require(p + "attn_v.weight"), e, kvWidth);
            shape(tc, tc.require(p + "attn_v.bias"), kvWidth);
            shape(tc, tc.require(p + "attn_out.weight"), qWidth, e);
            checkVector(tc, tc.get(p + "attn_out.bias"), e);
            TensorRef upW = tc.require(p + "ffn_up.weight");
            long ff = upW.ne(1);
            shape(tc, upW, e, ff);
            TensorRef gateW = tc.get(p + "ffn_gate.weight");
            if (gateW != null) {
                shape(tc, gateW, e, ff);
            }
            shape(tc, tc.require(p + "ffn_down.weight"), ff, e);
            checkVector(tc, tc.get(p + "ffn_down.bias"), e);
        }

        /** The layer over {@code [n][width]} rows of {@code x} at frame positions {@code pos}, in place. */
        void forward(CpuMatVec matVec, float[] x, int n, int[] pos, float eps) {
            float[] h = x.clone();
            AudioOps.rmsNorm(h, n, width, ln1, null, eps);
            AudioOps.addScaled(x, attention(matVec, h, n, pos), 1f);
            h = x.clone();
            AudioOps.rmsNorm(h, n, width, ln2, null, eps);
            float[] u = AudioOps.linear(matVec, up, null, h, n);
            if (gate != null) {
                float[] g = AudioOps.linear(matVec, gate, null, h, n);
                AudioOps.activate(ActOp.SILU, g);
                for (int i = 0; i < u.length; i++) {
                    u[i] *= g[i];
                }
            } else {
                AudioOps.activate(ActOp.SILU, u);
            }
            AudioOps.addScaled(x, AudioOps.linear(matVec, down, downB, u, n), 1f);
        }

        /** Interleaved RoPE on Q and K, each frame attending to itself and the {@code window - 1} before it. */
        private float[] attention(CpuMatVec matVec, float[] h, int n, int[] pos) {
            float[] qr = AudioOps.linear(matVec, q, qB, h, n);
            float[] kr = AudioOps.linear(matVec, k, null, h, n);
            float[] vr = AudioOps.linear(matVec, v, vB, h, n);
            AudioOps.ropeNormal(qr, n, heads, dh, pos, theta);
            AudioOps.ropeNormal(kr, n, kvHeads, dh, pos, theta);
            VisionRows.KeyRanges causal = new VisionRows.KeyRanges() {
                @Override
                public int begin(int row) {
                    return Math.max(0, row - window + 1);
                }

                @Override
                public int end(int row) {
                    return row + 1;
                }
            };
            float[] a = VisionRows.attend(qr, 0, n, kr, vr, 0, n, kvHeads * dh, heads, heads / kvHeads, dh,
                    (float) (1.0 / Math.sqrt(dh)), causal);
            return AudioOps.linear(matVec, o, oB, a, n);
        }
    }
}
