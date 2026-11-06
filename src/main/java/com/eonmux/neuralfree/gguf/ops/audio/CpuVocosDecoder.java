package com.eonmux.neuralfree.gguf.ops.audio;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * CPU executor of {@link VocosDecoder}, the soprano Vocos head of llama.cpp (tools/mtmd/models/soprano.cpp
 * build(), lines 5-59): bilinear time upsample with {@code align_corners}, the input projection (a same-padding
 * {@code conv1d} or a pointwise product), per-frame LayerNorms over channels, the Vocos blocks' depthwise conv,
 * pointwise pair and layer scale, and the polar head that turns log-magnitude and phase rows into interleaved
 * {@code (real, imaginary)} bin values. Every weight is read once at bind as F32 ({@link CpuWeights#floats}) or
 * kept as a reference for {@link CpuMatVec} ({@code pw1}, {@code pw2}, {@code output}); the math uses the shared
 * CPU kernels ({@link CpuConv1d}, {@link CpuOps}).
 *
 * <p>Layouts follow the graph's transposes: the residual signal runs channel-major {@code [C][T]} (element
 * {@code (c, t)} at {@code v[c*T + t]}); the LayerNorms and pointwise products run token-major {@code [T][C]}
 * between explicit copies. One call decodes one whole feature batch (the head is state-free; the reference
 * builds it per GEN_WAV call). Not thread-safe; a call allocates its work arrays.
 */
public final class CpuVocosDecoder implements AutoCloseable {

    /** Resolved tensors of one Vocos block. */
    public static final class Block {
        final float[] dwconvW;
        final float[] dwconvB;
        final float[] normW;
        final float[] normB;
        final CpuTensorRef pw1;
        final float[] pw1B;
        final CpuTensorRef pw2;
        final float[] pw2B;
        final float[] gamma;

        Block(float[] dwconvW, float[] dwconvB, float[] normW, float[] normB, CpuTensorRef pw1, float[] pw1B,
              CpuTensorRef pw2, float[] pw2B, float[] gamma) {
            this.dwconvW = dwconvW;
            this.dwconvB = dwconvB;
            this.normW = normW;
            this.normB = normB;
            this.pw1 = pw1;
            this.pw1B = pw1B;
            this.pw2 = pw2;
            this.pw2B = pw2B;
            this.gamma = gamma;
        }
    }

    /** Resolved tensors of the head. */
    public static final class Weights {
        final int convK;
        final float[] inputConvW;
        final CpuTensorRef inputProj;
        final float[] inputB;
        final float[] normW;
        final float[] normB;
        final List<Block> blocks;
        final float[] outputNormW;
        final float[] outputNormB;
        final CpuTensorRef output;
        final float[] outputB;

        Weights(int convK, float[] inputConvW, CpuTensorRef inputProj, float[] inputB, float[] normW, float[] normB,
                List<Block> blocks, float[] outputNormW, float[] outputNormB, CpuTensorRef output, float[] outputB) {
            this.convK = convK;
            this.inputConvW = inputConvW;
            this.inputProj = inputProj;
            this.inputB = inputB;
            this.normW = normW;
            this.normB = normB;
            this.blocks = blocks;
            this.outputNormW = outputNormW;
            this.outputNormB = outputNormB;
            this.output = output;
            this.outputB = outputB;
        }
    }

    private final VocosDecoder dec;
    private final Weights w;
    private final CpuMatVec matVec;
    private boolean closed;

    CpuVocosDecoder(VocosDecoder dec, Weights w, CpuMatVec matVec) {
        this.dec = Objects.requireNonNull(dec, "decoder");
        this.w = Objects.requireNonNull(w, "weights");
        this.matVec = Objects.requireNonNull(matVec, "matVec");
    }

    /** Resolves the geometry's tensors through the name lookups: F32 vectors and matmul references. */
    static Weights resolve(VocosDecoder dec, Function<String, float[]> floats,
                           Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(dec, "decoder");
        List<Block> blocks = new ArrayList<>();
        for (int il = 0; il < dec.blocks(); il++) {
            blocks.add(new Block(
                    floats.apply(VocosDecoder.dwconv(il, "weight")),
                    floats.apply(VocosDecoder.dwconv(il, "bias")),
                    floats.apply(VocosDecoder.block(il, "norm", "weight")),
                    floats.apply(VocosDecoder.block(il, "norm", "bias")),
                    refs.apply(VocosDecoder.block(il, "pw1", "weight")),
                    floats.apply(VocosDecoder.block(il, "pw1", "bias")),
                    refs.apply(VocosDecoder.block(il, "pw2", "weight")),
                    floats.apply(VocosDecoder.block(il, "pw2", "bias")),
                    floats.apply("a.gen.wav.up.blk." + il + ".gamma")));
        }
        return new Weights(dec.convK(),
                dec.convK() > 0 ? floats.apply(VocosDecoder.weight(VocosDecoder.INPUT)) : null,
                dec.convK() > 0 ? null : refs.apply(VocosDecoder.weight(VocosDecoder.INPUT)),
                floats.apply(VocosDecoder.bias(VocosDecoder.INPUT)),
                floats.apply(VocosDecoder.weight(VocosDecoder.NORM)),
                floats.apply(VocosDecoder.bias(VocosDecoder.NORM)),
                Collections.unmodifiableList(blocks),
                floats.apply(VocosDecoder.weight(VocosDecoder.OUTPUT_NORM)),
                floats.apply(VocosDecoder.bias(VocosDecoder.OUTPUT_NORM)),
                refs.apply(VocosDecoder.weight(VocosDecoder.OUTPUT)),
                floats.apply(VocosDecoder.bias(VocosDecoder.OUTPUT)));
    }

    /**
     * Binds to explicit weights, every tensor resolved by the geometry's names.
     *
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a tensor is missing
     */
    public static CpuVocosDecoder bind(VocosDecoder dec, CpuWeights weights) {
        return bind(dec, weights, CpuMatVec.PLAIN);
    }

    /** {@link #bind(VocosDecoder, CpuWeights)} with an explicit matmul channel (the tests' seam). */
    public static CpuVocosDecoder bind(VocosDecoder dec, CpuWeights weights, CpuMatVec matVec) {
        Objects.requireNonNull(weights, "weights");
        Objects.requireNonNull(matVec, "matVec");
        return new CpuVocosDecoder(dec, resolve(dec, weights::floats, weights::get), matVec);
    }

    /** The geometry this executor was bound to. */
    public VocosDecoder decoder() {
        return dec;
    }

    /**
     * Decodes one feature batch to polar spectra.
     *
     * @param features {@code [frames][nInp()]} normalized hidden-state rows
     * @return {@code [frames][nFrameVals()]} rows: {@code nBins()} interleaved {@code (real, imaginary)} values
     * @throws IllegalArgumentException when the batch is empty, too small, or of another width
     * @throws IllegalStateException    when closed
     */
    public float[][] spectrum(float[][] features) {
        ensureOpen();
        Objects.requireNonNull(features, "features");
        int frames = features.length;
        int inp = frames == 0 ? 0 : features[0].length;
        dec.checkFeatures(frames, inp);
        dec.checkCapacity(frames);
        int width = dec.width();
        int t = frames;
        int up = dec.upsampledFrames(frames);

        // [T][inp] -> channel-major [inp][T]
        float[] x = new float[inp * t];
        toChannelMajor(features, t, inp, x);
        // bilinear align-corners upsample along time: [inp][T] -> [inp][up]
        float[] upped = new float[inp * up];
        upsample(inp, t, up, x, upped);

        float[] y;
        if (w.convK > 0) {
            y = new float[width * up];
            CpuConv1d.same(up, inp, width, w.convK, 1, upped, 0, w.inputConvW, w.inputB, y, 0);
        } else {
            // y[c][u] = sum_i inputProj[c][i] * upped[i][u]
            float[] tok2 = new float[up * inp];
            toTokenMajor(upped, inp, up, tok2);
            float[] proj = new float[up * width];
            matVec.matmul(w.inputProj, tok2, 0, inp, proj, 0, width, up);
            CpuOps.addRowBias(proj, 0, up, width, w.inputB, 0);
            y = toChannelMajor(proj, up, width);
        }

        // LayerNorm over channels per frame, token-major
        float[] tok = new float[up * width];
        toTokenMajor(y, width, up, tok);
        CpuOps.layerNorm(tok, 0, tok, 0, up, width, w.normW, w.normB, dec.eps());

        for (Block b : w.blocks) {
            block(b, tok, up, width);
        }

        CpuOps.layerNorm(tok, 0, tok, 0, up, width, w.outputNormW, w.outputNormB, dec.eps());
        int out = dec.nFrameVals();
        float[] head = new float[up * out];
        matVec.matmul(w.output, tok, 0, width, head, 0, out, up);
        CpuOps.addRowBias(head, 0, up, out, w.outputB, 0);

        return polar(head, up, dec.nBins());
    }

    /** One Vocos block over token-major rows, written back into them (soprano.cpp:26-40). */
    private void block(Block b, float[] tok, int t, int c) {
        // depthwise conv over time, channel-major in and out
        float[] res = new float[t * c];
        System.arraycopy(tok, 0, res, 0, t * c);
        float[] cm = toChannelMajor(res, t, c);
        float[] dwd = new float[t * c];
        CpuConv1d.depthwise(t, c, VocosDecoder.DW_K, cm, 0, b.dwconvW, b.dwconvB, dwd, 0);
        float[] dwTok = new float[t * c];
        toTokenMajor(dwd, c, t, dwTok);
        CpuOps.layerNorm(dwTok, 0, dwTok, 0, t, c, b.normW, b.normB, dec.eps());

        int ff = dec.nFf();
        float[] h = new float[t * ff];
        matVec.matmul(b.pw1, dwTok, 0, c, h, 0, ff, t);
        CpuOps.addRowBias(h, 0, t, ff, b.pw1B, 0);
        CpuOps.unary(ActOp.GELU_ERF, ActParams.NONE, h, 0, h, 0, t * ff);
        float[] back = new float[t * c];
        matVec.matmul(b.pw2, h, 0, ff, back, 0, c, t);
        CpuOps.addRowBias(back, 0, t, c, b.pw2B, 0);

        // residual + gamma * back, per channel
        for (int ch = 0; ch < c; ch++) {
            float g = b.gamma[ch];
            for (int u = 0; u < t; u++) {
                tok[u * c + ch] = res[u * c + ch] + g * back[u * c + ch];
            }
        }
    }

    /** Polar head: log-magnitudes then phases per row become interleaved (real, imaginary) bins. */
    private static float[][] polar(float[] head, int t, int nBins) {
        float[][] rows = new float[t][2 * nBins];
        for (int u = 0; u < t; u++) {
            int off = u * 2 * nBins;
            for (int j = 0; j < nBins; j++) {
                float mag = (float) Math.exp(head[off + j]);
                mag = Math.max(0.0f, Math.min(100.0f, mag));
                float phase = head[off + nBins + j];
                rows[u][2 * j] = mag * (float) Math.cos(phase);
                rows[u][2 * j + 1] = mag * (float) Math.sin(phase);
            }
        }
        return rows;
    }

    /** Bilinear {@code align_corners} time upsample of a channel-major signal (soprano.cpp:14). */
    private static void upsample(int channels, int t, int up, float[] x, float[] y) {
        if (up == t) {
            System.arraycopy(x, 0, y, 0, channels * t);
            return;
        }
        double scale = (double) (t - 1) / (double) (up - 1);
        for (int c = 0; c < channels; c++) {
            int row = c * t;
            int outRow = c * up;
            for (int u = 0; u < up; u++) {
                double src = u * scale;
                int i0 = (int) Math.floor(src);
                if (i0 >= t - 1) {
                    y[outRow + u] = x[row + t - 1];
                    continue;
                }
                float frac = (float) (src - i0);
                y[outRow + u] = x[row + i0] + frac * (x[row + i0 + 1] - x[row + i0]);
            }
        }
    }

    private static void toChannelMajor(float[][] rows, int t, int c, float[] dst) {
        for (int u = 0; u < t; u++) {
            float[] row = rows[u];
            for (int ch = 0; ch < c; ch++) {
                dst[ch * t + u] = row[ch];
            }
        }
    }

    private static float[] toChannelMajor(float[] tok, int t, int c) {
        float[] cm = new float[t * c];
        for (int u = 0; u < t; u++) {
            for (int ch = 0; ch < c; ch++) {
                cm[ch * t + u] = tok[u * c + ch];
            }
        }
        return cm;
    }

    private static void toTokenMajor(float[] cm, int c, int t, float[] tok) {
        for (int u = 0; u < t; u++) {
            for (int ch = 0; ch < c; ch++) {
                tok[u * c + ch] = cm[ch * t + u];
            }
        }
    }

    /** Frees nothing today; present so a device-backed executor can share the contract. */
    @Override
    public void close() {
        closed = true;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the soprano vocos executor is closed");
        }
    }

    @Override
    public String toString() {
        return "CpuVocosDecoder(" + dec + ")";
    }
}
