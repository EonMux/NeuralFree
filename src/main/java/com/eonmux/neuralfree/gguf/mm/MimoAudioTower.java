package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.audio.MelChunk;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

import static com.eonmux.neuralfree.gguf.mm.ClipShapes.checkVector;
import static com.eonmux.neuralfree.gguf.mm.ClipShapes.shape;

/**
 * The MiMo audio tokenizer encoder and its projector (llama.cpp tools/mtmd/models/mimo-audio.cpp): two half-padded
 * kernel-3 convs with GELU-erf (the second striding by two), the ViT blocks with layer norms, NEOX RoPE over the
 * frame positions and per block ({@code clip.audio.wa_pattern_mode}) causal attention to all past frames
 * ({@code -1}) or to the {@code clip.audio.window_size} frames before it, the output of block
 * {@value #SKIP_LAYER} added back as a skip connection, the post layer norm, the stride-2 downsample conv with
 * GELU-erf and its layer norm, the residual vector quantization (per row and quantizer the codebook entry
 * maximizing {@code 2 c.r - |c|^2}, subtracted from the residual), the sum of the code embeddings, the local
 * transformer over groups of {@code clip.audio.local_group_size} rows (zero rows pad the last group) attending
 * within their group, and the GELU-erf MLP over each flattened group.
 * <p>
 * The skip connection adds the complete output of block {@value #SKIP_LAYER} (encoder_skip_layer_id 3 of the
 * tokenizer config), as the reference encoder does; mimo-audio.cpp states the same intent but captures the block's
 * stream before its FFN residual. Package-private executor support.
 */
final class MimoAudioTower implements FamilyEncoder {

    /** The block whose output the skip connection adds (zero-based; encoder_skip_layer_id 3). */
    static final int SKIP_LAYER = 2;
    /** RoPE base of the local transformer (audio_config.rope_theta in mimo-audio.cpp). */
    static final float LOCAL_ROPE_THETA = 640_000f;
    private static final int K = 3;
    private static final int DOWNSAMPLE = 2;

    private final int nMel;
    private final int width;
    private final int heads;
    private final int dh;
    private final float eps;
    private final float theta;
    private final int window;
    private final int[] waPattern;
    private final int group;
    private final CpuMatVec matVec;
    private final CpuTensorRef conv1;
    private final float[] conv1B;
    private final CpuTensorRef conv2;
    private final float[] conv2B;
    private final CpuVitTower tower;
    private final float[] postLnW;
    private final float[] postLnB;
    private final CpuTensorRef dsConv;
    private final float[] dsNormW;
    private final float[] dsNormB;
    private final Rvq rvq;
    private final LocalLayer[] localLayers;
    private final float[] localNorm;
    private final CpuTensorRef mm1;
    private final CpuTensorRef mm2;

    private MimoAudioTower(ClipHparams hp, TensorCatalog tc, CpuMatVec matVec, Function<String, float[]> floats,
                           Function<String, CpuTensorRef> refs) {
        ClipAudioHparams a = hp.audio();
        this.nMel = a.nMelBins();
        this.width = hp.nEmbd();
        this.heads = hp.nHead();
        this.dh = hp.dHead();
        this.eps = hp.eps();
        this.theta = a.ropeTheta();
        this.window = a.attnWindow();
        this.waPattern = a.waPattern();
        this.group = a.localGroup();
        this.matVec = matVec;
        this.conv1 = CpuVitTower.matrix(refs.apply("a.conv1d.1.weight"), width, (long) K * nMel);
        this.conv1B = floats.apply("a.conv1d.1.bias");
        this.conv2 = CpuVitTower.matrix(refs.apply("a.conv1d.2.weight"), width, (long) K * width);
        this.conv2B = floats.apply("a.conv1d.2.bias");
        List<ClipLayer> blocks = new ArrayList<>(hp.nLayer());
        for (int il = 0; il < hp.nLayer(); il++) {
            blocks.add(ClipLayer.resolve(tc, hp, il, false));
        }
        this.tower = new CpuVitTower(hp, matVec, blocks, tc.get("a.pre_ln.weight"), tc.get("a.pre_ln.bias"), null,
                null, floats, refs, null);
        this.postLnW = optional(tc, floats, "a.post_ln.weight");
        this.postLnB = optional(tc, floats, "a.post_ln.bias");
        this.dsConv = CpuVitTower.matrix(refs.apply("a.downsample.conv.weight"), width, (long) DOWNSAMPLE * width);
        this.dsNormW = floats.apply("a.downsample.norm.weight");
        this.dsNormB = floats.apply("a.downsample.norm.bias");
        this.rvq = new Rvq(tc, a.rvqCodebookSizes(), floats);
        this.localLayers = new LocalLayer[a.localLayers()];
        for (int il = 0; il < localLayers.length; il++) {
            localLayers[il] = new LocalLayer(hp, il, floats, refs);
        }
        this.localNorm = floats.apply("mm.a.local_norm.weight");
        this.mm1 = refs.apply("mm.a.mlp.1.weight");
        this.mm2 = refs.apply("mm.a.mlp.2.weight");
    }

    private static float[] optional(TensorCatalog tc, Function<String, float[]> floats, String name) {
        return tc.has(name) ? floats.apply(name) : null;
    }

    /** Output rows of a chunk of {@code nFrames} mel frames: the stride-2 stem, the downsample, the groups. */
    static int tokens(int nFrames, int group) {
        int n = (nFrames - 1) / 2 + 1;
        n = (n - DOWNSAMPLE) / DOWNSAMPLE + 1;
        return (n + group - 1) / group;
    }

    /** Shape-checks the stem, the blocks, the downsample, the RVQ tables, the local transformer and the MLP. */
    static void validate(ClipHparams hp, TensorCatalog tc) {
        int e = hp.nEmbd();
        ClipAudioHparams a = hp.audio();
        if (hp.nLayer() <= SKIP_LAYER) {
            throw new IllegalArgumentException("mimo_audio needs more than " + SKIP_LAYER + " encoder blocks for"
                    + " the skip connection, got " + hp.nLayer());
        }
        shape(tc, tc.require("a.conv1d.1.weight"), K, a.nMelBins(), e);
        shape(tc, tc.require("a.conv1d.1.bias"), 1, e);
        shape(tc, tc.require("a.conv1d.2.weight"), K, e, e);
        shape(tc, tc.require("a.conv1d.2.bias"), 1, e);
        for (int il = 0; il < hp.nLayer(); il++) {
            ClipLayer.resolve(tc, hp, il, false);
        }
        checkVector(tc, tc.get("a.pre_ln.weight"), e);
        checkVector(tc, tc.get("a.pre_ln.bias"), e);
        checkVector(tc, tc.get("a.post_ln.weight"), e);
        checkVector(tc, tc.get("a.post_ln.bias"), e);
        shape(tc, tc.require("a.downsample.conv.weight"), DOWNSAMPLE, e, e);
        shape(tc, tc.require("a.downsample.norm.weight"), e);
        shape(tc, tc.require("a.downsample.norm.bias"), e);
        Rvq.validate(tc, e, a.rvqCodebookSizes());
        for (int il = 0; il < a.localLayers(); il++) {
            LocalLayer.validate(tc, e, il);
        }
        shape(tc, tc.require("mm.a.local_norm.weight"), e);
        TensorRef m1 = tc.require("mm.a.mlp.1.weight");
        long hidden = m1.ne(1);
        shape(tc, m1, (long) e * a.localGroup(), hidden);
        shape(tc, tc.require("mm.a.mlp.2.weight"), hidden, hp.projectionDim());
    }

    /** Binds the executor over {@code model} with the weights of {@code floats} / {@code refs}. */
    static MimoAudioTower bind(ClipAudioModel model, CpuMatVec matVec, Function<String, float[]> floats,
                               Function<String, CpuTensorRef> refs) {
        Objects.requireNonNull(model, "model");
        return new MimoAudioTower(model.hparams(), model.catalog(), Objects.requireNonNull(matVec, "matVec"),
                floats, refs);
    }

    /** The encoder, the downsample, the RVQ codes, the local transformer and the projector over the chunk. */
    @Override
    public float[][] encode(MelChunk chunk) {
        if (chunk.nMel() != nMel) {
            throw new IllegalArgumentException("mel chunk has " + chunk.nMel() + " bins, the encoder needs " + nMel);
        }
        float[] c1 = conv(conv1, conv1B, chunk.data(), chunk.nLen(), nMel, 1);
        int n = AudioOps.convOut(chunk.nLen(), K, 2, 1);
        if (n < DOWNSAMPLE) {
            throw new IllegalArgumentException("mel chunk of " + chunk.nLen() + " frames gives " + n
                    + " encoder frames; the mimo_audio downsample needs at least " + DOWNSAMPLE);
        }
        float[] x = AudioOps.transpose(conv(conv2, conv2B, c1, chunk.nLen(), width, 2), width, n);
        x = encoder(x, n);
        int nDs = AudioOps.convOut(n, DOWNSAMPLE, DOWNSAMPLE, 0);
        float[] ds = AudioOps.conv2d(matVec, AudioOps.transpose(x, n, width), n, 1, width, dsConv, DOWNSAMPLE, 1,
                DOWNSAMPLE, 0, null);
        AudioOps.activate(ActOp.GELU_ERF, ds);
        ds = AudioOps.transpose(ds, width, nDs);
        AudioOps.layerNorm(ds, nDs, width, dsNormW, dsNormB, eps);
        int groups = (nDs + group - 1) / group;
        float[] local = new float[groups * group * width];
        System.arraycopy(rvq.embed(ds, nDs, width), 0, local, 0, nDs * width);
        localTransformer(local, groups * group);
        float[] h = AudioOps.linear(matVec, mm1, null, local, groups);
        AudioOps.activate(ActOp.GELU_ERF, h);
        return VisionRows.tokenRows(AudioOps.linear(matVec, mm2, null, h, groups), (int) mm2.rows());
    }

    /** The blocks with RoPE and their causal masks, the skip connection, the post layer norm. */
    private float[] encoder(float[] x, int n) {
        int[] pos = AudioOps.iota(n);
        VisionRows.KeyRanges all = keyRange(n);
        VisionRows.KeyRanges windowed = keyRange(window + 1);
        float[][] skip = new float[1][];
        CpuVitTower.Options opts = CpuVitTower.Options.NONE
                .withPos((rows, t, nHeads) -> AudioOps.ropeNeox(rows, t, nHeads, dh, dh, pos, theta))
                .withLayerWindows(il -> waPattern[il] == -1 ? all : windowed)
                .withTap((done, rows, t) -> {
                    if (done == SKIP_LAYER + 1) {
                        skip[0] = rows.clone();
                    }
                });
        float[] out = tower.forward(x, n, NormKind.LAYER, opts);
        AudioOps.addScaled(out, skip[0], 1f);
        AudioOps.layerNorm(out, n, width, postLnW, postLnB, eps);
        return out;
    }

    /** The causal key range of the {@code keys} most recent frames, the query frame included. */
    private static VisionRows.KeyRanges keyRange(int keys) {
        return new VisionRows.KeyRanges() {
            @Override
            public int begin(int row) {
                return Math.max(0, row - keys + 1);
            }

            @Override
            public int end(int row) {
                return row + 1;
            }
        };
    }

    /** The local layers over the padded {@code [n][width]} rows (in-group positions and keys), then the norm. */
    private void localTransformer(float[] x, int n) {
        int[] pos = new int[n];
        for (int i = 0; i < n; i++) {
            pos[i] = i % group;
        }
        VisionRows.KeyRanges inGroup = new VisionRows.KeyRanges() {
            @Override
            public int begin(int row) {
                return row - row % group;
            }

            @Override
            public int end(int row) {
                return row - row % group + group;
            }
        };
        for (LocalLayer l : localLayers) {
            l.forward(matVec, x, n, pos, inGroup, eps);
        }
        AudioOps.rmsNorm(x, n, width, localNorm, null, eps);
    }

    /**
     * {@code ggml_conv_1d(w, in, stride, 1, 1)} (kernel 3, half padding) plus the bias and the GELU-erf over a
     * channel-major {@code [nCh][len]} input; the output is channel-major.
     */
    private float[] conv(CpuTensorRef w, float[] bias, float[] in, int len, int nCh, int stride) {
        float[] out = AudioOps.conv2d(matVec, in, len, 1, nCh, w, K, 1, stride, 1, bias);
        AudioOps.activate(ActOp.GELU_ERF, out);
        return out;
    }

    /** The residual vector quantizer {@code a.rvq.codebook} and the code embeddings {@code mm.a.code_embd}. */
    private static final class Rvq {
        /** Per quantizer {@code [bins][dim]} codebook entries, their squared norms and the code embeddings. */
        private final float[][] codebooks;
        private final float[][] norms;
        private final float[][] embeddings;
        private final int dim;

        Rvq(TensorCatalog tc, int[] bins, Function<String, float[]> floats) {
            TensorRef cb = tc.require("a.rvq.codebook.weight");
            TensorRef ce = tc.require("mm.a.code_embd.weight");
            this.dim = (int) cb.ne(0);
            float[] cbAll = floats.apply(cb.name());
            float[] ceAll = floats.apply(ce.name());
            this.codebooks = new float[bins.length][];
            this.norms = new float[bins.length][];
            this.embeddings = new float[bins.length][];
            for (int q = 0; q < bins.length; q++) {
                int span = bins[q] * dim;
                codebooks[q] = Arrays.copyOfRange(cbAll, (int) (q * cb.ne(1) * dim),
                        (int) (q * cb.ne(1) * dim) + span);
                embeddings[q] = Arrays.copyOfRange(ceAll, (int) (q * ce.ne(1) * dim),
                        (int) (q * ce.ne(1) * dim) + span);
                norms[q] = new float[bins[q]];
                for (int b = 0; b < bins[q]; b++) {
                    double s = 0;
                    for (int c = 0; c < dim; c++) {
                        float v = codebooks[q][b * dim + c];
                        s += (double) v * v;
                    }
                    norms[q][b] = (float) s;
                }
            }
        }

        /** Codebook and code embedding {@code [dim, rows, quantizers]}, each with at least the configured bins. */
        static void validate(TensorCatalog tc, int dim, int[] bins) {
            TensorRef cb = tc.require("a.rvq.codebook.weight");
            TensorRef ce = tc.require("mm.a.code_embd.weight");
            for (TensorRef t : new TensorRef[]{cb, ce}) {
                if (t.rank() != 3 || t.ne(0) != dim || t.ne(2) != bins.length) {
                    throw new IllegalArgumentException("tensor " + t.name() + " has shape " + t + ", mimo_audio"
                            + " needs {" + dim + ", bins, " + bins.length + "}");
                }
            }
            for (int q = 0; q < bins.length; q++) {
                if (bins[q] > cb.ne(1) || bins[q] > ce.ne(1)) {
                    throw new IllegalArgumentException("clip.audio.rvq.codebook_size entry " + q + " (" + bins[q]
                            + ") exceeds the codebook or code embedding tables");
                }
            }
        }

        /** The summed code embeddings of the quantized {@code [n][dim]} rows. */
        float[] embed(float[] x, int n, int width) {
            float[] out = new float[n * width];
            float[] residual = x.clone();
            for (int t = 0; t < n; t++) {
                for (int q = 0; q < codebooks.length; q++) {
                    int best = nearest(q, residual, t * dim);
                    for (int c = 0; c < dim; c++) {
                        residual[t * dim + c] -= codebooks[q][best * dim + c];
                        out[t * width + c] += embeddings[q][best * dim + c];
                    }
                }
            }
            return out;
        }

        /** The first codebook entry of quantizer {@code q} maximizing {@code 2 c.r - |c|^2} ({@code ggml_argmax}). */
        private int nearest(int q, float[] r, int off) {
            int best = 0;
            float bestScore = Float.NEGATIVE_INFINITY;
            for (int b = 0; b < norms[q].length; b++) {
                float dot = 0f;
                for (int c = 0; c < dim; c++) {
                    dot += codebooks[q][b * dim + c] * r[off + c];
                }
                float score = 2f * dot - norms[q][b];
                if (score > bestScore) {
                    best = b;
                    bestScore = score;
                }
            }
            return best;
        }
    }

    /** One local transformer layer {@code mm.a.local_blk.N}: RMS norms, biased Q/K/V, NEOX RoPE, gated SiLU FFN. */
    private static final class LocalLayer {
        private final int width;
        private final int heads;
        private final int dh;
        private final float[] ln1;
        private final float[] ln2;
        private final CpuTensorRef q;
        private final float[] qB;
        private final CpuTensorRef k;
        private final float[] kB;
        private final CpuTensorRef v;
        private final float[] vB;
        private final CpuTensorRef o;
        private final CpuTensorRef gate;
        private final CpuTensorRef up;
        private final CpuTensorRef down;

        LocalLayer(ClipHparams hp, int il, Function<String, float[]> floats, Function<String, CpuTensorRef> refs) {
            String p = prefix(il);
            this.width = hp.nEmbd();
            this.heads = hp.nHead();
            this.dh = hp.dHead();
            this.ln1 = floats.apply(p + "ln1.weight");
            this.ln2 = floats.apply(p + "ln2.weight");
            this.q = refs.apply(p + "attn_q.weight");
            this.qB = floats.apply(p + "attn_q.bias");
            this.k = refs.apply(p + "attn_k.weight");
            this.kB = floats.apply(p + "attn_k.bias");
            this.v = refs.apply(p + "attn_v.weight");
            this.vB = floats.apply(p + "attn_v.bias");
            this.o = refs.apply(p + "attn_out.weight");
            this.gate = refs.apply(p + "ffn_gate.weight");
            this.up = refs.apply(p + "ffn_up.weight");
            this.down = refs.apply(p + "ffn_down.weight");
        }

        static String prefix(int il) {
            return "mm.a.local_blk." + il + ".";
        }

        static void validate(TensorCatalog tc, int e, int il) {
            String p = prefix(il);
            shape(tc, tc.require(p + "ln1.weight"), e);
            shape(tc, tc.require(p + "ln2.weight"), e);
            for (String proj : new String[]{"attn_q", "attn_k", "attn_v"}) {
                shape(tc, tc.require(p + proj + ".weight"), e, e);
                shape(tc, tc.require(p + proj + ".bias"), e);
            }
            shape(tc, tc.require(p + "attn_out.weight"), e, e);
            TensorRef upW = tc.require(p + "ffn_up.weight");
            long ff = upW.ne(1);
            shape(tc, upW, e, ff);
            shape(tc, tc.require(p + "ffn_gate.weight"), e, ff);
            shape(tc, tc.require(p + "ffn_down.weight"), ff, e);
        }

        void forward(CpuMatVec matVec, float[] x, int n, int[] pos, VisionRows.KeyRanges ranges, float eps) {
            float[] h = x.clone();
            AudioOps.rmsNorm(h, n, width, ln1, null, eps);
            float[] qr = AudioOps.linear(matVec, q, qB, h, n);
            float[] kr = AudioOps.linear(matVec, k, kB, h, n);
            float[] vr = AudioOps.linear(matVec, v, vB, h, n);
            AudioOps.ropeNeox(qr, n, heads, dh, dh, pos, LOCAL_ROPE_THETA);
            AudioOps.ropeNeox(kr, n, heads, dh, dh, pos, LOCAL_ROPE_THETA);
            float[] a = VisionRows.attend(qr, 0, n, kr, vr, 0, n, width, heads, 1, dh, (float) (1.0 / Math.sqrt(dh)),
                    ranges);
            AudioOps.addScaled(x, AudioOps.linear(matVec, o, null, a, n), 1f);
            h = x.clone();
            AudioOps.rmsNorm(h, n, width, ln2, null, eps);
            float[] u = AudioOps.linear(matVec, up, null, h, n);
            float[] g = AudioOps.linear(matVec, gate, null, h, n);
            AudioOps.activate(ActOp.SILU, g);
            for (int i = 0; i < u.length; i++) {
                u[i] *= g[i];
            }
            AudioOps.addScaled(x, AudioOps.linear(matVec, down, null, u, n), 1f);
        }
    }
}
