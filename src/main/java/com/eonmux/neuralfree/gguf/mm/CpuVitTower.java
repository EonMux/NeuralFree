package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.IntFunction;

/**
 * CPU executor of the pre-norm ViT tower every clip encoder shares, vision and audio alike (llama.cpp
 * {@code clip_graph::build_vit}, tools/mtmd/clip.cpp): optional pre-norm, then per block norm 1, fused or separate
 * Q/K/V (grouped-query heads when {@code n_head_kv < n_head}), the optional Q/K norms, an optional position hook on
 * Q and K, the optional weightless RMS norm of every value head, bidirectional softmax attention at
 * {@code 1/sqrt(d_head)} or the options' scale, each query row over all keys or over its per-layer key window, with
 * the optional per-head sinks, the output projection, the optional layer scale {@code ls1}, the optional
 * post-attention norm and the residual, norm 2, the gated or plain FFN with the encoder's activation (or the routed
 * experts of a MoE block), the optional post-FFN norm, the optional layer scale {@code ls2}, the residual and the
 * optional block output scale (or the post-norm layout of CogVLM), and an optional tap of the stream between
 * blocks; then the audio average pool of the projectors that pool, and the optional post-norm. A block matrix with
 * clamp bounds ({@link ClipClamp}) clamps its input and output. Matmuls run through {@link CpuMatVec} in the stored
 * weight types, so quantized mmproj weights run as they do for the text model. Activations are token-major
 * {@code [nTokens][nEmbd]}. Immutable weights; not thread-safe across calls.
 */
final class CpuVitTower {

    /** Rotary position hook applied to the Q and K rows of every block ({@code add_pos} of build_vit). */
    interface PosHook {
        /** Rotates {@code nTokens} token-major rows of {@code nHeads} heads each of {@code x} in place. */
        void apply(float[] x, int nTokens, int nHeads);
    }

    /** Observer of the residual stream between blocks. */
    interface LayerTap {
        /**
         * Called with the stream after {@code done} blocks (0: after the pre-norm, before any block); the rows are
         * the live buffer, so an observer that keeps them copies them and must not modify them.
         */
        void observe(int done, float[] rows, int nTokens);
    }

    /**
     * Per-call options of {@link #forward}: the attention windows per layer, a position hook, the audio average
     * pool, the post-norm block layout, the blocks to run, a tap between blocks, the attention scale, the value
     * norm and the attention sinks.
     */
    static final class Options {
        static final Options NONE = new Options(null, null, false, false, -1, null, -1, Float.NaN, false, false);

        private final IntFunction<VisionRows.KeyRanges> layerWindows;
        private final PosHook pos;
        private final boolean avgPool;
        private final boolean postNorm;
        private final int nLayers;
        private final LayerTap tap;
        private final int firstBlock;
        private final float attnScale;
        private final boolean valueNorm;
        private final boolean sinks;

        private Options(IntFunction<VisionRows.KeyRanges> layerWindows, PosHook pos, boolean avgPool,
                        boolean postNorm, int nLayers, LayerTap tap, int firstBlock, float attnScale,
                        boolean valueNorm, boolean sinks) {
            this.layerWindows = layerWindows;
            this.pos = pos;
            this.avgPool = avgPool;
            this.postNorm = postNorm;
            this.nLayers = nLayers;
            this.tap = tap;
            this.firstBlock = firstBlock;
            this.attnScale = attnScale;
            this.valueNorm = valueNorm;
            this.sinks = sinks;
        }

        private Options with(IntFunction<VisionRows.KeyRanges> windows, PosHook hook, boolean pool, boolean post,
                             int layers, LayerTap observer) {
            return new Options(windows, hook, pool, post, layers, observer, firstBlock, attnScale, valueNorm, sinks);
        }

        /**
         * The key range of every query row per layer index (a block-diagonal window attention); the function
         * returns null for a layer whose rows attend to all keys.
         */
        Options withLayerWindows(IntFunction<VisionRows.KeyRanges> windows) {
            return with(Objects.requireNonNull(windows, "windows"), pos, avgPool, postNorm, nLayers, tap);
        }

        Options withPos(PosHook hook) {
            return with(layerWindows, Objects.requireNonNull(hook, "hook"), avgPool, postNorm, nLayers, tap);
        }

        /** {@code ggml_pool_1d(AVG, 2, 2)} over the token axis after the blocks, before the post-norm. */
        Options withAvgPool(boolean on) {
            return with(layerWindows, pos, on, postNorm, nLayers, tap);
        }

        /**
         * The CogVLM block layout: attention and FFN read the residual stream unnormalized, and {@code ln1} /
         * {@code ln2} normalize their outputs before each residual add.
         */
        Options withPostNorm() {
            return with(layerWindows, pos, avgPool, true, nLayers, tap);
        }

        /** Runs only the first {@code n} blocks (the llava graph stops at its deepest feature layer). */
        Options withLayers(int n) {
            if (n < 0) {
                throw new IllegalArgumentException("block count " + n + " is negative");
            }
            return with(layerWindows, pos, avgPool, postNorm, n, tap);
        }

        /** Taps the residual stream before the first block and after every block (feature layers, deepstack). */
        Options withTap(LayerTap observer) {
            return with(layerWindows, pos, avgPool, postNorm, nLayers, Objects.requireNonNull(observer, "tap"));
        }

        /** Runs the blocks {@code [from, to)} as a segment, without the pre-norm, post-norm or average pool. */
        Options withBlocks(int from, int to) {
            if (from < 0 || to < from) {
                throw new IllegalArgumentException("block range [" + from + ", " + to + ") is invalid");
            }
            return new Options(layerWindows, pos, avgPool, postNorm, to, tap, from, attnScale, valueNorm, sinks);
        }

        /** The softmax scale {@code scale} instead of {@code 1/sqrt(d_head)}. */
        Options withAttnScale(float scale) {
            return new Options(layerWindows, pos, avgPool, postNorm, nLayers, tap, firstBlock, scale, valueNorm,
                    sinks);
        }

        /** RMS-normalizes every value head without a weight before the attention. */
        Options withValueNorm() {
            return new Options(layerWindows, pos, avgPool, postNorm, nLayers, tap, firstBlock, attnScale, true,
                    sinks);
        }

        /** Adds every block's per-head attention sinks to the softmax; every block run must carry them. */
        Options withSinks() {
            return new Options(layerWindows, pos, avgPool, postNorm, nLayers, tap, firstBlock, attnScale,
                    valueNorm, true);
        }
    }

    /** Resolved weights of one block; absent optionals are null. */
    private static final class LayerW {
        final boolean fusedQkv;
        final CpuTensorRef qkv;
        final CpuProjectionDeltas.Site qkvSite;
        final float[] qkvB;
        final CpuTensorRef q;
        final CpuProjectionDeltas.Site qSite;
        final float[] qB;
        final CpuTensorRef k;
        final CpuProjectionDeltas.Site kSite;
        final float[] kB;
        final CpuTensorRef v;
        final CpuProjectionDeltas.Site vSite;
        final float[] vB;
        final CpuTensorRef o;
        final CpuProjectionDeltas.Site oSite;
        final float[] oB;
        final float[] ln1W;
        final float[] ln1B;
        final float[] ln2W;
        final float[] ln2B;
        final CpuTensorRef ffnUp;
        final CpuProjectionDeltas.Site ffnUpSite;
        final float[] ffnUpB;
        final CpuTensorRef ffnGate;
        final CpuProjectionDeltas.Site ffnGateSite;
        final float[] ffnGateB;
        final CpuTensorRef ffnDown;
        final CpuProjectionDeltas.Site ffnDownSite;
        final float[] ffnDownB;
        final float[] ls1;
        final float[] ls2;
        final float[] qNorm;
        final float[] kNorm;
        final Moe moe;
        final float[] attnPostNorm;
        final float[] ffnPostNorm;
        final float[] outScale;
        final float[] sinks;
        final int dHead;
        /** Clamp bounds {@code {inMin, inMax, outMin, outMax}} per matrix, null when unclamped. */
        final float[] qkvClamp;
        final float[] qClamp;
        final float[] kClamp;
        final float[] vClamp;
        final float[] oClamp;
        final float[] upClamp;
        final float[] gateClamp;
        final float[] downClamp;

        LayerW(ClipLayer l, Function<String, float[]> floats, Function<String, CpuTensorRef> refs,
               CpuProjectionDeltas deltas, CpuMatVec matVec) {
            this.fusedQkv = l.fusedQkv();
            this.qkv = refOrNull(l.qkvWeight(), refs);
            this.qkvSite = loraSite(l.qkvWeight(), qkv, deltas, matVec);
            this.qkvB = floatsOrNull(l.qkvBias(), floats);
            this.q = refOrNull(l.qWeight(), refs);
            this.qSite = loraSite(l.qWeight(), q, deltas, matVec);
            this.qB = floatsOrNull(l.qBias(), floats);
            this.k = refOrNull(l.kWeight(), refs);
            this.kSite = loraSite(l.kWeight(), k, deltas, matVec);
            this.kB = floatsOrNull(l.kBias(), floats);
            this.v = refOrNull(l.vWeight(), refs);
            this.vSite = loraSite(l.vWeight(), v, deltas, matVec);
            this.vB = floatsOrNull(l.vBias(), floats);
            this.o = refs.apply(l.oWeight().name());
            this.oSite = loraSite(l.oWeight(), o, deltas, matVec);
            this.oB = floatsOrNull(l.oBias(), floats);
            this.ln1W = floats.apply(l.ln1Weight().name());
            this.ln1B = floatsOrNull(l.ln1Bias(), floats);
            this.ln2W = floats.apply(l.ln2Weight().name());
            this.ln2B = floatsOrNull(l.ln2Bias(), floats);
            this.moe = l.moe() ? new Moe(l, floats, refs, deltas, matVec) : null;
            this.ffnUp = moe != null ? null : refs.apply(l.ffnUpWeight().name());
            this.ffnUpSite = moe != null ? null : loraSite(l.ffnUpWeight(), ffnUp, deltas, matVec);
            this.ffnUpB = moe != null ? null : floatsOrNull(l.ffnUpBias(), floats);
            this.ffnGate = moe != null ? null : refOrNull(l.ffnGateWeight(), refs);
            this.ffnGateSite = moe != null ? null : loraSite(l.ffnGateWeight(), ffnGate, deltas, matVec);
            this.ffnGateB = moe != null ? null : floatsOrNull(l.ffnGateBias(), floats);
            this.ffnDown = moe != null ? null : refs.apply(l.ffnDownWeight().name());
            this.ffnDownSite = moe != null ? null : loraSite(l.ffnDownWeight(), ffnDown, deltas, matVec);
            this.ffnDownB = moe != null ? null : floatsOrNull(l.ffnDownBias(), floats);
            this.ls1 = floatsOrNull(l.ls1Weight(), floats);
            this.ls2 = floatsOrNull(l.ls2Weight(), floats);
            this.qNorm = floatsOrNull(l.qNormWeight(), floats);
            this.kNorm = floatsOrNull(l.kNormWeight(), floats);
            this.attnPostNorm = floatsOrNull(l.attnPostNormWeight(), floats);
            this.ffnPostNorm = floatsOrNull(l.ffnPostNormWeight(), floats);
            this.outScale = floatsOrNull(l.outScaleWeight(), floats);
            this.sinks = floatsOrNull(l.attnSinks(), floats);
            this.dHead = l.dHead();
            this.qkvClamp = clampOf(l, l.qkvWeight(), floats);
            this.qClamp = clampOf(l, l.qWeight(), floats);
            this.kClamp = clampOf(l, l.kWeight(), floats);
            this.vClamp = clampOf(l, l.vWeight(), floats);
            this.oClamp = clampOf(l, l.oWeight(), floats);
            this.upClamp = clampOf(l, l.ffnUpWeight(), floats);
            this.gateClamp = clampOf(l, l.ffnGateWeight(), floats);
            this.downClamp = clampOf(l, l.ffnDownWeight(), floats);
        }

        private static float[] clampOf(ClipLayer l, TensorRef weight, Function<String, float[]> floats) {
            ClipClamp c = l.clamp(weight);
            return c == null ? null : c.values(floats);
        }
    }

    /** The routed experts of one MoE block, the stacked weights viewed as {@code [n_expert * rows, cols]}. */
    private static final class Moe {
        final CpuTensorRef router;
        final CpuProjectionDeltas.Site routerSite;
        final CpuTensorRef gate;
        final CpuTensorRef up;
        final CpuTensorRef down;
        final float[] selectBias;
        final int nExpert;
        final int nFf;

        Moe(ClipLayer l, Function<String, float[]> floats, Function<String, CpuTensorRef> refs,
            CpuProjectionDeltas deltas, CpuMatVec matVec) {
            this.nExpert = l.nExpert();
            this.nFf = l.nFfExpert();
            long nEmbd = l.moeGateWeight().ne(0);
            this.router = matrix(refs.apply(l.moeRouterWeight().name()), nExpert, nEmbd);
            this.routerSite = deltas == null ? null
                    : deltas.serveModel(l.moeRouterWeight().name(), router, matVec);
            this.gate = matrix(refs.apply(l.moeGateWeight().name()), (long) nExpert * nFf, nEmbd);
            this.up = matrix(refs.apply(l.moeUpWeight().name()), (long) nExpert * nFf, nEmbd);
            this.down = matrix(refs.apply(l.moeDownWeight().name()), nExpert * nEmbd, nFf);
            this.selectBias = floatsOrNull(l.moeSelectBias(), floats);
        }
    }

    private final ClipHparams hp;
    private final CpuMatVec matVec;
    private final float[] preLnW;
    private final float[] preLnB;
    private final float[] postLnW;
    private final float[] postLnB;
    private final List<LayerW> layers;

    /**
     * The tower of {@code hp} over the resolved blocks and the optional pre/post norms (null when absent).
     */
    CpuVitTower(ClipHparams hp, CpuMatVec matVec, List<ClipLayer> blocks, TensorRef preLnW, TensorRef preLnB,
                TensorRef postLnW, TensorRef postLnB, Function<String, float[]> floats,
                Function<String, CpuTensorRef> refs, CpuProjectionDeltas deltas) {
        this.hp = Objects.requireNonNull(hp, "hparams");
        this.matVec = Objects.requireNonNull(matVec, "matVec");
        this.preLnW = floatsOrNull(preLnW, floats);
        this.preLnB = floatsOrNull(preLnB, floats);
        this.postLnW = floatsOrNull(postLnW, floats);
        this.postLnB = floatsOrNull(postLnB, floats);
        List<LayerW> ls = new ArrayList<>(blocks.size());
        for (ClipLayer l : blocks) {
            if (l.moe()) {
                requireMoeHparams(hp);
            }
            ls.add(new LayerW(l, floats, refs, deltas, matVec));
        }
        this.layers = Collections.unmodifiableList(ls);
    }

    /**
     * The LoRA site of one block matrix, served when the tower binds to a delta registry; null for an absent
     * optional or a tower bound without one.
     */
    private static CpuProjectionDeltas.Site loraSite(TensorRef t, CpuTensorRef ref, CpuProjectionDeltas deltas,
                                                     CpuMatVec matVec) {
        return t == null || deltas == null ? null : deltas.serveModel(t.name(), ref, matVec);
    }

    /** build_moe_ffn asserts a SiLU expert activation and a positive routed-expert count. */
    private static void requireMoeHparams(ClipHparams hp) {
        if (hp.ffnOp() != ClipFfnOp.SILU) {
            throw new IllegalArgumentException("the MoE vision blocks run SiLU experts (clip.use_silu), the file"
                    + " selects " + hp.ffnOp());
        }
        if (hp.nExpertUsed() <= 0) {
            throw new IllegalArgumentException("the MoE vision blocks need clip.vision.expert_used_count > 0, got "
                    + hp.nExpertUsed());
        }
    }

    /**
     * Runs the tower over {@code nTokens} token-major rows of {@code rows}, which it overwrites in place, with
     * every norm of the kind {@code norm}. A segment ({@link Options#withBlocks}) runs its blocks only.
     *
     * @return the output rows: {@code rows} itself, or a new array of {@code (nTokens - 2) / 2 + 1} rows when
     *         the options average-pool
     */
    float[] forward(float[] rows, int nTokens, NormKind norm, Options opts) {
        int nEmbd = hp.nEmbd();
        if (nTokens <= 0 || rows.length < nTokens * nEmbd) {
            throw new IllegalArgumentException("the tower needs " + nTokens + " rows of " + nEmbd + ", the buffer"
                    + " holds " + rows.length + " floats");
        }
        boolean whole = opts.firstBlock < 0;
        if (whole && preLnW != null) {
            normRows(rows, rows, nTokens, nEmbd, preLnW, preLnB, norm, hp.eps());
        }
        int from = Math.max(opts.firstBlock, 0);
        int to = opts.nLayers < 0 ? layers.size() : opts.nLayers;
        if (to > layers.size()) {
            throw new IllegalArgumentException("the tower holds " + layers.size() + " blocks, " + to
                    + " were requested");
        }
        if (opts.tap != null) {
            opts.tap.observe(from, rows, nTokens);
        }
        for (int il = from; il < to; il++) {
            VisionRows.KeyRanges windows = opts.layerWindows != null ? opts.layerWindows.apply(il) : null;
            if (opts.postNorm) {
                postNormBlock(rows, layers.get(il), nTokens, norm, windows, opts);
            } else {
                block(rows, layers.get(il), nTokens, norm, windows, opts);
            }
            if (opts.tap != null) {
                opts.tap.observe(il + 1, rows, nTokens);
            }
        }
        return whole ? finish(rows, nTokens, norm, opts) : rows;
    }

    /** The audio average pool when the options ask for it, then the post-norm. */
    private float[] finish(float[] rows, int nTokens, NormKind norm, Options opts) {
        int nEmbd = hp.nEmbd();
        int n = nTokens;
        float[] out = rows;
        if (opts.avgPool) {
            if (nTokens < 2) {
                throw new IllegalArgumentException("the average pool needs at least 2 tokens, got " + nTokens);
            }
            n = (nTokens - 2) / 2 + 1;
            out = avgPool(rows, n, nEmbd);
        }
        if (postLnW != null) {
            normRows(out, out, n, nEmbd, postLnW, postLnB, norm, hp.eps());
        }
        return out;
    }

    /**
     * One pre-norm block in place: attention (scaled by ls1, then the post-attention norm) and FFN (the post-FFN
     * norm, then ls2), each with a residual, then the block output scale.
     */
    private void block(float[] rows, LayerW l, int nTokens, NormKind norm, VisionRows.KeyRanges windows,
                       Options opts) {
        int nEmbd = hp.nEmbd();
        int n = nTokens * nEmbd;
        float[] cur = new float[n];
        normRows(rows, cur, nTokens, nEmbd, l.ln1W, l.ln1B, norm, hp.eps());
        float[] attn = attention(cur, l, nTokens, norm, windows, opts);
        VisionRows.mulCols(attn, l.ls1);
        postNorm(attn, l.attnPostNorm, nTokens, norm);
        CpuOps.add(attn, 0, rows, 0, rows, 0, n);
        cur = new float[n];
        normRows(rows, cur, nTokens, nEmbd, l.ln2W, l.ln2B, norm, hp.eps());
        blockFfn(cur, l, nTokens);
        postNorm(cur, l.ffnPostNorm, nTokens, norm);
        VisionRows.mulCols(cur, l.ls2);
        CpuOps.add(cur, 0, rows, 0, rows, 0, n);
        VisionRows.mulCols(rows, l.outScale);
    }

    /** A bias-free norm of the block's kind in place; nothing for a null weight. */
    private void postNorm(float[] x, float[] weight, int nTokens, NormKind norm) {
        if (weight != null) {
            normRows(x, x, nTokens, hp.nEmbd(), weight, null, norm, hp.eps());
        }
    }

    /** One post-norm block in place: {@code x += ln1(attn(x))}, then {@code x += ln2(ffn(x))}. */
    private void postNormBlock(float[] rows, LayerW l, int nTokens, NormKind norm, VisionRows.KeyRanges windows,
                               Options opts) {
        int nEmbd = hp.nEmbd();
        int n = nTokens * nEmbd;
        float[] attn = attention(rows, l, nTokens, norm, windows, opts);
        normRows(attn, attn, nTokens, nEmbd, l.ln1W, l.ln1B, norm, hp.eps());
        CpuOps.add(attn, 0, rows, 0, rows, 0, n);
        float[] cur = Arrays.copyOf(rows, n);
        blockFfn(cur, l, nTokens);
        normRows(cur, cur, nTokens, nEmbd, l.ln2W, l.ln2B, norm, hp.eps());
        CpuOps.add(cur, 0, rows, 0, rows, 0, n);
    }

    /** The FFN of a block in place: the routed experts of a MoE block, {@link #ffn} otherwise. */
    private void blockFfn(float[] cur, LayerW l, int nTokens) {
        if (l.moe != null) {
            moeFfn(cur, l.moe, nTokens);
        } else {
            ffn(cur, l, nTokens);
        }
    }

    private static float[] avgPool(float[] rows, int nOut, int width) {
        float[] out = new float[nOut * width];
        for (int t = 0; t < nOut; t++) {
            int a = 2 * t * width;
            int b = a + width;
            for (int e = 0; e < width; e++) {
                out[t * width + e] = (rows[a + e] + rows[b + e]) * 0.5f;
            }
        }
        return out;
    }

    /** {@code build_ffn} of a block in place: up (and gate), the encoder's activation, down. */
    private void ffn(float[] cur, LayerW l, int nTokens) {
        int nFf = hp.nFf();
        float[] up = new float[nTokens * nFf];
        product(l.ffnUp, l.ffnUpSite, l.upClamp, cur, up, nTokens);
        VisionRows.addBias(up, l.ffnUpB);
        if (l.ffnGate != null) {
            float[] gate = new float[nTokens * nFf];
            product(l.ffnGate, l.ffnGateSite, l.gateClamp, cur, gate, nTokens);
            VisionRows.addBias(gate, l.ffnGateB);
            CpuOps.gluRows(hp.ffnGlu(), hp.ffnGluParams(), nTokens, nFf, gate, 0, nFf, up, 0, nFf, up, 0, nFf);
        } else {
            VisionRows.activate(hp.ffnOp().unary(), up);
        }
        product(l.ffnDown, l.ffnDownSite, l.downClamp, up, cur, nTokens);
        VisionRows.addBias(cur, l.ffnDownB);
    }

    /**
     * {@code y = W x} over {@code nTokens} rows; with clamp bounds {@code {inMin, inMax, outMin, outMax}} the input
     * is clamped first (on a copy) and the output after (the Gemma 4 clippable linear).
     */
    private void product(CpuTensorRef w, CpuProjectionDeltas.Site site, float[] clamp, float[] x, float[] y,
                         int nTokens) {
        int in = (int) w.cols();
        int out = (int) w.rows();
        float[] src = x;
        if (clamp != null) {
            src = new float[nTokens * in];
            CpuOps.unary(ActOp.CLAMP, ActParams.builder().clamp(clamp[0], clamp[1]).build(), x, 0, src, 0,
                    src.length);
        }
        matVec.matmul(w, src, 0, y, 0, nTokens);
        if (site != null) {
            site.apply(nTokens, src, 0, in, y, 0, out);
        }
        if (clamp != null) {
            CpuOps.unary(ActOp.CLAMP, ActParams.builder().clamp(clamp[2], clamp[3]).build(), y, 0, y, 0,
                    nTokens * out);
        }
    }

    /**
     * {@code build_moe_ffn} in place: sigmoid router probabilities, the top {@code expert_used_count} experts by
     * probability plus the optional selection bias, weights renormalized over the chosen probabilities (the bias
     * only selects), and the weighted sum of the chosen {@code down(silu(gate(x)) * up(x))}.
     */
    private void moeFfn(float[] cur, Moe m, int nTokens) {
        int nEmbd = hp.nEmbd();
        int k = Math.min(hp.nExpertUsed(), m.nExpert);
        float[] probs = new float[nTokens * m.nExpert];
        matVec.matmul(m.router, cur, 0, probs, 0, nTokens);
        if (m.routerSite != null) {
            m.routerSite.apply(nTokens, cur, 0, nEmbd, probs, 0, m.nExpert);
        }
        VisionRows.activate(ActOp.SIGMOID, probs);
        float[] gate = new float[m.nFf];
        float[] up = new float[m.nFf];
        float[] down = new float[nEmbd];
        float[] out = new float[nEmbd];
        for (int t = 0; t < nTokens; t++) {
            int[] chosen = topK(probs, t * m.nExpert, m.nExpert, m.selectBias, k);
            float sum = 0f;
            for (int e : chosen) {
                sum += probs[t * m.nExpert + e];
            }
            Arrays.fill(out, 0f);
            for (int e : chosen) {
                matVec.matvecExpert(m.gate, e, m.nExpert, cur, t * nEmbd, gate, 0);
                matVec.matvecExpert(m.up, e, m.nExpert, cur, t * nEmbd, up, 0);
                VisionRows.activate(ActOp.SILU, gate);
                for (int i = 0; i < m.nFf; i++) {
                    gate[i] *= up[i];
                }
                matVec.matvecExpert(m.down, e, m.nExpert, gate, 0, down, 0);
                float w = probs[t * m.nExpert + e] / sum;
                for (int c = 0; c < nEmbd; c++) {
                    out[c] += down[c] * w;
                }
            }
            System.arraycopy(out, 0, cur, t * nEmbd, nEmbd);
        }
    }

    /** The {@code k} experts of the largest {@code probs + bias} values ({@code ggml_top_k}), highest first. */
    static int[] topK(float[] probs, int off, int n, float[] bias, int k) {
        int[] chosen = new int[k];
        boolean[] taken = new boolean[n];
        for (int i = 0; i < k; i++) {
            int best = -1;
            float bestV = Float.NEGATIVE_INFINITY;
            for (int e = 0; e < n; e++) {
                float v = probs[off + e] + (bias != null ? bias[e] : 0f);
                if (!taken[e] && (best < 0 || v > bestV)) {
                    best = e;
                    bestV = v;
                }
            }
            taken[best] = true;
            chosen[i] = best;
        }
        return chosen;
    }

    /**
     * Q/K/V projections (K and V {@code n_head_kv} heads wide), Q/K norms, the position hook, bidirectional
     * attention over the key windows (query head {@code h} reading KV head {@code h / (n_head / n_head_kv)}), then
     * the output projection.
     */
    private float[] attention(float[] cur, LayerW l, int nTokens, NormKind norm, VisionRows.KeyRanges windows,
                              Options opts) {
        int dh = l.dHead;
        int qWidth = hp.nHead() * dh;
        int kvWidth = hp.nHeadKv() * dh;
        float[] q = new float[nTokens * qWidth];
        float[] k = new float[nTokens * kvWidth];
        float[] v = new float[nTokens * kvWidth];
        projectQkv(cur, l, q, k, v, nTokens);
        qkNorm(q, l.qNorm, norm);
        qkNorm(k, l.kNorm, norm);
        if (opts.pos != null) {
            opts.pos.apply(q, nTokens, hp.nHead());
            opts.pos.apply(k, nTokens, hp.nHeadKv());
        }
        if (opts.valueNorm) {
            normRows(v, v, v.length / dh, dh, null, null, NormKind.RMS, hp.eps());
        }
        float[] sinks = null;
        if (opts.sinks) {
            if (l.sinks == null) {
                throw new IllegalArgumentException("the attention sinks of this tower call need attn_sinks in"
                        + " every block it runs");
            }
            sinks = l.sinks;
        }
        float scale = Float.isNaN(opts.attnScale) ? (float) (1.0 / Math.sqrt(dh)) : opts.attnScale;
        float[] out = VisionRows.attend(q, 0, nTokens, k, v, 0, nTokens, kvWidth, hp.nHead(),
                hp.nHead() / hp.nHeadKv(), dh, scale, windows, sinks);
        float[] proj = new float[nTokens * hp.nEmbd()];
        product(l.o, l.oSite, l.oClamp, out, proj, nTokens);
        VisionRows.addBias(proj, l.oB);
        return proj;
    }

    /** A Q/K norm of {@code d_head} width normalizes every head, a whole-row one every row. */
    private void qkNorm(float[] x, float[] weight, NormKind norm) {
        if (weight == null) {
            return;
        }
        int width = weight.length;
        normRows(x, x, x.length / width, width, weight, null, norm, hp.eps());
    }

    private void projectQkv(float[] cur, LayerW l, float[] q, float[] k, float[] v, int nTokens) {
        int qWidth = q.length / nTokens;
        int kvWidth = k.length / nTokens;
        if (l.fusedQkv) {
            int width = qWidth + 2 * kvWidth;
            float[] qkv = new float[nTokens * width];
            product(l.qkv, l.qkvSite, l.qkvClamp, cur, qkv, nTokens);
            VisionRows.addBias(qkv, l.qkvB);
            for (int t = 0; t < nTokens; t++) {
                int src = t * width;
                System.arraycopy(qkv, src, q, t * qWidth, qWidth);
                System.arraycopy(qkv, src + qWidth, k, t * kvWidth, kvWidth);
                System.arraycopy(qkv, src + qWidth + kvWidth, v, t * kvWidth, kvWidth);
            }
        } else {
            product(l.q, l.qSite, l.qClamp, cur, q, nTokens);
            product(l.k, l.kSite, l.kClamp, cur, k, nTokens);
            product(l.v, l.vSite, l.vClamp, cur, v, nTokens);
            VisionRows.addBias(q, l.qB);
            VisionRows.addBias(k, l.kB);
            VisionRows.addBias(v, l.vB);
        }
    }

    // ------------------------------------------------------------------ helpers shared by the encoders

    /**
     * Row norms of {@code x} into {@code y} ({@code build_norm}): layer norm (weight, optional bias) or RMS norm
     * (weight, then the optional bias).
     */
    static void normRows(float[] x, float[] y, int nRows, int width, float[] weight, float[] bias, NormKind kind,
                         float eps) {
        if (kind == NormKind.LAYER) {
            CpuOps.layerNorm(x, 0, y, 0, nRows, width, weight, bias, eps);
        } else {
            CpuOps.rmsNorm(x, 0, y, 0, nRows, width, weight, eps);
            VisionRows.addBias(y, bias);
        }
    }

    static float[] floatsOrNull(TensorRef r, Function<String, float[]> floats) {
        return r == null ? null : floats.apply(r.name());
    }

    static CpuTensorRef refOrNull(TensorRef r, Function<String, CpuTensorRef> refs) {
        return r == null ? null : refs.apply(r.name());
    }

    /**
     * A multi-dimensional weight (a patch or conv kernel) as the {@code [rows, cols]} matrix its im2col matmul
     * reads: the stored bytes are contiguous ggml, so a view that split them differently is re-wrapped.
     */
    static CpuTensorRef matrix(CpuTensorRef r, long rows, long cols) {
        if (r.rows() == rows && r.cols() == cols) {
            return r;
        }
        long bytes = Math.multiplyExact(r.rowBytes(), r.rows());
        if (bytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("weight " + r + " needs " + bytes + " bytes, above one buffer");
        }
        ByteBuffer flat = ByteBuffer.allocate((int) bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (long row = 0; row < r.rows(); row++) {
            ByteBuffer buf = r.rowBuffer(row).duplicate();
            buf.position(r.rowOffset(row));
            buf.limit(r.rowOffset(row) + (int) r.rowBytes());
            flat.put(buf);
        }
        flat.clear();
        return CpuTensorRef.of(r.type(), flat, rows, cols);
    }
}
