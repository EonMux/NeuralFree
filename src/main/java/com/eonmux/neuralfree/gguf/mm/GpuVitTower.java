package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.ops.act.ActKernels;
import com.eonmux.neuralfree.gguf.ops.attn.FlashAttnTiling;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.moe.MoeRouting;
import com.eonmux.neuralfree.gguf.ops.rope.RopeKernels;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;

import java.util.ArrayList;
import java.util.List;

/**
 * GPU executor of the shared ViT tower ({@link VisionOps#tower}), the counterpart of {@link CpuVitTower}: optional
 * pre-norm, then per block norm 1, Q/K/V (fused or separate; K and V {@code n_head_kv} heads wide), the optional
 * Q/K norms, the tower's rope on Q and K ({@code LLM_RoPE_Ext}: the qwen vision m-rope in VISION mode, the two
 * NORM-mode half-head calls of clip.cpp {@code build_rope_2d}, its NEOX-mode variant, or NEOX calls over dim
 * segments), the optional weightless RMS norm of every value head, bidirectional {@code LLM_FlashAttn_F32} at
 * {@code 1/sqrt(d_head)} or the options' scale, causal or bidirectional as the options select, over the whole grid
 * or one call per run of query rows that share a key window, the output projection, the optional layer scale and
 * post-attention norm, the residual, norm 2, the FFN (dense with the encoder's gate, clamped for GLM-5V, or the
 * routed experts of a MoE block: {@code LLM_MoE_Router} with sigmoid gating, {@code {T}_BatchMatMulId} over the
 * (token, expert) pairs, SwiGLU, {@code LLM_MoE_WeightedSum}), the optional post-FFN norm and layer scale, the
 * residual and the optional block output scale (the post-norm layout of CogVLM when selected), and the optional
 * stream tap; then the optional post-norm. A block matrix with clamp bounds clamps its input and output
 * ({@code LLM_Unary} CLAMP). A segment ({@link VisionOps.TowerOptions#withBlocks}) runs its blocks only. Work
 * buffers are scratch temporaries per role, reused block after block. Not thread-safe; one encode at a time.
 */
final class GpuVitTower {

    private final GpuVisionOps host;
    private final GpuOps ops;
    private final GpuVisionWeights w;
    private final ClipHparams hp;
    private final int nEmbd;
    private final int dHead;
    private final int kvWidth;
    private final float attnScale;

    GpuVitTower(GpuVisionOps host, GpuVisionWeights w) {
        this.host = host;
        this.ops = host.gpu();
        this.w = w;
        this.hp = host.hp();
        this.nEmbd = hp.nEmbd();
        this.dHead = host.model().dHead();
        this.kvWidth = hp.nHeadKv() * dHead;
        this.attnScale = host.model().attnScale();
    }

    /** One {@code LLM_RoPE_Ext} launch of a rope plan: its spec, the dim offset in every head, the positions. */
    private static final class RopeLaunch {
        final RopeSpec spec;
        final int nOffs;
        final ScratchSlot pos;

        RopeLaunch(RopeSpec spec, int nOffs, ScratchSlot pos) {
            this.spec = spec;
            this.nOffs = nOffs;
            this.pos = pos;
        }
    }

    /**
     * Runs the tower over the {@code nTokens} rows of {@code x} in place.
     *
     * @throws IllegalArgumentException when more blocks are requested than the tower holds
     */
    VisionOps.Block forward(GpuVisionOps.GpuBlock x, int nTokens, NormKind norm, VisionOps.TowerOptions opts) {
        if (nTokens <= 0 || (long) nTokens * nEmbd > x.n() * (long) x.width()) {
            throw new IllegalArgumentException("the tower needs " + nTokens + " rows of " + nEmbd + ", the block"
                    + " holds " + x.n() + " rows of " + x.width());
        }
        if (opts.sinks) {
            throw new IllegalArgumentException("the GPU flash attention has no attention sinks");
        }
        int nBlocks = opts.nLayers < 0 ? w.layers.size() : opts.nLayers;
        if (nBlocks > w.layers.size()) {
            throw new IllegalArgumentException("the tower holds " + w.layers.size() + " blocks, " + nBlocks
                    + " were requested");
        }
        boolean whole = opts.wholeTower();
        GpuBuffer cur = x.buffer();
        if (whole && w.preLnW != null) {
            norm(norm, w.preLnW, w.preLnB, cur, cur, nTokens);
        }
        if (opts.tap != null) {
            opts.tap.observe(opts.from(), x, nTokens);
        }
        List<RopeLaunch> rope = opts.rope == null ? null : rope(opts.rope, nTokens);
        for (int il = opts.from(); il < nBlocks; il++) {
            GpuVisionWeights.Layer l = w.layers.get(il);
            VisionRows.KeyRanges windows = opts.windows != null ? opts.windows.apply(il) : null;
            Call call = new Call(rope, windows, opts);
            if (opts.postNorm) {
                postNormBlock(cur, l, nTokens, norm, call);
            } else {
                block(cur, l, nTokens, norm, call);
            }
            if (opts.tap != null) {
                opts.tap.observe(il + 1, x, nTokens);
            }
        }
        if (whole && w.postLnW != null) {
            norm(norm, w.postLnW, w.postLnB, cur, cur, nTokens);
        }
        return x;
    }

    /** The per-block attention inputs of one tower call: the rope, the key windows, the scale, value norm, mask. */
    private final class Call {
        final List<RopeLaunch> rope;
        final VisionRows.KeyRanges windows;
        final float scale;
        final boolean valueNorm;
        final boolean causal;

        Call(List<RopeLaunch> rope, VisionRows.KeyRanges windows, VisionOps.TowerOptions opts) {
            this.rope = rope;
            this.windows = windows;
            this.scale = Float.isNaN(opts.attnScale) ? attnScale : opts.attnScale;
            this.valueNorm = opts.valueNorm;
            this.causal = opts.causal;
        }
    }

    /**
     * One pre-norm block in place: attention (scaled by ls1, then the post-attention norm) and FFN (the post-FFN
     * norm, then ls2), each with a residual, then the block output scale.
     */
    private void block(GpuBuffer x, GpuVisionWeights.Layer l, int n, NormKind norm, Call call) {
        int ne = n * nEmbd;
        ScratchSlot ln = host.temp("tower.ln", ne);
        ScratchSlot out = host.temp("tower.out", ne);
        norm(norm, l.ln1W, l.ln1B, x, ln.buffer(), n);
        attention(l, ln.buffer(), out.buffer(), n, norm, call);
        layerScale(out.buffer(), l.ls1, n);
        postNorm(norm, l.attnPostNorm, out.buffer(), n);
        ops.add(out.buffer(), 0, x, 0, x, 0, ne);
        norm(norm, l.ln2W, l.ln2B, x, ln.buffer(), n);
        ffn(l, ln.buffer(), out.buffer(), n);
        postNorm(norm, l.ffnPostNorm, out.buffer(), n);
        layerScale(out.buffer(), l.ls2, n);
        ops.add(out.buffer(), 0, x, 0, x, 0, ne);
        layerScale(x, l.outScale, n);
    }

    /** A bias-free norm of the block's kind in place; nothing for a null weight. */
    private void postNorm(NormKind kind, GpuWeight weight, GpuBuffer x, int n) {
        if (weight != null) {
            norm(kind, weight, null, x, x, n);
        }
    }

    /** One post-norm block in place: {@code x += ln1(attn(x))}, then {@code x += ln2(ffn(x))}. */
    private void postNormBlock(GpuBuffer x, GpuVisionWeights.Layer l, int n, NormKind norm, Call call) {
        int ne = n * nEmbd;
        ScratchSlot out = host.temp("tower.out", ne);
        attention(l, x, out.buffer(), n, norm, call);
        norm(norm, l.ln1W, l.ln1B, out.buffer(), out.buffer(), n);
        ops.add(out.buffer(), 0, x, 0, x, 0, ne);
        ffn(l, x, out.buffer(), n);
        norm(norm, l.ln2W, l.ln2B, out.buffer(), out.buffer(), n);
        ops.add(out.buffer(), 0, x, 0, x, 0, ne);
    }

    /**
     * {@code dst = W src + b} over {@code n} rows of {@code width}; with clamp bounds
     * {@code {inMin, inMax, outMin, outMax}} the input is clamped first (into a temporary) and the product after,
     * before the bias.
     */
    private void project(GpuWeight weight, GpuBuffer bias, float[] clamp, GpuBuffer src, GpuBuffer dst, int n,
                         int width) {
        GpuProjectionDeltas.Site site = w.site(weight.name());
        if (clamp == null) {
            host.projectBiased(weight, bias, src, dst, n, width);
            if (site != null) {
                site.apply(n, src, 0, Math.toIntExact(weight.cols()), dst, 0, width);
            }
            return;
        }
        int cols = Math.toIntExact(weight.cols());
        ScratchSlot in = host.temp("clamp.in." + cols, (long) n * cols);
        ops.unary(ActOp.CLAMP, ActParams.builder().clamp(clamp[0], clamp[1]).build(), src, 0, in.buffer(), 0,
                n * cols);
        ops.matmul(weight, in.buffer(), 0, dst, 0, n);
        if (site != null) {
            site.apply(n, in.buffer(), 0, cols, dst, 0, width);
        }
        ops.unary(ActOp.CLAMP, ActParams.builder().clamp(clamp[2], clamp[3]).build(), dst, 0, dst, 0, n * width);
        host.biasAdd(dst, bias, n, width);
    }

    // ------------------------------------------------------------------ attention

    /**
     * {@code dst = O . attn(Q, K, V) + b_o} over the rows of {@code cur}; Q/K norms run in {@code kind}. A causal
     * attention masks every key past the query's position (the flash kernel's causal flag over positions
     * {@code 0 .. n-1}).
     */
    private void attention(GpuVisionWeights.Layer l, GpuBuffer cur, GpuBuffer dst, int n, NormKind kind,
                           Call call) {
        int ne = n * nEmbd;
        int kvNe = n * kvWidth;
        ScratchSlot k = host.temp("tower.k", kvNe);
        ScratchSlot v = host.temp("tower.v", kvNe);
        ScratchSlot q;
        int qStride;
        if (l.fusedQkv()) {
            qStride = nEmbd + 2 * kvWidth;
            q = host.temp("tower.qkv", (long) n * qStride);
            project(l.qkv, l.qkvB, l.qkvClamp, cur, q.buffer(), n, qStride);
            ops.copy2d(q.buffer(), nEmbd, qStride, k.buffer(), 0, kvWidth, n, kvWidth);
            ops.copy2d(q.buffer(), nEmbd + kvWidth, qStride, v.buffer(), 0, kvWidth, n, kvWidth);
            if (l.qNorm != null) {
                // the Q norm reads contiguous rows
                ScratchSlot packed = host.temp("tower.q", ne);
                ops.copy2d(q.buffer(), 0, qStride, packed.buffer(), 0, nEmbd, n, nEmbd);
                q = packed;
                qStride = nEmbd;
            }
        } else {
            q = host.temp("tower.q", ne);
            qStride = nEmbd;
            project(l.q, l.qB, l.qClamp, cur, q.buffer(), n, nEmbd);
            project(l.k, l.kB, l.kClamp, cur, k.buffer(), n, kvWidth);
            project(l.v, l.vB, l.vClamp, cur, v.buffer(), n, kvWidth);
        }
        qkNorm(kind, l.qNorm, q.buffer(), n, qStride);
        qkNorm(kind, l.kNorm, k.buffer(), n, kvWidth);
        if (call.rope != null) {
            rotate(call.rope, q.buffer(), qStride, n, hp.nHead());
            rotate(call.rope, k.buffer(), kvWidth, n, hp.nHeadKv());
        }
        if (call.valueNorm) {
            ops.norm(NormKind.RMS, null, null, hp.eps(), v.buffer(), 0, v.buffer(), 0, ne / dHead, dHead);
        }
        ScratchSlot attn = host.temp("tower.attn", ne);
        ScratchSlot pos = host.positions(n);
        FlashAttnTiling plan = host.plan(hp.nHead(), hp.nHeadKv(), dHead);
        GpuVisionOps.AttnBuffers bufs = new GpuVisionOps.AttnBuffers(q.buffer(), qStride, k.buffer(), v.buffer(),
                kvWidth, attn.buffer(), nEmbd, pos.buffer());
        for (int[] run : VisionRows.keyRuns(call.windows, n)) {
            host.flash(plan, bufs, run[0], run[1] - run[0], run[0], run[2], run[3], call.scale, call.causal);
        }
        project(l.o, l.oB, l.oClamp, attn.buffer(), dst, n, nEmbd);
    }

    /**
     * The Q or K norm in place over {@code n} contiguous rows of {@code rowWidth} (the projection's width): one of
     * {@code d_head} width normalizes every head, one of the whole width every row; nothing for a null weight.
     */
    private void qkNorm(NormKind kind, GpuWeight weight, GpuBuffer x, int n, int rowWidth) {
        if (weight == null) {
            return;
        }
        int width = Math.toIntExact(weight.cols());
        ops.norm(kind, weight, null, hp.eps(), x, 0, x, 0, n * rowWidth / width, width);
    }

    /** {@code x[r][c] *= scale[c]} over {@code n} rows of {@code n_embd}; nothing for a null scale. */
    private void layerScale(GpuBuffer x, GpuBuffer scale, int n) {
        if (scale != null) {
            host.rowScale(x, scale, n, nEmbd);
        }
    }

    // ------------------------------------------------------------------ rope

    /**
     * Uploads the positions of {@code plan} and rotates the {@code x.n()} rows of {@code x}, {@code nHeads} heads
     * of {@code d_head} each, in place ({@link VisionOps#rope}): four sections of {@code n} for the qwen vision
     * m-rope, or the two single-position halves of {@code build_rope_2d} (the second half's frequencies scaled by
     * {@code base^(-2 / d_head)} when interleaved) or of the Gemma 4 NEOX-mode rope, or one NEOX launch per
     * segment.
     */
    void rope(GpuVisionOps.GpuBlock x, int nHeads, VisionOps.RopePlan plan) {
        if (nHeads <= 0 || x.width() != nHeads * dHead) {
            throw new IllegalArgumentException("rows of " + x.width() + " are not " + nHeads + " heads of " + dHead);
        }
        rotate(rope(plan, x.n()), x.buffer(), x.width(), x.n(), nHeads);
    }

    /**
     * The launches of {@code plan} over {@code n} rows, positions uploaded: the qwen vision m-rope (four sections
     * of {@code n}), the two single-position halves of {@code build_rope_2d} (the second half's frequencies
     * scaled by {@code base^(-2 / d_head)} when interleaved), or one NEOX launch per segment.
     */
    private List<RopeLaunch> rope(VisionOps.RopePlan plan, int n) {
        List<RopeLaunch> out = new ArrayList<>();
        if (plan.qwenPositions != null) {
            out.add(new RopeLaunch(QwenVisionGrid.rope(dHead, plan.qwenBase), 0,
                    positions("rope.pos", plan.qwenPositions, QwenVisionGrid.POSITION_SECTIONS * n)));
            return out;
        }
        if (plan.neox != null) {
            for (int i = 0; i < plan.neox.size(); i++) {
                VisionOps.NeoxSegment seg = plan.neox.get(i);
                if (seg.offset + seg.nDims > dHead) {
                    throw new IllegalArgumentException("NEOX dims [" + seg.offset + ", " + (seg.offset + seg.nDims)
                            + ") exceed the head width " + dHead);
                }
                RopeSpec spec = RopeSpec.builder(RopeType.NEOX).nDims(seg.nDims).freqBase(plan.neoxBase).build();
                out.add(new RopeLaunch(spec, seg.offset, positions("rope.pos" + i, seg.positions, n)));
            }
            return out;
        }
        if (dHead % 4 != 0) {
            throw new IllegalArgumentException("the 2D RoPE needs a head width divisible by 4, got " + dHead);
        }
        if (plan.neoxMode) {
            out.add(new RopeLaunch(halfRope(RopeType.NEOX, plan.base, 1f), 0,
                    positions("rope.pos", plan.posA, n)));
            out.add(new RopeLaunch(halfRope(RopeType.NEOX, plan.base, 1f), dHead / 2,
                    positions("rope.posB", plan.posB, n)));
            return out;
        }
        float scaleB = plan.interleaveFreq ? (float) Math.pow(plan.base, -2.0 / dHead) : 1f;
        out.add(new RopeLaunch(halfRope(RopeType.NORM, plan.base, 1f), 0, positions("rope.pos", plan.posA, n)));
        out.add(new RopeLaunch(halfRope(RopeType.NORM, plan.base, scaleB), dHead / 2,
                positions("rope.posB", plan.posB, n)));
        return out;
    }

    /** {@code values} uploaded to the scratch temporary {@code role}, refused unless {@code expected} long. */
    private ScratchSlot positions(String role, int[] values, int expected) {
        if (values.length != expected) {
            throw new IllegalArgumentException("the rope needs " + expected + " positions, got " + values.length);
        }
        ScratchSlot pos = host.temp(role, values.length);
        host.writeInts(pos.buffer(), values);
        return pos;
    }

    /**
     * A rope of {@code type} over half a head at {@code freqScale * pos * base^(-4i / d_head)}: adjacent pairs in
     * NORM mode, pairs {@code (i, i + d_head / 4)} in NEOX mode.
     */
    private RopeSpec halfRope(RopeType type, double base, float freqScale) {
        return RopeSpec.builder(type).nDims(dHead / 2).freqBase((float) base).freqScale(freqScale).build();
    }

    /** Rotates the {@code n} rows of {@code x}, {@code nHeads} heads at row stride {@code rowStride}, in place. */
    private void rotate(List<RopeLaunch> rope, GpuBuffer x, int rowStride, int n, int nHeads) {
        for (RopeLaunch r : rope) {
            int local = host.local();
            Object[] args = RopeKernels.ropeExtArgs(r.spec, n, nHeads, dHead, rowStride, r.nOffs, n, 0, false,
                    r.pos.buffer(), host.placeholders().readOnlyFloats(), x);
            long items = RopeKernels.ropeExtWorkItems(r.spec.type(), n, nHeads, r.spec.nDims());
            ops.enqueue(RopeKernels.ROPE_EXT, ActKernels.global1d(items, local), new long[]{local}, args);
        }
    }

    // ------------------------------------------------------------------ FFN

    /** {@code dst = down(act(up(cur)) [* gate]) + b_down}, or the routed experts of a MoE block. */
    private void ffn(GpuVisionWeights.Layer l, GpuBuffer cur, GpuBuffer dst, int n) {
        if (l.moe != null) {
            moeFfn(l.moe, cur, dst, n);
            return;
        }
        int nFf = Math.toIntExact(l.ffnUp.rows());
        ScratchSlot up = host.temp("ffn.up", (long) n * nFf);
        project(l.ffnUp, l.ffnUpB, l.upClamp, cur, up.buffer(), n, nFf);
        if (l.ffnGate != null) {
            ScratchSlot gate = host.temp("ffn.gate", (long) n * nFf);
            project(l.ffnGate, l.ffnGateB, l.gateClamp, cur, gate.buffer(), n, nFf);
            ops.glu(hp.ffnGlu(), hp.ffnGluParams(), gate.buffer(), 0, nFf, up.buffer(), 0, nFf, up.buffer(), 0,
                    nFf, n, nFf);
        } else {
            ops.unary(hp.ffnOp().unary(), ActParams.NONE, up.buffer(), 0, up.buffer(), 0, n * nFf);
        }
        project(l.ffnDown, l.ffnDownB, l.downClamp, up.buffer(), dst, n, nEmbd);
    }

    /**
     * {@code build_moe_ffn}: sigmoid router probabilities, the top {@code expert_used_count} experts by probability
     * plus the optional selection bias, weights renormalized over the chosen probabilities; then for every
     * (token, expert) pair {@code down(silu(gate(x)) * up(x))}, summed with the weights into {@code dst}.
     */
    private void moeFfn(GpuVisionWeights.Moe m, GpuBuffer cur, GpuBuffer dst, int n) {
        int k = Math.min(hp.nExpertUsed(), m.nExpert);
        int pairs = Math.multiplyExact(n, k);
        MoeRouting.Params params = new MoeRouting.Params(m.nExpert, k, GatingFunc.SIGMOID, 1, 0, true, 1f,
                m.selectBias != null, MoeKernels.SEL_PROBS, 0, false);
        int routerLocal = routerLocal(m.nExpert);
        MoeKernels.checkRouter(params, routerLocal);
        ScratchSlot logits = host.temp("moe.logits", (long) n * m.nExpert);
        ops.matmul(m.router, cur, 0, logits.buffer(), 0, n);
        GpuProjectionDeltas.Site routerSite = w.site(m.router.name());
        if (routerSite != null) {
            routerSite.apply(n, cur, 0, Math.toIntExact(m.router.cols()), logits.buffer(), 0, m.nExpert);
        }
        ScratchSlot ids = host.temp("moe.ids", pairs);
        ScratchSlot weights = host.temp("moe.weights", pairs);
        GpuBuffer bias = m.selectBias != null ? m.selectBias : host.placeholders().readOnlyFloats();
        ops.enqueue(MoeKernels.ROUTER, MoeKernels.routerGlobal(n, routerLocal), new long[]{routerLocal},
                MoeKernels.routerArgs(params, n, logits.buffer(), bias, ids.buffer(), weights.buffer()));
        ScratchSlot rows = host.temp("moe.rows", pairs);
        int[] pairIds = new int[pairs];
        for (int p = 0; p < pairs; p++) {
            pairIds[p] = p;
        }
        host.writeInts(rows.buffer(), pairIds);
        ScratchSlot gate = host.temp("moe.gate", (long) pairs * m.nFf);
        ScratchSlot up = host.temp("moe.up", (long) pairs * m.nFf);
        expertMatmul(m.gate, m.nFf, nEmbd, pairs, k, rows, ids, cur, gate.buffer());
        expertMatmul(m.up, m.nFf, nEmbd, pairs, k, rows, ids, cur, up.buffer());
        ops.glu(GluOp.SWIGLU, ActParams.NONE, gate.buffer(), 0, m.nFf, up.buffer(), 0, m.nFf, up.buffer(), 0, m.nFf,
                pairs, m.nFf);
        ScratchSlot down = host.temp("moe.down", (long) pairs * nEmbd);
        expertMatmul(m.down, nEmbd, m.nFf, pairs, 1, rows, ids, up.buffer(), down.buffer());
        int local = host.local();
        ops.enqueue(MoeKernels.WEIGHTED_SUM, MoeKernels.elementGlobal((long) n * nEmbd, local), new long[]{local},
                MoeKernels.weightedSumArgs(n, nEmbd, k, 0, down.buffer(), weights.buffer(), dst));
    }

    /**
     * {@code {T}_BatchMatMulId} over every pair row: pair {@code p} multiplies its expert's {@code [outDim][inDim]}
     * matrix with input row {@code p / xDiv}, one dispatch per shard of whole experts.
     */
    private void expertMatmul(GpuWeight experts, int outDim, int inDim, int pairs, int xDiv, ScratchSlot rows,
                              ScratchSlot ids, GpuBuffer input, GpuBuffer output) {
        int idLocal = idLocal(experts);
        GpuBuffer scales = host.placeholders().readOnlyFloats();
        for (int s = 0; s < experts.shardCount(); s++) {
            if (experts.shardRowStart(s) % outDim != 0 || experts.shardRows(s) % outDim != 0) {
                throw new IllegalArgumentException("expert tensor " + experts.name() + ": shard " + s
                        + " does not hold whole experts of " + outDim + " rows");
            }
            int base = Math.toIntExact(experts.shardRowStart(s) / outDim);
            int count = Math.toIntExact(experts.shardRows(s) / outDim);
            ops.enqueue(MoeKernels.batchMatMulId(experts.type()),
                    MoeKernels.batchMatMulIdGlobal(outDim, pairs, idLocal), new long[]{idLocal, 1},
                    MoeKernels.batchMatMulIdArgs(outDim, inDim, pairs, xDiv, 0, false, base, count, rows.buffer(),
                            ids.buffer(), experts.shard(s), input, scales, output));
        }
    }

    /** The id-matmul work-group: a power of two in {@code [32, MAX_LOCAL]} covering the type's block threads. */
    private int idLocal(GpuWeight experts) {
        int l = Math.max(32, Integer.highestOneBit(Math.max(1, host.local())));
        l = Math.min(Math.max(l, ceilPow2(MoeKernels.threadsPerBlock(experts.type()))), MoeKernels.MAX_LOCAL);
        MoeKernels.checkIdLocal(experts.type(), l);
        return l;
    }

    /** The router work-group: one lane per 32 experts at least, a power of two in {@code [32, MAX_LOCAL]}. */
    private int routerLocal(int nExpert) {
        int needed = ceilPow2((nExpert + 31) / 32);
        int preferred = Integer.highestOneBit(Math.max(host.local(), MoeKernels.MIN_ROUTER_LOCAL));
        return Math.min(MoeKernels.MAX_LOCAL, Math.max(needed, preferred));
    }

    private static int ceilPow2(int v) {
        return v <= 1 ? 1 : Integer.highestOneBit(v - 1) << 1;
    }

    private void norm(NormKind kind, GpuWeight weight, GpuWeight bias, GpuBuffer src, GpuBuffer dst, int rows) {
        ops.norm(kind, weight, bias, hp.eps(), src, 0, dst, 0, rows, nEmbd);
    }
}
