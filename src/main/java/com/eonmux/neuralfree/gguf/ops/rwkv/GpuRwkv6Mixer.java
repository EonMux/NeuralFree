package com.eonmux.neuralfree.gguf.ops.rwkv;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MissingTensorException;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The RWKV6 time mix on the GPU ({@code MixerKind.RWKV6} and {@code RWKV6_QWEN2}): llama.cpp
 * {@code llm_build_rwkv6_base::build_rwkv6_time_mix} (models/rwkv6-base.cpp:30-165).
 *
 * <p>Per step over {@code T} rows of {@code in} (the {@code attn_norm} output, {@code C = n_embd}, {@code H} heads
 * of {@code S}):
 * <ol>
 *   <li>{@code TOKEN_SHIFT_LERP} mode {@code LERP_X}: {@code xx = sx * lerp_x + cur} with
 *       {@code sx = x_prev - cur} (:52-57).</li>
 *   <li>{@code h = tanh(W1 xx)} ({@code [T][5][E]}), then plane {@code m} of {@code [5][T][C]} is
 *       {@code W2[m] h[.][m]} (:59-65). W2's five row blocks are copied into their own weights at bind time; for
 *       {@code T > 1} the {@code h} rows are de-interleaved to {@code [5][T][E]} with {@code LLM_Copy2D_F32}.</li>
 *   <li>{@code TOKEN_SHIFT_LERP} mode {@code LORA} in place: {@code x_m = (plane_m + lerp_m) * sx + cur} with the
 *       fused {@code time_mix_lerp_fused} or the packed {@code lerp_w/k/v/r/g} (:68-90). It stores the new shift.</li>
 *   <li>{@code r = Wr x_r (+b)}, {@code k = Wk x_k (+b)}, {@code v = Wv x_v (+b)}, raw gate {@code Wg x_g}
 *       (:92-104); {@code k} and {@code v} heads are repeated when {@code time_mix_key} has fewer heads (:109-117).</li>
 *   <li>{@code w = exp(-exp(Wd2 tanh(Wd1 x_w) + time_mix_decay))} ({@code LLM_RWKV_Decay6}, :122-127).</li>
 *   <li>QRWKV ({@code time_mix_first} absent): {@code k -= k * w}, then {@code LLM_GLA} with per-channel gates and
 *       scale {@code S^-0.5} (:129-137). Otherwise {@code LLM_RWKV_WKV6} with {@code time_mix_first} (:139), then
 *       the per-head group norm with {@code time_mix_ln} / {@code time_mix_ln.bias} (:148-156).</li>
 *   <li>{@code cur = y * act(g)} with {@code act} = sigmoid (QRWKV) or SiLU ({@code LLM_GLU}), then
 *       {@code out = Wo cur} (:105-108, :160-161).</li>
 * </ol>
 * State: the layer's {@code shift} row at offset 0 and its {@code state} matrices ({@code [h][key][value]}).
 * Not thread-safe.</p>
 */
public final class GpuRwkv6Mixer implements GpuMixer {

    private static final int N_MIX = 5;
    private static final int PLANE_W = 0;
    private static final int PLANE_K = 1;
    private static final int PLANE_V = 2;
    private static final int PLANE_R = 3;
    private static final int PLANE_G = 4;

    private RwkvBlocks.GpuKit kit;
    private GpuOps ops;
    private boolean closed;

    private int nEmbd;
    private int headSize;
    private int nHead;
    private int kvWidth;
    private int headGroups;
    private int extra;
    private int decayDim;
    private boolean qrwkv;

    private GpuWeight w1;
    private final GpuWeight[] w2 = new GpuWeight[N_MIX];
    private GpuWeight wr;
    private GpuWeight wk;
    private GpuWeight wv;
    private GpuWeight wg;
    private GpuWeight wo;
    /**
     * LoRA deltas of receptance, key, value, gate and output ({@code build_lora_mm}, models/rwkv6-base.cpp); the
     * low-rank lerp and decay products are plain {@code ggml_mul_mat} in llama.cpp and take none.
     */
    private GpuProjectionDeltas.Site rDeltas;
    private GpuProjectionDeltas.Site kDeltas;
    private GpuProjectionDeltas.Site vDeltas;
    private GpuProjectionDeltas.Site gDeltas;
    private GpuProjectionDeltas.Site oDeltas;
    private GpuWeight decayW1;
    private GpuWeight decayW2;

    private ScratchSlot xy;
    private ScratchSlot h;
    private ScratchSlot hd;
    private ScratchSlot planes;
    private ScratchSlot r;
    private ScratchSlot k;
    private ScratchSlot v;
    private ScratchSlot kRaw;
    private ScratchSlot vRaw;
    private ScratchSlot g;
    private ScratchSlot dh;
    private ScratchSlot wRaw;
    private ScratchSlot w;
    private List<GpuBuffer> working = Collections.emptyList();

    private PreparedDispatch shiftX;
    private PreparedDispatch tanhH;
    private PreparedDispatch shiftLora;
    private PreparedDispatch rBias;
    private PreparedDispatch kBias;
    private PreparedDispatch vBias;
    private PreparedDispatch tanhD;
    private PreparedDispatch decay;
    private PreparedDispatch kDecay;
    private PreparedDispatch wkv;
    private PreparedDispatch groupNorm;
    private PreparedDispatch gate;

    /**
     * Resolves the layer's tensors, derives the W2 row blocks (and the packed lerps without {@code lerp_fused}),
     * reserves the working temporaries and prepares every element-wise and recurrent dispatch.
     *
     * @throws IllegalStateException    when already bound or closed, the context has no recurrent memory, or a
     *                                  tensor's shape contradicts the spec
     * @throws IllegalArgumentException when the layer is not an RWKV6 / RWKV6_QWEN2 layer
     * @throws MissingTensorException   when a required tensor is absent
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        if (closed || kit != null) {
            throw new IllegalStateException("RWKV6 time mix is " + (closed ? "closed" : "already bound"));
        }
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        if (layer.mixerKind() != MixerKind.RWKV6 && layer.mixerKind() != MixerKind.RWKV6_QWEN2) {
            throw new IllegalArgumentException("GpuRwkv6Mixer cannot serve " + layer.mixerKind() + " layer "
                    + layer.il());
        }
        RecurrentSpec spec = RwkvBlocks.recurrent(layer, layer.mixerKind());
        RwkvBlocks.GpuKit k0 = new RwkvBlocks.GpuKit(ctx, layer, "RWKV6 time mix");
        try {
            bindChecked(k0, spec);
        } catch (RuntimeException e) {
            try {
                k0.close();
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
        kit = k0;
        ops = ctx.ops();
    }

    private void bindChecked(RwkvBlocks.GpuKit kt, RecurrentSpec spec) {
        nEmbd = spec.nEmbd();
        headSize = spec.wkvHeadSize();
        nHead = nEmbd / headSize;
        qrwkv = spec.qrwkv();
        int c = nEmbd;
        bindProjections(kt, c);
        bindTemporaries(kt, c);
        bindDispatches(kt, c);
    }

    private void bindProjections(RwkvBlocks.GpuKit kt, int c) {
        w1 = kt.matrix(TensorNames.TIME_MIX_W1, c, 0);
        if (w1.rows() % N_MIX != 0) {
            throw new IllegalStateException(kt.where + ": " + w1.name() + " has " + w1.rows()
                    + " rows, not a multiple of " + N_MIX);
        }
        extra = Math.toIntExact(w1.rows() / N_MIX);
        GpuWeight w2Full = kt.matrix(TensorNames.TIME_MIX_W2, extra, (long) N_MIX * c);
        for (int m = 0; m < N_MIX; m++) {
            w2[m] = kt.rowBlock(w2Full, (long) m * c, c, w2Full.name() + "[" + m + "]");
        }
        wr = kt.matrix(TensorNames.TIME_MIX_RECEPTANCE, c, c);
        wk = kt.matrix(TensorNames.TIME_MIX_KEY, c, 0);
        kvWidth = Math.toIntExact(wk.rows());
        if (kvWidth % headSize != 0 || kvWidth == 0 || nHead % (kvWidth / headSize) != 0) {
            throw new IllegalStateException(kt.where + ": " + wk.name() + " has " + kvWidth + " rows, not a head"
                    + " count dividing " + nHead + " heads of " + headSize);
        }
        headGroups = nHead / (kvWidth / headSize);
        wv = kt.matrix(TensorNames.TIME_MIX_VALUE, c, kvWidth);
        wg = kt.matrix(TensorNames.TIME_MIX_GATE, c, c);
        wo = kt.matrix(TensorNames.TIME_MIX_OUTPUT, c, c);
        decayW1 = kt.matrix(TensorNames.TIME_MIX_DECAY_W1, c, 0);
        decayDim = Math.toIntExact(decayW1.rows());
        decayW2 = kt.matrix(TensorNames.TIME_MIX_DECAY_W2, decayDim, c);
        GpuProjectionDeltas deltas = kt.ctx.projectionDeltas();
        rDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_RECEPTANCE), wr);
        kDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_KEY), wk);
        vDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_VALUE), wv);
        gDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_GATE), wg);
        oDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_OUTPUT), wo);
    }

    private void bindTemporaries(RwkvBlocks.GpuKit kt, int c) {
        xy = kt.temp("rwkv6.xy", c);
        h = kt.temp("rwkv6.h", (long) N_MIX * extra);
        hd = kt.temp("rwkv6.hd", (long) N_MIX * extra);
        planes = kt.temp("rwkv6.planes", (long) N_MIX * c);
        r = kt.temp("rwkv6.r", c);
        k = kt.temp("rwkv6.k", c);
        v = kt.temp("rwkv6.v", c);
        kRaw = headGroups > 1 ? kt.temp("rwkv6.k_raw", kvWidth) : k;
        vRaw = headGroups > 1 ? kt.temp("rwkv6.v_raw", kvWidth) : v;
        g = kt.temp("rwkv6.g", c);
        dh = kt.temp("rwkv6.dh", decayDim);
        wRaw = kt.temp("rwkv6.w_raw", c);
        w = kt.temp("rwkv6.w", c);
        List<GpuBuffer> b = new ArrayList<>();
        for (ScratchSlot s : Arrays.asList(xy, h, hd, planes, r, k, v, kRaw, vRaw, g, dh, wRaw, w)) {
            b.add(s.buffer());
        }
        working = Collections.unmodifiableList(b);
    }

    private void bindDispatches(RwkvBlocks.GpuKit kt, int c) {
        GpuBuffer shift = kt.state(LayerMemoryPlan.STATE_SHIFT, c);
        GpuBuffer state = kt.state(LayerMemoryPlan.STATE_SCAN, (long) c * headSize);
        GpuBuffer stateRows = kt.stateRows();
        shiftX = kt.tokenShift(c, 0, RwkvKernels.SHIFT_MODE_LERP_X, shift, 0,
                kt.vector(TensorNames.TIME_MIX_LERP_X, c), null, xy.buffer());
        tanhH = kt.unary(ActOp.TANH, h.buffer(), N_MIX * extra);
        shiftLora = kt.tokenShift(c, N_MIX, RwkvKernels.SHIFT_MODE_LORA, shift, 0, null, lerps(kt, c),
                planes.buffer());
        rBias = kt.biasAdd(kt.biasOrNull(TensorNames.TIME_MIX_RECEPTANCE, c), r.buffer(), c);
        kBias = kt.biasAdd(kt.biasOrNull(TensorNames.TIME_MIX_KEY, kvWidth), kRaw.buffer(), kvWidth);
        vBias = kt.biasAdd(kt.biasOrNull(TensorNames.TIME_MIX_VALUE, kvWidth), vRaw.buffer(), kvWidth);
        tanhD = kt.unary(ActOp.TANH, dh.buffer(), decayDim);
        decay = kt.decay(RwkvKernels.DECAY6, c, wRaw.buffer(), kt.vector(TensorNames.TIME_MIX_DECAY, c),
                w.buffer());
        if (qrwkv) {
            kDecay = kt.glu(GluOp.MUL, k.buffer(), w.buffer(), wRaw.buffer(), c);
            wkv = kt.wkv(RwkvKernels.GLA, nHead, headSize, kt.ubatch, c, nHead, RwkvKernels.GLA_G_PER_CHANNEL,
                    RwkvKernels.glaScale(headSize), k.buffer(), v.buffer(), r.buffer(), w.buffer(), state, stateRows,
                    xy.buffer());
        } else {
            wkv = kt.wkv(RwkvKernels.WKV6, nHead, headSize, kt.ubatch, c, nHead, k.buffer(), v.buffer(), r.buffer(),
                    kt.vector(TensorNames.TIME_MIX_FIRST, c), w.buffer(), state, stateRows, xy.buffer());
            GpuBuffer lnB = kt.biasOrNull(TensorNames.TIME_MIX_LN, c);
            if (lnB == null) {
                throw new MissingTensorException(kt.weights.archName(), kt.layer.il(),
                        kt.layer.tensorName(TensorNames.bias(TensorNames.TIME_MIX_LN)));
            }
            groupNorm = kt.groupNorm(xy.buffer(), nHead, headSize, kt.vector(TensorNames.TIME_MIX_LN, c), lnB);
        }
        gate = kt.glu(qrwkv ? GluOp.SIGMUL : GluOp.SWIGLU, g.buffer(), xy.buffer(), xy.buffer(), c);
    }

    /** {@code [5][C]} lerp vectors in plane order w, k, v, r, g: the fused tensor or the five packed ones. */
    private static GpuBuffer lerps(RwkvBlocks.GpuKit kt, int c) {
        GpuBuffer fused = kt.vectorOrNull(TensorNames.TIME_MIX_LERP_FUSED, (long) N_MIX * c);
        if (fused != null) {
            return fused;
        }
        return kt.pack(Arrays.asList(
                kt.vector(TensorNames.TIME_MIX_LERP_W, c),
                kt.vector(TensorNames.TIME_MIX_LERP_K, c),
                kt.vector(TensorNames.TIME_MIX_LERP_V, c),
                kt.vector(TensorNames.TIME_MIX_LERP_R, c),
                kt.vector(TensorNames.TIME_MIX_LERP_G, c)), c);
    }

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        prefill(pos, 1, in, out);
    }

    /**
     * Runs the time mix over {@code nTokens} rows of {@code in} into {@code out} and advances the layer's shift and
     * WKV state.
     *
     * @throws IllegalStateException    when not bound or closed
     * @throws IllegalArgumentException for a token count outside {@code [1, ubatch]}, a slot too small for the rows
     *                                  or a slot sharing a buffer with the working rows
     */
    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        if (closed || kit == null) {
            throw new IllegalStateException("RWKV6 time mix is " + (closed ? "closed" : "not bound"));
        }
        int c = nEmbd;
        int t = nTokens;
        kit.checkStep(t, in, out, c, working);
        kit.runShift(shiftX, t, c, in);
        ops.matmul(w1, xy.buffer(), 0, h.buffer(), 0, t);
        kit.runUnary(tanhH, t, N_MIX * extra);
        mixPlanes(t, c);
        kit.runShift(shiftLora, t, c, in);
        int plane = t * c;
        GpuBuffer x = planes.buffer();
        project(wr, rDeltas, rBias, x, PLANE_R * plane, r.buffer(), t);
        project(wk, kDeltas, kBias, x, PLANE_K * plane, kRaw.buffer(), t);
        project(wv, vDeltas, vBias, x, PLANE_V * plane, vRaw.buffer(), t);
        ops.matmul(wg, x, PLANE_G * plane, g.buffer(), 0, t);
        gDeltas.applyRows(t, x, PLANE_G * plane, g.buffer(), 0);
        if (headGroups > 1) {
            repeatHeads(kRaw.buffer(), k.buffer(), t);
            repeatHeads(vRaw.buffer(), v.buffer(), t);
        }
        ops.matmul(decayW1, x, PLANE_W * plane, dh.buffer(), 0, t);
        kit.runUnary(tanhD, t, decayDim);
        ops.matmul(decayW2, dh.buffer(), 0, wRaw.buffer(), 0, t);
        kit.runDecay(decay, t, c);
        if (qrwkv) {
            kit.runGlu(kDecay, t, c);
            ops.scaleAdd(wRaw.buffer(), 0, -1.0f, k.buffer(), 0, plane);
        }
        kit.runWkv(wkv, t);
        if (groupNorm != null) {
            kit.runGroupNorm(groupNorm, t, nHead);
        }
        kit.runGlu(gate, t, c);
        ops.matmul(wo, xy.buffer(), 0, out.buffer(), out.offset(), t);
        oDeltas.applyRows(t, xy.buffer(), 0, out.buffer(), out.offset());
    }

    /** {@code out[t] = W in[t] + deltas (+ b)} into rows at element 0 of {@code out}. */
    private void project(GpuWeight w, GpuProjectionDeltas.Site site, PreparedDispatch bias, GpuBuffer in, int inOff,
                         GpuBuffer out, int t) {
        ops.matmul(w, in, inOff, out, 0, t);
        site.applyRows(t, in, inOff, out, 0);
        kit.runBias(bias, t);
    }

    /** Plane {@code m} of {@code [5][T][C]} = {@code W2[m]} applied to the {@code m}-th {@code E} slice of h. */
    private void mixPlanes(int t, int c) {
        int e = extra;
        if (t == 1) {
            for (int m = 0; m < N_MIX; m++) {
                ops.matmul(w2[m], h.buffer(), m * e, planes.buffer(), m * c, 1);
            }
            return;
        }
        for (int m = 0; m < N_MIX; m++) {
            ops.copy2d(h.buffer(), m * e, N_MIX * e, hd.buffer(), m * t * e, e, t, e);
            ops.matmul(w2[m], hd.buffer(), m * t * e, planes.buffer(), m * t * c, t);
        }
    }

    /** {@code dst[t][h] = src[t][h / G]} per head of {@code S} for {@code G = n_head / n_head_kv} (ggml_repeat). */
    private void repeatHeads(GpuBuffer src, GpuBuffer dst, int t) {
        int nKv = kvWidth / headSize;
        for (int gi = 0; gi < headGroups; gi++) {
            ops.copy2d(src, 0, headSize, dst, gi * headSize, headGroups * headSize, t * nKv, headSize);
        }
    }

    /** Releases the prepared dispatches and the derived weight buffers. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (kit != null) {
            kit.close();
        }
    }

    @Override
    public String toString() {
        return "GpuRwkv6Mixer{" + (kit != null ? kit.where + (qrwkv ? ", qrwkv" : "") : "unbound")
                + (closed ? ", closed" : "") + '}';
    }
}
