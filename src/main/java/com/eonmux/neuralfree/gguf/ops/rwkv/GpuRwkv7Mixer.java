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
 * The RWKV7 time mix on the GPU ({@code MixerKind.RWKV7}, archs {@code rwkv7} and {@code arwkv7}): llama.cpp
 * {@code llm_build_rwkv7_base::build_rwkv7_time_mix} (models/rwkv7-base.cpp:32-137).
 *
 * <p>Per step over {@code T} rows of {@code in} ({@code C = n_embd}, {@code H} heads of {@code S}):
 * <ol>
 *   <li>{@code TOKEN_SHIFT_LERP} mode {@code LERP}: planes {@code x_m = sx * lerp_fused[m] + cur} in order
 *       r, w, k, v, a and, with gating, g (:51-64); stores the new shift.</li>
 *   <li>{@code r = Wr x_r}; {@code w = exp(-0.606531 * sigmoid(W2 tanh(W1 x_w) + w0))} ({@code LLM_RWKV_Decay7},
 *       :66-70); {@code k = Wk x_k}; {@code v = Wv x_v} (:72-73).</li>
 *   <li>Value residual: layer 0 keeps its {@code v} in {@link RwkvBlocks#V_FIRST}; later layers add
 *       {@code (v_first - v) * sigmoid(V2 V1 x_v + v0)} (:74-84).</li>
 *   <li>Gating ({@code time_mix_g1} and {@code g2}): {@code g = G2 sigmoid(G1 x_g)} (:85-88).
 *       {@code a = sigmoid(A2 A1 x_a + a0)} (:89-91).</li>
 *   <li>{@code LLM_RWKV7_KkPrep}: {@code kk = l2norm(k * k_k)}, {@code k += a * k * k_a - k * k_a}, the WKV7
 *       operands {@code -kk} and {@code kk * a} (:93-97), then {@code LLM_RWKV_WKV7} (:107), after the decay as the
 *       kernels require.</li>
 *   <li>Group norm with {@code time_mix_ln} / {@code .bias} when both exist (:116-125), then
 *       {@code LLM_RWKV7_Bonus} {@code y += v * sum_head(k * r * r_k)} (:127-130), {@code y *= g} with gating
 *       (:131-133) and {@code out = Wo y} (:134).</li>
 * </ol>
 * State: the layer's {@code shift} row at offset 0 and its {@code state} matrices ({@code [h][value][key]}).
 * Not thread-safe.</p>
 */
public final class GpuRwkv7Mixer implements GpuMixer {

    private static final int PLANE_R = 0;
    private static final int PLANE_W = 1;
    private static final int PLANE_K = 2;
    private static final int PLANE_V = 3;
    private static final int PLANE_A = 4;
    private static final int PLANE_G = 5;

    private RwkvBlocks.GpuKit kit;
    private GpuOps ops;
    private boolean closed;

    private int nEmbd;
    private int headSize;
    private int nHead;
    private boolean firstLayer;
    private boolean gating;
    private int decayDim;
    private int iclrDim;
    private int valueDim;
    private int gateDim;

    private GpuWeight wr;
    private GpuWeight wk;
    private GpuWeight wv;
    private GpuWeight wo;
    /**
     * LoRA deltas of receptance, key, value and output ({@code build_lora_mm}, models/rwkv7-base.cpp); the low-rank
     * w/a/v/g products are plain {@code ggml_mul_mat} in llama.cpp and take none.
     */
    private GpuProjectionDeltas.Site rDeltas;
    private GpuProjectionDeltas.Site kDeltas;
    private GpuProjectionDeltas.Site vDeltas;
    private GpuProjectionDeltas.Site oDeltas;
    private GpuWeight w1;
    private GpuWeight w2;
    private GpuWeight a1;
    private GpuWeight a2;
    private GpuWeight v1;
    private GpuWeight v2;
    private GpuWeight g1;
    private GpuWeight g2;

    private ScratchSlot planes;
    private ScratchSlot r;
    private ScratchSlot w;
    private ScratchSlot k;
    private ScratchSlot v;
    private ScratchSlot a;
    private ScratchSlot g;
    private ScratchSlot lora;
    private ScratchSlot tmp;
    private ScratchSlot tmp2;
    private ScratchSlot aa;
    private ScratchSlot bb;
    private ScratchSlot y;
    private ScratchSlot vFirst;
    private List<GpuBuffer> working = Collections.emptyList();

    private PreparedDispatch shift;
    private PreparedDispatch tanhW;
    private PreparedDispatch decay;
    private PreparedDispatch vBias;
    private PreparedDispatch vSigmoid;
    private PreparedDispatch vMix;
    private PreparedDispatch gSigmoid;
    private PreparedDispatch aBias;
    private PreparedDispatch aSigmoid;
    private PreparedDispatch kkPrep;
    private PreparedDispatch wkv;
    private PreparedDispatch groupNorm;
    private PreparedDispatch bonus;
    private PreparedDispatch gateMul;

    /**
     * Resolves the layer's tensors, reserves the working temporaries (and {@link RwkvBlocks#V_FIRST}) and prepares
     * every element-wise and recurrent dispatch.
     *
     * @throws IllegalStateException    when already bound or closed, the context has no recurrent memory, or a
     *                                  tensor's shape contradicts the spec
     * @throws IllegalArgumentException when the layer is not an RWKV7 layer
     * @throws MissingTensorException   when a required tensor is absent
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        if (closed || kit != null) {
            throw new IllegalStateException("RWKV7 time mix is " + (closed ? "closed" : "already bound"));
        }
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        RecurrentSpec spec = RwkvBlocks.recurrent(layer, MixerKind.RWKV7);
        RwkvBlocks.GpuKit k0 = new RwkvBlocks.GpuKit(ctx, layer, "RWKV7 time mix");
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
        firstLayer = kt.layer.il() == 0;
        int c = nEmbd;
        bindProjections(kt, c);
        bindTemporaries(kt, c);
        bindDispatches(kt, c);
    }

    private void bindProjections(RwkvBlocks.GpuKit kt, int c) {
        wr = kt.matrix(TensorNames.TIME_MIX_RECEPTANCE, c, c);
        wk = kt.matrix(TensorNames.TIME_MIX_KEY, c, c);
        wv = kt.matrix(TensorNames.TIME_MIX_VALUE, c, c);
        wo = kt.matrix(TensorNames.TIME_MIX_OUTPUT, c, c);
        GpuProjectionDeltas deltas = kt.ctx.projectionDeltas();
        rDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_RECEPTANCE), wr);
        kDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_KEY), wk);
        vDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_VALUE), wv);
        oDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.TIME_MIX_OUTPUT), wo);
        w1 = kt.matrix(TensorNames.TIME_MIX_W1, c, 0);
        decayDim = Math.toIntExact(w1.rows());
        w2 = kt.matrix(TensorNames.TIME_MIX_W2, decayDim, c);
        a1 = kt.matrix(TensorNames.TIME_MIX_A1, c, 0);
        iclrDim = Math.toIntExact(a1.rows());
        a2 = kt.matrix(TensorNames.TIME_MIX_A2, iclrDim, c);
        if (!firstLayer) {
            v1 = kt.matrix(TensorNames.TIME_MIX_V1, c, 0);
            valueDim = Math.toIntExact(v1.rows());
            v2 = kt.matrix(TensorNames.TIME_MIX_V2, valueDim, c);
        }
        g1 = kt.matrixOrNull(TensorNames.TIME_MIX_G1, c, 0);
        gateDim = g1 != null ? Math.toIntExact(g1.rows()) : 0;
        g2 = g1 != null ? kt.matrixOrNull(TensorNames.TIME_MIX_G2, gateDim, c) : null;
        gating = g1 != null && g2 != null;
    }

    private void bindTemporaries(RwkvBlocks.GpuKit kt, int c) {
        int nMix = gating ? 6 : 5;
        planes = kt.temp("rwkv7.planes", (long) nMix * c);
        r = kt.temp("rwkv7.r", c);
        w = kt.temp("rwkv7.w", c);
        k = kt.temp("rwkv7.k", c);
        v = kt.temp("rwkv7.v", c);
        a = kt.temp("rwkv7.a", c);
        g = gating ? kt.temp("rwkv7.g", c) : null;
        lora = kt.temp("rwkv7.lora", Math.max(Math.max(decayDim, iclrDim), Math.max(valueDim, gateDim)));
        tmp = kt.temp("rwkv7.tmp", c);
        tmp2 = kt.temp("rwkv7.tmp2", c);
        aa = kt.temp("rwkv7.aa", c);
        bb = kt.temp("rwkv7.bb", c);
        y = kt.temp("rwkv7.y", c);
        vFirst = kt.temp(RwkvBlocks.V_FIRST, c);
        List<GpuBuffer> b = new ArrayList<>();
        for (ScratchSlot s : Arrays.asList(planes, r, w, k, v, a, lora, tmp, tmp2, aa, bb, y, vFirst)) {
            b.add(s.buffer());
        }
        if (g != null) {
            b.add(g.buffer());
        }
        working = Collections.unmodifiableList(b);
    }

    private void bindDispatches(RwkvBlocks.GpuKit kt, int c) {
        int nMix = gating ? 6 : 5;
        GpuBuffer shiftState = kt.state(LayerMemoryPlan.STATE_SHIFT, c);
        GpuBuffer state = kt.state(LayerMemoryPlan.STATE_SCAN, (long) c * headSize);
        shift = kt.tokenShift(c, nMix, RwkvKernels.SHIFT_MODE_LERP, shiftState, 0, null,
                kt.vector(TensorNames.TIME_MIX_LERP_FUSED, (long) nMix * c), planes.buffer());
        tanhW = kt.unary(ActOp.TANH, lora.buffer(), decayDim);
        decay = kt.decay(RwkvKernels.DECAY7, c, tmp.buffer(), kt.vector(TensorNames.TIME_MIX_W0, c), w.buffer());
        if (!firstLayer) {
            vBias = kt.biasAdd(kt.vector(TensorNames.TIME_MIX_V0, c), tmp.buffer(), c);
            vSigmoid = kt.unary(ActOp.SIGMOID, tmp.buffer(), c);
            vMix = kt.glu(GluOp.MUL, tmp2.buffer(), tmp.buffer(), tmp2.buffer(), c);
        }
        if (gating) {
            gSigmoid = kt.unary(ActOp.SIGMOID, lora.buffer(), gateDim);
            gateMul = kt.glu(GluOp.MUL, y.buffer(), g.buffer(), y.buffer(), c);
        }
        aBias = kt.biasAdd(kt.vector(TensorNames.TIME_MIX_A0, c), a.buffer(), c);
        aSigmoid = kt.unary(ActOp.SIGMOID, a.buffer(), c);
        kkPrep = kt.headRows(RwkvKernels.KK_PREP, nHead, headSize, kt.ubatch, nHead, headSize, k.buffer(),
                kt.vector(TensorNames.TIME_MIX_K_K, c), kt.vector(TensorNames.TIME_MIX_K_A, c), a.buffer(),
                tmp.buffer(), tmp2.buffer(), aa.buffer(), bb.buffer());
        wkv = kt.wkv(RwkvKernels.WKV7, nHead, headSize, kt.ubatch, c, nHead, r.buffer(), w.buffer(), tmp2.buffer(),
                v.buffer(), aa.buffer(), bb.buffer(), state, kt.stateRows(), y.buffer());
        GpuBuffer ln = kt.vectorOrNull(TensorNames.TIME_MIX_LN, c);
        GpuBuffer lnB = kt.biasOrNull(TensorNames.TIME_MIX_LN, c);
        if (ln != null && lnB != null) {
            groupNorm = kt.groupNorm(y.buffer(), nHead, headSize, ln, lnB);
        }
        bonus = kt.headRows(RwkvKernels.BONUS, nHead, headSize, kt.ubatch, nHead, headSize, tmp2.buffer(),
                r.buffer(), kt.vector(TensorNames.TIME_MIX_R_K, c), v.buffer(), y.buffer());
    }

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        prefill(pos, 1, in, out);
    }

    /**
     * Runs the time mix over {@code nTokens} rows of {@code in} into {@code out} and advances the layer's shift and
     * WKV state; layer 0 also refreshes {@link RwkvBlocks#V_FIRST} for the later layers of the same step.
     *
     * @throws IllegalStateException    when not bound or closed
     * @throws IllegalArgumentException for a token count outside {@code [1, ubatch]}, a slot too small for the rows
     *                                  or a slot sharing a buffer with the working rows
     */
    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        if (closed || kit == null) {
            throw new IllegalStateException("RWKV7 time mix is " + (closed ? "closed" : "not bound"));
        }
        int c = nEmbd;
        int t = nTokens;
        kit.checkStep(t, in, out, c, working);
        int plane = t * c;
        GpuBuffer x = planes.buffer();
        kit.runShift(shift, t, c, in);
        ops.matmul(wr, x, PLANE_R * plane, r.buffer(), 0, t);
        rDeltas.applyRows(t, x, PLANE_R * plane, r.buffer(), 0);
        ops.matmul(w1, x, PLANE_W * plane, lora.buffer(), 0, t);
        kit.runUnary(tanhW, t, decayDim);
        ops.matmul(w2, lora.buffer(), 0, tmp.buffer(), 0, t);
        kit.runDecay(decay, t, c);
        ops.matmul(wk, x, PLANE_K * plane, k.buffer(), 0, t);
        kDeltas.applyRows(t, x, PLANE_K * plane, k.buffer(), 0);
        ops.matmul(wv, x, PLANE_V * plane, v.buffer(), 0, t);
        vDeltas.applyRows(t, x, PLANE_V * plane, v.buffer(), 0);
        valueResidual(t, plane, x);
        if (gating) {
            ops.matmul(g1, x, PLANE_G * plane, lora.buffer(), 0, t);
            kit.runUnary(gSigmoid, t, gateDim);
            ops.matmul(g2, lora.buffer(), 0, g.buffer(), 0, t);
        }
        ops.matmul(a1, x, PLANE_A * plane, lora.buffer(), 0, t);
        ops.matmul(a2, lora.buffer(), 0, a.buffer(), 0, t);
        kit.runBias(aBias, t);
        kit.runUnary(aSigmoid, t, c);
        kit.runHeadRows(kkPrep, t, nHead, headSize);
        kit.runWkv(wkv, t);
        if (groupNorm != null) {
            kit.runGroupNorm(groupNorm, t, nHead);
        }
        kit.runHeadRows(bonus, t, nHead, headSize);
        if (gating) {
            kit.runGlu(gateMul, t, c);
        }
        ops.matmul(wo, y.buffer(), 0, out.buffer(), out.offset(), t);
        oDeltas.applyRows(t, y.buffer(), 0, out.buffer(), out.offset());
    }

    /** Layer 0 stores {@code v}; later layers compute {@code v += (v_first - v) * sigmoid(V2 V1 x_v + v0)}. */
    private void valueResidual(int t, int plane, GpuBuffer x) {
        if (firstLayer) {
            ops.copy(v.buffer(), 0, vFirst.buffer(), 0, plane);
            return;
        }
        ops.matmul(v1, x, PLANE_V * plane, lora.buffer(), 0, t);
        ops.matmul(v2, lora.buffer(), 0, tmp.buffer(), 0, t);
        kit.runBias(vBias, t);
        kit.runUnary(vSigmoid, t, nEmbd);
        ops.copy(vFirst.buffer(), 0, tmp2.buffer(), 0, plane);
        ops.scaleAdd(v.buffer(), 0, -1.0f, tmp2.buffer(), 0, plane);
        kit.runGlu(vMix, t, nEmbd);
        ops.add(v.buffer(), 0, tmp2.buffer(), 0, v.buffer(), 0, plane);
    }

    /** Releases the prepared dispatches. Idempotent. */
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
        return "GpuRwkv7Mixer{" + (kit != null ? kit.where + (gating ? ", gated" : "") : "unbound")
                + (closed ? ", closed" : "") + '}';
    }
}
