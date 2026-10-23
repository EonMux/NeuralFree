package com.eonmux.neuralfree.gguf.ops.rwkv;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.cpu.block.CpuFfnBlock;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * CPU twin of {@link GpuRwkvChannelMix}: the RWKV6 channel mix
 * {@code out = sigmoid(Wr x_r) * (Wv relu(Wk x_k)^2)} (models/rwkv6-base.cpp:13-23) and the RWKV7 channel mix
 * {@code out = Wv relu(Wk x_k)^2} (models/rwkv7-base.cpp:9-27), with the previous token in the layer's shift row at
 * offset {@code n_embd}. Not thread-safe.
 */
public final class CpuRwkvChannelMix implements CpuFfnBlock {

    private RwkvBlocks.CpuKit kit;
    private CpuMatVec mv;
    private boolean closed;

    private int nEmbd;
    private int nFf;
    private boolean receptance;

    private CpuTensorRef wk;
    private CpuTensorRef wv;
    private CpuTensorRef wr;
    /**
     * LoRA deltas of key, value and receptance ({@code build_lora_mm}, models/rwkv6-base.cpp,
     * models/rwkv7-base.cpp).
     */
    private CpuProjectionDeltas.Site kDeltas;
    private CpuProjectionDeltas.Site vDeltas;
    private CpuProjectionDeltas.Site rDeltas;
    private float[] lerpK;
    private float[] lerpR;
    private float[] shift;

    private float[] x;
    private float[] kAct;
    private float[] vOut;
    private float[] rOut;

    /**
     * Resolves the layer's channel-mix tensors and shift row and reserves the working temporaries.
     *
     * @throws IllegalStateException    when already bound or closed, the context has no recurrent memory, or a
     *                                  tensor's shape contradicts the spec
     * @throws IllegalArgumentException when the layer is not an RWKV6 / RWKV7 channel-mix layer
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        if (closed || kit != null) {
            throw new IllegalStateException("RWKV channel mix is " + (closed ? "closed" : "already bound"));
        }
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        RecurrentSpec spec = RwkvBlocks.channelMixSpec(layer);
        RwkvBlocks.CpuKit kt = new RwkvBlocks.CpuKit(ctx, layer, "RWKV channel mix");
        int c = spec.nEmbd();
        nEmbd = c;
        receptance = spec.kind() == MixerKind.RWKV6;
        wk = kt.matrix(TensorNames.CHANNEL_MIX_KEY, c, 0);
        nFf = Math.toIntExact(wk.rows());
        wv = kt.matrix(TensorNames.CHANNEL_MIX_VALUE, nFf, c);
        lerpK = kt.vector(TensorNames.CHANNEL_MIX_LERP_K, c);
        if (receptance) {
            wr = kt.matrix(TensorNames.CHANNEL_MIX_RECEPTANCE, c, c);
            lerpR = kt.vector(TensorNames.CHANNEL_MIX_LERP_R, c);
        }
        CpuProjectionDeltas deltas = kt.ctx.projectionDeltas();
        kDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.CHANNEL_MIX_KEY), wk, kt.mv);
        vDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.CHANNEL_MIX_VALUE), wv, kt.mv);
        rDeltas = wr != null
                ? deltas.serve(kt.layer, TensorNames.weight(TensorNames.CHANNEL_MIX_RECEPTANCE), wr, kt.mv) : null;
        shift = kt.state(LayerMemoryPlan.STATE_SHIFT);
        CpuWkv.checkRows("RWKV shift state", shift, 0, 2, c);
        x = kt.temp("rwkv.cmix.x", (receptance ? 2 : 1) * c);
        kAct = kt.temp("rwkv.cmix.k", nFf);
        vOut = receptance ? kt.temp("rwkv.cmix.v", c) : null;
        rOut = receptance ? kt.temp("rwkv.cmix.r", c) : null;
        mv = kt.mv;
        kit = kt;
    }

    /**
     * Runs the channel mix over {@code n} rows of {@code in} into {@code out} and advances the layer's channel-mix
     * shift.
     *
     * @throws IllegalStateException    when not bound or closed
     * @throws IllegalArgumentException for a token count outside {@code [1, ubatch]} or arrays too small for the rows
     */
    @Override
    public void forward(int n, float[] in, float[] out) {
        if (closed || kit == null) {
            throw new IllegalStateException("RWKV channel mix is " + (closed ? "closed" : "not bound"));
        }
        int c = nEmbd;
        kit.checkStep(n, in, out, c);
        if (receptance) {
            CpuRwkvMix.tokenShiftLerp(n, c, 1, RwkvKernels.SHIFT_MODE_PAIR, in, 0, shift, c, lerpK, lerpR, x, 0);
        } else {
            CpuRwkvMix.tokenShiftLerp(n, c, 1, RwkvKernels.SHIFT_MODE_LERP, in, 0, shift, c, null, lerpK, x, 0);
        }
        mv.matmul(wk, x, 0, kAct, 0, n);
        kDeltas.applyRows(n, x, 0, kAct, 0);
        CpuOps.unary(ActOp.RELU_SQR, ActParams.NONE, kAct, 0, kAct, 0, n * nFf);
        if (!receptance) {
            mv.matmul(wv, kAct, 0, out, 0, n);
            vDeltas.applyRows(n, kAct, 0, out, 0);
            return;
        }
        mv.matmul(wv, kAct, 0, vOut, 0, n);
        vDeltas.applyRows(n, kAct, 0, vOut, 0);
        mv.matmul(wr, x, n * c, rOut, 0, n);
        rDeltas.applyRows(n, x, n * c, rOut, 0);
        CpuOps.glu(GluOp.SIGMUL, ActParams.NONE, rOut, 0, vOut, 0, out, 0, n * c);
    }

    @Override
    public void close() {
        closed = true;
    }

    @Override
    public String toString() {
        return "CpuRwkvChannelMix{" + (kit != null ? kit.where + (receptance ? ", rwkv6" : ", rwkv7") : "unbound")
                + (closed ? ", closed" : "") + '}';
    }
}
