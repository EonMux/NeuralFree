package com.eonmux.neuralfree.gguf.ops.rwkv;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The RWKV channel-mix FFN on the GPU ({@code FfnKind.RWKV_CHANNEL_MIX}) over the {@code attn_norm_2} rows, with the
 * channel mix's previous token in the layer's shift row at offset {@code n_embd}:
 * <ul>
 *   <li>RWKV6 (models/rwkv6-base.cpp:13-23): {@code TOKEN_SHIFT_LERP} mode
 *       {@code PAIR} gives {@code x_k = sx * lerp_k + cur} and {@code x_r = sx * lerp_r + cur};
 *       {@code out = sigmoid(Wr x_r) * (Wv relu(Wk x_k)^2)}.</li>
 *   <li>RWKV7 (models/rwkv7-base.cpp:9-27): mode {@code LERP} with {@code lerp_k};
 *       {@code out = Wv relu(Wk x_k)^2}.</li>
 * </ul>
 * Both store the new shift. Not thread-safe.
 */
public final class GpuRwkvChannelMix implements GpuFfnBlock {

    private RwkvBlocks.GpuKit kit;
    private GpuOps ops;
    private boolean closed;

    private int nEmbd;
    private int nFf;
    private boolean receptance;

    private GpuWeight wk;
    private GpuWeight wv;
    private GpuWeight wr;
    /**
     * LoRA deltas of key, value and receptance ({@code build_lora_mm}, models/rwkv6-base.cpp,
     * models/rwkv7-base.cpp).
     */
    private GpuProjectionDeltas.Site kDeltas;
    private GpuProjectionDeltas.Site vDeltas;
    private GpuProjectionDeltas.Site rDeltas;

    private ScratchSlot x;
    private ScratchSlot kAct;
    private ScratchSlot vOut;
    private ScratchSlot rOut;
    private List<GpuBuffer> working = Collections.emptyList();

    private PreparedDispatch shift;
    private PreparedDispatch reluSqr;

    /**
     * Resolves the layer's channel-mix tensors, reserves the working temporaries and prepares the token-shift and
     * activation dispatches.
     *
     * @throws IllegalStateException    when already bound or closed, the context has no recurrent memory, or a
     *                                  tensor's shape contradicts the spec
     * @throws IllegalArgumentException when the layer is not an RWKV6 / RWKV7 channel-mix layer
     * @throws com.eonmux.neuralfree.gguf.spec.MissingTensorException when a required tensor is absent
     */
    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        if (closed || kit != null) {
            throw new IllegalStateException("RWKV channel mix is " + (closed ? "closed" : "already bound"));
        }
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        RecurrentSpec spec = RwkvBlocks.channelMixSpec(layer);
        RwkvBlocks.GpuKit k0 = new RwkvBlocks.GpuKit(ctx, layer, "RWKV channel mix");
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
        int c = spec.nEmbd();
        nEmbd = c;
        wk = kt.matrix(TensorNames.CHANNEL_MIX_KEY, c, 0);
        nFf = Math.toIntExact(wk.rows());
        wv = kt.matrix(TensorNames.CHANNEL_MIX_VALUE, nFf, c);
        receptance = spec.kind() == MixerKind.RWKV6;
        wr = receptance ? kt.matrix(TensorNames.CHANNEL_MIX_RECEPTANCE, c, c) : null;
        GpuProjectionDeltas deltas = kt.ctx.projectionDeltas();
        kDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.CHANNEL_MIX_KEY), wk);
        vDeltas = deltas.serve(kt.layer, TensorNames.weight(TensorNames.CHANNEL_MIX_VALUE), wv);
        rDeltas = wr != null
                ? deltas.serve(kt.layer, TensorNames.weight(TensorNames.CHANNEL_MIX_RECEPTANCE), wr) : null;
        int planes = receptance ? 2 : 1;
        x = kt.temp("rwkv.cmix.x", (long) planes * c);
        kAct = kt.temp("rwkv.cmix.k", nFf);
        vOut = receptance ? kt.temp("rwkv.cmix.v", c) : null;
        rOut = receptance ? kt.temp("rwkv.cmix.r", c) : null;
        List<GpuBuffer> b = new ArrayList<>(Arrays.asList(x.buffer(), kAct.buffer()));
        if (receptance) {
            b.add(vOut.buffer());
            b.add(rOut.buffer());
        }
        working = Collections.unmodifiableList(b);
        GpuBuffer shiftState = kt.state(LayerMemoryPlan.STATE_SHIFT, 2L * c);
        GpuBuffer lerpK = kt.vector(TensorNames.CHANNEL_MIX_LERP_K, c);
        if (receptance) {
            shift = kt.tokenShift(c, 1, RwkvKernels.SHIFT_MODE_PAIR, shiftState, c, lerpK,
                    kt.vector(TensorNames.CHANNEL_MIX_LERP_R, c), x.buffer());
        } else {
            shift = kt.tokenShift(c, 1, RwkvKernels.SHIFT_MODE_LERP, shiftState, c, null, lerpK, x.buffer());
        }
        reluSqr = kt.unary(ActOp.RELU_SQR, kAct.buffer(), nFf);
    }

    /**
     * Runs the channel mix over {@code nTokens} rows of {@code in} into {@code out} and advances the layer's
     * channel-mix shift.
     *
     * @throws IllegalStateException    when not bound or closed
     * @throws IllegalArgumentException for a token count outside {@code [1, ubatch]}, a slot too small for the rows
     *                                  or a slot sharing a buffer with the working rows
     */
    @Override
    public void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
        if (closed || kit == null) {
            throw new IllegalStateException("RWKV channel mix is " + (closed ? "closed" : "not bound"));
        }
        int c = nEmbd;
        int t = nTokens;
        kit.checkStep(t, in, out, c, working);
        kit.runShift(shift, t, c, in);
        ops.matmul(wk, x.buffer(), 0, kAct.buffer(), 0, t);
        kDeltas.applyRows(t, x.buffer(), 0, kAct.buffer(), 0);
        kit.runUnary(reluSqr, t, nFf);
        if (!receptance) {
            ops.matmul(wv, kAct.buffer(), 0, out.buffer(), out.offset(), t);
            vDeltas.applyRows(t, kAct.buffer(), 0, out.buffer(), out.offset());
            return;
        }
        ops.matmul(wv, kAct.buffer(), 0, vOut.buffer(), 0, t);
        vDeltas.applyRows(t, kAct.buffer(), 0, vOut.buffer(), 0);
        ops.matmul(wr, x.buffer(), t * c, rOut.buffer(), 0, t);
        rDeltas.applyRows(t, x.buffer(), t * c, rOut.buffer(), 0);
        ops.glu(GluOp.SIGMUL, ActParams.NONE, rOut.buffer(), 0, c, vOut.buffer(), 0, c, out.buffer(), out.offset(),
                c, t, c);
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
        return "GpuRwkvChannelMix{" + (kit != null ? kit.where + (receptance ? ", rwkv6" : ", rwkv7") : "unbound")
                + (closed ? ", closed" : "") + '}';
    }
}
