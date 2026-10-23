package com.eonmux.neuralfree.gguf.ops.norm;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.spec.ActParams;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.Arrays;
import java.util.Objects;

/**
 * Device twin of {@link AdaNormMath}: per step and layer, {@code ada = up · gelu_erf(down · t_cond)} into a private
 * vector, then {@code x = x * 1 + x * ada} by {@value NormKernels#RESIDUAL_MIX} in {@link NormKernels.MixMode#ALPHA_BETA}
 * mode with the row period {@code width}. Owns its {@code t_cond}, ones, hidden and ada device vectors; the owner closes
 * it once the queue is idle. Not thread-safe.
 */
public final class GpuAdaNorm implements AutoCloseable {

    private final GpuForwardContext ctx;
    private final GpuOps ops;
    private final int width;
    private final int maxHidden;
    private final GpuBuffer placeholder;
    private GpuBuffer tCond;
    private GpuBuffer ones;
    private GpuBuffer hidden;
    private GpuBuffer ada;

    /**
     * Uploads {@code t_cond} of {@code delay}, allocates the step vectors and prepares
     * {@value NormKernels#RESIDUAL_MIX} for the launch size of {@link #apply} ({@link GpuOps#prewarm}).
     *
     * @param maxHidden largest hidden width of the layers' ada MLPs
     * @throws IllegalArgumentException when a width is not positive or the delay is negative
     */
    public GpuAdaNorm(GpuForwardContext ctx, int width, int delay, int maxHidden) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.ops = ctx.ops();
        if (maxHidden < 1) {
            throw new IllegalArgumentException("ada hidden width must be positive, got " + maxHidden);
        }
        float[] t = AdaNormMath.timeCondition(delay, width);
        float[] one = new float[width];
        Arrays.fill(one, 1.0f);
        this.width = width;
        this.maxHidden = maxHidden;
        this.placeholder = ctx.placeholders().readOnlyFloats();
        try {
            tCond = upload(t);
            ones = upload(one);
            hidden = ctx.backend().allocateBuffer((long) maxHidden * Float.BYTES);
            ada = ctx.backend().allocateBuffer((long) width * Float.BYTES);
            ops.prewarm(NormKernels.RESIDUAL_MIX, new long[]{ops.local()});
        } catch (RuntimeException e) {
            try {
                close();
            } catch (RuntimeException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
    }

    private GpuBuffer upload(float[] v) {
        GpuBuffer b = ctx.backend().allocateBuffer((long) v.length * Float.BYTES, true);
        try {
            GpuEvent e = ctx.queue().enqueueWriteBuffer(b, 0, (long) v.length * Float.BYTES, v);
            if (e != null) {
                e.close();
            }
        } catch (RuntimeException e) {
            b.close();
            throw e;
        }
        return b;
    }

    /** One layer's ada projections on the device, with their LoRA delta sites. */
    public static final class Layer {
        final GpuWeight down;
        final GpuWeight up;
        final GpuProjectionDeltas.Site downDeltas;
        final GpuProjectionDeltas.Site upDeltas;
        final int hidden;

        /**
         * @throws IllegalArgumentException when the projections are not a {@code [width, h]} / {@code [h, width]} pair
         */
        public Layer(GpuWeight down, GpuWeight up, GpuProjectionDeltas.Site downDeltas,
                     GpuProjectionDeltas.Site upDeltas, int width) {
            this.down = Objects.requireNonNull(down, "down");
            this.up = Objects.requireNonNull(up, "up");
            this.downDeltas = Objects.requireNonNull(downDeltas, "downDeltas");
            this.upDeltas = Objects.requireNonNull(upDeltas, "upDeltas");
            if (down.cols() != width || up.rows() != width || up.cols() != down.rows() || down.rows() < 1
                    || down.rows() > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("ada projections " + down.name() + " " + down.cols() + "x"
                        + down.rows() + " and " + up.name() + " " + up.cols() + "x" + up.rows()
                        + " do not map width " + width + " through a hidden width");
            }
            this.hidden = (int) down.rows();
        }

        /** Hidden width of the ada MLP. */
        public int hidden() {
            return hidden;
        }
    }

    /**
     * Enqueues {@code x = x + x * ada(layer)} over {@code nRows} rows of {@code width} floats at {@code xOff}.
     *
     * @throws IllegalArgumentException when the layer's hidden width exceeds the allocated one
     * @throws IllegalStateException    when closed
     */
    public void apply(Layer layer, GpuBuffer x, int xOff, int nRows) {
        Objects.requireNonNull(layer, "layer");
        if (ada == null) {
            throw new IllegalStateException("ada norm is closed");
        }
        if (layer.hidden > maxHidden) {
            throw new IllegalArgumentException("ada hidden width " + layer.hidden + " exceeds " + maxHidden);
        }
        ops.matmul(layer.down, tCond, 0, hidden, 0, 1);
        layer.downDeltas.applyRows(1, tCond, 0, hidden, 0);
        ops.unary(ActOp.GELU_ERF, ActParams.NONE, hidden, 0, hidden, 0, layer.hidden);
        ops.matmul(layer.up, hidden, 0, ada, 0, 1);
        layer.upDeltas.applyRows(1, hidden, 0, ada, 0);
        int n = Math.multiplyExact(nRows, width);
        int local = ops.local();
        ops.enqueue(NormKernels.RESIDUAL_MIX, new long[]{((long) n + local - 1) / local * local}, new long[]{local},
                NormKernels.residualMixArgs(NormKernels.MixMode.ALPHA_BETA, n, 1.0f, x, xOff, x, xOff, ones, ada,
                        width, x, xOff, placeholder));
    }

    /** Releases the device vectors. The queue must be idle. Idempotent. */
    @Override
    public void close() {
        GpuBuffer[] all = {tCond, ones, hidden, ada};
        tCond = null;
        ones = null;
        hidden = null;
        ada = null;
        RuntimeException failure = Closeables.closeAll(Arrays.asList(all), null);
        if (failure != null) {
            throw failure;
        }
    }
}
