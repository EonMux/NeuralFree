package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtra;
import com.eonmux.neuralfree.gguf.cpu.block.CpuLayerExtraFactory;
import com.eonmux.neuralfree.gguf.cpu.block.CpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraExtra.Projection;
import com.eonmux.neuralfree.gguf.ops.heads.SwitchedLoraExtra.Router;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Arrays;
import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

/**
 * CPU twin of {@link SwitchedLoraExtra}: the graniteswitch per-token LoRA delta of every layer projection
 * (models/granite-switch.cpp:242-427) with {@link CpuPooling#switchedLoraAdd}, the CPU twin of
 * {@value HeadsKernels#SWITCHED_LORA_ADD}.
 *
 * <p>{@link #bind} decodes the 14 stacked tensors of the layer to F32, reserves the shared {@code tmp} and slot
 * temporaries and registers {@link #addDelta} of each projection in the context's {@link CpuProjectionDeltas} under
 * {@code <base>.weight}, so the layer's attention mixer and dense FFN add it right after their base products (a
 * projection no bound block serves fails the bind). {@link #beforeBlock} fills the slots of the step's rows from the
 * model's {@link Router}; {@link #addDelta} adds {@code B[s_t] (A[s_t] x_t)}. {@link #close} unregisters the deltas.
 * Not thread-safe.</p>
 */
public final class CpuSwitchedLoraExtra implements CpuLayerExtra {

    static final String TEMP_TMP = "switched_lora.tmp";
    static final String TEMP_SLOTS = "switched_lora.slots";

    private final Router router;
    private final Map<Projection, Pair> pairs = new EnumMap<>(Projection.class);
    /** The projection deltas registered in {@link #registry}, removed by {@link #close}. */
    private final Map<Projection, CpuProjectionDeltas.Delta> registered = new EnumMap<>(Projection.class);
    private CpuProjectionDeltas registry;
    private LayerSpec layer;
    private int nSlots;
    private int ubatch;
    private float[] tmp;
    private int[] slots;

    CpuSwitchedLoraExtra(Router router) {
        this.router = Objects.requireNonNull(router, "router");
    }

    /** The factory of {@code LayerSpec.Extras.SWITCHED_LORA}; {@code router} null makes creation fail. */
    static CpuLayerExtraFactory factory(Router router) {
        return new CpuLayerExtraFactory() {
            @Override
            public String featureId() {
                return LayerSpec.Extras.SWITCHED_LORA;
            }

            @Override
            public CpuLayerExtra create(LayerSpec layer) {
                if (router == null) {
                    throw new UnsupportedOperationException("switched LoRA layer " + layer.il() + " needs the model's"
                            + " SwitchedLoraExtra.Router; build the extras with HeadsBlocks.cpuExtras(router)");
                }
                return new CpuSwitchedLoraExtra(router);
            }
        };
    }

    /** The model's adapter router shared by the layers. */
    public Router router() {
        return router;
    }

    @Override
    public void bind(CpuForwardContext ctx, LayerSpec layer) {
        Objects.requireNonNull(ctx, "ctx");
        Objects.requireNonNull(layer, "layer");
        close();
        this.layer = layer;
        this.nSlots = router.nSlots();
        this.ubatch = ctx.scratch().ubatch();
        int maxRank = 0;
        for (Projection p : Projection.values()) {
            TensorRef a = ctx.spec().tensors().require(layer.stack(), layer.weightLayer(), p.loraA());
            TensorRef b = ctx.spec().tensors().require(layer.stack(), layer.weightLayer(), p.loraB());
            long[] an = a.ne();
            long[] bn = b.ne();
            if (an.length != 3 || bn.length != 3 || an[1] != bn[0] || an[2] != nSlots || bn[2] != nSlots) {
                throw new IllegalArgumentException("layer " + layer.il() + " " + p.base() + ": lora_a "
                        + Arrays.toString(an) + " and lora_b " + Arrays.toString(bn) + " are not stacked {in, rank, "
                        + nSlots + "} / {rank, out, " + nSlots + "}");
            }
            int in = HeadsBlocks.toInt(an[0], a.name());
            int rank = HeadsBlocks.toInt(an[1], a.name());
            int out = HeadsBlocks.toInt(bn[1], b.name());
            HeadsKernels.checkLoraShape(in, out, rank, nSlots);
            pairs.put(p, new Pair(in, out, rank, ctx.weights().floats(a), ctx.weights().floats(b)));
            maxRank = Math.max(maxRank, rank);
        }
        this.tmp = ctx.scratch().temp(TEMP_TMP, ubatch * maxRank);
        this.slots = ctx.scratch().intTemp(TEMP_SLOTS, ubatch);
        this.registry = ctx.projectionDeltas();
        try {
            for (Projection p : Projection.values()) {
                CpuProjectionDeltas.Delta d = (n, x, xOff, xStride, y, yOff, yStride) ->
                        addDelta(p, n, x, xOff, xStride, y, yOff, yStride);
                registry.register(layer, TensorNames.weight(p.base()), d);
                registered.put(p, d);
            }
        } catch (RuntimeException e) {
            close();
            throw e;
        }
    }

    /** Fills the adapter slots of rows {@code [pos0, pos0 + n)} for this step. */
    @Override
    public void beforeBlock(int pos0, int n, float[] x) {
        ensureBound();
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("switched LoRA step of " + n + " rows, ubatch is " + ubatch);
        }
        router.slots(pos0, n, slots, 0);
    }

    /**
     * {@code y[t] += B[s_t] (A[s_t] x[t])} for the {@code n} rows of the current step after projection {@code p}'s
     * base product: row {@code t} of {@code x} at {@code xOff + t*xStride}, of {@code y} at {@code yOff + t*yStride}.
     */
    public void addDelta(Projection p, int n, float[] x, int xOff, int xStride, float[] y, int yOff, int yStride) {
        ensureBound();
        Pair pair = pairs.get(Objects.requireNonNull(p, "projection"));
        if (n < 1 || n > ubatch) {
            throw new IllegalArgumentException("switched LoRA delta of " + n + " rows, ubatch is " + ubatch);
        }
        CpuPooling.switchedLoraAdd(n, pair.in, pair.out, pair.rank, nSlots, slots, pair.a, pair.b, x, xOff, xStride,
                tmp, 0, y, yOff, yStride);
    }

    private void ensureBound() {
        if (layer == null) {
            throw new IllegalStateException("switched LoRA extra is not bound");
        }
    }

    /** Unregisters the projection deltas. Idempotent. */
    @Override
    public void close() {
        if (registry != null) {
            for (Map.Entry<Projection, CpuProjectionDeltas.Delta> e : registered.entrySet()) {
                registry.unregister(layer, TensorNames.weight(e.getKey().base()), e.getValue());
            }
            registered.clear();
            registry = null;
        }
    }

    @Override
    public String toString() {
        return "CpuSwitchedLoraExtra{" + (layer == null ? "unbound" : "layer " + layer.il()) + '}';
    }

    /** The decoded stacked tensors of one projection. */
    private static final class Pair {
        final int in;
        final int out;
        final int rank;
        final float[] a;
        final float[] b;

        Pair(int in, int out, int rank, float[] a, float[] b) {
            this.in = in;
            this.out = out;
            this.rank = rank;
            this.a = a;
            this.b = b;
        }
    }
}
