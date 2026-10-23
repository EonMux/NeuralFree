package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.ops.flow.FlowKernels;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Device deltas of one LoRA pair bound to its scale ({@link LoraAdapter.Delta}), as the three kinds of
 * {@link GpuProjectionDeltas}:
 * <ul>
 *   <li>{@link Projection}: {@code y[t] += scale * B (A x[t])} (llama-graph.cpp:1648-1664) through
 *       {@link SwitchedLoraExtra.GpuDelta#ofAdapter}, in chunks of at most {@code ubatch} rows;</li>
 *   <li>{@link Experts}: {@code y[p] += scale * B[e_p] (A[e_p] x_p)} per selected (token, expert) pair
 *       (build_lora_mm_id, :1734-1746) through {@value HeadsKernels#SWITCHED_LORA_ADD} with the expert ids as slots
 *       over the stacked {@code A {in, r, n_expert}} / {@code B {r, out, n_expert}};</li>
 *   <li>{@link Embedding}: {@code y[t] += scale * B A[id_t]} (build_inp_embd, :2694-2709) by
 *       {@value FlowKernels#GATHER_ROWS} of the {@code A} rows, then the {@value HeadsKernels#SWITCHED_LORA_ADD}
 *       up stage with one slot.</li>
 * </ul>
 * {@code A} and {@code scale * B} are uploaded once as F32 and owned by the delta, which releases them on
 * {@link AutoCloseable#close}. Every dispatch is enqueued on the queue of the {@link GpuOps} the delta was built with.
 */
public final class GpuLoraDeltas {

    /** Every kernel the deltas dispatch, for {@code KernelSet.prewarm} at attach time. */
    public static final List<String> KERNELS = Collections.unmodifiableList(Arrays.asList(
            HeadsKernels.SWITCHED_LORA_ADD, FlowKernels.GATHER_ROWS, GpuOps.COPY2D));

    /** Scratch temporary of the expert and embedding {@code tmp = A x} rows. */
    static final String TEMP_TMP = "lora.tmp";
    /** Scratch temporary of the per-pair input rows of a batched expert product over token rows. */
    static final String TEMP_PAIR_X = "lora.pair_x";

    private GpuLoraDeltas() {
    }

    /**
     * The device delta of {@code d} for its pair's kind.
     *
     * @throws IllegalArgumentException when the pair is too large for the kernel's 32-bit indexing
     */
    public static AutoCloseable of(GpuOps ops, LoraAdapter.Delta d) {
        switch (d.pair().kind()) {
            case EXPERTS:
                return new Experts(ops, d);
            case EMBEDDING:
                return new Embedding(ops, d);
            default:
                return new Projection(ops, d);
        }
    }

    /** A dense projection delta. */
    public static final class Projection implements GpuProjectionDeltas.Delta, AutoCloseable {
        private final SwitchedLoraExtra.GpuDelta delta;
        private final int ubatch;

        Projection(GpuOps ops, LoraAdapter.Delta d) {
            requireKind(d, LoraAdapter.Target.PROJECTION);
            this.delta = SwitchedLoraExtra.GpuDelta.ofAdapter(ops, d);
            this.ubatch = ops.scratch().ubatch();
        }

        @Override
        public void add(int nTokens, GpuBuffer x, int xOff, int xStride, GpuBuffer y, int yOff, int yStride) {
            for (int t = 0; t < nTokens; t += ubatch) {
                int n = Math.min(ubatch, nTokens - t);
                delta.add(n, x, xOff + t * xStride, xStride, y, yOff + t * yStride, yStride);
            }
        }

        @Override
        public void close() {
            delta.close();
        }
    }

    /** A per-expert delta over the selected pairs. */
    public static final class Experts implements GpuProjectionDeltas.ExpertDelta, AutoCloseable {
        private final GpuOps ops;
        private final LoraAdapter.Pair pair;
        private final List<GpuBuffer> owned = new ArrayList<>();
        private final GpuBuffer a;
        private final GpuBuffer b;

        Experts(GpuOps ops, LoraAdapter.Delta d) {
            requireKind(d, LoraAdapter.Target.EXPERTS);
            this.ops = Objects.requireNonNull(ops, "ops");
            this.pair = d.pair();
            HeadsKernels.checkLoraShape(pair.inDim(), pair.outDim(), pair.rank(), pair.experts());
            try {
                this.a = upload(ops, pair.target() + LoraAdapterLoader.SUFFIX_A, d.a(), owned);
                this.b = upload(ops, pair.target() + LoraAdapterLoader.SUFFIX_B, d.scaledB(), owned);
            } catch (RuntimeException e) {
                SwitchedLoraExtra.closeAll(owned);
                throw e;
            }
            ops.prewarm(HeadsKernels.SWITCHED_LORA_ADD, new long[]{HeadsKernels.LORA_LOCAL});
        }

        @Override
        public void add(int nTokens, int k, GpuBuffer ids, GpuBuffer x, int xOff, boolean tokenRows, GpuBuffer y,
                        int yOff) {
            int pairs = Math.multiplyExact(nTokens, k);
            int in = pair.inDim();
            GpuBuffer src = x;
            int srcOff = xOff;
            int xStride = in;
            if (tokenRows && nTokens == 1) {
                xStride = 0;
            } else if (tokenRows && k > 1) {
                ScratchSlot rows = ops.scratch().temp(TEMP_PAIR_X, (long) pairs * in);
                for (int j = 0; j < k; j++) {
                    ops.copy2d(x, xOff, in, rows.buffer(), rows.offset() + j * in, k * in, nTokens, in);
                }
                src = rows.buffer();
                srcOff = rows.offset();
            }
            ScratchSlot tmp = ops.scratch().temp(TEMP_TMP, (long) pairs * pair.rank());
            stages(pairs, ids, src, srcOff, xStride, tmp, y, yOff);
        }

        private void stages(int pairs, GpuBuffer ids, GpuBuffer x, int xOff, int xStride, ScratchSlot tmp,
                            GpuBuffer y, int yOff) {
            for (int stage : new int[]{HeadsKernels.LORA_STAGE_DOWN, HeadsKernels.LORA_STAGE_UP}) {
                long[] global = {HeadsKernels.loraGlobal(stage, pairs, pair.rank(), pair.outDim())};
                ops.enqueue(HeadsKernels.SWITCHED_LORA_ADD, global, new long[]{HeadsKernels.LORA_LOCAL},
                        pairs, pair.inDim(), pair.outDim(), pair.rank(), stage, pair.experts(), xStride, xOff,
                        pair.outDim(), yOff, tmp.offset(), ids, a, b, x, tmp.buffer(), y);
            }
        }

        @Override
        public void close() {
            SwitchedLoraExtra.closeAll(owned);
        }
    }

    /** A token-embedding delta over looked-up rows. */
    public static final class Embedding implements GpuProjectionDeltas.EmbeddingDelta, AutoCloseable {
        private final GpuOps ops;
        private final LoraAdapter.Pair pair;
        private final List<GpuBuffer> owned = new ArrayList<>();
        private final GpuBuffer a;
        private final GpuBuffer b;
        private final GpuBuffer zeroSlots;
        private final int ubatch;

        Embedding(GpuOps ops, LoraAdapter.Delta d) {
            requireKind(d, LoraAdapter.Target.EMBEDDING);
            this.ops = Objects.requireNonNull(ops, "ops");
            this.pair = d.pair();
            this.ubatch = ops.scratch().ubatch();
            HeadsKernels.checkLoraShape(pair.rank(), pair.outDim(), pair.rank(), 1);
            if ((long) pair.inDim() * pair.rank() > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("LoRA table of " + pair.target() + " (" + pair.inDim() + " x "
                        + pair.rank() + ") exceeds the 32-bit index range");
            }
            try {
                this.a = upload(ops, pair.target() + LoraAdapterLoader.SUFFIX_A, d.a(), owned);
                this.b = upload(ops, pair.target() + LoraAdapterLoader.SUFFIX_B, d.scaledB(), owned);
                this.zeroSlots = SwitchedLoraExtra.allocateFloats(ops, pair.target() + " LoRA slots", ubatch, false);
                owned.add(zeroSlots);
                ops.fill(zeroSlots, 0L, (long) ubatch * Integer.BYTES, 0);
            } catch (RuntimeException e) {
                SwitchedLoraExtra.closeAll(owned);
                throw e;
            }
            ops.prewarm(FlowKernels.GATHER_ROWS, new long[]{FlowKernels.GATHER_LOCAL});
            ops.prewarm(HeadsKernels.SWITCHED_LORA_ADD, new long[]{HeadsKernels.LORA_LOCAL});
        }

        @Override
        public void add(int nTokens, GpuBuffer ids, GpuBuffer y, int yOff, int yStride) {
            if (nTokens > ubatch) {
                throw new IllegalArgumentException("LoRA embedding delta of " + nTokens + " rows, ubatch is " + ubatch);
            }
            int r = pair.rank();
            ScratchSlot tmp = ops.scratch().temp(TEMP_TMP, (long) nTokens * r);
            ops.enqueue(FlowKernels.GATHER_ROWS, new long[]{FlowKernels.gatherGlobal(nTokens, r)},
                    new long[]{FlowKernels.GATHER_LOCAL}, nTokens, r, r, r, 0, tmp.offset(), ids, a, tmp.buffer());
            long[] global = {HeadsKernels.loraGlobal(HeadsKernels.LORA_STAGE_UP, nTokens, r, pair.outDim())};
            ops.enqueue(HeadsKernels.SWITCHED_LORA_ADD, global, new long[]{HeadsKernels.LORA_LOCAL},
                    nTokens, r, pair.outDim(), r, HeadsKernels.LORA_STAGE_UP, 1, r, 0, yStride, yOff, tmp.offset(),
                    zeroSlots, a, b, ops.placeholders().readOnlyFloats(), tmp.buffer(), y);
        }

        @Override
        public void close() {
            SwitchedLoraExtra.closeAll(owned);
        }
    }

    private static void requireKind(LoraAdapter.Delta d, LoraAdapter.Target kind) {
        if (Objects.requireNonNull(d, "delta").pair().kind() != kind) {
            throw new IllegalArgumentException(d.pair() + " is not a " + kind + " pair");
        }
    }

    private static GpuBuffer upload(GpuOps ops, String what, float[] values, List<GpuBuffer> owned) {
        GpuBuffer buf = SwitchedLoraExtra.allocateFloats(ops, what, values.length, true);
        owned.add(buf);
        GpuEvent e = ops.queue().enqueueWriteBuffer(buf, 0L, (long) values.length * Float.BYTES, values);
        if (e != null) {
            e.close();
        }
        return buf;
    }
}
