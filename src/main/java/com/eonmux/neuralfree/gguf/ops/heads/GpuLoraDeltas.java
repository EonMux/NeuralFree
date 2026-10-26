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
 *   <li>{@link Projection}: {@code y[t] += scale * B (A x[t])} (llama-graph.cpp:1648-1664) through the one-slot
 *       {@link SwitchedLoraExtra.GpuDelta}, in chunks of at most {@code ubatch} rows;</li>
 *   <li>{@link Experts}: {@code y[p] += scale * B[e_p] (A[e_p] x_p)} per selected (token, expert) pair
 *       (build_lora_mm_id, :1734-1746) through {@value HeadsKernels#SWITCHED_LORA_ADD} with the expert ids as slots
 *       over the stacked {@code A {in, r, n_expert}} / {@code B {r, out, n_expert}};</li>
 *   <li>{@link Embedding}: {@code y[t] += scale * B A[id_t]} (build_inp_embd, :2694-2709) by
 *       {@value FlowKernels#GATHER_ROWS} of the {@code A} rows, then the {@value HeadsKernels#SWITCHED_LORA_ADD}
 *       up stage with one slot.</li>
 * </ul>
 * {@code A} and {@code scale * B} are uploaded once as F32 and owned by the delta, which releases them on
 * {@link AutoCloseable#close}. Every dispatch is enqueued on the queue of the {@link GpuOps} the delta was built with.
 * Between steps, on the engine thread, the bound scale rewrites the host {@code scale * B} and re-uploads the buffer
 * in place ({@link Rescalable#rescale}, LR-1): no kernel ABI change, the device bytes of the same buffer change and
 * every following dispatch reads the new B. A rescale after the last dispatch of a step and before the first of the
 * next one is safe; the fencers of {@code LoraRegistry} events (PC-1's ParameterChanges) keep that between registry
 * events and steps.
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
     * The device delta of {@code d} for its pair's kind; the delta carries the rescale of its bound scale.
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

    /**
     * A device delta whose bound scale rewrites its {@code scale * B} buffer in place between steps (LR-1).
     */
    public interface Rescalable {
        /** Rewrites the bound scale to {@code userScale}; the next dispatch reads the new {@code B}. */
        void rescale(float userScale);
    }

    /** A dense projection delta. */
    public static final class Projection implements GpuProjectionDeltas.Delta, AutoCloseable, Rescalable {
        private final LoraAdapter.Delta delta;
        private final GpuOps ops;
        private final SwitchedLoraExtra.GpuDelta kernel;
        private final GpuBuffer b;
        private final List<GpuBuffer> owned = new ArrayList<>();
        private final int ubatch;

        Projection(GpuOps ops, LoraAdapter.Delta d) {
            requireKind(d, LoraAdapter.Target.PROJECTION);
            this.delta = d;
            this.ops = Objects.requireNonNull(ops, "ops");
            LoraAdapter.Pair pair = d.pair();
            this.ubatch = ops.scratch().ubatch();
            GpuBuffer zeros = SwitchedLoraExtra.allocateFloats(ops, pair.target() + " LoRA slots", ubatch, false);
            owned.add(zeros);
            ops.fill(zeros, 0L, (long) ubatch * Integer.BYTES, 0);
            try {
                GpuBuffer a = SwitchedLoraExtra.allocateFloats(ops, pair.target() + LoraAdapterLoader.SUFFIX_A,
                        d.a().length, true);
                owned.add(a);
                rewrite(ops, a, d.a());
                this.b = SwitchedLoraExtra.allocateFloats(ops, pair.target() + LoraAdapterLoader.SUFFIX_B,
                        d.scaledB().length, false);
                owned.add(b);
                rewrite(ops, b, d.scaledB());
                this.kernel = new SwitchedLoraExtra.GpuDelta(ops, a, b, pair.inDim(), pair.outDim(), pair.rank(), 1);
            } catch (RuntimeException e) {
                SwitchedLoraExtra.closeAll(owned);
                throw e;
            }
        }

        @Override
        public void add(int nTokens, GpuBuffer x, int xOff, int xStride, GpuBuffer y, int yOff, int yStride) {
            for (int t = 0; t < nTokens; t += ubatch) {
                int n = Math.min(ubatch, nTokens - t);
                kernel.add(n, zeroSlots(), x, xOff + t * xStride, xStride, y, yOff + t * yStride, yStride);
            }
        }

        private GpuBuffer zeroSlots() {
            // The first owned buffer is the zero-slot buffer of the one-slot adapter delta.
            return owned.get(0);
        }

        @Override
        public void rescale(float userScale) {
            delta.rescale(userScale);
            rewrite(ops, b, delta.scaledB());
        }

        @Override
        public void close() {
            kernel.close();
            SwitchedLoraExtra.closeAll(owned);
        }
    }

    /** A per-expert delta over the selected pairs. */
    public static final class Experts implements GpuProjectionDeltas.ExpertDelta, AutoCloseable, Rescalable {
        private final LoraAdapter.Delta delta;
        private final GpuOps ops;
        private final LoraAdapter.Pair pair;
        private final GpuBuffer b;
        private final List<GpuBuffer> owned = new ArrayList<>();
        private final GpuBuffer a;

        Experts(GpuOps ops, LoraAdapter.Delta d) {
            requireKind(d, LoraAdapter.Target.EXPERTS);
            this.delta = d;
            this.ops = Objects.requireNonNull(ops, "ops");
            this.pair = d.pair();
            HeadsKernels.checkLoraShape(pair.inDim(), pair.outDim(), pair.rank(), pair.experts());
            try {
                this.a = upload(ops, pair.target() + LoraAdapterLoader.SUFFIX_A, d.a(), owned, true);
                this.b = upload(ops, pair.target() + LoraAdapterLoader.SUFFIX_B, d.scaledB(), owned, false);
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
        public void rescale(float userScale) {
            delta.rescale(userScale);
            rewrite(ops, b, delta.scaledB());
        }

        @Override
        public void close() {
            SwitchedLoraExtra.closeAll(owned);
        }
    }

    /** A token-embedding delta over looked-up rows. */
    public static final class Embedding implements GpuProjectionDeltas.EmbeddingDelta, AutoCloseable, Rescalable {
        private final LoraAdapter.Delta delta;
        private final GpuOps ops;
        private final LoraAdapter.Pair pair;
        private final GpuBuffer b;
        private final GpuBuffer zeroSlots;
        private final int ubatch;
        private final List<GpuBuffer> owned = new ArrayList<>();
        private final GpuBuffer a;

        Embedding(GpuOps ops, LoraAdapter.Delta d) {
            requireKind(d, LoraAdapter.Target.EMBEDDING);
            this.delta = d;
            this.ops = Objects.requireNonNull(ops, "ops");
            this.pair = d.pair();
            this.ubatch = ops.scratch().ubatch();
            HeadsKernels.checkLoraShape(pair.rank(), pair.outDim(), pair.rank(), 1);
            if ((long) pair.inDim() * pair.rank() > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("LoRA table of " + pair.target() + " (" + pair.inDim() + " x "
                        + pair.rank() + ") exceeds the 32-bit index range");
            }
            try {
                this.a = upload(ops, pair.target() + LoraAdapterLoader.SUFFIX_A, d.a(), owned, true);
                this.b = upload(ops, pair.target() + LoraAdapterLoader.SUFFIX_B, d.scaledB(), owned, false);
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
        public void rescale(float userScale) {
            delta.rescale(userScale);
            rewrite(ops, b, delta.scaledB());
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

    private static GpuBuffer upload(GpuOps ops, String what, float[] values, List<GpuBuffer> owned, boolean readOnly) {
        GpuBuffer buf = SwitchedLoraExtra.allocateFloats(ops, what, values.length, readOnly);
        owned.add(buf);
        rewrite(ops, buf, values);
        return buf;
    }

    /** Writes {@code values} into {@code buf} on the queue of {@code ops}, in place (the rescale upload). */
    private static void rewrite(GpuOps ops, GpuBuffer buf, float[] values) {
        GpuEvent e = ops.queue().enqueueWriteBuffer(buf, 0L, (long) values.length * Float.BYTES, values);
        if (e != null) {
            e.close();
        }
    }
}
