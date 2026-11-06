package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.pipeline.SplitSpecs;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;

/**
 * The pooled FFN block: its input is copied to the owner stage, the wrapped block runs there (bound to the owner's
 * context, not the layer's), and its output comes back into the layer's branch slot. Two fake backends stand in for
 * the two GPUs; the delegate copies the rows it is given, so the test sees the exact values that crossed.
 */
public class PooledFfnBlockTest {

    private static final int UBATCH = 2;
    private static final int WIDTH = 64;

    private final List<AutoCloseable> closeables = new ArrayList<>();
    private PooledFfnBlock wrapper;
    private FakeComputeBackend local;

    @After
    public void tearDown() throws Exception {
        for (int i = closeables.size() - 1; i >= 0; i--) {
            closeables.get(i).close();
        }
    }

    @Test
    public void theBlockRunsOnTheOwnerStageAndItsRowsCrossBothWays() throws Exception {
        GpuForwardContext localCtx = context(false);
        GpuForwardContext ownerCtx = context(true);
        LayerSpec layer = SplitSpecs.trunk(2, MemoryKind.NONE).layers().get(1);
        RecordingFfn delegate = new RecordingFfn();
        wrapper = new PooledFfnBlock(delegate, 1, ownerCtx, WIDTH);
        wrapper.bind(localCtx, layer);

        try (GpuBuffer in = local.allocateBuffer((long) UBATCH * WIDTH * Float.BYTES, true);
             GpuBuffer out = local.allocateBuffer((long) UBATCH * WIDTH * Float.BYTES, true)) {
            ScratchSlot inSlot = ScratchSlot.of("test.pool.in", in, 0, (long) UBATCH * WIDTH);
            ScratchSlot outSlot = ScratchSlot.of("test.pool.out", out, 0, (long) UBATCH * WIDTH);
            float[] values = new float[UBATCH * WIDTH];
            for (int i = 0; i < values.length; i++) {
                values[i] = (float) Math.sin(i * 0.37 + 1.0);
            }
            write(local.computeQueue(), in, values);

            wrapper.forward(UBATCH, inSlot, outSlot);

            assertEquals("the wrapped block ran on the owner stage's context", ownerCtx, delegate.boundTo());
            assertArrayEquals("the owner's rows return into the layer's branch", values, read(local.computeQueue(),
                    out, values.length), 0f);
        }
        assertFalse("a decode step of a pooled block reads rows back", wrapper.capturableDecode());
        assertThrows(IllegalStateException.class, () -> wrapper.bind(localCtx, layer));
    }

    // ------------------------------------------------------------------ fixtures

    /** A context on a fake backend; {@code ownerStage} only names the backend in the diagnostics. */
    private GpuForwardContext context(boolean ownerStage) {
        FakeComputeBackend backend = FakeComputeBackend.builder().strictLlmArgs(false).build();
        closeables.add(backend);
        if (!ownerStage) {
            local = backend;
        }
        ModelSpec spec = SplitSpecs.trunk(2, MemoryKind.NONE);
        ComputeQueue queue = backend.computeQueue();
        LlmKernels kernels = LlmKernels.open(backend);
        GpuScratch scratch = GpuScratch.forModel(backend, spec, UBATCH);
        closeables.add(scratch);
        GpuPlaceholders placeholders = GpuPlaceholders.allocate(backend, queue);
        closeables.add(placeholders);
        GpuOps ops = new GpuOps(queue, kernels, scratch, placeholders);
        closeables.add(ops);
        return GpuForwardContext.builder(kernels, ops, name -> null, spec).build();
    }

    /** Copies the rows it is given into its output on whichever context it was bound to. */
    private static final class RecordingFfn implements GpuFfnBlock {
        private GpuForwardContext ctx;

        @Override
        public void bind(GpuForwardContext ctx, LayerSpec layer) {
            this.ctx = ctx;
        }

        @Override
        public void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
            int floats = nTokens * WIDTH;
            float[] rows = new float[floats];
            GpuEvent read = ctx.queue().enqueueReadBuffer(in.buffer(), 4L * in.offset(),
                    4L * floats, rows);
            await(read);
            ctx.queue().enqueueWriteBuffer(out.buffer(), 4L * out.offset(), 4L * floats, rows);
            ctx.queue().finish();
        }

        private static void await(GpuEvent event) {
            if (event == null) {
                return;
            }
            try {
                event.awaitCompletion();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while waiting for the recorded rows", e);
            } finally {
                event.close();
            }
        }

        GpuForwardContext boundTo() {
            return ctx;
        }
    }

    private static void write(ComputeQueue queue, GpuBuffer buffer, float[] values) {
        try (GpuEvent e = queue.enqueueWriteBuffer(buffer, 0L, values.length * (long) Float.BYTES, values)) {
            queue.finish();
        }
    }

    private static float[] read(ComputeQueue queue, GpuBuffer buffer, int floats) {
        float[] out = new float[floats];
        try (GpuEvent e = queue.enqueueReadBuffer(buffer, 0L, floats * (long) Float.BYTES, out)) {
            queue.finish();
        }
        return out;
    }
}
