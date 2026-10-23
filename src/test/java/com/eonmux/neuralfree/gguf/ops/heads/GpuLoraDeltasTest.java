package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.ops.flow.FlowKernels;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Dispatch;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Device LoRA deltas ({@link GpuLoraDeltas}) on a fake backend whose {@value HeadsKernels#SWITCHED_LORA_ADD},
 * {@value FlowKernels#GATHER_ROWS} and {@value GpuOps#COPY2D} dispatches are emulated on the host from their kernel
 * contracts (heads.cl, flow.cl): the dispatch traces of the projection, expert and embedding deltas, and their
 * results against the scalar reference of {@code build_lora_mm}, {@code build_lora_mm_id} and the embedding-row delta
 * (llama-graph.cpp:1648-1664, 1734-1746, 2694-2709).
 */
public class GpuLoraDeltasTest {

    private static final float EPS = 1e-5f;
    private static final int UBATCH = 8;
    private static final String Q = "blk.0.attn_q.weight";
    private static final String UP_EXPS = "blk.0.ffn_up_exps.weight";
    private static final String EMBD = "token_embd.weight";

    private FakeComputeBackend backend;
    private GpuOps ops;

    @Before
    public void open() {
        backend = FakeComputeBackend.builder().strictLlmArgs(true)
                .emulate(HeadsKernels.SWITCHED_LORA_ADD, GpuLoraDeltasTest::switchedLoraAdd)
                .emulate(FlowKernels.GATHER_ROWS, GpuLoraDeltasTest::gatherRows)
                .emulate(GpuOps.COPY2D, GpuLoraDeltasTest::copy2d)
                .build();
        GpuScratch scratch = GpuScratch.builder(backend, UBATCH).slot(ScratchSlots.X, (long) UBATCH * 16).build();
        GpuPlaceholders placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
    }

    @After
    public void close() {
        if (ops != null) {
            ops.close();
        }
        if (backend != null) {
            backend.close();
        }
    }

    // ------------------------------------------------------------------ emulated kernels

    /** LLM_SwitchedLoraAdd (heads.cl): stage 0 {@code tmp = A[s] x}, stage 1 {@code y += B[s] tmp}. */
    private static void switchedLoraAdd(Dispatch d) {
        int n = d.intArg(0);
        int in = d.intArg(1);
        int out = d.intArg(2);
        int rank = d.intArg(3);
        int stage = d.intArg(4);
        int nSlots = d.intArg(5);
        int xStride = d.intArg(6);
        int xOff = d.intArg(7);
        int yStride = d.intArg(8);
        int yOff = d.intArg(9);
        int tmpOff = d.intArg(10);
        FakeGpuBuffer slots = d.bufferArg(11);
        FakeGpuBuffer a = d.bufferArg(12);
        FakeGpuBuffer b = d.bufferArg(13);
        FakeGpuBuffer x = d.bufferArg(14);
        FakeGpuBuffer tmp = d.bufferArg(15);
        FakeGpuBuffer y = d.bufferArg(16);
        for (int t = 0; t < n; t++) {
            int s = Math.max(0, Math.min(nSlots - 1, slots.getInt(t)));
            if (stage == HeadsKernels.LORA_STAGE_DOWN) {
                for (int r = 0; r < rank; r++) {
                    float acc = 0f;
                    for (int i = 0; i < in; i++) {
                        acc += a.getFloat(((long) s * rank + r) * in + i) * x.getFloat(xOff + (long) t * xStride + i);
                    }
                    tmp.putFloat(tmpOff + (long) t * rank + r, acc);
                }
            } else {
                for (int o = 0; o < out; o++) {
                    long at = yOff + (long) t * yStride + o;
                    float acc = y.getFloat(at);
                    for (int r = 0; r < rank; r++) {
                        acc += b.getFloat(((long) s * out + o) * rank + r) * tmp.getFloat(tmpOff + (long) t * rank + r);
                    }
                    y.putFloat(at, acc);
                }
            }
        }
    }

    /** LLM_GatherRows (flow.cl). */
    private static void gatherRows(Dispatch d) {
        int nRows = d.intArg(0);
        int rowLen = d.intArg(1);
        int srcStride = d.intArg(2);
        int dstStride = d.intArg(3);
        int srcOff = d.intArg(4);
        int dstOff = d.intArg(5);
        FakeGpuBuffer rows = d.bufferArg(6);
        FakeGpuBuffer src = d.bufferArg(7);
        FakeGpuBuffer dst = d.bufferArg(8);
        for (int r = 0; r < nRows; r++) {
            int s = rows.getInt(r);
            for (int c = 0; c < rowLen; c++) {
                dst.putFloat(dstOff + (long) r * dstStride + c,
                        s >= 0 ? src.getFloat(srcOff + (long) s * srcStride + c) : 0f);
            }
        }
    }

    /** LLM_Copy2D_F32: {@code (rows, cols, src_stride, src_off, dst_stride, dst_off, src, dst)}. */
    private static void copy2d(Dispatch d) {
        int rows = d.intArg(0);
        int cols = d.intArg(1);
        int srcStride = d.intArg(2);
        int srcOff = d.intArg(3);
        int dstStride = d.intArg(4);
        int dstOff = d.intArg(5);
        FakeGpuBuffer src = d.bufferArg(6);
        FakeGpuBuffer dst = d.bufferArg(7);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                dst.putFloat(dstOff + (long) r * dstStride + c, src.getFloat(srcOff + (long) r * srcStride + c));
            }
        }
    }

    // ------------------------------------------------------------------ helpers

    private FakeGpuBuffer floats(float[] values) {
        FakeGpuBuffer buf = (FakeGpuBuffer) backend.allocateBuffer((long) values.length * Float.BYTES);
        for (int i = 0; i < values.length; i++) {
            buf.putFloat(i, values[i]);
        }
        return buf;
    }

    private FakeGpuBuffer ints(int[] values) {
        FakeGpuBuffer buf = (FakeGpuBuffer) backend.allocateBuffer((long) values.length * Integer.BYTES);
        for (int i = 0; i < values.length; i++) {
            buf.putInt(i, values[i]);
        }
        return buf;
    }

    private List<String> dispatched() {
        List<String> names = new ArrayList<>();
        for (Dispatch d : backend.dispatches()) {
            names.add(d.kernelName());
        }
        return names;
    }

    // ------------------------------------------------------------------ tests

    @Test
    public void projectionDeltaAddsOnStridedRowsInTwoStages() throws Exception {
        int in = 5;
        int out = 3;
        LoraAdapter.Pair p = LoraFixtures.projection(Q, in, out, 2, 1);
        LoraAdapter a = LoraFixtures.adapter("q.gguf", 4.0f, p);
        int n = 3;
        int xOff = 2;
        int xStride = in + 1;
        int yOff = 1;
        int yStride = out + 2;
        float[] x = LoraFixtures.random(2, xOff + n * xStride);
        float[] y0 = LoraFixtures.random(3, yOff + n * yStride);
        float[] expected = y0.clone();
        for (int t = 0; t < n; t++) {
            LoraFixtures.project(p, a.scale(Q, 0.5f), 0, x, xOff + t * xStride, expected, yOff + t * yStride);
        }
        FakeGpuBuffer xb = floats(x);
        FakeGpuBuffer yb = floats(y0);
        try (AutoCloseable delta = GpuLoraDeltas.of(ops, a.delta(Q, 0.5f))) {
            backend.clearLog();
            ((GpuProjectionDeltas.Delta) delta).add(n, xb, xOff, xStride, yb, yOff, yStride);
            assertEquals(Arrays.asList(HeadsKernels.SWITCHED_LORA_ADD, HeadsKernels.SWITCHED_LORA_ADD), dispatched());
            assertArrayEquals(expected, yb.getFloats(0, expected.length), EPS);
        }
    }

    @Test
    public void expertDeltaOverTokenRowsGathersThePairRowsAndUsesTheExpertIdsAsSlots() throws Exception {
        int in = 4;
        int out = 3;
        int k = 2;
        int nTokens = 3;
        int pairs = nTokens * k;
        LoraAdapter.Pair p = LoraFixtures.experts(UP_EXPS, in, out, 2, 3, 11);
        LoraAdapter a = LoraFixtures.adapter("moe.gguf", 2.0f, p);
        float s = a.scale(UP_EXPS, 1.0f);
        int[] ids = {2, 0, 1, 2, 0, 1};
        int xOff = 1;
        float[] x = LoraFixtures.random(12, xOff + nTokens * in);
        float[] y0 = LoraFixtures.random(13, pairs * out);
        float[] expected = y0.clone();
        for (int q = 0; q < pairs; q++) {
            LoraFixtures.project(p, s, ids[q], x, xOff + (q / k) * in, expected, q * out);
        }
        FakeGpuBuffer xb = floats(x);
        FakeGpuBuffer yb = floats(y0);
        try (AutoCloseable delta = GpuLoraDeltas.of(ops, a.delta(UP_EXPS, 1.0f))) {
            backend.clearLog();
            ((GpuProjectionDeltas.ExpertDelta) delta).add(nTokens, k, ints(ids), xb, xOff, true, yb, 0);
            assertEquals(Arrays.asList(GpuOps.COPY2D, GpuOps.COPY2D, HeadsKernels.SWITCHED_LORA_ADD,
                    HeadsKernels.SWITCHED_LORA_ADD), dispatched());
            assertEquals(pairs, backend.dispatchesOf(HeadsKernels.SWITCHED_LORA_ADD).get(0).intArg(0));
            assertEquals(3, backend.dispatchesOf(HeadsKernels.SWITCHED_LORA_ADD).get(0).intArg(5));
            assertArrayEquals(expected, yb.getFloats(0, expected.length), EPS);
        }
    }

    @Test
    public void expertDeltaOfOneTokenBroadcastsItsRowWithAZeroStride() throws Exception {
        int in = 4;
        int out = 3;
        int k = 3;
        LoraAdapter.Pair p = LoraFixtures.experts(UP_EXPS, in, out, 2, 4, 21);
        LoraAdapter a = LoraFixtures.adapter("moe.gguf", 0.0f, p);
        int[] ids = {3, 1, 2};
        float[] x = LoraFixtures.random(22, in);
        float[] y0 = LoraFixtures.random(23, k * out);
        float[] expected = y0.clone();
        for (int q = 0; q < k; q++) {
            LoraFixtures.project(p, 0.5f, ids[q], x, 0, expected, q * out);
        }
        FakeGpuBuffer yb = floats(y0);
        try (AutoCloseable delta = GpuLoraDeltas.of(ops, a.delta(UP_EXPS, 0.5f))) {
            backend.clearLog();
            ((GpuProjectionDeltas.ExpertDelta) delta).add(1, k, ints(ids), floats(x), 0, true, yb, 0);
            assertEquals(Arrays.asList(HeadsKernels.SWITCHED_LORA_ADD, HeadsKernels.SWITCHED_LORA_ADD), dispatched());
            assertEquals(0, backend.dispatchesOf(HeadsKernels.SWITCHED_LORA_ADD).get(0).intArg(6));
            assertArrayEquals(expected, yb.getFloats(0, expected.length), EPS);
        }
    }

    @Test
    public void expertDeltaOverPairRowsReadsOneRowPerPair() throws Exception {
        int in = 4;
        int out = 3;
        int k = 2;
        int pairs = 4;
        LoraAdapter.Pair p = LoraFixtures.experts(UP_EXPS, in, out, 3, 3, 31);
        LoraAdapter a = LoraFixtures.adapter("moe.gguf", 3.0f, p);
        float s = a.scale(UP_EXPS, 1.0f);
        int[] ids = {1, 2, 0, 1};
        float[] x = LoraFixtures.random(32, pairs * in);
        float[] y0 = LoraFixtures.random(33, pairs * out);
        float[] expected = y0.clone();
        for (int q = 0; q < pairs; q++) {
            LoraFixtures.project(p, s, ids[q], x, q * in, expected, q * out);
        }
        FakeGpuBuffer yb = floats(y0);
        try (AutoCloseable delta = GpuLoraDeltas.of(ops, a.delta(UP_EXPS, 1.0f))) {
            backend.clearLog();
            ((GpuProjectionDeltas.ExpertDelta) delta).add(pairs / k, k, ints(ids), floats(x), 0, false, yb, 0);
            assertEquals(Arrays.asList(HeadsKernels.SWITCHED_LORA_ADD, HeadsKernels.SWITCHED_LORA_ADD), dispatched());
            assertArrayEquals(expected, yb.getFloats(0, expected.length), EPS);
        }
    }

    @Test
    public void embeddingDeltaGathersTheARowsAndAddsTheUpStage() throws Exception {
        int nVocab = 7;
        int width = 4;
        LoraAdapter.Pair p = LoraFixtures.embedding(EMBD, nVocab, width, 2, 41);
        LoraAdapter a = LoraFixtures.adapter("embd.gguf", 1.0f, p);
        float s = a.scale(EMBD, 2.0f);
        int[] ids = {6, 0, 3};
        int yOff = 2;
        int yStride = width + 1;
        float[] y0 = LoraFixtures.random(42, yOff + ids.length * yStride);
        float[] expected = y0.clone();
        for (int t = 0; t < ids.length; t++) {
            LoraFixtures.embed(p, s, ids[t], expected, yOff + t * yStride);
        }
        FakeGpuBuffer yb = floats(y0);
        try (AutoCloseable delta = GpuLoraDeltas.of(ops, a.delta(EMBD, 2.0f))) {
            backend.clearLog();
            ((GpuProjectionDeltas.EmbeddingDelta) delta).add(ids.length, ints(ids), yb, yOff, yStride);
            assertEquals(Arrays.asList(FlowKernels.GATHER_ROWS, HeadsKernels.SWITCHED_LORA_ADD), dispatched());
            assertEquals(HeadsKernels.LORA_STAGE_UP, backend.dispatchesOf(HeadsKernels.SWITCHED_LORA_ADD).get(0)
                    .intArg(HeadsKernels.ARG_LORA_STAGE));
            assertArrayEquals(expected, yb.getFloats(0, expected.length), EPS);
        }
    }
}
