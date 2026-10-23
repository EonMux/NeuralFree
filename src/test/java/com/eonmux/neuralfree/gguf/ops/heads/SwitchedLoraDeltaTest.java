package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Dispatch;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.OpKind;
import org.junit.After;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;

/**
 * Dispatch traces of the switched and generic LoRA deltas against the host-only fake backend with strict LLM argument
 * rules, plus the host router and the CPU LoRA delta:
 * <ul>
 *   <li>graniteswitch: {@code LLM_SwitchedLoraAdd} stage 0 then stage 1 on one queue, the shared slot upload written
 *       once per step, and the router's closed form;</li>
 *   <li>generic LoRA: one-slot delta with zero slots, and the CPU delta's {@code scale = alpha / rank}.</li>
 * </ul>
 */
public class SwitchedLoraDeltaTest {

    private static final int N_EMBD = 64;
    private static final int UBATCH = 8;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuOps ops;

    private void open() {
        backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
        scratch = GpuScratch.builder(backend, UBATCH).slot(ScratchSlots.X, (long) UBATCH * N_EMBD).build();
        GpuPlaceholders placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders);
    }

    @After
    public void tearDown() {
        if (ops != null) {
            ops.close();
        }
        if (backend != null) {
            backend.close();
        }
    }

    private GpuBuffer floats(long n) {
        return backend.allocateBuffer(n * Float.BYTES);
    }

    private static List<String> names(List<Dispatch> ds) {
        List<String> out = new ArrayList<>();
        for (Dispatch d : ds) {
            out.add(d.kernelName());
        }
        return out;
    }

    private static int[] ints(Dispatch d, int from, int to) {
        int[] out = new int[to - from];
        for (int i = from; i < to; i++) {
            out[i - from] = d.intArg(i);
        }
        return out;
    }

    // ================================================================== switched LoRA

    @Test
    public void switchedLoraEnqueuesStageZeroThenStageOne() {
        open();
        int out = 16;
        int rank = 4;
        int nSlots = 3;
        GpuBuffer a = floats((long) N_EMBD * rank * nSlots);
        GpuBuffer b = floats((long) rank * out * nSlots);
        GpuBuffer slots = backend.allocateBuffer((long) UBATCH * Integer.BYTES);
        GpuBuffer x = floats((long) UBATCH * N_EMBD);
        GpuBuffer y = floats((long) UBATCH * out);
        try (SwitchedLoraExtra.GpuDelta delta = new SwitchedLoraExtra.GpuDelta(ops, a, b, N_EMBD, out, rank,
                nSlots)) {
            backend.clearLog();
            delta.add(3, slots, x, 0, N_EMBD, y, 0, out);

            List<Dispatch> ds = backend.dispatches();
            assertEquals(Arrays.asList(HeadsKernels.SWITCHED_LORA_ADD, HeadsKernels.SWITCHED_LORA_ADD), names(ds));
            ScratchSlot tmp = scratch.temp(SwitchedLoraExtra.GpuDelta.TEMP_TMP, (long) UBATCH * rank);
            for (int stage = 0; stage < 2; stage++) {
                Dispatch d = ds.get(stage);
                assertArrayEquals(new int[]{3, N_EMBD, out, rank, stage, nSlots, N_EMBD, 0, out, 0, tmp.offset()},
                        ints(d, 0, 11));
                assertSame(slots, d.bufferArg(11));
                assertSame(a, d.bufferArg(12));
                assertSame(b, d.bufferArg(13));
                assertSame(x, d.bufferArg(14));
                assertSame(tmp.buffer(), d.bufferArg(15));
                assertSame(y, d.bufferArg(16));
                assertArrayEquals(new long[]{HeadsKernels.loraGlobal(stage, 3, rank, out)}, d.global());
                assertArrayEquals(new long[]{HeadsKernels.LORA_LOCAL}, d.local());
                assertEquals(FakeComputeBackend.COMPUTE_QUEUE, d.queue());
            }
        }
    }

    @Test
    public void switchedLoraRebindsTheSharedTemporaryAfterALargerRank() {
        open();
        int out = 8;
        GpuBuffer slots = backend.allocateBuffer((long) UBATCH * Integer.BYTES);
        GpuBuffer x = floats((long) UBATCH * N_EMBD);
        GpuBuffer y = floats((long) UBATCH * out);
        try (SwitchedLoraExtra.GpuDelta small = new SwitchedLoraExtra.GpuDelta(ops, floats(N_EMBD * 2L),
                floats(2L * out), N_EMBD, out, 2, 1);
             SwitchedLoraExtra.GpuDelta large = new SwitchedLoraExtra.GpuDelta(ops, floats(N_EMBD * 32L),
                     floats(32L * out), N_EMBD, out, 32, 1)) {
            GpuBuffer grown = scratch.temp(SwitchedLoraExtra.GpuDelta.TEMP_TMP, (long) UBATCH * 32).buffer();
            backend.clearLog();
            small.add(1, slots, x, 0, N_EMBD, y, 0, out);

            for (Dispatch d : backend.dispatches()) {
                assertSame(grown, d.bufferArg(15));
            }
            assertEquals(2, backend.dispatches().size());
            large.add(1, slots, x, 0, N_EMBD, y, 0, out);
        }
    }

    @Test
    public void slotUploadWritesOncePerStep() {
        open();
        SwitchedLoraExtra.Router router = SwitchedLoraExtra.Router.of(4.0f, new int[]{100, 101}, new int[]{5, 6});
        SwitchedLoraExtra.SlotUpload upload = new SwitchedLoraExtra.SlotUpload();
        upload.reserve(UBATCH);
        ScratchSlot slots = scratch.temp(SwitchedLoraExtra.TEMP_SLOTS, UBATCH);
        router.accept(0, new int[]{1, 100, 2});
        backend.clearLog();
        upload.write(ops, router, 0, 3, slots);
        upload.write(ops, router, 0, 3, slots);
        assertEquals(1, backend.ops().stream().filter(o -> o.kind() == OpKind.WRITE).count());

        router.accept(3, new int[]{7});
        upload.write(ops, router, 3, 1, slots);
        assertEquals(2, backend.ops().stream().filter(o -> o.kind() == OpKind.WRITE).count());
    }

    @Test
    public void routerFollowsTheAttentionClosedForm() {
        SwitchedLoraExtra.Router router = SwitchedLoraExtra.Router.of(4.0f, new int[]{100, 101}, new int[]{5, 6});
        assertEquals(3, router.nSlots());
        assertArrayEquals(new int[]{1, 5, 6, 2}, router.substitute(new int[]{1, 100, 101, 2}));

        router.accept(0, new int[]{1, 100, 2, 3});
        int[] slots = new int[4];
        router.slots(0, 4, slots, 0);
        assertArrayEquals(new int[]{0, 1, 1, 1}, slots);

        router.accept(0, new int[]{101, 100});
        assertEquals(2, router.length());
        assertEquals(2, router.slot(0));
        assertEquals(2, router.slot(1));

        router.reset();
        router.accept(0, new int[]{100});
        router.shift(0, 1);
        assertEquals(0, router.length());
        assertThrows(IllegalArgumentException.class, () -> router.accept(2, new int[]{1}));
    }

    // ================================================================== generic LoRA

    private static LoraAdapter adapter(String target, int in, int out, int rank, float alpha) {
        float[] a = new float[rank * in];
        float[] b = new float[out * rank];
        for (int i = 0; i < a.length; i++) {
            a[i] = i + 1;
        }
        for (int i = 0; i < b.length; i++) {
            b[i] = i + 3;
        }
        Map<String, LoraAdapter.Pair> pairs = new HashMap<>();
        pairs.put(target, new LoraAdapter.Pair(target, false, in, out, rank, a, b));
        return new LoraAdapter("test.gguf", "llama", alpha, null, null, null, pairs, Collections.emptyMap());
    }

    @Test
    public void cpuDeltaScalesByAlphaOverRank() {
        LoraAdapter lora = adapter("blk.0.attn_q.weight", 2, 2, 1, 2.0f);
        assertEquals(2.0f, lora.scale("blk.0.attn_q.weight", 1.0f), 0.0f);
        LoraAdapter.Delta delta = lora.delta("blk.0.attn_q.weight", 1.0f);
        float[] x = {1.0f, 1.0f};
        float[] y = {1.0f, 1.0f};
        delta.add(1, x, 0, 2, new float[1], 0, y, 0, 2);
        // A x = 1 + 2 = 3; scale * B * 3 = 2 * {3, 4} * 3.
        assertArrayEquals(new float[]{19.0f, 25.0f}, y, 1e-6f);
    }

    @Test
    public void adapterDeltaUsesOneSlotOfZeros() {
        open();
        int out = 8;
        int rank = 2;
        LoraAdapter lora = adapter("blk.0.attn_q.weight", N_EMBD, out, rank, 0.0f);
        GpuBuffer x = floats((long) UBATCH * N_EMBD);
        GpuBuffer y = floats((long) UBATCH * out);
        scratch.temp(SwitchedLoraExtra.GpuDelta.TEMP_TMP, (long) UBATCH * rank);
        int liveBefore = backend.liveBuffers().size();
        SwitchedLoraExtra.GpuDelta delta = SwitchedLoraExtra.GpuDelta.ofAdapter(ops,
                lora.delta("blk.0.attn_q.weight", 0.5f));
        try {
            assertEquals(1, backend.ops().stream().filter(o -> o.kind() == OpKind.FILL).count());
            backend.clearLog();
            delta.add(2, x, 0, N_EMBD, y, 0, out);

            List<Dispatch> ds = backend.dispatches();
            assertEquals(Arrays.asList(HeadsKernels.SWITCHED_LORA_ADD, HeadsKernels.SWITCHED_LORA_ADD), names(ds));
            for (Dispatch d : ds) {
                assertEquals(1, d.intArg(5));
                assertSame(ds.get(0).bufferArg(11), d.bufferArg(11));
                assertSame(ds.get(0).bufferArg(12), d.bufferArg(12));
                assertSame(ds.get(0).bufferArg(13), d.bufferArg(13));
            }
        } finally {
            delta.close();
        }
        assertEquals("the A, scaled B and zero-slot uploads are released with the delta", liveBefore,
                backend.liveBuffers().size());
    }

    @Test
    public void adaptersOfOneTargetKeepSeparateUploads() {
        open();
        LoraAdapter lora = adapter("blk.0.attn_q.weight", N_EMBD, 8, 2, 0.0f);
        try (SwitchedLoraExtra.GpuDelta half = SwitchedLoraExtra.GpuDelta.ofAdapter(ops,
                lora.delta("blk.0.attn_q.weight", 0.5f));
             SwitchedLoraExtra.GpuDelta full = SwitchedLoraExtra.GpuDelta.ofAdapter(ops,
                     lora.delta("blk.0.attn_q.weight", 1.0f))) {
            GpuBuffer x = floats((long) UBATCH * N_EMBD);
            GpuBuffer y = floats((long) UBATCH * 8);
            backend.clearLog();
            half.add(1, x, 0, N_EMBD, y, 0, 8);
            full.add(1, x, 0, N_EMBD, y, 0, 8);

            List<Dispatch> ds = backend.dispatches();
            assertNotSame(ds.get(0).bufferArg(13), ds.get(2).bufferArg(13));
        }
    }
}
