package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.gpu.block.GpuProjectionDeltas;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link LoraAttachment#gpu(List, List)} over the stages of a layer split: every pair is registered on, and uploaded
 * to, the device whose blocks serve its base tensor; the device bytes of one stage count only its pairs; a pair no
 * device serves or two devices serve fails the attachment by name with nothing left registered; a rescale rewrites
 * each pair on its own device; closing removes the deltas of every device.
 */
public class LoraAttachmentSplitTest {

    private static final int UBATCH = 4;
    private static final int N_EMBD = 6;
    private static final int N_VOCAB = 9;
    private static final int RANK = 2;
    private static final String OUTPUT = "output.weight";
    private static final String Q0 = "blk.0.attn_q.weight";
    private static final String Q1 = "blk.1.attn_q.weight";

    private final List<FakeComputeBackend> backends = new ArrayList<>();
    private final List<GpuScratch> scratches = new ArrayList<>();
    private final List<GpuOps> ops = new ArrayList<>();
    private GpuProjectionDeltas first;
    private GpuProjectionDeltas second;

    @Before
    public void open() {
        for (int d = 0; d < 2; d++) {
            FakeComputeBackend backend = FakeComputeBackend.builder().strictLlmArgs(true).build();
            GpuScratch scratch = GpuScratch.builder(backend, UBATCH).slot(ScratchSlots.X, (long) UBATCH * 16)
                    .build();
            // the deltas share one scratch temporary per device; growing it first leaves only their own uploads
            scratch.temp(SwitchedLoraExtra.GpuDelta.TEMP_TMP, (long) UBATCH * RANK);
            GpuPlaceholders placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
            backends.add(backend);
            scratches.add(scratch);
            ops.add(new GpuOps(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders));
        }
        // stage 0 runs layer 0 and the output head, stage 1 runs layer 1
        first = new GpuProjectionDeltas();
        first.serveModel(OUTPUT, N_EMBD, N_VOCAB, null);
        first.serveModel(Q0, N_EMBD, N_EMBD, null);
        second = new GpuProjectionDeltas();
        second.serveModel(Q1, N_EMBD, N_EMBD, null);
    }

    @After
    public void close() {
        for (GpuOps o : ops) {
            o.close();
        }
        for (GpuScratch sc : scratches) {
            sc.close();
        }
        for (FakeComputeBackend b : backends) {
            b.close();
        }
    }

    private List<LoraAttachment.GpuDevice> devices() {
        return Arrays.asList(new LoraAttachment.GpuDevice(ops.get(0), first),
                new LoraAttachment.GpuDevice(ops.get(1), second));
    }

    private static List<LoraAttachment.Binding> bindings(LoraAdapter.Pair... pairs) {
        return Collections.singletonList(new LoraAttachment.Binding(LoraFixtures.adapter("a.gguf", 4f, pairs), 1f));
    }

    private static LoraAdapter.Pair q(String target, long seed) {
        return LoraFixtures.projection(target, N_EMBD, N_EMBD, RANK, seed);
    }

    @Test
    public void everyPairGoesToTheDeviceThatServesItsTensor() {
        int before0 = backends.get(0).liveBuffers().size();
        int before1 = backends.get(1).liveBuffers().size();
        List<LoraAttachment.Binding> b = bindings(LoraFixtures.projection(OUTPUT, N_EMBD, N_VOCAB, RANK, 1),
                q(Q0, 3), q(Q1, 5));
        LoraAttachment att = LoraAttachment.gpu(b, devices());
        assertEquals(3, att.deltaCount());
        assertFalse(first.served(OUTPUT).isEmpty());
        assertFalse(first.served(Q0).isEmpty());
        assertFalse(second.served(Q1).isEmpty());
        int grown0 = backends.get(0).liveBuffers().size() - before0;
        int grown1 = backends.get(1).liveBuffers().size() - before1;
        assertTrue("the second device holds the uploads of its pair", grown1 > 0);
        assertEquals("the first device holds the uploads of its two pairs", 2 * grown1, grown0);

        att.close();
        assertTrue(first.served(OUTPUT).isEmpty());
        assertTrue(first.served(Q0).isEmpty());
        assertTrue(second.served(Q1).isEmpty());
        assertEquals(before0, backends.get(0).liveBuffers().size());
        assertEquals(before1, backends.get(1).liveBuffers().size());
    }

    @Test
    public void aDeviceServingNoPairReceivesNoUpload() {
        int before0 = backends.get(0).liveBuffers().size();
        LoraAttachment att = LoraAttachment.gpu(bindings(q(Q1, 5)), devices());
        assertEquals(1, att.deltaCount());
        assertEquals(before0, backends.get(0).liveBuffers().size());
        assertTrue(first.served(Q0).isEmpty());
        assertFalse(second.served(Q1).isEmpty());
        att.close();
        assertTrue(second.served(Q1).isEmpty());
    }

    @Test
    public void aRescaleRewritesEveryPairOnItsOwnDevice() {
        List<LoraAttachment.Binding> b = bindings(q(Q0, 3), q(Q1, 5));
        LoraAttachment att = LoraAttachment.gpu(b, devices());
        backends.get(0).clearLog();
        backends.get(1).clearLog();
        att.rescale(b.get(0).adapter(), 2f);
        for (int d = 0; d < 2; d++) {
            long writes = backends.get(d).ops().stream()
                    .filter(op -> op.kind() == FakeComputeBackend.OpKind.WRITE).count();
            assertEquals("device " + d + " rewrites the scaled B of its one pair", 1L, writes);
        }
        att.close();
    }

    @Test
    public void deviceBytesOfAStageCountItsPairsOnly() {
        LoraAdapter.Pair p1 = q(Q1, 5);
        List<LoraAttachment.Binding> b = bindings(q(Q0, 3), p1);
        long stage1 = LoraAttachment.gpuDeviceBytes(b, UBATCH, Q1::equals);
        assertEquals((long) (p1.a.length + p1.b.length + UBATCH) * Float.BYTES, stage1);
        assertEquals(LoraAttachment.gpuDeviceBytes(b, UBATCH),
                stage1 + LoraAttachment.gpuDeviceBytes(b, UBATCH, Q0::equals));
    }

    @Test
    public void aPairNoDeviceServesFailsTheAttachmentByName() {
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> LoraAttachment.gpu(bindings(q(Q0, 3), q("blk.2.attn_q.weight", 7)), devices()));
        assertTrue(e.getMessage(), e.getMessage().contains("blk.2.attn_q.weight"));
        assertTrue("nothing stays registered", first.served(Q0).isEmpty());
    }

    @Test
    public void aPairTwoDevicesServeIsRefused() {
        second.serveModel(Q0, N_EMBD, N_EMBD, null);
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> LoraAttachment.gpu(bindings(q(Q0, 3)), devices()));
        assertTrue(e.getMessage(), e.getMessage().contains("both serve"));
        assertTrue(first.served(Q0).isEmpty());
        assertTrue(second.served(Q0).isEmpty());
    }

    @Test
    public void anAttachmentNeedsADevice() {
        assertThrows(IllegalArgumentException.class,
                () -> LoraAttachment.gpu(bindings(q(Q0, 3)), Collections.<LoraAttachment.GpuDevice>emptyList()));
    }
}
