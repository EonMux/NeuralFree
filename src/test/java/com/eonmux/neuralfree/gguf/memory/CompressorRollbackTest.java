package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan.CompressedKvPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.memory.cpu.RecurrentStateAccessor;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuCompressedKv;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Rollback of DSV4 compressor state as llama.cpp's {@code seq_rm} allows it (llama-kv-cache-dsv4.cpp:1463-1510): at
 * most {@code n_rs_seq} positions, once until the next step, within the last step's positions; the rollback restores
 * snapshot plane {@code d} into the live plane of every compressor (llama-kv-cache-dsv4.cpp:657-681).
 */
public class CompressorRollbackTest {

    @Test
    public void blockerFollowsTheSnapshotRules() {
        CompressorRollback none = new CompressorRollback(0);
        none.onStep(4);
        assertNotNull(none.blocker(1));

        CompressorRollback r = new CompressorRollback(3);
        assertNotNull("no step stored yet", r.blocker(1));
        r.onStep(5);
        assertNull(r.blocker(1));
        assertNull(r.blocker(3));
        assertNotNull("beyond n_rs_seq", r.blocker(4));
        assertEquals(2, r.onRollback(2));
        assertNotNull("one rollback per step", r.blocker(1));
        assertThrows(IllegalStateException.class, () -> r.onRollback(1));
        r.onStep(1);
        assertNull(r.blocker(1));
        assertNotNull("the last step stored one position", r.blocker(2));
        r.reset();
        assertNotNull(r.blocker(1));

        assertThrows(IllegalArgumentException.class, () -> new CompressorRollback(-1));
        assertThrows(IllegalArgumentException.class, () -> r.onStep(0));
        assertThrows(IllegalStateException.class, () -> r.onRollback(0));
    }

    private static MemoryPlan dsv4Plan(int snapshots) {
        MemoryConfig cfg = MemoryConfig.builder().typeK(KvDType.F16).typeV(KvDType.F16).ubatch(8).build()
                .forRollback(snapshots);
        return MemoryPlanner.planForContext(MemoryPlannerTest.deepseek4Csa(), cfg, 256);
    }

    private static void step(HeapModelMemory mem, int pos0, int n) {
        mem.beginStep(pos0, n);
        int[] tokens = new int[n];
        Arrays.fill(tokens, 7);
        mem.sequence().append(tokens, 0, n);
    }

    @Test
    public void heapMemoryRollsBackWithinTheSnapshots() {
        ModelSpec spec = MemoryPlannerTest.deepseek4Csa();
        try (HeapModelMemory mem = HeapModelMemory.allocate(dsv4Plan(3), spec, null)) {
            assertFalse(mem.canRollback(1));
            step(mem, 0, 8);
            step(mem, 8, 5);
            assertTrue(mem.canRollback(1));
            assertTrue(mem.canRollback(3));
            assertFalse(mem.canRollback(4));

            CompressedKvPlan.Kind[] kinds = {CompressedKvPlan.Kind.CSA, CompressedKvPlan.Kind.INDEXER};
            for (CompressedKvPlan.Kind kind : kinds) {
                RecurrentStateAccessor state = mem.compressorState(0, kind);
                for (String name : new String[]{CpuMemory.COMPRESSOR_KV, CpuMemory.COMPRESSOR_SCORE}) {
                    float[] t = state.tensor(name);
                    int plane = t.length / 4;
                    for (int d = 0; d < 4; d++) {
                        Arrays.fill(t, d * plane, (d + 1) * plane, 10f * d + name.length());
                    }
                }
            }

            mem.rollback(2);
            assertEquals(11, mem.sequence().nPast());
            for (CompressedKvPlan.Kind kind : kinds) {
                RecurrentStateAccessor state = mem.compressorState(0, kind);
                for (String name : new String[]{CpuMemory.COMPRESSOR_KV, CpuMemory.COMPRESSOR_SCORE}) {
                    float[] t = state.tensor(name);
                    int plane = t.length / 4;
                    float[] want = new float[plane];
                    Arrays.fill(want, 20f + name.length());
                    assertArrayEquals(kind + " " + name + " live plane", want, Arrays.copyOf(t, plane), 0f);
                }
            }
            assertFalse("a second rollback waits for a step", mem.canRollback(1));
            assertThrows(IllegalStateException.class, () -> mem.rollback(1));

            step(mem, 11, 2);
            assertTrue(mem.canRollback(2));
            assertFalse(mem.canRollback(3));
            mem.reset();
            assertFalse(mem.canRollback(1));
        }
    }

    private static void step(DeviceModelMemory mem, int pos0, int n) {
        mem.stepInputs().upload(pos0, n);
        int[] tokens = new int[n];
        Arrays.fill(tokens, 7);
        mem.sequence().append(tokens, 0, n);
    }

    /** Fills plane {@code d} of both compressor tensors of every compressed cache of layer 0 with {@code 10 d + k}. */
    private static void markPlanes(DeviceModelMemory mem, int planes) {
        for (GpuCompressedKv c : mem.compressed(0)) {
            FakeGpuBuffer[] tensors = {(FakeGpuBuffer) c.stateKv(), (FakeGpuBuffer) c.stateScore()};
            for (int k = 0; k < tensors.length; k++) {
                int plane = c.plan().planeFloats();
                assertEquals(planes * (long) plane * Float.BYTES, tensors[k].sizeInBytes());
                for (int d = 0; d < planes; d++) {
                    float[] v = new float[plane];
                    Arrays.fill(v, 10f * d + k);
                    tensors[k].putFloats((long) d * plane, v);
                }
            }
        }
    }

    /** The device twin of {@link #heapMemoryRollsBackWithinTheSnapshots}: same rules, plane copied on the queue. */
    @Test
    public void deviceMemoryRollsBackWithinTheSnapshots() {
        try (FakeComputeBackend backend = FakeComputeBackend.create();
             DeviceModelMemory mem = DeviceModelMemory.allocate(backend, backend.computeQueue(), dsv4Plan(3),
                     MropeTextPositions.PPP0, null)) {
            assertEquals(2, mem.compressed(0).size());
            assertFalse(mem.canRollback(1));
            step(mem, 0, 8);
            step(mem, 8, 5);
            assertTrue(mem.canRollback(3));
            assertFalse(mem.canRollback(4));

            markPlanes(mem, 4);
            mem.rollback(2);
            assertEquals(11, mem.sequence().nPast());
            for (GpuCompressedKv c : mem.compressed(0)) {
                int plane = c.plan().planeFloats();
                FakeGpuBuffer[] tensors = {(FakeGpuBuffer) c.stateKv(), (FakeGpuBuffer) c.stateScore()};
                for (int k = 0; k < tensors.length; k++) {
                    float[] want = new float[plane];
                    Arrays.fill(want, 20f + k);
                    assertArrayEquals(c.plan().kind() + " tensor " + k + " live plane", want,
                            tensors[k].getFloats(0, plane), 0f);
                    float[] kept = new float[plane];
                    Arrays.fill(kept, 30f + k);
                    assertArrayEquals(c.plan().kind() + " tensor " + k + " plane 3 is kept", kept,
                            tensors[k].getFloats(3L * plane, plane), 0f);
                }
            }
            assertFalse("a second rollback waits for a step", mem.canRollback(1));
            assertThrows(IllegalStateException.class, () -> mem.rollback(1));

            step(mem, 11, 2);
            assertTrue(mem.canRollback(2));
            assertFalse(mem.canRollback(3));
            mem.reset();
            assertFalse(mem.canRollback(1));
        }
    }

    @Test
    public void heapMemoryWithoutSnapshotsRefusesCompressedRollback() {
        try (HeapModelMemory mem = HeapModelMemory.allocate(dsv4Plan(0), MemoryPlannerTest.deepseek4Csa(), null)) {
            step(mem, 0, 4);
            assertFalse(mem.canRollback(1));
            assertThrows(IllegalStateException.class, () -> mem.rollback(1));
        }
    }
}
