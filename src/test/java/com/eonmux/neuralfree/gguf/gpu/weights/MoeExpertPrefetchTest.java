package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.Op;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend.OpKind;
import com.eonmux.neuralfree.gguf.testutil.FakeGpuBuffer;
import org.junit.After;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The prefill expert prefetch of {@link MoeExpertPrefetch} on the host-backed fake backend, against Avalon's
 * {@code GGML_SCHED_PREFETCH_EXPERTS} with the cache's lent zones: a prefill-sized step borrows zones from the tail
 * of the cache's banks, the experts the cache holds are read in place and never copied, the others land at their
 * natural slot of the layer's zone, the next layer is staged ahead behind the compute that last read its zone, the
 * prefill map points every expert at its slot, and a smaller step gives the zones back.
 */
public class MoeExpertPrefetchTest {

    private static final int N_EXPERT = 8;
    private static final int K = 2;
    private static final int N_FF = 4;
    private static final int COLS = 2;
    private static final int LAYERS = 3;
    private static final int TENSORS = 3;
    /** One expert of one tensor: {@code cols * rows * 4} bytes of F32. */
    private static final int TENSOR_BYTES = COLS * N_FF * Float.BYTES;
    /** Tokens of a prefill step: above the cache's planning size, so the step only feeds the histogram. */
    private static final int PREFILL = MoeExpertCache.MAX_BATCH + 8;

    private FakeComputeBackend backend;
    private MoeExpertCache cache;
    private MoeExpertPrefetch pool;

    @After
    public void tearDown() {
        if (pool != null) {
            pool.close();
        }
        if (cache != null) {
            cache.close();
        }
        if (backend != null) {
            backend.close();
        }
    }

    /** The byte every element of expert {@code e} of tensor {@code t} of layer {@code l} holds. */
    private static byte marker(int l, int t, int e) {
        return (byte) (1 + l * 64 + t * 16 + e);
    }

    private static final class ArrayStore implements WeightStore {
        private final byte[] data;

        ArrayStore(byte[] data) {
            this.data = data;
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) {
            System.arraycopy(data, Math.toIntExact(tensorOffset), dst, 0, size);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) {
            System.arraycopy(data, Math.toIntExact(tensorOffset), dst, dstOffset, size);
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) {
            dst.put(ByteBuffer.wrap(data, Math.toIntExact(tensorOffset), size));
        }

        @Override
        public void close() {
        }
    }

    private static String name(int l, int t) {
        return "blk." + l + ".ffn_" + new String[]{"gate", "up", "down"}[t] + "_exps.weight";
    }

    /** The layers' gate, up and down stacks, every expert filled with its {@link #marker}. */
    private static WeightDataProvider provider() {
        List<GGUFTensorInfo> infos = new ArrayList<>();
        List<List<String>> names = new ArrayList<>();
        byte[] all = new byte[LAYERS * TENSORS * N_EXPERT * TENSOR_BYTES];
        int offset = 0;
        for (int l = 0; l < LAYERS; l++) {
            List<String> layerNames = new ArrayList<>();
            for (int t = 0; t < TENSORS; t++) {
                infos.add(new GGUFTensorInfo(name(l, t), new long[]{COLS, N_FF, N_EXPERT}, GGMLType.F32, offset));
                for (int e = 0; e < N_EXPERT; e++) {
                    Arrays.fill(all, offset + e * TENSOR_BYTES, offset + (e + 1) * TENSOR_BYTES, marker(l, t, e));
                }
                offset += N_EXPERT * TENSOR_BYTES;
                layerNames.add(name(l, t));
            }
            names.add(layerNames);
        }
        return new WeightDataProvider(new ArrayStore(all), infos, names, 0, 0L);
    }

    private static List<MoeExpertCache.LayerPlan> plans() {
        List<MoeExpertCache.LayerPlan> out = new ArrayList<>();
        for (int l = 0; l < LAYERS; l++) {
            List<MoeExpertCache.TensorPlan> tensors = new ArrayList<>();
            for (int t = 0; t < TENSORS; t++) {
                tensors.add(new MoeExpertCache.TensorPlan(name(l, t), COLS, N_FF, N_EXPERT, GGMLType.F32));
            }
            out.add(new MoeExpertCache.LayerPlan(l, K, tensors));
        }
        return out;
    }

    /** A cache of {@code slots} slots per group over the three layers. */
    private void open(int slots, int zones) {
        open(slots, zones, 0);
    }

    private void open(int slots, int zones, int hostedMaxRows) {
        backend = FakeComputeBackend.builder().build();
        cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(), plans(),
                (long) slots * TENSORS * TENSOR_BYTES, PREFILL * K, null, "m");
        pool = MoeExpertPrefetch.open(backend, cache, plans(), zones, hostedMaxRows, "m");
    }

    /** Decode steps of layer {@code il} that make {@code experts} resident (their fills completed). */
    private void makeResident(int il, int... experts) {
        for (int e : experts) {
            cache.beginStep(il, new int[]{e, e}, K, 1);
            cache.endStep(il);
        }
        cache.beginStep(il, new int[]{experts[0], experts[0]}, K, 1);
        cache.endStep(il);
    }

    /** Routed ids of a prefill step: every token routes to two experts, every expert is demanded. */
    private static int[] prefillIds() {
        int[] ids = new int[PREFILL * K];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = i % N_EXPERT;
        }
        return ids;
    }

    /** One prefetched step of layer {@code il}: begin, the cache's step, cover, end. Returns the prefill map. */
    private int[] prefetchedStep(int il) {
        int[] ids = prefillIds();
        assertTrue(pool.begin(il, PREFILL, ids.length));
        MoeExpertCache.Step step = cache.beginStep(il, ids, ids.length, PREFILL);
        GpuBuffer map = pool.cover(il, ids, step.missPairs(), step.missCount());
        assertNotSame(cache.mapBuffer(il), map);
        pool.end(il);
        cache.endStep(il);
        return ((FakeGpuBuffer) map).getInts(0, N_EXPERT);
    }

    /** The first byte of slot {@code slot} of tensor {@code t}'s bank. */
    private byte slotByte(int il, int t, int slot) {
        byte[] b = new byte[1];
        ((FakeGpuBuffer) cache.bank(il, t)).read((long) slot * TENSOR_BYTES, b, 0, 1);
        return b[0];
    }

    private List<Op> transferWrites() {
        List<Op> out = new ArrayList<>();
        for (Op op : backend.ops()) {
            if (op.kind() == OpKind.WRITE && op.queue().equals(FakeComputeBackend.TRANSFER_QUEUE)) {
                out.add(op);
            }
        }
        return out;
    }

    @Test
    public void aPrefillStepReadsResidentsInPlaceAndCopiesTheOthersIntoItsZone() {
        open(20, 2);
        makeResident(0, 0, 1);
        assertEquals(2, pool.zoneCount(0));
        backend.clearLog();

        int[] map = prefetchedStep(0);

        assertEquals("zones 4..11 and 12..19 are lent", 16, cache.lentSlots(0));
        int zone0 = 20 - 2 * N_EXPERT;
        assertArrayEquals(new int[]{0, 1, zone0 + 2, zone0 + 3, zone0 + 4, zone0 + 5, zone0 + 6, zone0 + 7}, map);
        for (int t = 0; t < TENSORS; t++) {
            for (int e = 2; e < N_EXPERT; e++) {
                assertEquals(marker(0, t, e), slotByte(0, t, zone0 + e));
            }
            assertEquals("a resident expert is never copied", 0, slotByte(0, t, zone0));
            assertEquals(marker(0, t, 0), slotByte(0, t, 0));
        }
        assertEquals(6, pool.stepExperts());
        assertEquals(2, pool.residentReads());
        // experts 2..7 of each tensor are one run: one write per tensor for the step, one per tensor ahead
        assertEquals(2 * TENSORS, transferWrites().size());
        boolean gated = false;
        for (Op op : backend.ops()) {
            gated |= op.kind() == OpKind.MARKER && op.queue().equals(FakeComputeBackend.COMPUTE_QUEUE)
                    && op.waitCount() == 1;
        }
        assertTrue("the compute queue waits for the zone's uploads", gated);
    }

    @Test
    public void theNextLayerIsStagedAheadSoItsStepCopiesNothing() {
        open(20, 2);
        prefetchedStep(0);
        assertEquals("layer 1 is staged into the other zone during layer 0's step", N_EXPERT, pool.stagedExperts());
        int zone1 = 20 - N_EXPERT;
        for (int t = 0; t < TENSORS; t++) {
            for (int e = 0; e < N_EXPERT; e++) {
                assertEquals(marker(1, t, e), slotByte(1, t, zone1 + e));
            }
        }
        long atStep = pool.stepExperts();
        backend.clearLog();

        int[] map = prefetchedStep(1);

        assertEquals("layer 1's step copies nothing", atStep, pool.stepExperts());
        for (int e = 0; e < N_EXPERT; e++) {
            assertEquals(zone1 + e, map[e]);
        }
        // layer 2 takes layer 0's zone: its copy waits for the compute that read layer 0
        List<Op> writes = transferWrites();
        assertEquals(TENSORS, writes.size());
        assertEquals(1, writes.get(0).waitCount());
        assertEquals(2L * N_EXPERT, pool.stagedExperts());
        for (int t = 0; t < TENSORS; t++) {
            assertEquals(marker(2, t, 5), slotByte(2, t, 20 - 2 * N_EXPERT + 5));
        }
    }

    @Test
    public void aDecodeStepGivesTheZonesBackToTheCache() {
        open(20, 2);
        prefetchedStep(0);
        assertTrue(pool.lent(0));

        assertFalse(pool.begin(0, 1, K));
        assertFalse(pool.lent(0));
        assertEquals(0, cache.lentSlots(0));
        assertThrows(IllegalStateException.class, () -> pool.cover(0, new int[]{1, 2}, new int[]{0}, 1));

        // the next prefill lends again and re-copies what the restored zones lost
        prefetchedStep(0);
        assertTrue(pool.stats().contains("lends 2"));
    }

    @Test
    public void aSmallStepOrOneTokenIsNotPrefetched() {
        open(20, 2);
        assertFalse(pool.begin(0, 1, 2 * N_EXPERT));
        assertFalse("pairs below twice the experts", pool.begin(0, 4, 2 * N_EXPERT - 1));
        assertTrue(pool.begin(0, 8, 2 * N_EXPERT));
        assertTrue(MoeExpertPrefetch.prefillSized(8, 16, 8));
        assertFalse(MoeExpertPrefetch.prefillSized(1, 16, 8));
    }

    @Test
    public void stepsUpToTheHostedRowBoundKeepTheHostMisses() {
        open(20, 2, 2 * N_EXPERT * 2);
        assertFalse(pool.begin(0, 16, 2 * N_EXPERT * 2));
        assertEquals(0, cache.lentSlots(0));
        assertTrue(pool.begin(0, 17, 2 * N_EXPERT * 2 + 2));
    }

    @Test
    public void oneZoneCopiesEachLayerAtItsStepBehindThePreviousRead() {
        open(12, 2);
        assertEquals("12 slots hold one layer stack of 8", 1, pool.zoneCount(0));
        prefetchedStep(0);
        assertEquals(0, pool.stagedExperts());
        assertEquals(N_EXPERT, pool.stepExperts());
        backend.clearLog();

        int[] map = prefetchedStep(1);
        assertEquals(2L * N_EXPERT, pool.stepExperts());
        assertEquals(1, transferWrites().get(0).waitCount());
        assertEquals(12 - N_EXPERT + 3, map[3]);
        assertEquals(marker(1, 2, 3), slotByte(1, 2, 12 - N_EXPERT + 3));
    }

    @Test
    public void aCacheSmallerThanALayerStackKeepsTheHostMisses() {
        open(5, 2);
        assertEquals(0, pool.zoneCount(0));
        assertFalse(pool.begin(0, PREFILL, PREFILL * K));
        assertEquals(0, cache.lentSlots(0));
    }

    @Test
    public void closingReleasesEveryEvent() {
        open(20, 2);
        prefetchedStep(0);
        prefetchedStep(1);
        pool.close();
        pool = null;
        cache.close();
        cache = null;
        assertEquals(0, backend.openEventCount());
    }

    @Test
    public void refusesInvalidArguments() {
        open(20, 2);
        assertThrows(IllegalArgumentException.class,
                () -> MoeExpertPrefetch.open(backend, cache, plans(), 0, 0, "m"));
        assertThrows(IllegalArgumentException.class,
                () -> MoeExpertPrefetch.open(backend, cache, plans(), 1, -1, "m"));
        assertThrows(IllegalArgumentException.class, () -> pool.begin(7, PREFILL, PREFILL * K));
        assertTrue(pool.begin(0, PREFILL, PREFILL * K));
        int[] ids = prefillIds();
        ids[3] = N_EXPERT;
        assertThrows(IllegalArgumentException.class, () -> pool.cover(0, ids, new int[]{3}, 1));
    }
}
