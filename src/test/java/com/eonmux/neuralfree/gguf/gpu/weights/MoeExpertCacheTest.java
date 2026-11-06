package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.ops.moe.MoeCacheRanking;
import com.eonmux.neuralfree.gguf.ops.moe.MoeSplitDispatcher;
import com.eonmux.neuralfree.gguf.ops.moe.MoeTailSkip;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.After;
import org.junit.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The step protocol of {@link MoeExpertCache} on the host-backed fake backend: plan-driven fills become visible at
 * the layer's next step (the map the kernels read), the histogram of a long prefill flushes at the first small
 * step, the ranking preloads the slots before the first step, the disabled group serves nothing, an evicted
 * expert loses its map entry (also one the split dispatcher copied in), a prefetched step never splits, and lent
 * tail slots leave the cache until they are restored.
 */
public class MoeExpertCacheTest {

    private static final int N_EXPERT = 8;
    private static final int N_FF = 4;
    private static final int COLS = 2;
    /** One expert of one tensor: {@code cols * rows * 4} bytes of F32. */
    private static final int TENSOR_BYTES = COLS * N_FF * Float.BYTES;

    private FakeComputeBackend backend;

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
    }

    /** One in-memory tensor per entry, laid back to back in one byte[] store. */
    private static final class ArrayStore implements WeightStore {
        private final byte[] data;

        ArrayStore(byte[] data) {
            this.data = data;
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst) throws IOException {
            System.arraycopy(data, Math.toIntExact(tensorOffset), dst, 0, size);
        }

        @Override
        public void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) throws IOException {
            System.arraycopy(data, Math.toIntExact(tensorOffset), dst, dstOffset, size);
        }

        @Override
        public void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException {
            dst.put(ByteBuffer.wrap(data, Math.toIntExact(tensorOffset), size));
        }

        @Override
        public void close() {
        }
    }

    /** A provider over two layers' expert tensor stacks {@code blk.{0,1}.ffn_{gate,up,down}_exps.weight} of F32. */
    private static WeightDataProvider provider(int layers) {
        long offset = 0;
        List<GGUFTensorInfo> infos = new ArrayList<>();
        List<List<String>> names = new ArrayList<>();
        byte[][] bytes = new byte[layers][];
        for (int l = 0; l < layers; l++) {
            String[] tensors = {"blk." + l + ".ffn_gate_exps.weight", "blk." + l + ".ffn_up_exps.weight",
                    "blk." + l + ".ffn_down_exps.weight"};
            byte[] layerBytes = new byte[tensors.length * N_EXPERT * TENSOR_BYTES];
            List<String> layerNames = new ArrayList<>();
            for (int t = 0; t < tensors.length; t++) {
                infos.add(new GGUFTensorInfo(tensors[t], new long[]{COLS, N_FF, N_EXPERT}, GGMLType.F32, offset));
                for (int e = 0; e < N_EXPERT; e++) {
                    Arrays.fill(layerBytes, (t * N_EXPERT + e) * TENSOR_BYTES, (t * N_EXPERT + e + 1) * TENSOR_BYTES,
                            (byte) (l * 16 + t + 1));
                }
                offset += N_EXPERT * TENSOR_BYTES;
                layerNames.add(tensors[t]);
            }
            names.add(layerNames);
            bytes[l] = layerBytes;
        }
        byte[] all = new byte[(int) offset];
        for (int l = 0; l < layers; l++) {
            System.arraycopy(bytes[l], 0, all, l * bytes[l].length, bytes[l].length);
        }
        WeightStore store = new ArrayStore(all);
        return new WeightDataProvider(store, infos, names, 0, 0L);
    }

    private static MoeExpertCache.LayerPlan plan(int weightLayer, int k) {
        List<MoeExpertCache.TensorPlan> tensors = Arrays.asList(
                new MoeExpertCache.TensorPlan("blk." + weightLayer + ".ffn_gate_exps.weight", COLS, N_FF, N_EXPERT,
                        GGMLType.F32),
                new MoeExpertCache.TensorPlan("blk." + weightLayer + ".ffn_up_exps.weight", COLS, N_FF, N_EXPERT,
                        GGMLType.F32),
                new MoeExpertCache.TensorPlan("blk." + weightLayer + ".ffn_down_exps.weight", COLS, N_FF, N_EXPERT,
                        GGMLType.F32));
        return new MoeExpertCache.LayerPlan(weightLayer, k, tensors);
    }

    private static byte[] intBytes(int[] v) {
        ByteBuffer buf = ByteBuffer.allocate(v.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int x : v) {
            buf.putInt(x);
        }
        return buf.array();
    }

    @Test
    public void aFillBecomesVisibleAtTheNextStep() {
        backend = FakeComputeBackend.builder().build();
        MoeExpertCache cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(1),
                Collections.singletonList(plan(0, 1)), 1L * TENSOR_BYTES * 3, 8, null, "m");

        MoeExpertCache.Step first = cache.beginStep(0, new int[]{3}, 1, 1);
        assertEquals(0, first.hitCount());
        assertEquals(1, first.missCount());
        assertEquals(0, first.missPairs()[0]);
        cache.endStep(0);

        // the fill ran while the host computed the miss: the next step reads it from the slot
        MoeExpertCache.Step second = cache.beginStep(0, new int[]{3}, 1, 1);
        assertEquals(1, second.hitCount());
        assertEquals(0, second.missCount());
        assertEquals(1, cache.slotCount(0));
        cache.close();
    }

    @Test
    public void aPrefillOnlyAccumulatesAndTheHistogramFlushesAtTheNextDecode() {
        backend = FakeComputeBackend.builder().build();
        MoeExpertCache cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(1),
                Collections.singletonList(plan(0, 1)), 1L * TENSOR_BYTES * 3, 64, null, "m");

        int[] prefill = new int[64];
        Arrays.fill(prefill, 5);
        MoeExpertCache.Step step = cache.beginStep(0, prefill, prefill.length, 32 + 1);
        assertEquals(0, step.hitCount());
        assertEquals(prefill.length, step.missCount());
        cache.endStep(0);

        // the histogram flushes at the first small step: expert 5 takes a slot, invisible until the next step
        MoeExpertCache.Step decode = cache.beginStep(0, new int[]{1}, 1, 1);
        assertEquals(0, decode.hitCount());
        cache.endStep(0);
        MoeExpertCache.Step next = cache.beginStep(0, new int[]{5}, 1, 1);
        assertEquals(1, next.hitCount());
        assertEquals(0, next.missCount());
        cache.close();
    }

    @Test
    public void theRankingPreloadsTheSlotsBeforeTheFirstStep() throws IOException {
        backend = FakeComputeBackend.builder().build();
        Path ranking = Files.createTempFile("moe-cache-ranking", ".bin");
        ByteBuffer buf = ByteBuffer.allocate(24 + 2 * 8).order(ByteOrder.LITTLE_ENDIAN);
        buf.put("STRP".getBytes("US-ASCII")).putInt(2).putInt(1).putInt(N_EXPERT).putInt(2).putInt(2);
        buf.putShort((short) 0).putShort((short) 6).putInt(10);
        buf.putShort((short) 0).putShort((short) 2).putInt(5);
        Files.write(ranking, buf.array());
        try {
            MoeExpertCache cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(1),
                    Collections.singletonList(plan(0, 1)), 2L * TENSOR_BYTES * 3, 8, MoeCacheRanking.open(null,
                            N_EXPERT, new MoeCacheRanking.Options(MoeCacheRanking.PreloadMode.PATH, ranking, null)),
                    "m");
                cache.endStep(0); // no step ran yet: nothing to fill

            MoeExpertCache.Step step = cache.beginStep(0, new int[]{6, 2}, 2, 1);
            assertEquals(2, step.hitCount());
            assertEquals(0, step.missCount());
            assertEquals(2, cache.slotCount(0));
            cache.close();
        } finally {
            Files.deleteIfExists(ranking);
        }
    }

    @Test
    public void aBankTooSmallForTheBudgetDisablesTheLayer() {
        backend = FakeComputeBackend.builder().build();
        MoeExpertCache cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(1),
                Collections.singletonList(plan(0, 1)), TENSOR_BYTES * 3 - 1, 8, null, "m");

        assertFalse(cache.serves(0));
        MoeExpertCache.Step step = cache.beginStep(0, new int[]{3}, 1, 1);
        assertEquals(0, step.hitCount());
        assertEquals(1, step.missCount());
        cache.endStep(0);
        cache.close();
    }

    @Test
    public void twoLayersOfOneGroupShareTheSlots() {
        backend = FakeComputeBackend.builder().build();
        MoeExpertCache cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(2),
                Arrays.asList(plan(0, 1), plan(1, 1)), 2L * TENSOR_BYTES * 3, 8, null, "m");

        assertEquals(3, cache.tensorCount(1));
        assertEquals("each layer keeps its own tensor names", "blk.1.ffn_gate_exps.weight",
                cache.tensor(1, 0).name());
        assertEquals("blk.0.ffn_down_exps.weight", cache.tensor(0, 2).name());
        assertEquals(2, cache.slotCount(0));
        MoeExpertCache.Step l0 = cache.beginStep(0, new int[]{1}, 1, 1);
        assertEquals(1, l0.missCount());
        cache.endStep(0);
        MoeExpertCache.Step l1 = cache.beginStep(1, new int[]{3}, 1, 1);
        assertEquals(1, l1.missCount());
        cache.endStep(1);
        MoeExpertCache.Step l0b = cache.beginStep(0, new int[]{1}, 1, 1);
        assertEquals(1, l0b.hitCount());
        cache.endStep(0);
        assertTrue(cache.stats().contains("hits 1"));
        assertTrue(cache.stats().contains("misses 2"));
        cache.close();
    }

    @Test
    public void theMapBufferHoldsMinusOneForUncachedExperts() {
        backend = FakeComputeBackend.builder().build();
        MoeExpertCache cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(1),
                Collections.singletonList(plan(0, 1)), 1L * TENSOR_BYTES * 3, 8, null, "m");
        cache.beginStep(0, new int[]{3}, 1, 1);
        cache.endStep(0);
        cache.beginStep(0, new int[]{3}, 1, 1);
        cache.endStep(0);
        cache.beginStep(0, new int[]{3}, 1, 1);
        cache.endStep(0);

        byte[] bytes = new byte[N_EXPERT * Integer.BYTES];
        backend.computeQueue().enqueueReadBuffer(cache.mapBuffer(0), 0, bytes.length, bytes).close();
        backend.computeQueue().finish();
        int[] map = new int[N_EXPERT];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(map);
        assertEquals(0, map[3]);
        assertEquals(-1, map[0]);
        assertEquals(-1, map[N_EXPERT - 1]);
        cache.close();
    }

    @Test
    public void planArgumentsAreValidated() {
        backend = FakeComputeBackend.builder().build();
        MoeExpertCache cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(1),
                Collections.singletonList(plan(0, 2)), 1L * TENSOR_BYTES * 3, 4, null, "m");

        org.junit.Assert.assertThrows(IllegalArgumentException.class,
                () -> cache.beginStep(0, new int[]{N_EXPERT}, 1, 1));
        org.junit.Assert.assertThrows(IllegalArgumentException.class,
                () -> cache.beginStep(0, new int[]{1, 2, 3}, 3, 1));
        org.junit.Assert.assertThrows(IllegalArgumentException.class,
                () -> cache.beginStep(7, new int[]{1}, 1, 1));
        cache.close();
    }

    @Test
    public void theTailSkipDropsTheMissesOfLowRankExperts() {
        backend = FakeComputeBackend.builder().build();
        MoeExpertCache cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(1),
                Collections.singletonList(plan(0, 2)), 2L * TENSOR_BYTES * 3, 8, null, "m");

        // two tokens: (0 first, 1 second), (2 first, 3 second); rank 1 drops only the second picks
        int[] ids = {0, 1, 2, 3};
        MoeExpertCache.Step step = cache.beginStep(0, ids, ids.length, 2, true, 1);
        assertEquals(0, step.hitCount());
        assertEquals(2, step.missCount());
        assertArrayEquals(new int[]{0, 2}, Arrays.copyOf(step.missPairs(), step.missCount()));
        assertEquals(2, step.skippedCount());
        assertArrayEquals(new int[]{1, 3}, Arrays.copyOf(step.skippedPairs(), step.skippedCount()));
        cache.close();
    }

    @Test
    public void aTailSkippedExpertIsNotMovedByTheSplit() {
        MoeExpertCache cache = oneSlotCacheWithDispatcher();
        // every token routes expert 1 first and expert 2 second: rank 1 drops expert 2, never expert 1
        MoeExpertCache.Step step = cache.beginStep(0, TWO_EXPERTS, TWO_EXPERTS.length, 8, true, 1);
        assertEquals("the 8 second picks of expert 2", 8, step.skippedCount());
        assertEquals("the dropped expert is never copied into the cache", -1, cache.residentSlot(0, 2));
        assertEquals(-1, deviceMap(cache, 0)[2]);
        assertEquals("the kept expert still runs on the device or the host", 8, step.hitCount() + step.missCount());
        cache.endStep(0);
        cache.close();
    }

    @Test
    public void aCachedExpertIsNeverTailSkipped() {
        backend = FakeComputeBackend.builder().build();
        MoeExpertCache cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(1),
                Collections.singletonList(plan(0, 2)), 2L * TENSOR_BYTES * 3, 8, null, "m");
        cache.beginStep(0, new int[]{3, 3}, 2, 1);
        cache.endStep(0);

        // expert 3 is a hit although it is the token's second pick; expert 0 is a first pick and stays a miss
        MoeExpertCache.Step step = cache.beginStep(0, new int[]{0, 3}, 2, 1, true, 1);
        assertEquals(1, step.hitCount());
        assertEquals(1, step.missCount());
        assertEquals(0, step.skippedCount());
        cache.endStep(0);
        cache.close();
    }

    @Test
    public void theTailSkipLeavesThePrefetchStepsAlone() {
        MoeExpertCache cache = oneSlotCacheWithDispatcher();
        MoeExpertCache.Step step = cache.beginStep(0, TWO_EXPERTS, TWO_EXPERTS.length, 8, false, 1);
        assertEquals("a prefetched step uploads every miss, so none is skipped", 0, step.skippedCount());
        assertEquals(TWO_EXPERTS.length, step.missCount());
        cache.endStep(0);
        cache.close();
    }

    @Test
    public void tailSkipRanksAreValidated() {
        backend = FakeComputeBackend.builder().build();
        MoeExpertCache cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(1),
                Collections.singletonList(plan(0, 1)), 1L * TENSOR_BYTES * 3, 8, null, "m");
        org.junit.Assert.assertThrows(IllegalArgumentException.class,
                () -> cache.beginStep(0, new int[]{1}, 1, 1, true, MoeTailSkip.MAX_RANK + 1));
        org.junit.Assert.assertThrows(IllegalArgumentException.class,
                () -> cache.beginStep(0, new int[]{1}, 1, 1, true, -1));
        cache.close();
    }

    @Test
    public void bankBytesMatchTheTensorPlan() {
        backend = FakeComputeBackend.builder().build();
        MoeExpertCache cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(1),
                Collections.singletonList(plan(0, 1)), 1L * TENSOR_BYTES * 3, 8, null, "m");

        assertEquals(TENSOR_BYTES, cache.bank(0, 0).sizeInBytes());
        assertArrayEquals(new String[]{"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                "blk.0.ffn_down_exps.weight"}, new String[]{cache.tensor(0, 0).name(), cache.tensor(0, 1).name(),
                cache.tensor(0, 2).name()});
        cache.close();
    }

    /** The device map of layer {@code il}. */
    private int[] deviceMap(MoeExpertCache cache, int il) {
        byte[] bytes = new byte[N_EXPERT * Integer.BYTES];
        backend.computeQueue().enqueueReadBuffer(cache.mapBuffer(il), 0, bytes.length, bytes).close();
        backend.computeQueue().finish();
        int[] map = new int[N_EXPERT];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(map);
        return map;
    }

    @Test
    public void anExpertEvictedByAnotherLayerLosesItsMapEntry() {
        backend = FakeComputeBackend.builder().build();
        MoeExpertCache cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(2),
                Arrays.asList(plan(0, 1), plan(1, 1)), 1L * TENSOR_BYTES * 3, 8, null, "m");
        cache.beginStep(0, new int[]{1}, 1, 1);
        cache.endStep(0);
        assertEquals(1, cache.beginStep(0, new int[]{1}, 1, 1).hitCount());
        cache.endStep(0);

        // layer 1's miss takes the only slot: layer 0's expert 1 must no longer read it
        cache.beginStep(1, new int[]{3}, 1, 1);
        cache.endStep(1);
        MoeExpertCache.Step again = cache.beginStep(0, new int[]{1}, 1, 1);
        assertEquals(0, again.hitCount());
        assertEquals(1, again.missCount());
        assertEquals(-1, deviceMap(cache, 0)[1]);
        assertEquals(-1, cache.residentSlot(0, 1));
        cache.endStep(0);
        cache.close();
    }

    @Test
    public void lentTailSlotsEvictTheirExpertsUntilTheyAreRestored() {
        backend = FakeComputeBackend.builder().build();
        MoeExpertCache cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(1),
                Collections.singletonList(plan(0, 1)), 4L * TENSOR_BYTES * 3, 8, null, "m");
        for (int e : new int[]{0, 1, 2, 3}) {
            cache.beginStep(0, new int[]{e}, 1, 1);
            cache.endStep(0);
        }
        // expert 3's fill is still pending: the lend completes it first, then evicts the tail slots 2 and 3
        cache.lendSlots(0, 2);
        assertEquals(2, cache.lentSlots(0));
        assertEquals(0, cache.residentSlot(0, 0));
        assertEquals(1, cache.residentSlot(0, 1));
        assertEquals(-1, cache.residentSlot(0, 2));
        assertEquals(-1, cache.residentSlot(0, 3));

        MoeExpertCache.Step step = cache.beginStep(0, new int[]{2}, 1, 1);
        assertEquals(1, step.missCount());
        assertEquals(-1, deviceMap(cache, 0)[2]);
        cache.endStep(0);
        assertEquals(1, cache.beginStep(0, new int[]{2}, 1, 1).hitCount());
        cache.endStep(0);
        assertTrue("a lent slot is never filled", cache.residentSlot(0, 2) < 2);
        org.junit.Assert.assertThrows(IllegalStateException.class, () -> cache.lendSlots(0, 1));

        cache.restoreSlots(0);
        assertEquals(0, cache.lentSlots(0));
        cache.beginStep(0, new int[]{5}, 1, 1);
        cache.endStep(0);
        assertEquals(1, cache.beginStep(0, new int[]{5}, 1, 1).hitCount());
        cache.endStep(0);
        int slot = cache.residentSlot(0, 5);
        assertTrue("expert 5 takes a restored tail slot, got " + slot, slot >= 2);
        cache.close();
    }

    @Test
    public void lendingValidatesTheSlotCount() {
        backend = FakeComputeBackend.builder().build();
        MoeExpertCache cache = MoeExpertCache.open(backend, backend.computeQueue(), provider(1),
                Collections.singletonList(plan(0, 1)), 2L * TENSOR_BYTES * 3, 8, null, "m");
        org.junit.Assert.assertThrows(IllegalArgumentException.class, () -> cache.lendSlots(0, 0));
        org.junit.Assert.assertThrows(IllegalArgumentException.class, () -> cache.lendSlots(0, 3));
        cache.close();
    }

    /** A cache of two layers sharing one slot, with the split dispatcher of an 8-thread host. */
    private MoeExpertCache oneSlotCacheWithDispatcher() {
        backend = FakeComputeBackend.builder().build();
        return MoeExpertCache.open(backend, backend.computeQueue(), provider(2),
                Arrays.asList(plan(0, 2), plan(1, 2)), 1L * TENSOR_BYTES * 3, 16, null, "m",
                new MoeSplitDispatcher(MoeSplitDispatcher.Model.of(8)));
    }

    /** Eight tokens over experts 1 and 2 (two per token): wider than the one slot, so nothing is planned. */
    private static final int[] TWO_EXPERTS = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2};

    /** The expert of layer {@code il} the cache holds among {@code experts}, or -1. */
    private static int residentOf(MoeExpertCache cache, int il, int... experts) {
        for (int e : experts) {
            if (cache.residentSlot(il, e) >= 0) {
                return e;
            }
        }
        return -1;
    }

    @Test
    public void anExpertTheSplitCopiedInLosesItsMapEntryWhenAnotherLayerEvictsIt() {
        MoeExpertCache cache = oneSlotCacheWithDispatcher();
        MoeExpertCache.Step split = cache.beginStep(0, TWO_EXPERTS, TWO_EXPERTS.length, 8);
        int moved = residentOf(cache, 0, 1, 2);
        assertTrue("the dispatcher copied one expert into the empty slot", moved >= 0);
        assertEquals(8, split.hitCount());
        assertEquals(0, deviceMap(cache, 0)[moved]);
        cache.endStep(0);

        // layer 1's miss takes the only slot: layer 0's copied expert must no longer read it
        cache.beginStep(1, new int[]{3, 3}, 2, 1);
        cache.endStep(1);
        MoeExpertCache.Step again = cache.beginStep(0, new int[]{moved, moved}, 2, 1);
        assertEquals(0, again.hitCount());
        assertEquals(-1, cache.residentSlot(0, moved));
        assertEquals(-1, deviceMap(cache, 0)[moved]);
        cache.endStep(0);
        cache.close();
    }

    @Test
    public void aStepWhoseMissesThePrefetchUploadsNeverSplits() {
        MoeExpertCache cache = oneSlotCacheWithDispatcher();
        MoeExpertCache.Step step = cache.beginStep(0, TWO_EXPERTS, TWO_EXPERTS.length, 8, false);
        assertEquals("no expert is copied for the host's sake", -1, residentOf(cache, 0, 1, 2));
        assertEquals(TWO_EXPERTS.length, step.missCount());
        cache.endStep(0);
        cache.close();
    }
}
