package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PageCopy;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The KV chunk reuse on the CPU backend ({@link ForwardBackend#reuseChunk}): a run of one sequence's cached rows
 * re-appends into another sequence's paged pages at the prompt's own positions. An unshifted run whose donor context
 * is the prompt's own context copies bit-exact K/V, so its logits equal a full prefill's; a shifted run re-rotates
 * the copied keys by the run's delta exactly as the context shift rotates them (the shift kernel's composition with
 * a fresh rotation is proven in {@code CpuRopeTest}), and keeps the values bit-exact — the re-appended rows'
 * deeper-layer state still reflects the donor's own prefix, the accuracy trade llama.cpp's opt-in
 * {@code n_cache_reuse} accepts, which is why the shifted comparison is against the kernel, not a full prefill.
 * A run the memory cannot serve refuses by name before any state changed. The synthetic two-layer llama opens
 * paged with pages of 8, and unpaged for the refusals.
 */
public class CpuChunkReuseTest {

    private static final int PAGE = 8;
    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float TOL = 1e-3f;

    /** The paged memory: two sequences, pages of 8, ubatch 8. */
    private static EngineConfig paged() {
        return EngineConfig.builder().memory(MemoryConfig.builder()
                .nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .parallel(2).kvPageSize(PAGE).build()).build();
    }

    private static EngineConfig unpaged() {
        return EngineConfig.builder().memory(MemoryConfig.builder()
                .nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    /** The cached prompt: 24 distinct tokens (three pages). */
    private static int[] cached() {
        int[] t = new int[3 * PAGE];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (i * 13 + 5) % 250;
        }
        return t;
    }

    /** A batch over {@code prompt[from..to)} at its own positions for {@code seq}, the last row its output row. */
    private static ForwardBatch batch(int[] prompt, int from, int to, int seq) {
        ForwardBatch.Builder batch = ForwardBatch.builder(to - from);
        for (int i = from; i < to; i++) {
            batch.add(prompt[i], i, seq, i == to - 1);
        }
        return batch.build();
    }

    @Test
    public void anUnshiftedReAppendMatchesTheFullPrefill() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] cache = cached();
        // the claimant's prompt: the cache's rows [0, 12), the cache's rows [12, 20) re-appended, a two-row tail
        int[] prompt = new int[22];
        System.arraycopy(cache, 0, prompt, 0, 12);
        System.arraycopy(cache, 12, prompt, 12, 8);
        prompt[20] = 200;
        prompt[21] = 210;

        // the reference: the same prompt prefilled whole on a paged instance of the same shape
        float[] reference;
        try (GGUFModel one = new GGUFModel(model.toString(), paged())) {
            reference = one.forwardBackend().forwardBatch(batch(prompt, 0, prompt.length, SEQ_A))[0];
        }
        try (GGUFModel two = new GGUFModel(model.toString(), paged())) {
            ForwardBackend f = two.forwardBackend();
            PagedSequences mem = paged(f);
            f.forwardBatch(batch(cache, 0, cache.length, SEQ_A));        // the cache side: three pages
            int[] mid = mem.blockPages(SEQ_A, 1, PAGE);
            int[] top = mem.blockPages(SEQ_A, 2, PAGE);
            f.forwardBatch(batch(prompt, 0, 12, SEQ_B));                 // the claimant's own first 12 rows
            // rows [12, 20) re-append source rows [12, 20): page row 4 of block 1, then all of block 2
            f.reuseChunk(SEQ_B, new int[][]{mid, top}, 4, 12, 8, 0,
                    Arrays.copyOfRange(prompt, 12, 20), -1);
            float[] out = f.forwardBatch(batch(prompt, 20, prompt.length, SEQ_B))[0];
            assertArrayEquals("the re-appended rows compute like a full prefill", reference, out, 0f);
            assertTrue(mem.pendingCopies().isEmpty());
        }
    }

    /**
     * A shifted re-append: the claimant prefills rows [0, 12) and re-appends the cache's rows [16, 24) at rows
     * [12, 20), a shift of -4. Every copied key equals the source key rotated by the same shift kernel the context
     * shift uses, and every copied value equals the source value bit-exact (only the keys carry positions); the
     * re-appended rows hold the prompt's positions [12, 20).
     */
    @Test
    public void aShiftedReAppendReRotatesTheCopiedKeysAndKeepsTheValues() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] cache = cached();
        int[] prompt = new int[21];
        System.arraycopy(cache, 0, prompt, 0, 12);
        System.arraycopy(cache, 16, prompt, 12, 8);
        prompt[20] = 220;

        try (GGUFModel two = new GGUFModel(model.toString(), paged())) {
            ForwardBackend f = two.forwardBackend();
            PagedSequences mem = paged(f);
            HeapModelMemory hm = (HeapModelMemory) f.memory();
            f.forwardBatch(batch(cache, 0, cache.length, SEQ_A));
            int[] top = mem.blockPages(SEQ_A, 2, PAGE);
            f.forwardBatch(batch(prompt, 0, 12, SEQ_B));
            int heads = two.spec().layer(0).attention().nHeadKv();
            int headDim = two.spec().layer(0).attention().headDimK();
            // the source cells before the edit, per layer and row
            float[][][][] srcK = new float[2][heads][PAGE][headDim];
            float[][][][] srcV = new float[2][heads][PAGE][headDim];
            for (int il = 0; il < 2; il++) {
                for (int h = 0; h < heads; h++) {
                    for (int r = 0; r < PAGE; r++) {
                        hm.kv(il).readK(top[0] * PAGE + r, h, srcK[il][h][r], 0);
                        hm.kv(il).readV(top[0] * PAGE + r, h, srcV[il][h][r], 0);
                    }
                }
            }
            // rows [12, 20) re-append source rows [16, 24) (block 2 whole): every copied key re-rotates by -4
            f.reuseChunk(SEQ_B, new int[][]{top}, 0, 12, 8, -4,
                    Arrays.copyOfRange(prompt, 12, 20), -1);
            // the copied rows span two pages of the claimant: rows [12, 16) on its block 1's rows [4, 8),
            // rows [16, 20) on its block 2's rows [0, 4)
            int[] dstA = mem.blockPages(SEQ_B, 1, 4, new int[]{8, 9, 10, 11});
            int[] dstB = mem.blockPages(SEQ_B, 2, 4, new int[]{16, 17, 18, 19});
            assertEquals(1, dstA.length);
            assertEquals(1, dstB.length);
            com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable shift =
                    com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable.forShift(two.spec().layer(0).rope(), null);
            float[] want = new float[headDim];
            float[] got = new float[headDim];
            for (int il = 0; il < 2; il++) {
                for (int h = 0; h < heads; h++) {
                    for (int r = 0; r < PAGE; r++) {
                        int dst = r < 4 ? dstA[0] * PAGE + 4 + r : dstB[0] * PAGE + r - 4;
                        System.arraycopy(srcK[il][h][r], 0, want, 0, headDim);
                        com.eonmux.neuralfree.gguf.ops.rope.CpuRope.shiftF32(shift, want, 1, 1, headDim, headDim,
                                0, new int[]{-4});
                        hm.kv(il).readK(dst, h, got, 0);
                        assertArrayEquals("layer " + il + " head " + h + " row " + r + " key", want, got, 1e-5f);
                        hm.kv(il).readV(dst, h, got, 0);
                        assertArrayEquals("layer " + il + " head " + h + " row " + r + " value",
                                srcV[il][h][r], got, 0f);
                    }
                }
            }
            assertArrayEquals("the re-appended rows hold the prompt's positions",
                    new int[]{8, 9, 10, 11, 12, 13, 14, 15}, mem.pagePositions(0, dstA[0], PAGE));
            assertArrayEquals(new int[]{16, 17, 18, 19, -1, -1, -1, -1},
                    mem.pagePositions(0, dstB[0], PAGE));
            mem.releaseCopies(mem.pendingCopies().toArray(new PageCopy[0]));
            assertTrue(mem.pendingCopies().isEmpty());
        }
    }

    @Test
    public void anUnpagedMemoryRefusesTheReuseByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel one = new GGUFModel(model.toString(), unpaged())) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> one.forwardBackend().reuseChunk(SEQ_A, new int[][]{new int[]{0}}, 0, 0, 1, 0,
                            new int[]{1}, -1));
            assertTrue(e.getMessage(), e.getMessage().contains("paged"));
        }
    }

    @Test
    public void aRunTheMemoryCannotServeRefusesByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel two = new GGUFModel(model.toString(), paged())) {
            ForwardBackend f = two.forwardBackend();
            PagedSequences mem = paged(f);
            f.forwardBatch(batch(cached(), 0, PAGE, SEQ_A));             // the cache side: one page
            int[] page = mem.blockPages(SEQ_A, 0, PAGE);
            // a run whose page list does not cover its rows
            assertThrows(IllegalArgumentException.class, () -> f.reuseChunk(SEQ_B, new int[][]{page},
                    0, 0, 9, 0, new int[9], -1));
            // a run of no rows
            assertThrows(IllegalArgumentException.class, () -> f.reuseChunk(SEQ_B, new int[][]{page},
                    0, 0, 0, 0, new int[0], -1));
            assertTrue(mem.pendingCopies().isEmpty());
        }
    }

    private static PagedSequences paged(ForwardBackend f) {
        PagedSequences mem = ((HeapModelMemory) f.memory()).pagedSequences();
        if (mem == null) {
            throw new IllegalStateException("the test needs a paged memory");
        }
        return mem;
    }
}
