package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertTrue;

/**
 * GPU parity of the grouped expert paths against the existing per-pair kernels, run on each backend by a subclass
 * that opens the device and the {@link GroupedGemmHarness}:
 * <ul>
 *   <li>the multi-work-group sort ({@code LLM_MoE_BucketCount} + {@code LLM_MoE_BucketScan} +
 *       {@code LLM_MoE_BucketScatter}) writes exactly the counts, offsets and sorted rows of the one-work-group
 *       {@code LLM_MoE_Bucketize} and of {@link MoeCombine#bucketize};</li>
 *   <li>{@code LLM_MoE_AlignBlocks} on one and on many work-groups writes the layout of
 *       {@link MoeCombine#alignBlocks};</li>
 *   <li>the grouped GEMM over the MLA per-head shape (heads as experts, pair rows {@code x_stride} apart, input and
 *       output offsets) equals the per-head {@code {T}_MatVecId} of the mixer;</li>
 *   <li>the grouped GEMM and the per-head {@code {T}_MatVecId} stay the fp64 dots over MLA pair rows whose byte
 *       phase alternates along an odd stride, for the block formats with vectorized activation readers (Q8_0,
 *       Q4_0);</li>
 *   <li>the transposed per-head read {@code {T}_MatVecIdT} over one broadcast latent row equals the fp64 column
 *       products of the deepseek2 absorbed {@code attn_k_b} slot blocks;</li>
 *   <li>the grouped GEMM over the value-expert shape (token rows, two expert shards) equals
 *       {@code {T}_BatchMatMulId};</li>
 *   <li>the expert-cache form ({@code {T}_GroupedGemmInd}, and the kernel {@link GpuGroupedRows#groupedKernel}
 *       picks for it on the device) over a bank that holds some experts in permuted slots equals
 *       {@code {T}_BatchMatMulIdInd} and the fp64 products: pairs of uncached experts are left untouched, pairs of
 *       out-of-range ids are zeroed;</li>
 *   <li>the direct grouped GEMM {@link GpuGroupedRows#groupedKernel} picks (the tensor-core form on CUDA and Vulkan
 *       devices with cooperative matrices) equals {@code {T}_BatchMatMulId} over two scaled shards.</li>
 * </ul>
 */
public abstract class MoeGroupedPathsParity {

    /** The open device fixtures; set by the subclass before each test. */
    protected GroupedGemmHarness h;

    @Test
    public void multiGroupSortMatchesTheOneGroupBucketize() {
        int nTokens = 100;
        int k = 3;
        int nExpert = 13;
        int chunk = 64;
        int rows = nTokens * k;
        int[] idValues = GroupedGemmHarness.ids(rows, nExpert, 5, true);
        GpuBuffer ids = h.ints(idValues);
        GroupedGemmHarness.Sorted single = h.sort(nTokens, k, nExpert, ids);

        int groups = MoeKernels.bucketGroups(rows, chunk);
        assertTrue("the case needs several work-groups", groups > 1);
        GpuBuffer hist = h.out((int) MoeKernels.bucketHistInts(rows, nExpert, chunk), 0);
        GpuBuffer counts = h.out(nExpert, 0);
        GpuBuffer offsets = h.out(nExpert, 0);
        GpuBuffer rowToken = h.out(rows, 0);
        GpuBuffer rowSlot = h.out(rows, 0);
        long[] local = {32};
        long[] grid = {(long) groups * 32};
        h.dispatch(MoeKernels.BUCKET_COUNT, grid, local,
                MoeKernels.bucketCountArgs(nTokens, k, nExpert, chunk, ids, hist));
        h.dispatch(MoeKernels.BUCKET_SCAN, local, local,
                MoeKernels.bucketScanArgs(nTokens, k, nExpert, chunk, hist, counts, offsets));
        h.dispatch(MoeKernels.BUCKET_SCATTER, grid, local,
                MoeKernels.bucketScatterArgs(nTokens, k, nExpert, chunk, ids, hist, rowToken, rowSlot));
        h.finish();

        MoeCombine.Buckets cpu = MoeCombine.bucketize(nTokens, k, nExpert, idValues);
        assertArrayEquals("counts", cpu.counts(), h.readInts(counts, nExpert));
        assertArrayEquals("offsets", cpu.offsets(), h.readInts(offsets, nExpert));
        assertArrayEquals("row_token", cpu.rowToken(), h.readInts(rowToken, rows));
        assertArrayEquals("row_slot", cpu.rowSlot(), h.readInts(rowSlot, rows));
        assertArrayEquals("one-group row_token", cpu.rowToken(), h.readInts(single.rowToken, rows));
        assertArrayEquals("one-group row_slot", cpu.rowSlot(), h.readInts(single.rowSlot, rows));
    }

    @Test
    public void alignBlocksOnOneAndOnManyWorkGroupsMatchesTheCpuLayout() {
        int nTokens = 90;
        int k = 2;
        int nExpert = 7;
        int rows = nTokens * k;
        int[] idValues = GroupedGemmHarness.ids(rows, nExpert, 11, true);
        GroupedGemmHarness.Sorted many = h.sort(nTokens, k, nExpert, h.ints(idValues));
        int positions = many.tiles * MoeKernels.ALIGN_TILE;
        assertTrue("the case needs several align work-groups",
                MoeKernels.alignGlobal(many.tiles, GroupedGemmHarness.LOCAL)[0] > GroupedGemmHarness.LOCAL);
        GpuBuffer token1 = h.out(positions, 0);
        GpuBuffer slot1 = h.out(positions, 0);
        long[] one = {GroupedGemmHarness.LOCAL};
        h.dispatch(MoeKernels.ALIGN_BLOCKS, one, one, MoeKernels.alignBlocksArgs(nTokens, k, nExpert, many.tiles,
                many.counts, many.offsets, many.rowToken, many.rowSlot, token1, slot1));
        h.finish();

        MoeCombine.Buckets b = MoeCombine.bucketize(nTokens, k, nExpert, idValues);
        MoeCombine.Aligned cpu = MoeCombine.alignBlocks(nTokens, k, nExpert, b.counts(), b.offsets(), b.rowToken(),
                b.rowSlot());
        int[] wantToken = padded(cpu.gemmToken(), positions);
        int[] wantSlot = padded(cpu.gemmSlot(), positions);
        assertArrayEquals("gemm_token, many groups", wantToken, h.readInts(many.gemmToken, positions));
        assertArrayEquals("gemm_slot, many groups", wantSlot, h.readInts(many.gemmSlot, positions));
        assertArrayEquals("gemm_token, one group", wantToken, h.readInts(token1, positions));
        assertArrayEquals("gemm_slot, one group", wantSlot, h.readInts(slot1, positions));
    }

    /**
     * The transposed per-head read over the deepseek2 absorbed {@code attn_k_b} shape ({@code {T}_MatVecIdT}): each
     * head's slot block is [kv_lora x n_nope] row-major and every slot dots a column of its block with one broadcast
     * latent row ({@code x_stride = 0}), against the fp64 column products.
     */
    @Test
    public void transposedMatVecIdMatchesTheColumnDotsOfTheAbsorbedShape() {
        for (GGMLType type : new GGMLType[]{GGMLType.F16, GGMLType.Q8_0}) {
            int nHead = 4;
            int kvLora = 64;
            int nope = 32;
            int inOff = 5;
            int outOff = 9;
            int[] slots = new int[nHead];
            for (int s = 0; s < nHead; s++) {
                slots[s] = s;
            }
            byte[] raw = GroupedGemmHarness.weight(type, nHead * kvLora, nope, 3);
            GpuBuffer weight = h.bytes(raw);
            GpuBuffer input = h.floats(GroupedGemmHarness.values(inOff + kvLora, 5, 1.5f));
            GpuBuffer ph = h.floats(new float[]{1.0f});
            int outLen = outOff + nHead * nope;
            GpuBuffer got = h.out(outLen, -7.0f);

            h.perRowT(type, nHead, nope, kvLora, 0, inOff, outOff, nHead, h.ints(slots), weight, input, ph, got);
            h.finish();

            float[] w = GroupedGemmHarness.decode(type, raw, nHead * kvLora, nope);
            float[] x = h.readFloats(input, inOff + kvLora);
            float[] have = h.readFloats(got, outLen);
            for (int s = 0; s < nHead; s++) {
                for (int r = 0; r < nope; r++) {
                    double acc = 0.0;
                    double terms = 0.0;
                    for (int l = 0; l < kvLora; l++) {
                        double t = (double) w[(s * kvLora + l) * nope + r] * x[inOff + l];
                        acc += t;
                        terms += Math.abs(t);
                    }
                    assertNear(type + " head " + s + " row " + r, acc, have[outOff + s * nope + r],
                            2e-5 * terms + 1e-3);
                }
            }
        }
    }

    /**
     * The latent MLA absorption shape: {@code n} tokens of {@code n_head} slots ({@code s = t * n_head + h -> h}),
     * pair rows {@code x_stride > in_dim} apart from {@code in_off}, outputs from {@code out_off}.
     */
    @Test
    public void groupedGemmMatchesThePerHeadMatVecIdOfTheMlaShape() {
        GGMLType type = GGMLType.Q8_0;
        int nHead = 4;
        int nTokens = 9;
        int inDim = 64;
        int outDim = 40;
        int xStride = 80;
        int inOff = 7;
        int outOff = 11;
        int rows = nTokens * nHead;
        int[] slots = new int[rows];
        for (int s = 0; s < rows; s++) {
            slots[s] = s % nHead;
        }
        GpuBuffer slotBuf = h.ints(slots);
        GpuBuffer weight = h.bytes(GroupedGemmHarness.weight(type, nHead * outDim, inDim, 3));
        GpuBuffer input = h.floats(GroupedGemmHarness.values(inOff + rows * xStride, 9, 2.0f));
        GpuBuffer ph = h.floats(new float[]{1.0f});
        int outLen = outOff + rows * outDim;
        GpuBuffer ref = h.out(outLen, -7.0f);
        GpuBuffer got = h.out(outLen, -7.0f);

        h.perRow(type, rows, outDim, inDim, xStride, inOff, outOff, nHead, slotBuf, weight, input, ph, ref);
        GroupedGemmHarness.Sorted s = h.sort(nTokens, nHead, nHead, slotBuf);
        h.grouped(MoeKernels.groupedGemm(type), s, outDim, inDim, 1, xStride, inOff, outOff, false, 0, nHead,
                weight, input, ph, got);
        h.finish();
        assertClose(h.readFloats(ref, outLen), h.readFloats(got, outLen), 1e-4f);
    }

    /**
     * The latent MLA absorption shape over rows whose byte phase alternates along an odd stride, for the block
     * formats whose activation readers use {@code nf_read_float4} (Q8_0 and Q4_0): {@code n} tokens of
     * {@code n_head} slots ({@code s = t * n_head + h -> h}), pair rows {@code x_stride = in_dim + 1} apart from an
     * odd {@code in_off} (some rows 16-byte aligned, the rest not), outputs from an odd {@code out_off}; the grouped
     * GEMM and the per-head {@code {T}_MatVecId} against the fp64 column dots.
     */
    @Test
    public void groupedGemmAndMatVecIdFollowTheMisaligningMlaRows() {
        for (GGMLType type : new GGMLType[]{GGMLType.Q8_0, GGMLType.Q4_0}) {
            int nHead = 4;
            int nTokens = 5;
            int inDim = 64;
            int outDim = 48;
            int xStride = inDim + 1;
            int inOff = 7;
            int outOff = 11;
            int rows = nTokens * nHead;
            int[] slots = new int[rows];
            for (int s = 0; s < rows; s++) {
                slots[s] = s % nHead;
            }
            GpuBuffer slotBuf = h.ints(slots);
            byte[] raw = GroupedGemmHarness.weight(type, nHead * outDim, inDim, 4);
            float[] x = GroupedGemmHarness.values(inOff + rows * xStride, 9, 2.0f);
            GpuBuffer weight = h.bytes(raw);
            GpuBuffer input = h.floats(x);
            GpuBuffer ph = h.floats(new float[]{1.0f});
            int outLen = outOff + rows * outDim;
            GpuBuffer ref = h.out(outLen, -7.0f);
            GpuBuffer got = h.out(outLen, -7.0f);

            h.perRow(type, rows, outDim, inDim, xStride, inOff, outOff, nHead, slotBuf, weight, input, ph, ref);
            GroupedGemmHarness.Sorted s = h.sort(nTokens, nHead, nHead, slotBuf);
            h.grouped(MoeKernels.groupedGemm(type), s, outDim, inDim, 1, xStride, inOff, outOff, false, 0, nHead,
                    weight, input, ph, got);
            h.finish();

            float[] w = GroupedGemmHarness.decode(type, raw, nHead * outDim, inDim);
            float[] matVec = h.readFloats(ref, outLen);
            float[] have = h.readFloats(got, outLen);
            for (int p = 0; p < rows; p++) {
                for (int r = 0; r < outDim; r++) {
                    double[] dot = dot(w, slots[p] * outDim + r, inDim, x, inOff + p * xStride);
                    int at = outOff + p * outDim + r;
                    assertNear("MatVecId " + type + " pair " + p + " row " + r, dot[0], matVec[at],
                            2e-5 * dot[1] + 1e-3);
                    assertNear("GroupedGemm " + type + " pair " + p + " row " + r, dot[0], have[at],
                            2e-5 * dot[1] + 1e-3);
                }
            }
        }
    }

    /** The MoVA value-expert shape: token rows ({@code x_div = k}), the expert tensor in two shards. */
    @Test
    public void groupedGemmMatchesTheBatchMatMulIdOfTheValueExpertShape() {
        GGMLType type = GGMLType.F16;
        int nExpert = 6;
        int k = 2;
        int nTokens = 21;
        int inDim = 48;
        int outDim = 70;
        int rows = nTokens * k;
        int[] idValues = GroupedGemmHarness.ids(rows, nExpert, 2, false);
        GpuBuffer ids = h.ints(idValues);
        byte[] raw = GroupedGemmHarness.weight(type, nExpert * outDim, inDim, 4);
        int half = raw.length / 2;
        GpuBuffer shard0 = h.bytes(Arrays.copyOfRange(raw, 0, half));
        GpuBuffer shard1 = h.bytes(Arrays.copyOfRange(raw, half, raw.length));
        GpuBuffer input = h.floats(GroupedGemmHarness.values(nTokens * inDim, 6, 1.5f));
        GpuBuffer ph = h.floats(new float[]{1.0f});
        GpuBuffer ref = h.out(rows * outDim, -7.0f);
        GpuBuffer got = h.out(rows * outDim, -7.0f);

        GroupedGemmHarness.Sorted s = h.sort(nTokens, k, nExpert, ids);
        int per = nExpert / 2;
        h.perPair(type, s, rows, outDim, inDim, k, false, 0, per, shard0, input, ph, ref);
        h.perPair(type, s, rows, outDim, inDim, k, false, per, per, shard1, input, ph, ref);
        h.grouped(MoeKernels.groupedGemm(type), s, outDim, inDim, k, inDim, 0, 0, false, 0, per, shard0, input, ph,
                got);
        h.grouped(MoeKernels.groupedGemm(type), s, outDim, inDim, k, inDim, 0, 0, false, per, per, shard1, input,
                ph, got);
        h.finish();
        assertClose(h.readFloats(ref, rows * outDim), h.readFloats(got, rows * outDim), 1e-4f);
    }

    /** Token rows ({@code x_div = k}, the gate/up shape) over F16 experts held by a cache bank. */
    @Test
    public void mappedGroupedGemmMatchesTheIndKernelOverTokenRows() {
        checkMapped(GGMLType.F16, true);
    }

    /** Pair rows ({@code x_div = 1}, the down shape) over Q8_0 experts held by a cache bank. */
    @Test
    public void mappedGroupedGemmMatchesTheIndKernelOverPairRows() {
        checkMapped(GGMLType.Q8_0, false);
    }

    @Test
    public void selectedGroupedKernelMatchesTheBatchMatMulIdOfShardedExperts() {
        for (GGMLType type : new GGMLType[]{GGMLType.F16, GGMLType.Q8_0}) {
            checkSelectedDirect(type);
        }
    }

    private static final int MAP_EXPERTS = 7;
    private static final int MAP_K = 2;
    private static final int MAP_TOKENS = 37;
    private static final int MAP_IN = 64;
    private static final int MAP_OUT = 70;
    /** The experts the bank holds, in slot order; the others are misses the host would compute. */
    private static final int[] CACHED = {5, 1, 3, 6};

    /**
     * The bank of {@link #CACHED} over random ids with out-of-range entries: the scalar {@code {T}_GroupedGemmInd},
     * the device's chosen map-form kernel and the per-pair {@code {T}_BatchMatMulIdInd} against fp64 products.
     */
    private void checkMapped(GGMLType type, boolean tokenRows) {
        int rows = MAP_TOKENS * MAP_K;
        int xDiv = tokenRows ? MAP_K : 1;
        int[] ids = GroupedGemmHarness.ids(rows, MAP_EXPERTS, 7, true);
        byte[] raw = GroupedGemmHarness.weight(type, MAP_EXPERTS * MAP_OUT, MAP_IN, 12);
        int perExpert = raw.length / MAP_EXPERTS;
        byte[] bank = new byte[CACHED.length * perExpert];
        int[] map = new int[MAP_EXPERTS];
        Arrays.fill(map, -1);
        for (int slot = 0; slot < CACHED.length; slot++) {
            System.arraycopy(raw, CACHED[slot] * perExpert, bank, slot * perExpert, perExpert);
            map[CACHED[slot]] = slot;
        }
        float[] x = GroupedGemmHarness.values((rows / xDiv) * MAP_IN, 4, 1.5f);
        float[] scales = {0.5f, 1.25f, 2.0f, 0.75f, 1.5f, 1.0f, 0.25f};
        GpuBuffer mapBuf = h.ints(map);
        GpuBuffer bankBuf = h.bytes(bank);
        GpuBuffer input = h.floats(x);
        GpuBuffer scaleBuf = h.floats(scales);
        String chosen = GpuGroupedRows.groupedKernel(h.ops(), type, true);
        String[] kernels = {MoeKernels.groupedGemmInd(type), chosen};
        GpuBuffer[] outs = {h.out(rows * MAP_OUT, -7.0f), h.out(rows * MAP_OUT, -7.0f)};
        GpuBuffer perPair = h.out(rows * MAP_OUT, -7.0f);

        GroupedGemmHarness.Sorted s = h.sort(MAP_TOKENS, MAP_K, MAP_EXPERTS, h.ints(ids));
        for (int i = 0; i < kernels.length; i++) {
            h.groupedInd(kernels[i], s, MAP_OUT, MAP_IN, xDiv, true, mapBuf, bankBuf, input, scaleBuf, outs[i]);
        }
        h.perPairInd(type, s, rows, MAP_OUT, MAP_IN, xDiv, true, mapBuf, bankBuf, input, scaleBuf, perPair);
        h.finish();

        float[] w = GroupedGemmHarness.decode(type, raw, MAP_EXPERTS * MAP_OUT, MAP_IN);
        float[] ref = h.readFloats(perPair, rows * MAP_OUT);
        for (int i = 0; i < kernels.length; i++) {
            float[] got = h.readFloats(outs[i], rows * MAP_OUT);
            for (int p = 0; p < rows; p++) {
                int e = ids[p];
                for (int r = 0; r < MAP_OUT; r++) {
                    int at = p * MAP_OUT + r;
                    String where = kernels[i] + " pair " + p + " (expert " + e + ") row " + r;
                    if (e < 0 || e >= MAP_EXPERTS) {
                        assertTrue(where + ": out-of-range ids write 0, got " + got[at], got[at] == 0.0f);
                    } else if (map[e] < 0) {
                        assertTrue(where + ": a miss stays the host's, got " + got[at], got[at] == -7.0f);
                    } else {
                        double[] dot = dot(w, e * MAP_OUT + r, MAP_IN, x, (p / xDiv) * MAP_IN);
                        double tol = 2e-5 * dot[1] * scales[e] + 1e-3;
                        assertNear(where, dot[0] * scales[e], got[at], tol);
                    }
                    assertNear(where + " against BatchMatMulIdInd", ref[at], got[at],
                            2e-5 * Math.abs(ref[at]) + 2e-3);
                }
            }
        }
    }

    /** The device's chosen direct grouped GEMM over two scaled shards and out-of-range ids. */
    private void checkSelectedDirect(GGMLType type) {
        int nExpert = 6;
        int k = 2;
        int nTokens = 41;
        int inDim = 96;
        int outDim = 70;
        int rows = nTokens * k;
        int[] idValues = GroupedGemmHarness.ids(rows, nExpert, 3, true);
        byte[] raw = GroupedGemmHarness.weight(type, nExpert * outDim, inDim, 9);
        int half = raw.length / 2;
        int per = nExpert / 2;
        GpuBuffer[] shards = {h.bytes(Arrays.copyOfRange(raw, 0, half)),
                h.bytes(Arrays.copyOfRange(raw, half, raw.length))};
        GpuBuffer input = h.floats(GroupedGemmHarness.values(nTokens * inDim, 5, 1.5f));
        GpuBuffer scales = h.floats(new float[]{0.5f, 1.25f, 2.0f, 0.75f, 1.5f, 1.0f});
        GpuBuffer ref = h.out(rows * outDim, -7.0f);
        GpuBuffer got = h.out(rows * outDim, -7.0f);
        String kernel = GpuGroupedRows.groupedKernel(h.ops(), type, false);

        GroupedGemmHarness.Sorted s = h.sort(nTokens, k, nExpert, h.ints(idValues));
        for (int i = 0; i < 2; i++) {
            h.perPair(type, s, rows, outDim, inDim, k, true, i * per, per, shards[i], input, scales, ref);
            h.grouped(kernel, s, outDim, inDim, k, inDim, 0, 0, true, i * per, per, shards[i], input, scales, got);
        }
        h.finish();
        float[] want = h.readFloats(ref, rows * outDim);
        float[] have = h.readFloats(got, rows * outDim);
        for (int i = 0; i < want.length; i++) {
            assertNear(kernel + " element " + i, want[i], have[i], 2e-5 * Math.abs(want[i]) + 2e-3);
        }
    }

    /** {@code {sum, sum of |terms|}} of weight row {@code row} against {@code x[xOff ..]}. */
    private static double[] dot(float[] w, int row, int inDim, float[] x, int xOff) {
        double acc = 0.0;
        double terms = 0.0;
        for (int i = 0; i < inDim; i++) {
            double t = (double) w[row * inDim + i] * x[xOff + i];
            acc += t;
            terms += Math.abs(t);
        }
        return new double[]{acc, terms};
    }

    private static void assertNear(String what, double expected, float actual, double tol) {
        assertTrue(what + ": expected " + expected + ", got " + actual, Math.abs(expected - actual) <= tol);
    }

    private static int[] padded(int[] used, int positions) {
        int[] out = Arrays.copyOf(used, positions);
        Arrays.fill(out, used.length, positions, -1);
        return out;
    }

    static void assertClose(float[] expected, float[] actual, float rel) {
        for (int i = 0; i < expected.length; i++) {
            float tol = rel * Math.max(1.0f, Math.abs(expected[i]));
            assertTrue("element " + i + ": expected " + expected[i] + ", got " + actual[i],
                    Math.abs(expected[i] - actual[i]) <= tol);
        }
    }
}
