package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.tensor.HalfFloats;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Device fixtures of the grouped-GEMM GPU parity tests: buffer upload and read-back, raw kernel dispatches, the
 * expert sort ({@code LLM_MoE_Bucketize} + {@code LLM_MoE_AlignBlocks}), the per-pair reference kernels
 * ({@code {T}_BatchMatMulId}, {@code {T}_BatchMatMulIdInd}, {@code {T}_MatVecId}, {@code {T}_MatVecIdT}) and the
 * grouped GEMM over one backend, a {@link GpuOps} for the bind-time kernel choice, plus small synthetic expert
 * weights (F16, BF16, Q4_0, Q8_0) and the CPU decoder of their values.
 */
final class GroupedGemmHarness implements AutoCloseable {

    /** Work-group size of the sort, align and id-matvec dispatches. */
    static final int LOCAL = 64;

    private final ComputeBackend backend;
    private final ComputeQueue queue;
    private final LlmKernels kernels;
    private final List<GpuBuffer> buffers = new ArrayList<>();
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;

    GroupedGemmHarness(ComputeBackend backend, LlmKernels kernels) {
        this.backend = backend;
        this.queue = backend.computeQueue();
        this.kernels = kernels;
    }

    LlmKernels kernels() {
        return kernels;
    }

    /** The device's {@link GpuOps}, opened on first use; the grouped kernel choice reads it. */
    GpuOps ops() {
        if (ops == null) {
            scratch = GpuScratch.builder(backend, 1).build();
            placeholders = GpuPlaceholders.allocate(backend, queue);
            ops = new GpuOps(queue, kernels, scratch, placeholders);
        }
        return ops;
    }

    GpuBuffer ints(int[] values) {
        ByteBuffer bb = ByteBuffer.allocate(4 * values.length).order(ByteOrder.LITTLE_ENDIAN);
        bb.asIntBuffer().put(values);
        return bytes(bb.array());
    }

    GpuBuffer floats(float[] values) {
        GpuBuffer b = alloc(4L * values.length, true);
        close(queue.enqueueWriteBuffer(b, 0, 4L * values.length, values));
        return b;
    }

    GpuBuffer bytes(byte[] values) {
        GpuBuffer b = alloc(values.length, true);
        close(queue.enqueueWriteBuffer(b, 0, values.length, values));
        return b;
    }

    /** A writable buffer of {@code count} 32-bit elements, filled with {@code fill} bits. */
    GpuBuffer out(int count, float fill) {
        float[] init = new float[count];
        Arrays.fill(init, fill);
        GpuBuffer b = alloc(4L * count, false);
        close(queue.enqueueWriteBuffer(b, 0, 4L * count, init));
        return b;
    }

    int[] readInts(GpuBuffer b, int count) {
        byte[] raw = new byte[4 * count];
        GpuEvent e = queue.enqueueReadBuffer(b, 0, raw.length, raw);
        queue.finish();
        close(e);
        int[] out = new int[count];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(out);
        return out;
    }

    float[] readFloats(GpuBuffer b, int count) {
        float[] out = new float[count];
        GpuEvent e = queue.enqueueReadBuffer(b, 0, 4L * count, out);
        queue.finish();
        close(e);
        return out;
    }

    void dispatch(String kernel, long[] global, long[] local, Object... args) {
        close(queue.enqueueKernel(kernels.get(kernel), global, local, args));
    }

    void finish() {
        queue.finish();
    }

    /** The sorted rows and their aligned layout of one step. */
    static final class Sorted {
        final GpuBuffer counts;
        final GpuBuffer offsets;
        final GpuBuffer rowToken;
        final GpuBuffer rowSlot;
        final GpuBuffer gemmToken;
        final GpuBuffer gemmSlot;
        final int tiles;

        Sorted(GpuBuffer counts, GpuBuffer offsets, GpuBuffer rowToken, GpuBuffer rowSlot, GpuBuffer gemmToken,
               GpuBuffer gemmSlot, int tiles) {
            this.counts = counts;
            this.offsets = offsets;
            this.rowToken = rowToken;
            this.rowSlot = rowSlot;
            this.gemmToken = gemmToken;
            this.gemmSlot = gemmSlot;
            this.tiles = tiles;
        }
    }

    /** {@code LLM_MoE_Bucketize} then {@code LLM_MoE_AlignBlocks} (multi-work-group grid) over {@code ids}. */
    Sorted sort(int nTokens, int k, int nExpert, GpuBuffer ids) {
        int rows = nTokens * k;
        int tiles = MoeKernels.groupedTiles(rows, nExpert);
        int positions = tiles * MoeKernels.ALIGN_TILE;
        Sorted s = new Sorted(out(nExpert, 0), out(nExpert, 0), out(rows, 0), out(rows, 0), out(positions, 0),
                out(positions, 0), tiles);
        long[] group = {LOCAL};
        dispatch(MoeKernels.BUCKETIZE, group, group, MoeKernels.bucketizeArgs(nTokens, k, nExpert, ids, s.counts,
                s.offsets, s.rowToken, s.rowSlot));
        dispatch(MoeKernels.ALIGN_BLOCKS, MoeKernels.alignGlobal(tiles, LOCAL), group,
                MoeKernels.alignBlocksArgs(nTokens, k, nExpert, tiles, s.counts, s.offsets, s.rowToken, s.rowSlot,
                        s.gemmToken, s.gemmSlot));
        return s;
    }

    /** One grouped GEMM dispatch ({@code {T}_GroupedGemm} or {@code {T}_GroupedGemmMma}) into {@code output}. */
    void grouped(String kernel, Sorted s, int outDim, int inDim, int xDiv, int xStride, int inOff, int outOff,
                 boolean scaled, int expertBase, int expertCount, GpuBuffer weight, GpuBuffer input,
                 GpuBuffer scales, GpuBuffer output) {
        dispatch(kernel, MoeKernels.groupedGemmGlobal(outDim, s.tiles), MoeKernels.groupedGemmLocal(),
                MoeKernels.groupedGemmArgs(outDim, inDim, s.tiles, xDiv, xStride, inOff, outOff, scaled, expertBase,
                        expertCount, s.gemmToken, s.gemmSlot, weight, input, scales, output));
    }

    /** The per-pair reference {@code {T}_BatchMatMulId} over the sorted rows into {@code output}. */
    void perPair(GGMLType type, Sorted s, int nRows, int outDim, int inDim, int xDiv, boolean scaled, int expertBase,
                 int expertCount, GpuBuffer weight, GpuBuffer input, GpuBuffer scales, GpuBuffer output) {
        dispatch(MoeKernels.batchMatMulId(type), MoeKernels.batchMatMulIdGlobal(outDim, nRows, LOCAL),
                new long[]{LOCAL, 1}, MoeKernels.batchMatMulIdArgs(outDim, inDim, nRows, xDiv, 0, scaled, expertBase,
                        expertCount, s.rowToken, s.rowSlot, weight, input, scales, output));
    }

    /** The per-pair reference {@code {T}_BatchMatMulIdInd} over the sorted rows and a cache bank's map. */
    void perPairInd(GGMLType type, Sorted s, int nRows, int outDim, int inDim, int xDiv, boolean scaled,
                    GpuBuffer map, GpuBuffer bank, GpuBuffer input, GpuBuffer scales, GpuBuffer output) {
        dispatch(MoeKernels.batchMatMulIdInd(type), MoeKernels.batchMatMulIdGlobal(outDim, nRows, LOCAL),
                new long[]{LOCAL, 1}, MoeKernels.batchMatMulIdIndArgs(outDim, inDim, nRows, xDiv, 0, scaled,
                        s.rowToken, s.rowSlot, map, bank, input, scales, output));
    }

    /** One map-form grouped GEMM dispatch ({@code {T}_GroupedGemmInd} or {@code {T}_GroupedGemmMmaInd}). */
    void groupedInd(String kernel, Sorted s, int outDim, int inDim, int xDiv, boolean scaled, GpuBuffer map,
                    GpuBuffer bank, GpuBuffer input, GpuBuffer scales, GpuBuffer output) {
        dispatch(kernel, MoeKernels.groupedGemmGlobal(outDim, s.tiles), MoeKernels.groupedGemmLocal(),
                MoeKernels.groupedGemmIndArgs(outDim, inDim, s.tiles, xDiv, inDim, 0, 0, scaled, s.gemmToken,
                        s.gemmSlot, map, bank, input, scales, output));
    }

    /** The per-row reference {@code {T}_MatVecId}: slot {@code s} multiplies expert {@code slots[s]}. */
    void perRow(GGMLType type, int nSel, int outDim, int inDim, int xStride, int inOff, int outOff, int nExpert,
                GpuBuffer slots, GpuBuffer weight, GpuBuffer input, GpuBuffer scales, GpuBuffer output) {
        dispatch(MoeKernels.matVecId(type), MoeKernels.idMatVecGlobal(outDim, nSel, LOCAL), new long[]{LOCAL, 1},
                MoeKernels.matVecIdArgs(outDim, inDim, nSel, xStride, inOff, outOff, false, 0, nExpert, slots,
                        weight, input, scales, output));
    }

    /** The per-row transposed read {@code {T}_MatVecIdT} with the same arguments. */
    void perRowT(GGMLType type, int nSel, int outDim, int inDim, int xStride, int inOff, int outOff, int nExpert,
                 GpuBuffer slots, GpuBuffer weight, GpuBuffer input, GpuBuffer scales, GpuBuffer output) {
        dispatch(MoeKernels.matVecIdT(type), MoeKernels.idMatVecGlobal(outDim, nSel, LOCAL), new long[]{LOCAL, 1},
                MoeKernels.matVecIdArgs(outDim, inDim, nSel, xStride, inOff, outOff, false, 0, nExpert, slots,
                        weight, input, scales, output));
    }

    private GpuBuffer alloc(long bytes, boolean readOnly) {
        GpuBuffer b = backend.allocateBuffer(Math.max(4L, bytes), readOnly);
        buffers.add(b);
        return b;
    }

    private static void close(GpuEvent e) {
        if (e != null) {
            e.close();
        }
    }

    @Override
    public void close() {
        for (AutoCloseable c : new AutoCloseable[]{ops, placeholders, scratch}) {
            if (c != null) {
                try {
                    c.close();
                } catch (Exception ignored) {
                    // the backend closes what remains
                }
            }
        }
        ops = null;
        for (GpuBuffer b : buffers) {
            try {
                b.close();
            } catch (RuntimeException ignored) {
                // the backend closes what remains
            }
        }
        buffers.clear();
    }

    // ================================================================== synthetic data

    /** Deterministic values in {@code [-amp, amp)} on a 1/64 grid (exact in fp16 for amp <= 32). */
    static float[] values(int n, long seed, float amp) {
        float[] v = new float[n];
        long x = seed * 0x9E3779B97F4A7C15L + 1;
        for (int i = 0; i < n; i++) {
            x ^= x << 13;
            x ^= x >>> 7;
            x ^= x << 17;
            int q = (int) ((x >>> 20) & 127) - 64;
            v[i] = amp * q / 64.0f;
        }
        return v;
    }

    /** Expert ids of {@code rows} pairs over {@code nExpert} experts, with {@code bad} ids outside the range. */
    static int[] ids(int rows, int nExpert, long seed, boolean bad) {
        int[] ids = new int[rows];
        for (int p = 0; p < rows; p++) {
            int v = (int) ((p * 7 + seed * 13 + (p * p) % 5) % (nExpert + (bad ? 2 : 0)));
            ids[p] = bad && v >= nExpert ? (v == nExpert ? -1 : nExpert + 3) : v;
        }
        return ids;
    }

    /** Raw weight bytes of {@code rows x cols} values of {@code type} (F16, BF16, Q4_0 or Q8_0) built from
     * {@code seed}. */
    static byte[] weight(GGMLType type, int rows, int cols, long seed) {
        float[] v = values(rows * cols, seed, 1.0f);
        ByteBuffer b;
        switch (type) {
            case F16:
                b = ByteBuffer.allocate(2 * v.length).order(ByteOrder.LITTLE_ENDIAN);
                for (float f : v) {
                    b.putShort((short) HalfFloats.floatToHalf(f));
                }
                return b.array();
            case BF16:
                b = ByteBuffer.allocate(2 * v.length).order(ByteOrder.LITTLE_ENDIAN);
                for (float f : v) {
                    b.putShort((short) (Float.floatToRawIntBits(f) >>> 16));
                }
                return b.array();
            case Q4_0:
                if (cols % 32 != 0) {
                    throw new IllegalArgumentException("Q4_0 rows need a multiple of 32 columns, got " + cols);
                }
                b = ByteBuffer.allocate(rows * (cols / 32) * 18).order(ByteOrder.LITTLE_ENDIAN);
                for (int blk = 0; blk < rows * cols / 32; blk++) {
                    float amax = 0.0f;
                    for (int e = 0; e < 32; e++) {
                        amax = Math.max(amax, Math.abs(v[blk * 32 + e]));
                    }
                    final float d = amax == 0.0f ? 0.0078125f : amax / 8;
                    b.putShort((short) HalfFloats.floatToHalf(d));
                    for (int e = 0; e < 16; e++) {
                        b.put((byte) (q4Code(v[blk * 32 + e], d) | (q4Code(v[blk * 32 + e + 16], d) << 4)));
                    }
                }
                return b.array();
            case Q8_0:
                if (cols % 32 != 0) {
                    throw new IllegalArgumentException("Q8_0 rows need a multiple of 32 columns, got " + cols);
                }
                b = ByteBuffer.allocate(rows * (cols / 32) * 34).order(ByteOrder.LITTLE_ENDIAN);
                for (int blk = 0; blk < rows * cols / 32; blk++) {
                    b.putShort((short) HalfFloats.floatToHalf(0.0078125f * (1 + blk % 5)));
                    for (int e = 0; e < 32; e++) {
                        b.put((byte) Math.round(v[blk * 32 + e] * 100));
                    }
                }
                return b.array();
            default:
                throw new IllegalArgumentException("no synthetic weights of type " + type);
        }
    }

    /** The 4-bit code of {@code v} under block scale {@code d}: {@code clamp(round(v / d), -8, 7) + 8}. */
    private static int q4Code(float v, float d) {
        return Math.max(-8, Math.min(7, Math.round(v / d))) + 8;
    }

    /** The decoded values of {@link #weight}. */
    static float[] decode(GGMLType type, byte[] raw, int rows, int cols) {
        return CpuBlockDecoders.dequantize(raw, type, (long) rows * cols);
    }
}
