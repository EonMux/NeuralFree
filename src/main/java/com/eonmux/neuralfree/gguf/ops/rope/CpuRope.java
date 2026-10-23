package com.eonmux.neuralfree.gguf.ops.rope;

import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * CPU twins of the RoPE family kernels, with the kernels' index math and preconditions ({@link RopeKernels}):
 * <ul>
 *   <li>{@link #apply} is {@code LLM_RoPE_Ext}: NORM, NEOX, MROPE, IMROPE and VISION with the full sector selection
 *       of {@code ggml_mrope_cache_init} (no NEOX shortcut for multi-section types: text tokens still rotate the
 *       e section by its own position, usually 0);</li>
 *   <li>{@code store*} are {@code LLM_KV_Store_*} ({@code ggml_set_rows});</li>
 *   <li>{@code shift*} are {@code LLM_KV_RopeShift_*} (llama-kv-cache.cpp:1945-1993).</li>
 * </ul>
 * Block-type caches (Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1) are {@link ByteBuffer}s of ggml blocks addressed with
 * absolute byte indices, (de)quantized by {@link KvBlockCodec} (the ggml reference quantizers). F16 and BF16 caches
 * are {@code short[]} half / bfloat16 bits converted with round-to-nearest-even.
 */
public final class CpuRope {

    /** Tokens or cells per parallel task. */
    private static final int GRAIN = 4;

    private CpuRope() {
    }

    // ------------------------------------------------------------------ LLM_RoPE_Ext

    /**
     * Rotates, in place, dims {@code [nOffs, nOffs + table.rotatedWidth())} of the {@code nHeads} heads of
     * {@code nTokens} rows: head {@code h} of token {@code t} starts at {@code x[xOff + t*rowStride + h*headStride]};
     * section {@code k} of token {@code t}'s position is {@code pos[posOff + t + k*posStride]} (single-section types
     * read {@code pos[posOff + t]}).
     *
     * @throws IllegalArgumentException when the geometry violates the kernel preconditions or a range lies outside
     *                                  an array
     */
    public static void apply(RopeFreqTable table, float[] x, int xOff, int nTokens, int nHeads, int headStride,
                             int rowStride, int nOffs, int[] pos, int posOff, int posStride) {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(pos, "pos");
        RopeKernels.validateRopeExt(table.type(), table.nDims(), nOffs, nTokens, nHeads, headStride, rowStride,
                posStride, xOff);
        if (nTokens == 0 || nHeads == 0) {
            return;
        }
        if (posOff < 0) {
            throw new IllegalArgumentException("pos offset must be >= 0, got " + posOff);
        }
        final long lastX = xOff + (long) (nTokens - 1) * rowStride + (long) (nHeads - 1) * headStride + nOffs
                + table.rotatedWidth();
        requireRange("x", lastX, x.length);
        final int sections = table.positionsPerToken();
        requireRange("pos", posOff + (long) (sections - 1) * posStride + nTokens, pos.length);

        final int pairs = table.pairs();
        final int span = table.pairSpan();
        CpuParallel.parallelFor(nTokens, GRAIN, (from, to) -> {
            final float[] cs = new float[2 * pairs];
            for (int t = from; t < to; t++) {
                table.cosSin(pos, posOff, t, posStride, cs);
                for (int h = 0; h < nHeads; h++) {
                    rotate(table, cs, pairs, span, x, xOff + t * rowStride + h * headStride + nOffs);
                }
            }
        });
    }

    // ------------------------------------------------------------------ LLM_KV_Store_*

    /** {@code LLM_KV_Store_F32}: {@code dst[cellIdx[t]*dstRowStride + e] = src[srcOff + t*srcRowStride + e]}. */
    public static void storeF32(float[] src, int srcOff, int srcRowStride, int nTokens, int rowElems, int[] cellIdx,
                                float[] dst, int dstRowStride) {
        Objects.requireNonNull(dst, "dst");
        checkStore(KvDType.F32, src, srcOff, srcRowStride, nTokens, rowElems, cellIdx, dstRowStride);
        for (int t = 0; t < nTokens; t++) {
            final int cell = cellIdx[t];
            if (cell < 0) {
                continue;
            }
            requireRange("dst", (long) cell * dstRowStride + rowElems, dst.length);
            System.arraycopy(src, srcOff + t * srcRowStride, dst, cell * dstRowStride, rowElems);
        }
    }

    /** {@code LLM_KV_Store_F16}: as {@link #storeF32} with each value rounded to half (nearest even). */
    public static void storeF16(float[] src, int srcOff, int srcRowStride, int nTokens, int rowElems, int[] cellIdx,
                                short[] dst, int dstRowStride) {
        Objects.requireNonNull(dst, "dst");
        checkStore(KvDType.F16, src, srcOff, srcRowStride, nTokens, rowElems, cellIdx, dstRowStride);
        for (int t = 0; t < nTokens; t++) {
            final int cell = cellIdx[t];
            if (cell < 0) {
                continue;
            }
            requireRange("dst", (long) cell * dstRowStride + rowElems, dst.length);
            final int s = srcOff + t * srcRowStride;
            final int d = cell * dstRowStride;
            for (int e = 0; e < rowElems; e++) {
                dst[d + e] = (short) FpBlockDecoders.fp32ToFp16(src[s + e]);
            }
        }
    }

    /** {@code LLM_KV_Store_BF16}: as {@link #storeF32} with each value rounded to bfloat16 (nearest even). */
    public static void storeBf16(float[] src, int srcOff, int srcRowStride, int nTokens, int rowElems, int[] cellIdx,
                                 short[] dst, int dstRowStride) {
        Objects.requireNonNull(dst, "dst");
        checkStore(KvDType.BF16, src, srcOff, srcRowStride, nTokens, rowElems, cellIdx, dstRowStride);
        for (int t = 0; t < nTokens; t++) {
            final int cell = cellIdx[t];
            if (cell < 0) {
                continue;
            }
            requireRange("dst", (long) cell * dstRowStride + rowElems, dst.length);
            final int s = srcOff + t * srcRowStride;
            final int d = cell * dstRowStride;
            for (int e = 0; e < rowElems; e++) {
                dst[d + e] = (short) KvBlockCodec.fp32ToBf16(src[s + e]);
            }
        }
    }

    /**
     * {@code LLM_KV_Store_<block type>}: each 32-value block of a source row quantized into cache row
     * {@code cellIdx[t]}.
     *
     * @throws IllegalArgumentException when {@code type} is not a block type or the geometry is invalid
     */
    public static void storeBlocks(KvDType type, float[] src, int srcOff, int srcRowStride, int nTokens, int rowElems,
                                   int[] cellIdx, ByteBuffer dst, int dstRowStride) {
        Objects.requireNonNull(dst, "dst");
        requireBlockType(type);
        checkStore(type, src, srcOff, srcRowStride, nTokens, rowElems, cellIdx, dstRowStride);
        final int bs = type.blockSize();
        final int bb = type.blockBytes();
        final int nb = rowElems / bs;
        final int rowBlocks = dstRowStride / bs;
        for (int t = 0; t < nTokens; t++) {
            final int cell = cellIdx[t];
            if (cell < 0) {
                continue;
            }
            requireRange("dst", ((long) cell * rowBlocks + nb) * bb, dst.limit());
            for (int b = 0; b < nb; b++) {
                KvBlockCodec.quantize(type, src, srcOff + t * srcRowStride + b * bs, dst, (cell * rowBlocks + b) * bb);
            }
        }
    }

    // ------------------------------------------------------------------ LLM_KV_RopeShift_*

    /**
     * {@code LLM_KV_RopeShift_F32}: for every cell {@code c}, rotates dims {@code [nOffs, nOffs + nDims)} of each
     * head of row {@code c} by RoPE at {@code delta[c]}. A cell with delta 0 is rotated too, as llama.cpp shifts every
     * cell with its per-cell shift (llama-kv-cache.cpp:1545-1556): the identity rotation scaled by the YaRN magnitude.
     *
     * @param shiftTable a table from {@link RopeFreqTable#forShift} (NORM or NEOX)
     */
    public static void shiftF32(RopeFreqTable shiftTable, float[] cache, int nCells, int nHeads, int headStride,
                                int rowStride, int nOffs, int[] delta) {
        Objects.requireNonNull(cache, "cache");
        checkShift(KvDType.F32, shiftTable, nCells, nHeads, headStride, rowStride, nOffs, 0, delta);
        requireRange("cache", (long) nCells * rowStride, cache.length);
        final int pairs = shiftTable.pairs();
        final int span = shiftTable.pairSpan();
        CpuParallel.parallelFor(nCells, GRAIN, (from, to) -> {
            final float[] cs = new float[2 * pairs];
            for (int c = from; c < to; c++) {
                shiftTable.cosSin(delta[c], cs);
                for (int h = 0; h < nHeads; h++) {
                    rotate(shiftTable, cs, pairs, span, cache, c * rowStride + h * headStride + nOffs);
                }
            }
        });
    }

    /** {@code LLM_KV_RopeShift_F16}: {@link #shiftF32} on half values, each result rounded to half (nearest even). */
    public static void shiftF16(RopeFreqTable shiftTable, short[] cache, int nCells, int nHeads, int headStride,
                                int rowStride, int nOffs, int[] delta) {
        Objects.requireNonNull(cache, "cache");
        checkShift(KvDType.F16, shiftTable, nCells, nHeads, headStride, rowStride, nOffs, 0, delta);
        requireRange("cache", (long) nCells * rowStride, cache.length);
        final int pairs = shiftTable.pairs();
        final int span = shiftTable.pairSpan();
        CpuParallel.parallelFor(nCells, GRAIN, (from, to) -> {
            final float[] cs = new float[2 * pairs];
            for (int c = from; c < to; c++) {
                shiftTable.cosSin(delta[c], cs);
                for (int h = 0; h < nHeads; h++) {
                    final int base = c * rowStride + h * headStride + nOffs;
                    for (int i = 0; i < pairs; i++) {
                        final int a = base + shiftTable.pairFirst(i);
                        final int b = a + span;
                        final float x0 = FpBlockDecoders.fp16ToFp32(cache[a]);
                        final float x1 = FpBlockDecoders.fp16ToFp32(cache[b]);
                        cache[a] = (short) FpBlockDecoders.fp32ToFp16(x0 * cs[2 * i] - x1 * cs[2 * i + 1]);
                        cache[b] = (short) FpBlockDecoders.fp32ToFp16(x0 * cs[2 * i + 1] + x1 * cs[2 * i]);
                    }
                }
            }
        });
    }

    /** {@code LLM_KV_RopeShift_BF16}: {@link #shiftF32} on bfloat16 values, each result rounded to nearest even. */
    public static void shiftBf16(RopeFreqTable shiftTable, short[] cache, int nCells, int nHeads, int headStride,
                                 int rowStride, int nOffs, int[] delta) {
        Objects.requireNonNull(cache, "cache");
        checkShift(KvDType.BF16, shiftTable, nCells, nHeads, headStride, rowStride, nOffs, 0, delta);
        requireRange("cache", (long) nCells * rowStride, cache.length);
        final int pairs = shiftTable.pairs();
        final int span = shiftTable.pairSpan();
        CpuParallel.parallelFor(nCells, GRAIN, (from, to) -> {
            final float[] cs = new float[2 * pairs];
            for (int c = from; c < to; c++) {
                shiftTable.cosSin(delta[c], cs);
                for (int h = 0; h < nHeads; h++) {
                    final int base = c * rowStride + h * headStride + nOffs;
                    for (int i = 0; i < pairs; i++) {
                        final int a = base + shiftTable.pairFirst(i);
                        final int b = a + span;
                        final float x0 = FpBlockDecoders.bf16ToFp32(cache[a]);
                        final float x1 = FpBlockDecoders.bf16ToFp32(cache[b]);
                        cache[a] = (short) KvBlockCodec.fp32ToBf16(x0 * cs[2 * i] - x1 * cs[2 * i + 1]);
                        cache[b] = (short) KvBlockCodec.fp32ToBf16(x0 * cs[2 * i + 1] + x1 * cs[2 * i]);
                    }
                }
            }
        });
    }

    /**
     * {@code LLM_KV_RopeShift_<block type>}: per (cell, head), dequantizes the
     * {@link RopeKernels#shiftWindowStart work window}, undoes the Hadamard rotation of each {@code rotK} chunk when
     * {@code rotK > 1}, rotates the dims {@code [nOffs, nOffs + nDims)}, re-applies the Hadamard rotation and
     * requantizes each block of the window; blocks outside the window keep their bytes.
     *
     * @throws IllegalArgumentException when {@code type} is not a block type or the geometry is invalid
     */
    public static void shiftBlocks(KvDType type, RopeFreqTable shiftTable, ByteBuffer cache, int nCells, int nHeads,
                                   int headStride, int rowStride, int nOffs, int rotK, int[] delta) {
        Objects.requireNonNull(cache, "cache");
        requireBlockType(type);
        checkShift(type, shiftTable, nCells, nHeads, headStride, rowStride, nOffs, rotK, delta);
        final int bs = type.blockSize();
        final int bb = type.blockBytes();
        requireRange("cache", (long) nCells * (rowStride / bs) * bb, cache.limit());
        final int w0 = RopeKernels.shiftWindowStart(nOffs, rotK);
        final int width = RopeKernels.shiftWindowEnd(shiftTable.nDims(), nOffs, rotK) - w0;
        final int r0 = nOffs - w0;
        final int pairs = shiftTable.pairs();
        final int span = shiftTable.pairSpan();
        final int nblk = width / bs;
        CpuParallel.parallelFor(nCells, GRAIN, (from, to) -> {
            final float[] cs = new float[2 * pairs];
            final float[] buf = new float[width];
            for (int c = from; c < to; c++) {
                shiftTable.cosSin(delta[c], cs);
                for (int h = 0; h < nHeads; h++) {
                    final int blk0 = (c * rowStride + h * headStride + w0) / bs;
                    for (int k = 0; k < nblk; k++) {
                        KvBlockCodec.dequantize(type, cache, (blk0 + k) * bb, buf, k * bs);
                    }
                    hadamardChunks(buf, width, rotK);
                    rotate(shiftTable, cs, pairs, span, buf, r0);
                    hadamardChunks(buf, width, rotK);
                    for (int k = 0; k < nblk; k++) {
                        KvBlockCodec.quantize(type, buf, k * bs, cache, (blk0 + k) * bb);
                    }
                }
            }
        });
    }

    // ------------------------------------------------------------------ helpers

    /** Rotates the pairs of one head whose first rotated dim is {@code v[base]} with the row {@code cs}. */
    private static void rotate(RopeFreqTable table, float[] cs, int pairs, int span, float[] v, int base) {
        for (int i = 0; i < pairs; i++) {
            final int a = base + table.pairFirst(i);
            final int b = a + span;
            final float x0 = v[a];
            final float x1 = v[b];
            v[a] = x0 * cs[2 * i] - x1 * cs[2 * i + 1];
            v[b] = x0 * cs[2 * i + 1] + x1 * cs[2 * i];
        }
    }

    /** Normalized FWHT of each {@code rotK} chunk of {@code v[0, n)} (no-op for {@code rotK <= 1}). */
    private static void hadamardChunks(float[] v, int n, int rotK) {
        if (rotK <= 1) {
            return;
        }
        for (int c = 0; c < n; c += rotK) {
            CpuHadamard.fwht(v, c, rotK);
        }
    }

    private static void requireBlockType(KvDType type) {
        if (!Objects.requireNonNull(type, "type").quantized()) {
            throw new IllegalArgumentException(type + " is not a KV block type");
        }
    }

    private static void checkStore(KvDType type, float[] src, int srcOff, int srcRowStride, int nTokens,
                                   int rowElems, int[] cellIdx, int dstRowStride) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(cellIdx, "cellIdx");
        RopeKernels.validateKvStore(type, nTokens, rowElems, srcRowStride, srcOff, dstRowStride);
        if (nTokens > cellIdx.length) {
            throw new IllegalArgumentException("cellIdx holds " + cellIdx.length + " cells, need " + nTokens);
        }
        if (nTokens > 0) {
            requireRange("src", srcOff + (long) (nTokens - 1) * srcRowStride + rowElems, src.length);
        }
    }

    private static void checkShift(KvDType type, RopeFreqTable table, int nCells, int nHeads, int headStride,
                                   int rowStride, int nOffs, int rotK, int[] delta) {
        Objects.requireNonNull(table, "shiftTable");
        Objects.requireNonNull(delta, "delta");
        if (table.type() != RopeType.NORM && table.type() != RopeType.NEOX) {
            throw new IllegalArgumentException("a K-cache shift needs a NORM or NEOX table (RopeFreqTable.forShift),"
                    + " got " + table.type());
        }
        RopeKernels.validateKvShift(type, table.nDims(), nOffs, nCells, nHeads, headStride, rowStride, rotK);
        if (nCells > delta.length) {
            throw new IllegalArgumentException("delta holds " + delta.length + " cells, need " + nCells);
        }
    }

    private static void requireRange(String what, long end, int length) {
        if (end > length) {
            throw new IllegalArgumentException(what + " needs " + end + " elements, holds " + length);
        }
    }
}
