package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * The conversion of NeuralFree extension tensors ({@link GGMLType#isExtension()}) to the llama.cpp types a
 * COMPAT export writes, and the streaming conversion of their bytes. The rules are those of the export:
 * <ul>
 *   <li>F8_E4M3 → F16, exact; the tensor's {@code .scale} sibling is a separate tensor and stays.</li>
 *   <li>MXFP8 → BF16 when every block's E8M0 exponent satisfies {@code e ∈ [3, 246]}, so that
 *       {@code 2^(e-127) × [2^-9, 448]} lies in the bf16 normal range exactly; some {@code e ∈ [0, 2]}
 *       needs a subnormal target, so the tensor becomes F32. Some {@code e ≥ 247} overflows f32 too, so
 *       the export fails with an {@link IOException} naming the tensor. The exponent bytes are scanned
 *       once from the store.</li>
 *   <li>F8_E4M3_B128 and I8_B128 → Q8_0 (lossy: a per-32 requantization of a per-128 scale; the scale is
 *       rounded to fp16 by the Q8_0 format itself).</li>
 *   <li>NF4 and FP4_BNB → F16 (lossy: the codebook values round).</li>
 * </ul>
 * In {@link GgufWriter.Mode#COMPAT_EXACT} the lossy rows become F32 instead; only the MXFP8 rows are
 * shared with COMPAT, because their conversion is bit-exact either way. Decoding goes through
 * {@link CpuBlockDecoders}, which serves every storage type, extension and ggml alike, and the Q8_0
 * requantization through {@link KvBlockCodec#quantize(KvDType, float[], int, byte[], int)}.
 * <p>
 * Converted bytes are written in one sequential pass in chunks that bound the read buffer and the float
 * window to well below 64 MiB, so no tensor is ever held in the heap whole. Not thread-safe.
 */
public final class GgufCompatConverter {

    /** Largest MXFP8 E8M0 exponent whose block range fits bf16 (and f32) as a normal number. */
    public static final int MAX_FITTING_EXPONENT = 246;

    /** Largest float window of one conversion chunk, in elements (16 Mi floats). */
    private static final int CHUNK_ELEMENTS = 1 << 24;

    /** Largest byte chunk of the read and write buffers (64 MiB). */
    private static final int CHUNK_BYTES = 1 << 26;

    private GgufCompatConverter() {
    }

    /**
     * Target ggml type of {@code t} in {@code mode}; the type itself when the tensor carries no extension
     * type and so needs no conversion.
     *
     * @throws IOException naming the tensor for a tensor no llama.cpp consumer reads or for an MXFP8
     *                     exponent byte the COMPAT types cannot represent
     */
    public static GGMLType targetType(GGUFTensorInfo t, WeightStore store, GgufWriter.Mode mode)
            throws IOException {
        GGMLType from = t.type();
        if (!from.isExtension()) {
            return from;
        }
        checkNoConsumer(t);
        switch (from) {
            case F8_E4M3:
                return GGMLType.F16;
            case MXFP8:
                return mxfp8Target(t, store);
            case F8_E4M3_B128:
            case I8_B128:
                if (mode == GgufWriter.Mode.COMPAT_EXACT) {
                    return GGMLType.F32;
                }
                return GGMLType.Q8_0;
            case NF4:
            case FP4_BNB:
                if (mode == GgufWriter.Mode.COMPAT_EXACT) {
                    return GGMLType.F32;
                }
                return GGMLType.F16;
            default:
                throw new IOException("tensor '" + t.name() + "' of extension type " + from
                        + " has no llama.cpp conversion");
        }
    }

    /** True for the lossy or approximate COMPAT rows that {@code COMPAT_EXACT} exports as F32. */
    public static boolean isLossy(GGMLType from, GGMLType to) {
        switch (from) {
            case F8_E4M3_B128:
            case I8_B128:
            case NF4:
            case FP4_BNB:
                return to == GGMLType.Q8_0 || to == GGMLType.F16;
            default:
                return false;
        }
    }

    /**
     * Fails for {@code .input_perm} and {@code .input_zero_point} tensors: llama.cpp has no consumer for a
     * NeuralFree input permutation, so the export must either drop the tensor or stay NATIVE.
     */
    private static void checkNoConsumer(GGUFTensorInfo t) throws IOException {
        String name = t.name();
        if (name.endsWith(".input_perm") || name.endsWith(".input_zero_point")) {
            throw new IOException("tensor '" + name + "' is a NeuralFree input permutation, for which "
                    + "llama.cpp has no consumer: export with mode NATIVE instead of COMPAT");
        }
    }

    /** Exponent byte of MXFP8 block {@code i} (values {@code ne / 32}) in raw tensor bytes. */
    public static int mxfp8ExponentAt(byte[] raw, int block) {
        return raw[block * BlockLayout.MXFP8_BYTES] & 0xFF;
    }

    /**
     * Scans the E8M0 exponent bytes of an MXFP8 tensor once and selects its target type.
     *
     * @throws IOException naming the tensor when any exponent is {@code ≥ 247} (including the reserved
     *                     NaN byte 0xFF), whose value overflows f32 on decode
     */
    private static GGMLType mxfp8Target(GGUFTensorInfo t, WeightStore store) throws IOException {
        BlockLayout layout = BlockLayout.of(GGMLType.MXFP8);
        long blocks = layout.blocksForElements(t.numElements());
        int chunkBlocks = chunkBlocks(GGMLType.MXFP8, GGMLType.F32);
        byte[] raw = new byte[Math.multiplyExact(chunkBlocks, layout.blockBytes())];
        long fitting = 0;
        long readAt = 0;
        while (readAt < blocks) {
            int n = (int) Math.min(chunkBlocks, blocks - readAt);
            int bytes = Math.multiplyExact(n, layout.blockBytes());
            store.readTensor(t.offset() + Math.multiplyExact(readAt, layout.blockBytes()), bytes, raw, 0);
            for (int i = 0; i < n; i++) {
                int e = mxfp8ExponentAt(raw, i);
                if (e > MAX_FITTING_EXPONENT) {
                    throw new IOException("tensor '" + t.name() + "' of MXFP8 blocks has exponent byte " + e
                            + " (the value 2^(e-127) overflows f32) in block " + (readAt + i)
                            + "; a COMPAT export cannot represent it");
                }
                if (e > 2) {
                    fitting++;
                }
            }
            readAt += n;
        }
        return fitting == blocks ? GGMLType.BF16 : GGMLType.F32;
    }

    /**
     * Streams {@code src}'s data through {@code store}, converts it to {@code dst}'s type and writes it to
     * {@code enc}. The two tensor infos must share name and element count; whether the conversion is exact
     * is the caller's {@link #isLossy} note, reported in its log.
     */
    public static void convert(GGUFTensorInfo src, GGUFTensorInfo dst, WeightStore store, GgufEncoder enc)
            throws IOException {
        Objects.requireNonNull(store, "store");
        Objects.requireNonNull(enc, "enc");
        GGMLType from = src.type();
        GGMLType to = dst.type();
        if (!src.name().equals(dst.name())) {
            throw new IOException("tensor conversion renames '" + src.name() + "' to '" + dst.name() + "'");
        }
        long nElements = src.numElements();
        if (nElements != dst.numElements()) {
            throw new IOException("tensor '" + src.name() + "' conversion changes its element count from "
                    + nElements + " to " + dst.numElements());
        }
        checkConversion(src.name(), from, to);
        if (nElements == 0) {
            return;
        }
        BlockLayout srcLayout = BlockLayout.of(from);
        int srcBlockBytes = srcLayout.blockBytes();
        int srcBlockElements = srcLayout.blockElements();
        long blocks = srcLayout.blocksForElements(nElements);
        int chunkBlocks = chunkBlocks(from, to);
        byte[] raw = new byte[Math.multiplyExact(chunkBlocks, srcBlockBytes)];
        float[] x = new float[Math.multiplyExact(chunkBlocks, srcBlockElements)];
        byte[] outBytes = new byte[Math.multiplyExact(chunkBlocks, srcBlockElements) * 4];
        int q8BlockBytes = to == GGMLType.Q8_0 ? BlockLayout.of(GGMLType.Q8_0).blockBytes() : 0;
        long done = 0;
        while (done < blocks) {
            int nBlocks = (int) Math.min(chunkBlocks, blocks - done);
            int srcBytes = Math.multiplyExact(nBlocks, srcBlockBytes);
            store.readTensor(src.offset() + Math.multiplyExact(done, srcBlockBytes), srcBytes, raw, 0);
            CpuBlockDecoders.decode(from, ByteBuffer.wrap(raw), 0L, x, 0, nBlocks);
            int chunkElements = nBlocks * srcBlockElements;
            int outLen = encode(to, x, chunkElements, outBytes, q8BlockBytes);
            enc.writeData(outBytes, 0, outLen);
            done += nBlocks;
        }
    }

    /** The conversions a COMPAT export ever writes; anything else is the writer's bug. */
    private static void checkConversion(String name, GGMLType from, GGMLType to) throws IOException {
        boolean ok = from.isExtension()
                && (to == GGMLType.F16 || to == GGMLType.F32 || to == GGMLType.Q8_0)
                && (from != GGMLType.MXFP8 || to == GGMLType.BF16 || to == GGMLType.F32);
        if (!ok) {
            throw new IOException("tensor '" + name + "' has no COMPAT conversion from " + from + " to " + to);
        }
    }

    private static int encode(GGMLType to, float[] x, int n, byte[] out, int q8BlockBytes)
            throws IOException {
        switch (to) {
            case F32:
                for (int i = 0; i < n; i++) {
                    putInt(out, 4 * i, Float.floatToRawIntBits(x[i]));
                }
                return 4 * n;
            case F16:
                for (int i = 0; i < n; i++) {
                    putInt(out, 2 * i, FpBlockDecoders.fp32ToFp16(x[i]));
                }
                return 2 * n;
            case BF16:
                for (int i = 0; i < n; i++) {
                    putInt(out, 2 * i, KvBlockCodec.fp32ToBf16(x[i]));
                }
                return 2 * n;
            case Q8_0: {
                int nb = n / KvBlockCodec.BLOCK;
                for (int b = 0; b < nb; b++) {
                    KvBlockCodec.quantize(KvDType.Q8_0, x, b * KvBlockCodec.BLOCK, out, b * q8BlockBytes);
                }
                return nb * q8BlockBytes;
            }
            default:
                throw new IOException("no COMPAT encoder for ggml type " + to);
        }
    }

    private static void putInt(byte[] out, int off, int v) {
        out[off] = (byte) v;
        out[off + 1] = (byte) (v >>> 8);
        out[off + 2] = (byte) (v >>> 16);
        out[off + 3] = (byte) (v >>> 24);
    }

    /**
     * Source blocks per conversion chunk: the source bytes, the decoded elements and the written bytes all
     * stay within their chunk bounds, as does every int cast.
     */
    private static int chunkBlocks(GGMLType from, GGMLType to) {
        BlockLayout srcLayout = BlockLayout.of(from);
        int srcBlockBytes = srcLayout.blockBytes();
        int srcBlockElements = srcLayout.blockElements();
        int dstBytesPerSrcBlock = (int) BlockLayout.of(to).bytesForElements(srcBlockElements);
        long blocks = Math.min(CHUNK_BYTES / srcBlockBytes, CHUNK_BYTES / (long) dstBytesPerSrcBlock);
        blocks = Math.min(blocks, CHUNK_ELEMENTS / (long) srcBlockElements);
        return (int) Math.max(1, blocks);
    }
}
