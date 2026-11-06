package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hf.names.HfNameTables;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.TernaryBlockQuantizers;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.tensor.HalfFloats;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.Predicate;

/**
 * The GGUF type of each planned tensor: a port of step 5 of llama.cpp's {@code ModelBase.prepare_tensors}
 * (LC/conversion/base.py:1240-1330) and of its {@code --outtype auto} guess (base.py:180-197). Immutable.
 *
 * <p>{@link #typeFor} applies, in the converter's order:</p>
 * <ol>
 *   <li>A source marked with {@link ElementTransforms#keepInteger} keeps its integer type (EAGLE3 {@code d2t}).</li>
 *   <li>Step 1: a tensor of rank 1 or less, or a name ending in {@code _norm.weight}, is F32 even when the
 *       architecture forces another type (base.py:1246-1250 overrides {@code tensor_force_quant} there).</li>
 *   <li>Step 2: an architecture override ({@code tensor_force_quant}) wins over the rest, also for a name without a
 *       weight suffix (base.py:1254 tests the suffix only when there is no override).</li>
 *   <li>Step 3: without an override, a forced-F32 name, or a name that does not end in {@code .weight},
 *       {@code .lora_a} or {@code .lora_b}, is F32 (base.py:1254-1284).</li>
 *   <li>Step 4: a float source (F32, F16, BF16, F64) takes the output type; a ggml block source keeps its type.
 *       Under a ternary output type ({@code TQ1_0}/{@code TQ2_0}) the converter's embedding, output, altup-router
 *       and laurel weights stay F16 instead (base.py:1513-1525), and a row that is not whole blocks of 256 falls
 *       back to F16 the way a failed quantization does (base.py:1561-1564).</li>
 *   <li>Step 5: an E4M3FN source keeps its codes as ggml's F8_E4M3FN when the output type is AUTO (the converter
 *       preserves float8_e4m3fn weights, base.py:1475-1483), and is dequantized and typed as a float source
 *       (step 4) otherwise.</li>
 *   <li>Step 6: any other integer source is rejected.</li>
 *   <li>Step 7: any other eight-, six- or four-bit float, unsigned, boolean or complex source is rejected.</li>
 * </ol>
 *
 * <p>The class also holds the value codec the transforms convert with ({@link #decode} reads any element format as
 * floats, {@link #encode} writes floats as F32, F16, BF16, Q8_0, TQ1_0 or TQ2_0) and the producers built on it: the
 * element-wise conversion behind {@link ElementTransforms#convert} and its element ops, and the integer tensors
 * {@link ElementTransforms#keepInteger} marks.</p>
 */
public final class DtypePolicy {

    private static final String[] WEIGHT_SUFFIXES = {".weight", ".lora_a", ".lora_b"};

    /** The names that stay F16 under a ternary output type ({@code HfNameTables.ternaryKeptF16()}). */
    private static final Predicate<String> TERNARY_F16 = HfNameTables.ternaryKeptF16();

    private final Predicate<String> forcedF32;
    private final OutType outType;
    private final GGMLType autoType;

    /**
     * A policy whose output type is not AUTO, or AUTO to be resolved with {@link #resolveAuto}.
     *
     * @param forcedF32 the GGUF names the converter always stores as F32 ({@code HfNameTables.forcedF32()})
     */
    public DtypePolicy(Predicate<String> forcedF32, OutType outType) {
        this(forcedF32, outType, null);
    }

    private DtypePolicy(Predicate<String> forcedF32, OutType outType, GGMLType autoType) {
        this.forcedF32 = Objects.requireNonNull(forcedF32, "forcedF32");
        this.outType = Objects.requireNonNull(outType, "outType");
        this.autoType = autoType;
    }

    /** The configured output type. */
    public OutType outType() {
        return outType;
    }

    /**
     * This policy with AUTO resolved from the checkpoint's stored tensors by {@link #guessAuto}; other output types
     * are returned unchanged.
     */
    public DtypePolicy resolveAuto(HfTensorIndex checkpoint) {
        if (outType != OutType.AUTO) {
            return this;
        }
        return new DtypePolicy(forcedF32, outType, guessAuto(checkpoint));
    }

    /**
     * The converter's {@code --outtype auto} guess: BF16 or F16 for the first stored tensor of rank 2 or more with
     * that dtype, F16 when there is none.
     */
    public static GGMLType guessAuto(HfTensorIndex checkpoint) {
        for (String name : checkpoint.names()) {
            RowSource s = checkpoint.get(name);
            if (s.shape().length < 2) {
                continue;
            }
            StDType d = s.format().stDType();
            if (d == StDType.BF16) {
                return GGMLType.BF16;
            }
            if (d == StDType.F16) {
                return GGMLType.F16;
            }
        }
        return GGMLType.F16;
    }

    /**
     * The GGUF type of {@code ggufName} planned from {@code src}.
     *
     * @param forced the architecture override for this tensor, or null
     * @throws GgufFormatException   when the source dtype has no GGUF type and nothing rewrote it
     * @throws IllegalStateException when the output type is AUTO and {@link #resolveAuto} was not applied
     */
    public GGMLType typeFor(String ggufName, RowSource src, GGMLType forced) {
        Objects.requireNonNull(ggufName, "ggufName");
        ElementFormat f = src.format();
        if (src instanceof KeptInteger) {
            return f.ggmlType();
        }
        if (src.shape().length <= 1 || ggufName.endsWith("_norm.weight")) {
            return GGMLType.F32;
        }
        if (forced != null) {
            return forced;
        }
        if (forcedF32.test(ggufName) || !endsWithWeightSuffix(ggufName)) {
            return GGMLType.F32;
        }
        if (f.isFloat()) {
            return weightTarget(ggufName, f.ggmlType());
        }
        if (f.isBlock()) {
            return f.ggmlType();
        }
        if (f.ggmlType() == GGMLType.F8_E4M3FN) {
            // The stored E4M3FN codes are ggml's per-element F8_E4M3FN (llama.cpp GGML_TYPE_F8_E4M3):
            // --outtype auto keeps them (base.py:1475-1483 preserves float8_e4m3fn weights), the named
            // output types dequantize them and type them as any float weight.
            return outType == OutType.AUTO ? GGMLType.F8_E4M3FN : weightTarget(ggufName, f.ggmlType());
        }
        if (isInteger(f)) {
            throw new GgufFormatException(ggufName + ": integer tensor of dtype " + f + " from " + src.describe()
                    + " is only stored when the architecture marks it as an integer tensor");
        }
        throw new GgufFormatException(ggufName + ": dtype " + f + " of " + src.describe() + " has no GGUF type and"
                + " no quantization rewrite applies to it");
    }

    /**
     * {@code src} in the type {@link #typeFor} chooses: converted when the type differs from the source's, with the
     * converter's F16 fallback for a Q8_0 target whose rows are not whole blocks (base.py:1318-1323) and for a
     * TQ1_0 / TQ2_0 target whose rows are not whole blocks of 256 (a failed ternary quantization, base.py:1561-1564).
     */
    public RowSource apply(String ggufName, RowSource src, GGMLType forced) {
        GGMLType t = typeFor(ggufName, src, forced);
        if ((t == GGMLType.Q8_0 || t == GGMLType.TQ1_0 || t == GGMLType.TQ2_0)
                && src.format().ggmlType() != t) {
            long[] shape = src.shape();
            long inner = shape.length == 0 ? 1 : shape[shape.length - 1];
            if (inner % t.getBlockSize() != 0) {
                t = GGMLType.F16;
            }
        }
        if (t == src.format().ggmlType()) {
            return src;
        }
        return ElementTransforms.convert(src, t);
    }

    /**
     * The type of a float (or dequantized) weight {@code ggufName}: F16 for the names a ternary output type keeps
     * (base.py:1513-1525), the output type otherwise.
     */
    private GGMLType weightTarget(String ggufName, GGMLType source) {
        if ((outType == OutType.TQ1_0 || outType == OutType.TQ2_0) && TERNARY_F16.test(ggufName)) {
            return GGMLType.F16;
        }
        return floatTarget(source);
    }

    private GGMLType floatTarget(GGMLType source) {
        switch (outType) {
            case SOURCE:
                return source == GGMLType.F64 ? GGMLType.F32 : source;
            case AUTO:
                if (autoType == null) {
                    throw new IllegalStateException("output type AUTO needs DtypePolicy.resolveAuto(checkpoint)"
                            + " before tensors are planned");
                }
                return autoType;
            case F32:
                return GGMLType.F32;
            case F16:
                return GGMLType.F16;
            case BF16:
                return GGMLType.BF16;
            case Q8_0:
                return GGMLType.Q8_0;
            case TQ1_0:
                return GGMLType.TQ1_0;
            case TQ2_0:
                return GGMLType.TQ2_0;
            default:
                throw new IllegalStateException("unhandled output type " + outType);
        }
    }

    private static boolean endsWithWeightSuffix(String name) {
        for (String s : WEIGHT_SUFFIXES) {
            if (name.endsWith(s)) {
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------ value codec

    /** True for signed, unsigned and boolean integer element formats. */
    static boolean isInteger(ElementFormat f) {
        GGMLType t = f.ggmlType();
        if (t == GGMLType.I8 || t == GGMLType.I16 || t == GGMLType.I32 || t == GGMLType.I64) {
            return true;
        }
        StDType d = f.stDType();
        return d == StDType.U8 || d == StDType.U16 || d == StDType.U32 || d == StDType.U64 || d == StDType.BOOL;
    }

    /** True when {@link #decode} reads {@code f}. */
    static boolean decodes(ElementFormat f) {
        if (f.ggmlType() != null) {
            return CpuBlockDecoders.supports(f.ggmlType());
        }
        StDType d = f.stDType();
        return d != StDType.C64 && d != StDType.F6_E2M3 && d != StDType.F6_E3M2;
    }

    /**
     * Decodes {@code n} elements of format {@code f} starting at absolute byte {@code byteOff} of {@code src} (whole
     * blocks for a block type, a whole number of bytes for F4) into {@code dst[dstOff, dstOff + n)}.
     *
     * @throws IllegalArgumentException for a format without a decoder (F6, C64, unsupported ggml types)
     */
    static void decode(ElementFormat f, ByteBuffer src, int byteOff, float[] dst, int dstOff, int n) {
        GGMLType t = f.ggmlType();
        if (t != null) {
            if (!CpuBlockDecoders.supports(t)) {
                throw new IllegalArgumentException("no decoder for ggml type " + t);
            }
            CpuBlockDecoders.decodeElements(t, src, byteOff, dst, dstOff, n);
            return;
        }
        StDType d = f.stDType();
        for (int i = 0; i < n; i++) {
            dst[dstOff + i] = decodeOne(d, src, byteOff, i);
        }
    }

    private static float decodeOne(StDType d, ByteBuffer b, int off, int i) {
        switch (d) {
            case BOOL:
                return b.get(off + i) != 0 ? 1f : 0f;
            case U8:
                return b.get(off + i) & 0xFF;
            case U16:
                return b.getShort(off + 2 * i) & 0xFFFF;
            case U32:
                return b.getInt(off + 4 * i) & 0xFFFF_FFFFL;
            case U64: {
                long v = b.getLong(off + 8 * i);
                return v >= 0 ? (float) v : ((float) ((v >>> 1) | (v & 1))) * 2f;
            }
            case F8_E4M3:
                return FpBlockDecoders.e4m3ToFp32(b.get(off + i));
            case F8_E5M2:
                return FpBlockDecoders.e5m2ToFp32(b.get(off + i));
            case F8_E8M0:
                return FpBlockDecoders.e8m0ToFp32(b.get(off + i));
            case F8_E4M3FNUZ:
                return fnuz(b.get(off + i) & 0xFF, 3, 8, 10);
            case F8_E5M2FNUZ:
                return fnuz(b.get(off + i) & 0xFF, 2, 16, 17);
            case F4: {
                int v = b.get(off + i / 2) & 0xFF;
                return FpBlockDecoders.e2m1ToFp32((i & 1) == 0 ? v & 0xF : v >>> 4);
            }
            default:
                throw new IllegalArgumentException("no decoder for dtype " + d);
        }
    }

    /**
     * An FNUZ eight-bit float (no infinities, no negative zero, NaN at {@code 0x80}): {@code mBits} mantissa bits,
     * exponent bias {@code bias}, subnormals {@code m * 2^-subShift}.
     */
    private static float fnuz(int code, int mBits, int bias, int subShift) {
        if (code == 0x80) {
            return Float.NaN;
        }
        int e = (code & 0x7F) >>> mBits;
        int m = code & ((1 << mBits) - 1);
        float v = e == 0 ? Math.scalb((float) m, -subShift) : Math.scalb(1f + (float) m / (1 << mBits), e - bias);
        return (code & 0x80) != 0 ? -v : v;
    }

    /**
     * Writes {@code v[off, off + n)} as {@code target} at {@code out}'s position: F32, F16 and BF16 rounded to
     * nearest even; Q8_0 with the reference quantizer ({@code quantize_row_q8_0_ref}, rounding half away from zero),
     * {@code n} a multiple of 32; TQ1_0 and TQ2_0 with the reference ternary quantizers
     * ({@code quantize_row_tq1_0_ref} / {@code quantize_row_tq2_0_ref}), {@code n} a multiple of 256.
     *
     * @throws IllegalArgumentException for another target, or a quantized run that is not whole blocks
     */
    static void encode(GGMLType target, float[] v, int off, int n, ByteBuffer out) {
        switch (target) {
            case F32:
                for (int i = 0; i < n; i++) {
                    out.putFloat(v[off + i]);
                }
                return;
            case F16:
                for (int i = 0; i < n; i++) {
                    out.putShort((short) FpBlockDecoders.fp32ToFp16(v[off + i]));
                }
                return;
            case BF16:
                for (int i = 0; i < n; i++) {
                    out.putShort((short) HalfFloats.floatToBFloat16(v[off + i]));
                }
                return;
            case Q8_0: {
                int qk = GGMLType.Q8_0.getBlockSize();
                if (n % qk != 0) {
                    throw new IllegalArgumentException(n + " elements are not whole Q8_0 blocks of " + qk);
                }
                byte[] block = new byte[GGMLType.Q8_0.getTypeSize()];
                for (int i = 0; i < n; i += qk) {
                    KvBlockCodec.quantize(KvDType.Q8_0, v, off + i, block, 0);
                    out.put(block);
                }
                return;
            }
            case TQ1_0: {
                int qk = GGMLType.TQ1_0.getBlockSize();
                if (n % qk != 0) {
                    throw new IllegalArgumentException(n + " elements are not whole TQ1_0 blocks of " + qk);
                }
                byte[] block = new byte[GGMLType.TQ1_0.getTypeSize()];
                for (int i = 0; i < n; i += qk) {
                    TernaryBlockQuantizers.quantizeTq1_0(v, off + i, block, 0);
                    out.put(block);
                }
                return;
            }
            case TQ2_0: {
                int qk = GGMLType.TQ2_0.getBlockSize();
                if (n % qk != 0) {
                    throw new IllegalArgumentException(n + " elements are not whole TQ2_0 blocks of " + qk);
                }
                byte[] block = new byte[GGMLType.TQ2_0.getTypeSize()];
                for (int i = 0; i < n; i += qk) {
                    TernaryBlockQuantizers.quantizeTq2_0(v, off + i, block, 0);
                    out.put(block);
                }
                return;
            }
            default:
                throw new IllegalArgumentException("cannot convert to " + target + "; the targets are F32, F16,"
                        + " BF16, Q8_0, TQ1_0 and TQ2_0");
        }
    }

    /** True when {@link #constantRow} encodes any value in {@code f}. */
    static boolean encodesConstant(ElementFormat f) {
        return f.isFloat();
    }

    /** One row of {@code n} elements of {@code fill} in {@code f}; zero bytes for 0, which every format reads as 0. */
    static byte[] constantRow(ElementFormat f, long n, float fill) {
        long bytes = f.rowBytes(n);
        if (bytes > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("a fill row of " + n + " " + f + " elements is too large");
        }
        byte[] row = new byte[(int) bytes];
        if (fill == 0f && Float.floatToRawIntBits(fill) == 0) {
            return row;
        }
        if (!encodesConstant(f)) {
            throw new IllegalArgumentException("cannot encode the fill value " + fill + " as " + f);
        }
        ByteBuffer out = ByteBuffer.wrap(row).order(ByteOrder.LITTLE_ENDIAN);
        float[] v = new float[(int) n];
        Arrays.fill(v, fill);
        if (f.ggmlType() == GGMLType.F64) {
            for (float x : v) {
                out.putDouble(x);
            }
        } else {
            encode(f.ggmlType(), v, 0, v.length, out);
        }
        return row;
    }

    // ------------------------------------------------------------------ conversion producers

    /**
     * Decodes rows to f32 (multiplying in the output scale when asked), applies an op and encodes the result as the
     * target type.
     *
     * <p>A {@linkplain #widened() widened} producer computes in f32 what the converter computes on a tensor that
     * {@code prepare_tensors} hands its hooks as F32: every dtype except F16 and F32, BF16 in practice
     * (LC/conversion/base.py:1229-1231). Its target is then only the view's storage choice, the source's BF16, so
     * {@link #storedAs} rounds the f32 values once into another type instead of converting the BF16 values.</p>
     */
    static final class Mapped extends RowTransforms.Derived {
        private final RowSource src;
        private final String what;
        private final GGMLType target;
        private final ElementTransforms.RowFactor scale;
        private final ElementTransforms.RowOp op;
        private final ElementTransforms.ChunkOp chunkOp;
        private final boolean widened;
        private final int inner;

        Mapped(RowSource src, String what, GGMLType target, boolean foldScale, ElementTransforms.RowOp op,
               ExpertTransforms.Siblings sib) {
            this(src, what, target, foldScale && src.outputScale() != null
                    ? ElementTransforms.expertFactor(src, src.outputScale()) : null, op, null, sib);
        }

        /**
         * @param scale   the per-row output-scale factor multiplied into the decoded values, or null
         * @param op      the per-row op, or null
         * @param chunkOp the op over each decoded chunk, applied after the row ops, or null
         */
        Mapped(RowSource src, String what, GGMLType target, ElementTransforms.RowFactor scale,
               ElementTransforms.RowOp op, ElementTransforms.ChunkOp chunkOp,
               ExpertTransforms.Siblings sib) {
            this(src, what, target, scale, op, chunkOp, sib, false, src.describe() + what);
        }

        private Mapped(RowSource src, String what, GGMLType target, ElementTransforms.RowFactor scale,
                       ElementTransforms.RowOp op, ElementTransforms.ChunkOp chunkOp,
                       ExpertTransforms.Siblings sib, boolean widened, String describe) {
            super(describe, ElementFormat.of(target), src.shape(), sib);
            if (!DtypePolicy.decodes(src.format())) {
                throw new IllegalArgumentException(src.describe() + ": no decoder for its format " + src.format());
            }
            long[] shape = src.shape();
            this.src = src;
            this.what = what;
            this.target = target;
            this.scale = scale;
            this.op = op;
            this.chunkOp = chunkOp;
            this.widened = widened;
            this.inner = Math.toIntExact(shape.length == 0 ? 1 : shape[shape.length - 1]);
        }

        /** This producer marked as computing in f32 values the converter computes in F32. */
        Mapped widen() {
            return new Mapped(src, what, target, scale, op, chunkOp, siblings(), true, describe());
        }

        /** Whether the values are f32 results stored in {@link #format()} only as this view's choice. */
        boolean widened() {
            return widened;
        }

        /**
         * The same values computed in f32 and rounded once into {@code t}, still {@linkplain #widened() widened}.
         *
         * @throws IllegalStateException when this producer is not widened
         */
        Mapped storedAs(GGMLType t) {
            if (!widened) {
                throw new IllegalStateException(describe() + " is not computed in f32; convert its values instead");
            }
            if (t == target) {
                return this;
            }
            return new Mapped(src, what, t, scale, op, chunkOp, siblings(), true, src.describe() + what + " as " + t);
        }

        @Override
        public boolean rowLocal() {
            return src.rowLocal();
        }

        @Override
        void produce(long first, int count, ByteBuffer out) throws IOException {
            int per = RowTransforms.chunkRows(Math.max(Math.max(src.rowBytes(), rowBytes()), 4L * inner));
            for (int done = 0; done < count; ) {
                int c = Math.min(per, count - done);
                long r0 = first + done;
                ByteBuffer in = RowTransforms.readRows(src, r0, c);
                float[] v = new float[c * inner];
                int srb = (int) src.rowBytes();
                for (int i = 0; i < c; i++) {
                    DtypePolicy.decode(src.format(), in, i * srb, v, i * inner, inner);
                    if (scale != null) {
                        float s = scale.at(r0 + i);
                        for (int j = 0; j < inner; j++) {
                            v[i * inner + j] *= s;
                        }
                    }
                    if (op != null) {
                        op.apply(r0 + i, v, i * inner, inner);
                    }
                }
                if (chunkOp != null) {
                    chunkOp.apply(r0, c, v);
                }
                DtypePolicy.encode(target, v, 0, v.length, out);
                done += c;
            }
        }
    }

    /** An integer tensor stored as a signed ggml integer type, widening unsigned and boolean values. */
    static final class KeptInteger extends RowTransforms.Derived {
        private final RowSource src;
        private final int inBytes;
        private final int outBytes;

        KeptInteger(RowSource src) {
            super(src.describe() + " kept as an integer tensor", ElementFormat.of(widened(src.format())), src.shape(),
                    ExpertTransforms.Siblings.NONE);
            this.src = src;
            this.inBytes = Math.max(1, src.format().stDType().bits() / 8);
            this.outBytes = format().stDType().bits() / 8;
        }

        private static GGMLType widened(ElementFormat f) {
            if (f.ggmlType() != null) {
                return f.ggmlType();
            }
            StDType d = f.stDType();
            switch (d) {
                case BOOL:
                    return GGMLType.I8;
                case U8:
                    return GGMLType.I16;
                case U16:
                    return GGMLType.I32;
                default:
                    return GGMLType.I64;
            }
        }

        @Override
        public boolean rowLocal() {
            return src.rowLocal();
        }

        @Override
        public long directVirtualOffset() {
            return inBytes == outBytes && src.format().ggmlType() != null ? src.directVirtualOffset() : -1;
        }

        @Override
        void produce(long first, int count, ByteBuffer out) throws IOException {
            if (src.format().ggmlType() != null) {
                src.readRows(first, count, out);
                return;
            }
            boolean bool = src.format().stDType() == StDType.BOOL;
            int per = RowTransforms.chunkRows(Math.max(rowBytes(), src.rowBytes()));
            for (int done = 0; done < count; ) {
                int c = Math.min(per, count - done);
                ByteBuffer in = RowTransforms.readRows(src, first + done, c);
                while (in.hasRemaining()) {
                    long v;
                    switch (inBytes) {
                        case 1:
                            v = bool ? (in.get() != 0 ? 1 : 0) : in.get() & 0xFFL;
                            break;
                        case 2:
                            v = in.getShort() & 0xFFFFL;
                            break;
                        case 4:
                            v = in.getInt() & 0xFFFF_FFFFL;
                            break;
                        default:
                            v = in.getLong();
                            if (v < 0) {
                                throw new IOException(src.describe() + ": U64 value "
                                        + Long.toUnsignedString(v) + " does not fit I64");
                            }
                    }
                    switch (outBytes) {
                        case 1:
                            out.put((byte) v);
                            break;
                        case 2:
                            out.putShort((short) v);
                            break;
                        case 4:
                            out.putInt((int) v);
                            break;
                        default:
                            out.putLong(v);
                    }
                }
                done += c;
            }
        }
    }
}
