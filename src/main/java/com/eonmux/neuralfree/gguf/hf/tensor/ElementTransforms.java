package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.utils.numeric.Fp8Codec;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

/**
 * Element-wise transforms, scale folding, ternary and low-rank rewrites, and dtype conversion. Values are
 * computed in f32 and rounded once into the output format: the source's float format (F64 becomes F32), or F32 for
 * a block, eight-bit or integer source, which is dequantized first with one warning. A result kept in a BF16 source's
 * format stays an f32 result for the converter's purposes ({@code prepare_tensors} hands its hooks BF16 tensors as
 * F32): later element ops read its f32 values, and {@link #convert} rounds those once into another type. Scaling a
 * block format with a per-block scale ({@code d}, and {@code m} where present) rewrites only those fields: exact for
 * f32 scales, exact up to the fp16 rounding of {@code d} otherwise. All producers here are row-local and read their
 * source in chunks of at most {@link RowTransforms#CHUNK_BYTES} output bytes.
 */
public final class ElementTransforms {

    private static final Logger LOG = LogManager.getLogger(ElementTransforms.class);

    private ElementTransforms() {
    }

    // ------------------------------------------------------------------ functions and reshapes

    /** {@code -torch.exp(src)} (Mamba {@code A_log}, LC/conversion/mamba.py). */
    public static RowSource negExp(RowSource src) {
        return nonLinear(src, "-exp", v -> (float) -Math.exp(v));
    }

    /** {@code torch.exp(src)}. */
    public static RowSource exp(RowSource src) {
        return nonLinear(src, "exp", v -> (float) Math.exp(v));
    }

    /** {@code src.squeeze()}: every dimension of size 1 removed, keeping at least one (Mamba {@code conv1d}). */
    public static RowSource squeezeConv(RowSource src) {
        long[] shape = src.shape();
        long[] out = Arrays.stream(shape).filter(d -> d != 1).toArray();
        return reshape(src, out.length == 0 ? new long[]{1} : out);
    }

    /**
     * {@code src.reshape(shape)} in row-major order; one dimension may be {@code -1}. The bytes are unchanged, so
     * a tensor stored unchanged stays mapped in place. IllegalArgumentException when the element counts differ, the
     * new innermost dimension is not a whole number of blocks, or the source is a low-rank factor pair whose input
     * dimension changes.
     */
    public static RowSource reshape(RowSource src, long... shape) {
        long numel = numel(src.shape());
        long[] out = inferShape(src, shape.clone(), numel);
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.reshape((LoraFactorSource) src, out);
        }
        long[] in = src.shape();
        long inInner = in.length == 0 ? 1 : in[in.length - 1];
        long outInner = out.length == 0 ? 1 : out[out.length - 1];
        ExpertTransforms.Siblings sib;
        if (inInner == outInner) {
            sib = ExpertTransforms.Siblings.of(src);
        } else {
            src = ExpertTransforms.Siblings.withoutPerm(src);
            sib = ExpertTransforms.Siblings.of(src);
            if (sib.hasActivation()) {
                LOG.warn("{}: the reshape changes the input dimension; activation quantization values are dropped",
                        src.describe());
                sib = sib.withoutActivation();
            }
        }
        float[] os = sib.outputScale();
        if (os != null && os.length > 1 && !(out.length >= 3 && in.length >= 3 && out[0] == in[0])) {
            src = ExpertTransforms.Siblings.withoutScale(src);
            sib = sib.withoutOutputScale();
        }
        return new Reshaped(src, out, sib);
    }

    // ------------------------------------------------------------------ scalars, rows and columns

    /** {@code src + value}. */
    public static RowSource addScalar(RowSource src, float value) {
        return affine(src, 1f, value);
    }

    /** {@code src * factor}: a source with an output scale gets the scale multiplied, F8_E4M3 gets one, block
     *  formats with per-block scales are scaled in place, other formats are computed element by element. */
    public static RowSource mulScalar(RowSource src, float factor) {
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.onB((LoraFactorSource) src, b -> mulScalar(b, factor));
        }
        float[] os = src.outputScale();
        if (os != null || src.format().ggmlType() == GGMLType.F8_E4M3) {
            float[] scaled = os == null ? new float[]{factor} : os.clone();
            if (os != null) {
                for (int i = 0; i < scaled.length; i++) {
                    scaled[i] *= factor;
                }
            }
            return new Rescaled(src, ExpertTransforms.Siblings.of(src).withOutputScale(scaled));
        }
        if (scalesInPlace(src.format().ggmlType(), new float[]{factor})) {
            return new BlockScaled(src, " * " + factor, r -> factor, ExpertTransforms.Siblings.of(src));
        }
        return mapped(src, " * " + factor, false, (row, v, off, n) -> {
            for (int i = 0; i < n; i++) {
                v[off + i] *= factor;
            }
        }, ExpertTransforms.Siblings.of(src));
    }

    /** {@code src * a + b}. */
    public static RowSource affine(RowSource src, float a, float b) {
        if (src instanceof LoraFactorSource) {
            throw LoraFactorOps.unsupported(src, "the affine map x * " + a + " + " + b);
        }
        return nonLinear(src, " * " + a + " + " + b, v -> v * a + b);
    }

    /**
     * Multiplies row {@code i} by {@code factors[i]}: one factor per row of a 2-D tensor, per row of each matrix
     * of a stack ({@code factors.length == shape[-2]}) or per flat row; for a 1-D tensor, one factor per element.
     * Block formats with per-block scales are scaled in place; IllegalArgumentException when the factor count
     * matches no row count.
     */
    public static RowSource mulRowsBy(RowSource src, float[] factors) {
        float[] f = factors.clone();
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.onB((LoraFactorSource) src, b -> mulRowsBy(b, f));
        }
        long[] shape = src.shape();
        if (shape.length <= 1) {
            return mulColsBy(src, f);
        }
        long rows = src.rows();
        long perMatrix = shape[shape.length - 2];
        if (f.length != rows && f.length != perMatrix) {
            throw new IllegalArgumentException(src.describe() + ": " + f.length + " row factors match neither its "
                    + rows + " rows nor the " + perMatrix + " rows of each matrix");
        }
        RowFactor factor = r -> f[(int) (r % f.length)];
        if (scalesInPlace(src.format().ggmlType(), f)) {
            return new BlockScaled(src, " with scaled rows", factor, ExpertTransforms.Siblings.of(src));
        }
        return mapped(src, " with scaled rows", false, (row, v, off, n) -> {
            float s = factor.at(row);
            for (int i = 0; i < n; i++) {
                v[off + i] *= s;
            }
        }, ExpertTransforms.Siblings.of(src));
    }

    /**
     * Multiplies column {@code j} (innermost index) by {@code factors[j]}; IllegalArgumentException when the
     * factor count is not the innermost dimension.
     */
    public static RowSource mulColsBy(RowSource src, float[] factors) {
        float[] g = factors.clone();
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.onA((LoraFactorSource) src, a -> mulColsBy(a, g));
        }
        long[] shape = src.shape();
        long inner = shape.length == 0 ? 1 : shape[shape.length - 1];
        if (g.length != inner) {
            throw new IllegalArgumentException(src.describe() + ": " + g.length + " column factors for rows of "
                    + inner + " elements");
        }
        src = ExpertTransforms.Siblings.withoutPerm(src);
        return mapped(src, " with scaled columns", false, (row, v, off, n) -> {
            for (int i = 0; i < n; i++) {
                v[off + i] *= g[i];
            }
        }, ExpertTransforms.Siblings.of(src));
    }

    /**
     * {@code src[..., index]}: output element {@code j} of each row is input element {@code index[j]}. Exact for
     * element formats; a block format is dequantized unless the gather moves whole blocks.
     */
    public static RowSource gatherInner(RowSource src, int[] index) {
        return ColumnTransforms.gather(src, index);
    }

    // ------------------------------------------------------------------ dtype conversion

    /**
     * {@code src.to(target)} for F32, F16 and BF16 (round to nearest even) or Q8_0 (the reference quantizer,
     * rounding half away from zero). Any float, integer, eight-bit float or ggml block source is decoded first;
     * an output scale and the activation values are kept; IllegalArgumentException for another target or a
     * source format without a decoder.
     */
    public static RowSource convert(RowSource src, GGMLType target) {
        Objects.requireNonNull(target, "target");
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.convert((LoraFactorSource) src, target);
        }
        if (src.format().ggmlType() == target) {
            return src;
        }
        if (target != GGMLType.F32 && target != GGMLType.F16 && target != GGMLType.BF16 && target != GGMLType.Q8_0) {
            throw new IllegalArgumentException(src.describe() + ": cannot convert to " + target
                    + "; the targets are F32, F16, BF16 and Q8_0");
        }
        if (src instanceof DtypePolicy.Mapped && ((DtypePolicy.Mapped) src).widened()) {
            return ((DtypePolicy.Mapped) src).storedAs(target);
        }
        return new DtypePolicy.Mapped(src, " as " + target, target, false, null, ExpertTransforms.Siblings.of(src));
    }

    /** The values of {@code src} in {@code floatTarget} (F32, F16 or BF16), with its output scale and
     *  act-order permutation folded in. */
    public static RowSource dequant(RowSource src, GGMLType floatTarget) {
        if (floatTarget != GGMLType.F32 && floatTarget != GGMLType.F16 && floatTarget != GGMLType.BF16) {
            throw new IllegalArgumentException("dequant targets F32, F16 or BF16, not " + floatTarget);
        }
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.convert((LoraFactorSource) src, floatTarget);
        }
        RowSource s = opInput(ExpertTransforms.Siblings.withoutPerm(src));
        return new DtypePolicy.Mapped(s, " dequantized to " + floatTarget, floatTarget, true, null,
                ExpertTransforms.Siblings.of(s).withoutOutputScale());
    }

    /** Marks an integer tensor to be stored as an integer (EAGLE3 {@code d2t}): signed types keep their type;
     *  BOOL becomes I8, U8 I16, U16 I32, and U32 and U64 I64; IllegalArgumentException for a non-integer format. */
    public static RowSource keepInteger(RowSource src) {
        if (!DtypePolicy.isInteger(src.format())) {
            throw new IllegalArgumentException(src.describe() + ": keepInteger needs an integer tensor, not "
                    + src.format());
        }
        return new DtypePolicy.KeptInteger(src);
    }

    // ------------------------------------------------------------------ ternary re-quantization and low-rank merges

    /**
     * BitNet's {@code weight_quant} (LC/conversion/bitnet.py:24-33) stored as TQ2_0: {@code s = mean|W|} (at
     * least 1e-5), {@code q = round_half_even(W / s).clamp(-1, 1)}, block scale 1 and output scale {@code s}, so
     * {@code q * s} is the converter's value; the mean is computed here, reading the whole tensor once. The rows
     * must be whole TQ2_0 blocks of 256.
     */
    public static RowSource ternaryRequant(RowSource src) throws IOException {
        if (src instanceof LoraFactorSource) {
            throw LoraFactorOps.unsupported(src, "a ternary requantization");
        }
        long[] shape = src.shape();
        long inner = shape.length == 0 ? 1 : shape[shape.length - 1];
        if (inner % GGMLType.TQ2_0.getBlockSize() != 0) {
            throw new IllegalArgumentException(src.describe() + ": rows of " + inner + " elements are not whole TQ2_0"
                    + " blocks of " + GGMLType.TQ2_0.getBlockSize());
        }
        RowSource s = dequant(src, GGMLType.F32);
        double sum = 0;
        long n = 0;
        int chunk = RowTransforms.chunkRows(s.rowBytes());
        for (long r = 0; r < s.rows(); r += chunk) {
            int c = (int) Math.min(chunk, s.rows() - r);
            ByteBuffer b = RowTransforms.readRows(s, r, c);
            while (b.hasRemaining()) {
                sum += Math.abs(b.getFloat());
                n++;
            }
        }
        float scale = Math.max(n == 0 ? 0f : (float) (sum / n), 1e-5f);
        float iscale = 1f / scale;
        return new Ternary(s, iscale, ExpertTransforms.Siblings.of(s).withOutputScale(new float[]{1f / iscale}));
    }

    /** {@code base + scale * (b @ a)} in f32 (a merged LoRA delta, LC/conversion/lev.py), stored in the base's
     *  float format; {@code a} is {@code [r, in]} and is read whole per call, {@code b} is {@code [out, r]};
     *  IllegalArgumentException when the shapes do not compose. */
    public static RowSource addLowRank(RowSource base, RowSource a, RowSource b, float scale) {
        for (RowSource s : new RowSource[]{base, a, b}) {
            if (s instanceof LoraFactorSource) {
                throw LoraFactorOps.unsupported(s, "a dense low-rank merge");
            }
        }
        long[] w = base.shape();
        long[] as = a.shape();
        long[] bs = b.shape();
        if (w.length != 2 || as.length != 2 || bs.length != 2 || as[1] != w[1] || bs[0] != w[0] || bs[1] != as[0]
                || as[0] * as[1] > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("addLowRank: " + Arrays.toString(bs) + " @ " + Arrays.toString(as)
                    + " does not give the shape " + Arrays.toString(w) + " of " + base.describe());
        }
        RowSource bf = dequant(b, GGMLType.F32);
        RowSource af = dequant(a, GGMLType.F32);
        RowSource w0 = ExpertTransforms.Siblings.withoutPerm(base);
        int rank = (int) as[0];
        int in = (int) as[1];
        float[] os = w0.outputScale();
        RowFactor baseScale = os == null ? null : expertFactor(w0, os);
        ChunkOp delta = (first, count, v) -> {
            float[] av = LoraFactorSource.readAll(af);
            float[] bv = new float[count * rank];
            RowTransforms.readRows(bf, first, count).asFloatBuffer().get(bv);
            LoraFactorSource.multiplyAdd(bv, av, count, rank, in, scale, v);
        };
        return widenedIfBf16(new DtypePolicy.Mapped(opInput(w0), " + " + scale + " * B @ A", floatOut(w0.format()),
                baseScale, null, delta, ExpertTransforms.Siblings.of(w0).withoutOutputScale()));
    }

    // ------------------------------------------------------------------ scale folding

    /** {@code src} with its output scale multiplied into the weight: into the per-block scales when the format
     *  has them (exact for f32 scales, up to fp16 rounding of {@code d} otherwise), else element by element. */
    public static RowSource foldScale(RowSource src) {
        float[] os = src.outputScale();
        if (os == null) {
            return src;
        }
        ExpertTransforms.Siblings sib = ExpertTransforms.Siblings.of(src).withoutOutputScale();
        RowFactor factor = expertFactor(src, os);
        if (foldsIntoBlocks(src)) {
            return new BlockScaled(src, " with its scale folded", factor, sib);
        }
        if (src.format().isBlock()) {
            LOG.warn("{}: format {} has no block scale to fold its output scale into; dequantized to F32",
                    src.describe(), src.format());
        }
        return widenedIfBf16(new DtypePolicy.Mapped(opInput(src), " with its scale folded", floatOut(src.format()),
                true, null, sib));
    }

    /** True when {@link #foldScale} keeps the source's block format. */
    static boolean foldsIntoBlocks(RowSource src) {
        float[] os = src.outputScale();
        return os != null && scalesInPlace(src.format().ggmlType(), os);
    }

    /** The output-scale factor of each flat row: one value, or one per expert of the outermost dimension. */
    static RowFactor expertFactor(RowSource src, float[] os) {
        if (os.length == 1) {
            float s = os[0];
            return r -> s;
        }
        long[] shape = src.shape();
        if (shape.length < 2 || shape[0] != os.length) {
            throw new IllegalArgumentException(src.describe() + ": " + os.length + " output scales do not match its"
                    + " shape " + Arrays.toString(shape));
        }
        long perExpert = src.rows() / shape[0];
        return r -> os[(int) (r / perExpert)];
    }

    // ------------------------------------------------------------------ package helpers

    /** {@code src} dequantized to F32 for an op that needs element access, with one warning. */
    static RowSource dequantFallback(RowSource src, String why) {
        LOG.warn("{}: {} needs element access to {} data; dequantized to F32", src.describe(), why, src.format());
        return dequant(src, GGMLType.F32);
    }

    /** The element-wise output format for a source: its float format (F64 as F32), else F32. */
    static GGMLType floatOut(ElementFormat f) {
        return f.isFloat() && f.ggmlType() != GGMLType.F64 ? f.ggmlType() : GGMLType.F32;
    }

    private static RowSource nonLinear(RowSource src, String what, FloatOp op) {
        if (src instanceof LoraFactorSource) {
            throw LoraFactorOps.unsupported(src, what.trim());
        }
        return mapped(src, " " + what.trim(), true, (row, v, off, n) -> {
            for (int i = 0; i < n; i++) {
                v[off + i] = op.apply(v[off + i]);
            }
        }, ExpertTransforms.Siblings.of(src).withoutOutputScale());
    }

    private static DtypePolicy.Mapped mapped(RowSource src, String what, boolean foldScale, RowOp op,
                                 ExpertTransforms.Siblings sib) {
        if (src.format().isBlock()) {
            LOG.warn("{}: '{}' needs element access to {} data; computed in F32", src.describe(), what.trim(),
                    src.format());
        }
        return widenedIfBf16(new DtypePolicy.Mapped(opInput(src), what, floatOut(src.format()), foldScale, op, sib));
    }

    /** The input of an element op: a {@linkplain DtypePolicy.Mapped#widened() widened} producer is read as its
     *  f32 values, as the converter chains ops on F32 tensors; any other source as stored. */
    private static RowSource opInput(RowSource src) {
        return src instanceof DtypePolicy.Mapped && ((DtypePolicy.Mapped) src).widened()
                ? ((DtypePolicy.Mapped) src).storedAs(GGMLType.F32) : src;
    }

    /**
     * {@code m}, widened when it keeps a BF16 source's format: {@code prepare_tensors} hands the converter's hooks
     * every dtype except F16 and F32 as F32 (LC/conversion/base.py:1229-1231), so the op's f32 result is rounded only
     * into the planned type ({@link #convert}).
     */
    private static DtypePolicy.Mapped widenedIfBf16(DtypePolicy.Mapped m) {
        return m.format().ggmlType() == GGMLType.BF16 ? m.widen() : m;
    }

    private static long numel(long[] shape) {
        long n = 1;
        for (long d : shape) {
            n = Math.multiplyExact(n, d);
        }
        return n;
    }

    private static long[] inferShape(RowSource src, long[] shape, long numel) {
        int unknown = -1;
        long known = 1;
        for (int i = 0; i < shape.length; i++) {
            if (shape[i] == -1 && unknown < 0) {
                unknown = i;
            } else if (shape[i] < 0) {
                throw new IllegalArgumentException(src.describe() + ": invalid reshape target "
                        + Arrays.toString(shape));
            } else {
                known = Math.multiplyExact(known, shape[i]);
            }
        }
        if (unknown >= 0) {
            if (known == 0 || numel % known != 0) {
                throw new IllegalArgumentException(src.describe() + ": cannot infer -1 in " + Arrays.toString(shape)
                        + " for " + numel + " elements");
            }
            shape[unknown] = numel / known;
            known = numel;
        }
        if (known != numel) {
            throw new IllegalArgumentException(src.describe() + ": cannot reshape " + Arrays.toString(src.shape())
                    + " (" + numel + " elements) to " + Arrays.toString(shape));
        }
        return shape;
    }

    // ------------------------------------------------------------------ block scale fields

    /** Byte offsets of the fp16 ({@code [0]}) and f32 ({@code [1]}) per-block scales of the types scaled in place. */
    private static final Map<GGMLType, int[][]> SCALE_FIELDS = scaleFieldTable();

    private static Map<GGMLType, int[][]> scaleFieldTable() {
        int[] none = {};
        Map<GGMLType, int[][]> m = new EnumMap<>(GGMLType.class);
        m.put(GGMLType.Q4_0, new int[][]{{BlockLayout.Q4_0_D}, none});
        m.put(GGMLType.Q4_1, new int[][]{{BlockLayout.Q4_1_D, BlockLayout.Q4_1_M}, none});
        m.put(GGMLType.Q5_0, new int[][]{{BlockLayout.Q5_0_D}, none});
        m.put(GGMLType.Q5_1, new int[][]{{BlockLayout.Q5_1_D, BlockLayout.Q5_1_M}, none});
        m.put(GGMLType.Q8_0, new int[][]{{BlockLayout.Q8_0_D}, none});
        m.put(GGMLType.Q2_0, new int[][]{{BlockLayout.Q2_0_D}, none});
        m.put(GGMLType.Q1_0, new int[][]{{BlockLayout.Q1_0_D}, none});
        m.put(GGMLType.TQ1_0, new int[][]{{BlockLayout.TQ1_0_D}, none});
        m.put(GGMLType.TQ2_0, new int[][]{{BlockLayout.TQ2_0_D}, none});
        m.put(GGMLType.F8_E4M3_B128, new int[][]{none, {BlockLayout.F8B128_D}});
        m.put(GGMLType.I8_B128, new int[][]{none, {BlockLayout.I8B128_D}});
        m.put(GGMLType.NF4, new int[][]{none, {BlockLayout.BNB4_D}});
        m.put(GGMLType.FP4_BNB, new int[][]{none, {BlockLayout.BNB4_D}});
        return Collections.unmodifiableMap(m);
    }

    /** True when every factor rescales blocks of {@code t} in place (NVFP4: positive powers of two only). */
    private static boolean scalesInPlace(GGMLType t, float[] factors) {
        if (t == GGMLType.NVFP4) {
            for (float f : factors) {
                if (!(f > 0f) || Float.isInfinite(f) || Math.scalb(1f, Math.getExponent(f)) != f) {
                    return false;
                }
            }
            return true;
        }
        return t != null && SCALE_FIELDS.containsKey(t);
    }

    // ------------------------------------------------------------------ producers

    @FunctionalInterface
    interface RowFactor {
        float at(long row);
    }

    @FunctionalInterface
    private interface FloatOp {
        float apply(float v);
    }

    /** Transforms the decoded values {@code v[off, off + n)} of flat row {@code row} in place. */
    @FunctionalInterface
    interface RowOp {
        void apply(long row, float[] v, int off, int n);
    }

    /** Transforms the decoded values of {@code count} rows from {@code first}, packed in {@code v}, in place. */
    @FunctionalInterface
    interface ChunkOp {
        void apply(long first, int count, float[] v) throws IOException;
    }

    /** The bytes of a source under new sibling values. */
    private static final class Rescaled extends RowTransforms.Derived {
        private final RowSource src;

        Rescaled(RowSource src, ExpertTransforms.Siblings sib) {
            super(src.describe() + " rescaled", src.format(), src.shape(), sib);
            this.src = src;
        }

        @Override
        public boolean rowLocal() {
            return src.rowLocal();
        }

        @Override
        public long directVirtualOffset() {
            return src.directVirtualOffset();
        }

        @Override
        void produce(long first, int count, ByteBuffer out) throws IOException {
            src.readRows(first, count, out);
        }
    }

    /** A source's bytes in a new shape with the same row-major element order. */
    private static final class Reshaped extends RowTransforms.Derived {
        private final RowSource src;

        Reshaped(RowSource src, long[] shape, ExpertTransforms.Siblings sib) {
            super(src.describe() + " reshaped to " + Arrays.toString(shape), src.format(), shape, sib);
            this.src = src;
            if (Math.multiplyExact(rows(), rowBytes()) != Math.multiplyExact(src.rows(), src.rowBytes())) {
                throw new IllegalArgumentException(describe() + ": the reshape changes the byte size of "
                        + src.format() + " data");
            }
        }

        @Override
        public boolean rowLocal() {
            return src.rowLocal();
        }

        @Override
        public long directVirtualOffset() {
            return src.directVirtualOffset();
        }

        @Override
        void produce(long first, int count, ByteBuffer out) throws IOException {
            long srb = src.rowBytes();
            if (srb == rowBytes()) {
                src.readRows(first, count, out);
                return;
            }
            long start = first * rowBytes();
            long end = start + (long) count * rowBytes();
            int chunk = RowTransforms.chunkRows(srb);
            for (long r = start / srb; r * srb < end; r += chunk) {
                int c = (int) Math.min(chunk, Math.min(src.rows() - r, (end - r * srb + srb - 1) / srb));
                ByteBuffer in = RowTransforms.readRows(src, r, c);
                long from = Math.max(start, r * srb);
                long to = Math.min(end, (r + c) * srb);
                in.position((int) (from - r * srb)).limit((int) (to - r * srb));
                out.put(in);
            }
        }
    }

    /** Block bytes copied with their per-block scale fields multiplied by a per-row factor. */
    private static final class BlockScaled extends RowTransforms.Derived {
        private final RowSource src;
        private final RowFactor factor;
        private final int[][] fields;
        private final int blockBytes;

        BlockScaled(RowSource src, String what, RowFactor factor, ExpertTransforms.Siblings sib) {
            super(src.describe() + what, src.format(), src.shape(), sib);
            this.src = src;
            this.factor = factor;
            GGMLType t = src.format().ggmlType();
            this.fields = SCALE_FIELDS.get(t);
            this.blockBytes = t.getTypeSize();
        }

        @Override
        public boolean rowLocal() {
            return src.rowLocal();
        }

        @Override
        void produce(long first, int count, ByteBuffer out) throws IOException {
            int base = out.position();
            src.readRows(first, count, out);
            long rb = rowBytes();
            for (int i = 0; i < count; i++) {
                float f = factor.at(first + i);
                int rowStart = (int) (base + i * rb);
                for (int b = rowStart; b < rowStart + rb; b += blockBytes) {
                    if (fields == null) {
                        scaleNvfp4(out, b, f, first + i);
                        continue;
                    }
                    for (int o : fields[0]) {
                        float d0 = FpBlockDecoders.fp16ToFp32(out.getShort(b + o) & 0xFFFF);
                        int h = FpBlockDecoders.fp32ToFp16(d0 * f);
                        if (Float.isFinite(d0) && !Float.isFinite(FpBlockDecoders.fp16ToFp32(h))) {
                            throw overflow(first + i, d0 * f, "fp16");
                        }
                        out.putShort(b + o, (short) h);
                    }
                    for (int o : fields[1]) {
                        float d0 = out.getFloat(b + o);
                        if (Float.isFinite(d0) && !Float.isFinite(d0 * f)) {
                            throw overflow(first + i, d0 * f, "f32");
                        }
                        out.putFloat(b + o, d0 * f);
                    }
                }
            }
        }

        private IOException overflow(long row, float value, String field) {
            return new IOException(src.describe() + ": a block scale of row " + row + " becomes " + value
                    + ", which overflows its " + field + " field");
        }

        /**
         * Multiplies the four UE4M3 sub-scales of an NVFP4 block by the power of two {@code f}. A sub-scale is the
         * E4M3FN code of a non-negative value (bit 7 unused, {@code 0x7F} decoding to 0 as in ggml); the scaled value
         * must again be an exact non-NaN E4M3FN value.
         */
        private void scaleNvfp4(ByteBuffer out, int block, float f, long row) throws IOException {
            for (int s = 0; s < BlockLayout.NVFP4_QS - BlockLayout.NVFP4_D; s++) {
                int at = block + BlockLayout.NVFP4_D + s;
                int u = out.get(at) & 0xFF;
                int code = u & 0x7F;
                if (code == 0 || code == 0x7F) {
                    continue;
                }
                float scaled = Fp8Codec.E4M3FN.decode(code) * f;
                int nc = Fp8Codec.E4M3FN.encode(scaled);
                if (nc <= 0 || nc >= 0x7F || Fp8Codec.E4M3FN.decode(nc) != scaled) {
                    throw new IOException(src.describe() + ": NVFP4 sub-scale 0x" + Integer.toHexString(u) + " of row "
                            + row + " times " + f + " is not representable in UE4M3");
                }
                out.put(at, (byte) (nc | (u & 0x80)));
            }
        }
    }

    /** F32 rows quantized to TQ2_0 ternary codes with block scale 1. */
    private static final class Ternary extends RowTransforms.Derived {
        private final RowSource f32;
        private final float iscale;

        Ternary(RowSource f32, float iscale, ExpertTransforms.Siblings sib) {
            super(f32.describe() + " as ternary TQ2_0", ElementFormat.of(GGMLType.TQ2_0), f32.shape(), sib);
            this.f32 = f32;
            this.iscale = iscale;
        }

        @Override
        public boolean rowLocal() {
            return f32.rowLocal();
        }

        @Override
        void produce(long first, int count, ByteBuffer out) throws IOException {
            final int qk = GGMLType.TQ2_0.getBlockSize();
            int per = RowTransforms.chunkRows(f32.rowBytes());
            byte[] block = new byte[BlockLayout.TQ2_0_BYTES];
            short one = (short) FpBlockDecoders.fp32ToFp16(1f);
            for (int done = 0; done < count; ) {
                int c = Math.min(per, count - done);
                ByteBuffer in = RowTransforms.readRows(f32, first + done, c);
                int n = in.remaining() / 4;
                for (int b0 = 0; b0 < n; b0 += qk) {
                    Arrays.fill(block, (byte) 0);
                    for (int k = 0; k < qk; k++) {
                        float q = Math.max(-1f, Math.min(1f, (float) Math.rint(in.getFloat((b0 + k) * 4) * iscale)));
                        int at = (k / 128) * 32 + k % 32;
                        int shift = 2 * ((k % 128) / 32);
                        block[BlockLayout.TQ2_0_QS + at] |= (byte) (((int) q + 1) << shift);
                    }
                    block[BlockLayout.TQ2_0_D] = (byte) one;
                    block[BlockLayout.TQ2_0_D + 1] = (byte) (one >>> 8);
                    out.put(block);
                }
                done += c;
            }
        }
    }
}
