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

    /** {@code torch.exp(-src)} (the LeRoPE factors of Mistral Large 4, mistral4.py:92-95). */
    public static RowSource expNeg(RowSource src) {
        return nonLinear(src, "exp(-x)", v -> (float) Math.exp(-v));
    }

    /** {@code torch.tanh(src)} (Limite's XSA alpha, limite.py:170-173). */
    public static RowSource tanh(RowSource src) {
        return nonLinear(src, "tanh", v -> (float) Math.tanh(v));
    }

    /** {@code src.squeeze()}: every dimension of size 1 removed, keeping at least one (Mamba {@code conv1d}). */
    public static RowSource squeezeConv(RowSource src) {
        long[] shape = src.shape();
        long[] out = Arrays.stream(shape).filter(d -> d != 1).toArray();
        return reshape(src, out.length == 0 ? new long[]{1} : out);
    }

    /**
     * {@code src.squeeze(dim)}: the dimension {@code dim} (negative counts from the last) removed when it has one
     * element, otherwise the source unchanged, as torch's one-dimension squeeze.
     *
     * @throws IllegalArgumentException when {@code dim} lies outside the shape
     */
    public static RowSource squeezeDim(RowSource src, int dim) {
        long[] shape = src.shape();
        int rank = shape.length;
        int d = dim < 0 ? rank + dim : dim;
        if (d < 0 || d >= rank) {
            throw new IllegalArgumentException("cannot squeeze dimension " + dim + " of the rank-" + rank
                    + " tensor " + src.describe());
        }
        if (shape[d] != 1) {
            return src;
        }
        long[] out = new long[rank - 1];
        System.arraycopy(shape, 0, out, 0, d);
        System.arraycopy(shape, d + 1, out, d, rank - d - 1);
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

    /** {@code src * factor}: a source with an output scale gets the scale multiplied, the per-element and block
     *  F8_E4M3 types get one, block formats with per-block scales are scaled in place, other formats are computed
     *  element by element. */
    public static RowSource mulScalar(RowSource src, float factor) {
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.onB((LoraFactorSource) src, b -> mulScalar(b, factor));
        }
        float[] os = src.outputScale();
        GGMLType t = src.format().ggmlType();
        if (os != null || t == GGMLType.F8_E4M3 || t == GGMLType.F8_E4M3FN) {
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

    /**
     * {@code src - src.mean()}: each row minus the mean of its elements, the zero-mean router bias Sarvam's
     * converter applies to {@code e_score_correction_bias} before the tensor is written (sarvam.py
     * {@code _zero_mean_tensor}). For the one-dimensional bias vector the row is the whole tensor.
     */
    public static RowSource subtractRowMean(RowSource src) {
        if (src instanceof LoraFactorSource) {
            throw LoraFactorOps.unsupported(src, "the zero-mean map");
        }
        return mapped(src, " zero-mean", false, (row, v, off, n) -> {
            float mean = 0.0f;
            for (int i = 0; i < n; i++) {
                mean += v[off + i];
            }
            mean /= n;
            for (int i = 0; i < n; i++) {
                v[off + i] -= mean;
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

    // ------------------------------------------------------------------ layout producers

    /**
     * {@code src.permute(*order)}: the axes of the tensor in the given order, as the converter classes that reshape a
     * patch embedding from a transformer layout into a convolution kernel do (LC/conversion/lfm2.py:391-392,
     * phi.py:334-340). Values are computed in f32 and stored in the source's float format (BF16 as F32, as the
     * converter hands such tensors to its hooks).
     *
     * @throws IllegalArgumentException when {@code order} is not a permutation of the tensor's axes
     */
    public static RowSource permuteDims(RowSource src, int... order) throws IOException {
        long[] in = src.shape();
        if (order.length != in.length) {
            throw new IllegalArgumentException(src.describe() + ": permute order " + Arrays.toString(order)
                    + " does not name all " + in.length + " axes");
        }
        boolean[] seen = new boolean[in.length];
        long[] out = new long[in.length];
        for (int i = 0; i < order.length; i++) {
            int a = order[i];
            if (a < 0 || a >= in.length || seen[a]) {
                throw new IllegalArgumentException(src.describe() + ": permute order " + Arrays.toString(order)
                        + " is not a permutation of its axes");
            }
            seen[a] = true;
            out[i] = in[a];
        }
        float[] values = allFloats(src);
        long[] inStride = strides(in);
        float[] result = new float[values.length];
        for (int o = 0; o < result.length; o++) {
            long rem = o;
            long at = 0;
            for (int k = out.length - 1; k >= 0; k--) {
                long c = rem % out[k];
                rem /= out[k];
                at += c * inStride[order[k]];
            }
            result[o] = values[(int) at];
        }
        return new ArraySource(src, " permuted " + Arrays.toString(order), out, result);
    }

    /**
     * {@code src.sum(dim=axis)}: the converter's reduction of a patch embedding's temporal slabs
     * (LC/conversion/muse_glimmer.py:117-119). Values are computed in f32 and stored in the source's float format.
     *
     * @throws IllegalArgumentException when {@code axis} is not an axis of the tensor
     */
    public static RowSource sumAxis(RowSource src, int axis) throws IOException {
        long[] in = src.shape();
        int d = RowTransforms.normalizeDim(src, axis);
        long[] out = new long[in.length - 1];
        for (int i = 0, j = 0; i < in.length; i++) {
            if (i != d) {
                out[j++] = in[i];
            }
        }
        long groups = in[d];
        float[] values = allFloats(src);
        long[] inStride = strides(in);
        long outer = 1;
        for (int i = 0; i < d; i++) {
            outer = Math.multiplyExact(outer, in[i]);
        }
        long inner = 1;
        for (int i = d + 1; i < in.length; i++) {
            inner = Math.multiplyExact(inner, in[i]);
        }
        long outSize = Math.multiplyExact(outer, inner);
        if (outSize > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException(src.describe() + ": the summed tensor is too large");
        }
        float[] result = new float[(int) outSize];
        for (long o = 0; o < outSize; o++) {
            long outerIdx = o / inner;
            long innerIdx = o % inner;
            long base = outerIdx * groups * inner + innerIdx;
            float sum = 0f;
            for (long g = 0; g < groups; g++) {
                sum += values[(int) (base + g * inner)];
            }
            result[(int) o] = sum;
        }
        return new ArraySource(src, " summed over axis " + d, out, result);
    }

    /**
     * The bilinear resampling of a {@code [1, h, w, c]} tensor to {@code [1, toH, toW, c]} with
     * {@code align_corners=True}, as the converter resamples a RADIO position table to the image grid
     * (LC/conversion/nemotron.py:184-194). Values are computed in f32 and stored in the source's float format.
     *
     * @throws IllegalArgumentException when the source shape is not {@code [1, h, w, c]} or a target side is not
     *                                  positive
     */
    public static RowSource resizeBilinear(RowSource src, int toH, int toW) throws IOException {
        long[] in = src.shape();
        if (in.length != 4 || in[0] != 1 || toH <= 0 || toW <= 0) {
            throw new IllegalArgumentException(src.describe() + ": bilinear resampling needs a [1, h, w, c] tensor,"
                    + " got " + Arrays.toString(in) + " to " + toH + "x" + toW);
        }
        int h = Math.toIntExact(in[1]);
        int w = Math.toIntExact(in[2]);
        int c = Math.toIntExact(in[3]);
        long[] out = {1, toH, toW, c};
        float[] values = allFloats(src);
        float[] result = new float[Math.multiplyExact(Math.multiplyExact(toH, toW), c)];
        for (int oy = 0; oy < toH; oy++) {
            float y = toH > 1 ? oy * (h - 1f) / (toH - 1) : 0f;
            int y0 = Math.min((int) y, Math.max(h - 1, 0));
            int y1 = Math.min(y0 + 1, Math.max(h - 1, 0));
            float wy = h > 1 ? y - y0 : 0f;
            for (int ox = 0; ox < toW; ox++) {
                float x = toW > 1 ? ox * (w - 1f) / (toW - 1) : 0f;
                int x0 = Math.min((int) x, Math.max(w - 1, 0));
                int x1 = Math.min(x0 + 1, Math.max(w - 1, 0));
                float wx = w > 1 ? x - x0 : 0f;
                for (int ch = 0; ch < c; ch++) {
                    float v00 = values[((y0 * w) + x0) * c + ch];
                    float v01 = values[((y0 * w) + x1) * c + ch];
                    float v10 = values[((y1 * w) + x0) * c + ch];
                    float v11 = values[((y1 * w) + x1) * c + ch];
                    float top = v00 + (v01 - v00) * wx;
                    float bottom = v10 + (v11 - v10) * wx;
                    result[((oy * toW) + ox) * c + ch] = top + (bottom - top) * wy;
                }
            }
        }
        return new ArraySource(src, " resampled to " + toH + "x" + toW, out, result);
    }

    /** The row-major strides of {@code shape}. */
    private static long[] strides(long[] shape) {
        long[] s = new long[shape.length];
        long step = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            s[i] = step;
            step = Math.multiplyExact(step, shape[i]);
        }
        return s;
    }

    /** Every element of {@code src} as f32, row-major. */
    private static float[] allFloats(RowSource src) throws IOException {
        if (!DtypePolicy.decodes(src.format())) {
            throw new IllegalArgumentException(src.describe() + ": no decoder for its format " + src.format());
        }
        long[] shape = src.shape();
        long inner = shape.length == 0 ? 1 : shape[shape.length - 1];
        long rows = src.rows();
        long total = Math.multiplyExact(rows, inner);
        if (rows > Integer.MAX_VALUE - 8 || total > Integer.MAX_VALUE - 8 || inner > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException(src.describe() + ": the tensor is too large to materialize");
        }
        if (total == 0) {
            return new float[0];
        }
        ByteBuffer in = RowTransforms.readRows(src, 0, (int) rows);
        float[] values = new float[(int) total];
        int rowBytes = (int) src.rowBytes();
        for (int r = 0; r < rows; r++) {
            DtypePolicy.decode(src.format(), in, r * rowBytes, values, r * (int) inner, (int) inner);
        }
        return values;
    }

    // ------------------------------------------------------------------ dtype conversion

    /**
     * {@code src.to(target)} for F32, F16 and BF16 (round to nearest even), Q8_0 (the reference quantizer,
     * rounding half away from zero) or TQ1_0 / TQ2_0 (the reference ternary quantizers). Any float, integer,
     * eight-bit float or ggml block source is decoded first; an output scale and the activation values are kept;
     * IllegalArgumentException for another target or a source format without a decoder.
     */
    public static RowSource convert(RowSource src, GGMLType target) {
        Objects.requireNonNull(target, "target");
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.convert((LoraFactorSource) src, target);
        }
        if (src.format().ggmlType() == target) {
            return src;
        }
        if (target != GGMLType.F32 && target != GGMLType.F16 && target != GGMLType.BF16
                && target != GGMLType.Q8_0 && target != GGMLType.TQ1_0 && target != GGMLType.TQ2_0) {
            throw new IllegalArgumentException(src.describe() + ": cannot convert to " + target
                    + "; the targets are F32, F16, BF16, Q8_0, TQ1_0 and TQ2_0");
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
        m.put(GGMLType.F8_E4M3_B32, new int[][]{none, {BlockLayout.F8B32_D}});
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

    /** A materialized float array in the source's float format (BF16 as F32). */
    private static final class ArraySource extends RowTransforms.Derived {
        private final float[] values;
        private final GGMLType target;
        private final int inner;

        ArraySource(RowSource src, String what, long[] shape, float[] values) {
            super(src.describe() + what, ElementFormat.of(targetOf(src.format())), shape,
                    ExpertTransforms.Siblings.NONE);
            this.target = targetOf(src.format());
            this.values = values;
            long expected = Math.multiplyExact(rows(), shape.length == 0 ? 1 : shape[shape.length - 1]);
            if (expected != values.length) {
                throw new IllegalArgumentException(describe() + ": " + values.length + " values do not fill "
                        + expected + " elements");
            }
            this.inner = Math.toIntExact(shape.length == 0 ? 1 : shape[shape.length - 1]);
        }

        private static GGMLType targetOf(ElementFormat f) {
            GGMLType t = floatOut(f);
            return t == GGMLType.BF16 ? GGMLType.F32 : t;
        }

        @Override
        public boolean rowLocal() {
            return false;
        }

        @Override
        void produce(long first, int count, ByteBuffer out) throws IOException {
            DtypePolicy.encode(target, values, Math.toIntExact(first) * inner, count * inner, out);
        }
    }
}
