package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.LongUnaryOperator;

/**
 * Mixture-of-experts layouts: stacking per-expert tensors into one 3-D tensor (LC/conversion/llama.py:268-295),
 * splitting fused expert tensors ({@code [E, 2F, D]} gate/up checkpoints) and transposing the last two dimensions of
 * experts stored input-major ({@code [E, D, F]}: Llama4, Qwen3-VL, gpt-oss, LC/conversion/gpt_oss.py:107-119).
 *
 * <p>{@link Siblings} is the sibling-value bookkeeping every transform producer shares: the output and activation
 * scales, the zero point, the act-order permutation and the activation scheme a quantized source carries, and the
 * rules that keep them correct across moves, concatenations and stacks.</p>
 */
public final class ExpertTransforms {

    private static final Logger LOG = LogManager.getLogger(ExpertTransforms.class);

    private ExpertTransforms() {
    }

    /**
     * {@code torch.stack(parts, dim=0)}: a new outermost dimension of {@code parts.size()} experts. Parts must have
     * equal shapes and formats. Per-tensor output scales merge into one value per expert; when only some parts have
     * one, or a part's scale is not a single value, each part's scale is folded into its weight first. Activation
     * scales and zero points of one value each merge the same way.
     *
     * @throws IllegalArgumentException when the parts differ in shape or format
     */
    public static RowSource stack(List<? extends RowSource> parts) {
        List<RowSource> ps = new ArrayList<>(Objects.requireNonNull(parts, "parts"));
        if (ps.isEmpty()) {
            throw new IllegalArgumentException("stack needs at least one part");
        }
        if (ps.get(0) instanceof LoraFactorSource) {
            return LoraFactorOps.stack(ps);
        }
        RowSource first = ps.get(0);
        for (RowSource p : ps) {
            if (!Arrays.equals(p.shape(), first.shape()) || !p.format().equals(first.format())) {
                throw new IllegalArgumentException("stack: " + p.describe() + " does not match the shape "
                        + Arrays.toString(first.shape()) + " and format " + first.format() + " of " + first.describe());
            }
        }
        ps = Siblings.reconcile(ps, Siblings.Merge.STACK);
        long[] inner = ps.get(0).shape();
        long[] shape = new long[inner.length + 1];
        shape[0] = ps.size();
        System.arraycopy(inner, 0, shape, 1, inner.length);
        long rowsPerPart = ps.get(0).rows();
        RowTransforms.RowIndex index = new RowTransforms.RowIndex() {
            @Override
            public int part(long r) {
                return (int) (r / rowsPerPart);
            }

            @Override
            public long row(long r) {
                return r % rowsPerPart;
            }
        };
        return new RowTransforms.RowMap("stack of " + ps.size() + " experts starting with " + first.describe(), ps,
                shape, index, null, Siblings.stacked(ps));
    }

    /**
     * Splits {@code src} into {@code parts} equal slices along {@code dim} ({@code torch.chunk} of a fused expert
     * tensor, for example {@code [E, 2F, D]} along {@code -2} into gate and up).
     *
     * @throws IllegalArgumentException when the dimension does not divide into {@code parts}
     */
    public static RowSource[] splitExperts(RowSource src, int dim, int parts) {
        int d = RowTransforms.normalizeDim(src, dim);
        long size = src.shape()[d];
        if (parts <= 0 || size % parts != 0) {
            throw new IllegalArgumentException(src.describe() + ": dimension " + d + " of size " + size
                    + " does not split into " + parts + " equal parts");
        }
        long each = size / parts;
        RowSource[] out = new RowSource[parts];
        for (int i = 0; i < parts; i++) {
            out[i] = RowTransforms.sliceDim(src, d, i * each, each);
        }
        return out;
    }

    /**
     * {@code src.transpose(-1, -2)}. Element formats are transposed byte for byte, and so is F8_E4M3 (one code per
     * element, no block scale) when the new rows are whole blocks of 32. F8_E4M3_B128 and I8_B128 tensors whose last
     * two dimensions are multiples of 128 are re-blocked exactly, which requires the block scale to be the same across
     * each 128 x 128 tile (block-128 FP8 / int8 checkpoints); other block formats are dequantized to F32 with a
     * warning. Output scales are kept; activation values are dropped with a warning because the input
     * dimension becomes the output one, and an act-order permutation is folded first. Not row-local.
     *
     * @throws IllegalArgumentException for a tensor of rank below 2
     */
    public static RowSource transposeLast2(RowSource src) {
        if (src instanceof LoraFactorSource) {
            return LoraFactorOps.transposeLast2((LoraFactorSource) src);
        }
        long[] shape = src.shape();
        if (shape.length < 2) {
            throw new IllegalArgumentException(src.describe() + ": transposeLast2 needs a rank of at least 2");
        }
        src = Siblings.withoutPerm(src);
        Siblings sib = Siblings.of(src);
        if (sib.hasActivation()) {
            LOG.warn("{}: activation quantization values are dropped by the transpose; the weight loads weight-only",
                    src.describe());
            sib = sib.withoutActivation();
        }
        long rowsIn = shape[shape.length - 2];
        long cols = shape[shape.length - 1];
        GGMLType t = src.format().ggmlType();
        boolean reblock = (t == GGMLType.F8_E4M3_B128 || t == GGMLType.I8_B128) && rowsIn % 128 == 0;
        boolean codes = t == GGMLType.F8_E4M3 && rowsIn % t.getBlockSize() == 0;
        if (src.format().isBlock() && !reblock && !codes) {
            src = ElementTransforms.dequantFallback(src, "transposeLast2");
            sib = Siblings.of(src).withoutActivation();
        } else if (!src.format().isBlock() && src.format().stDType() != null
                && src.format().stDType().bits() % 8 != 0) {
            src = ElementTransforms.dequantFallback(src, "transposeLast2 of a sub-byte format");
            sib = Siblings.of(src).withoutActivation();
        }
        long[] out = shape.clone();
        out[out.length - 2] = cols;
        out[out.length - 1] = rowsIn;
        return new Transposed(src, out, sib);
    }

    /** {@code [..., R, C]} to {@code [..., C, R]}, reading the source in row chunks and extracting column slabs. */
    private static final class Transposed extends RowTransforms.Derived {
        private final RowSource src;
        private final long rowsIn;
        private final long cols;
        private final boolean reblock;
        private final int elementBytes;

        Transposed(RowSource src, long[] shape, Siblings sib) {
            super(src.describe() + " transposed", src.format(), shape, sib);
            this.src = src;
            this.rowsIn = shape[shape.length - 1];
            this.cols = shape[shape.length - 2];
            boolean codes = src.format().ggmlType() == GGMLType.F8_E4M3;
            this.reblock = src.format().isBlock() && !codes;
            this.elementBytes = codes ? 1 : reblock ? 0 : src.format().stDType().bits() / 8;
            if (cols > Integer.MAX_VALUE || rowsIn > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(describe() + ": a matrix side exceeds 2^31 elements");
            }
        }

        @Override
        public boolean rowLocal() {
            return false;
        }

        @Override
        void produce(long first, int count, ByteBuffer out) throws IOException {
            long end = first + count;
            long r = first;
            while (r < end) {
                long batch = r / cols;
                long c0 = r % cols;
                long c1 = Math.min(cols, c0 + (end - r));
                if (reblock) {
                    reblock(batch, c0, c1, out);
                } else {
                    elements(batch, c0, c1, out);
                }
                r += c1 - c0;
            }
        }

        /** Output rows {@code [c0, c1)} of matrix {@code batch} for an element format. */
        private void elements(long batch, long c0, long c1, ByteBuffer out) throws IOException {
            int es = elementBytes;
            long srcRb = src.rowBytes();
            int width = (int) (c1 - c0);
            int base = out.position();
            int chunk = RowTransforms.chunkRows(srcRb);
            for (long i = 0; i < rowsIn; i += chunk) {
                int n = (int) Math.min(chunk, rowsIn - i);
                ByteBuffer in = RowTransforms.readRows(src, batch * rowsIn + i, n);
                for (int k = 0; k < n; k++) {
                    int inRow = (int) (k * srcRb + c0 * es);
                    for (int c = 0; c < width; c++) {
                        int o = (int) (base + ((long) c * rowsIn + i + k) * es);
                        for (int b = 0; b < es; b++) {
                            out.put(o + b, in.get(inRow + c * es + b));
                        }
                    }
                }
            }
            out.position((int) (base + (long) width * rowsIn * es));
        }

        /**
         * Output rows {@code [c0, c1)} of matrix {@code batch} for F8_E4M3_B128 / I8_B128: output block {@code j} of
         * output row {@code c} holds column {@code c} of input rows {@code [128 j, 128 j + 128)} and their common scale
         * (both types store {@code {f32 d; 128 one-byte codes}}).
         */
        private void reblock(long batch, long c0, long c1, ByteBuffer out) throws IOException {
            final int qk = 128;
            final int bb = BlockLayout.F8B128_BYTES;
            long srcRb = src.rowBytes();
            int width = (int) (c1 - c0);
            int base = out.position();
            long outRb = rowBytes();
            int tilesPerChunk = Math.max(1, RowTransforms.chunkRows(srcRb * qk));
            for (long tile = 0; tile < rowsIn / qk; tile += tilesPerChunk) {
                int nt = (int) Math.min(tilesPerChunk, rowsIn / qk - tile);
                ByteBuffer in = RowTransforms.readRows(src, batch * rowsIn + tile * qk, nt * qk);
                for (int t = 0; t < nt; t++) {
                    for (int c = 0; c < width; c++) {
                        long col = c0 + c;
                        int inBlock = (int) (col / qk);
                        int inCode = (int) (col % qk);
                        int d0 = in.getInt((int) ((long) t * qk * srcRb + (long) inBlock * bb));
                        int o = (int) (base + c * outRb + (tile + t) * bb);
                        out.putInt(o, d0);
                        for (int k = 0; k < qk; k++) {
                            int rowOff = (int) (((long) t * qk + k) * srcRb + (long) inBlock * bb);
                            if (in.getInt(rowOff) != d0) {
                                throw new IOException(src.describe() + ": block scales differ inside the 128 x 128"
                                        + " tile at rows " + (batch * rowsIn + (tile + t) * qk) + ", column block "
                                        + inBlock + "; the transpose cannot keep the " + src.format()
                                        + " blocks exact");
                            }
                            out.put(o + BlockLayout.F8B128_QS + k, in.get(rowOff + BlockLayout.F8B128_QS + inCode));
                        }
                    }
                }
            }
            out.position((int) (base + width * outRb));
        }
    }

    /**
     * The sibling values a quantized source carries ({@link RowSource#outputScale()} and the activation values), with
     * the rules that keep them correct when sources are moved, merged or folded. Immutable; arrays are copied in and
     * out.
     */
    static final class Siblings {

        /** How parts are merged. */
        enum Merge { CONCAT_ROWS, CONCAT_COLS, STACK }

        static final Siblings NONE = new Siblings(null, null, null, null, null);

        private final float[] outputScale;
        private final float[] inputScale;
        private final int[] inputZeroPoint;
        private final int[] inputPerm;
        private final ActQuantScheme actQuant;

        Siblings(float[] outputScale, float[] inputScale, int[] inputZeroPoint, int[] inputPerm,
                 ActQuantScheme actQuant) {
            this.outputScale = outputScale == null ? null : outputScale.clone();
            this.inputScale = inputScale == null ? null : inputScale.clone();
            this.inputZeroPoint = inputZeroPoint == null ? null : inputZeroPoint.clone();
            this.inputPerm = inputPerm == null ? null : inputPerm.clone();
            this.actQuant = actQuant;
        }

        /** The sibling values of {@code s}. */
        static Siblings of(RowSource s) {
            if (s instanceof RowTransforms.Derived) {
                return ((RowTransforms.Derived) s).siblings();
            }
            return new Siblings(s.outputScale(), s.inputScale(), s.inputZeroPoint(), s.inputPerm(), s.actQuant());
        }

        float[] outputScale() {
            return outputScale == null ? null : outputScale.clone();
        }

        float[] inputScale() {
            return inputScale == null ? null : inputScale.clone();
        }

        int[] inputZeroPoint() {
            return inputZeroPoint == null ? null : inputZeroPoint.clone();
        }

        int[] inputPerm() {
            return inputPerm == null ? null : inputPerm.clone();
        }

        ActQuantScheme actQuant() {
            return actQuant;
        }

        /** True when an activation scale, zero point or scheme is present. */
        boolean hasActivation() {
            return inputScale != null || inputZeroPoint != null || actQuant != null;
        }

        Siblings withOutputScale(float[] scale) {
            return new Siblings(scale, inputScale, inputZeroPoint, inputPerm, actQuant);
        }

        Siblings withoutOutputScale() {
            return withOutputScale(null);
        }

        Siblings withoutInputPerm() {
            return new Siblings(outputScale, inputScale, inputZeroPoint, null, actQuant);
        }

        Siblings withoutActivation() {
            return new Siblings(outputScale, null, null, inputPerm, null);
        }

        /**
         * These values after dim 0 of {@code src} is moved to {@code outD0} entries, output entry {@code e} taking
         * input entry {@code srcEntry(e)} ({@code -1} for a new entry). Single values stay; per-expert values of a
         * 3-D stack follow their experts.
         *
         * @throws IllegalArgumentException when per-expert values do not match the experts, or new entries would
         *                                  need values
         */
        Siblings remapDim0(RowSource src, long outD0, LongUnaryOperator srcEntry) {
            long d0 = src.shape()[0];
            int rank = src.shape().length;
            float[] os = remapF(src, outputScale, d0, rank, outD0, srcEntry);
            float[] is = remapF(src, inputScale, d0, rank, outD0, srcEntry);
            int[] zp = null;
            if (inputZeroPoint != null) {
                float[] f = new float[inputZeroPoint.length];
                for (int i = 0; i < f.length; i++) {
                    f[i] = inputZeroPoint[i];
                }
                float[] g = remapF(src, f, d0, rank, outD0, srcEntry);
                zp = new int[g.length];
                for (int i = 0; i < g.length; i++) {
                    zp[i] = (int) g[i];
                }
            }
            return new Siblings(os, is, zp, inputPerm, actQuant);
        }

        private static float[] remapF(RowSource src, float[] a, long d0, int rank, long outD0,
                                      LongUnaryOperator srcEntry) {
            if (a == null || a.length == 1) {
                return a;
            }
            if (rank < 3 || a.length != d0) {
                throw new IllegalArgumentException(src.describe() + ": " + a.length + " per-expert values do not"
                        + " match its shape " + Arrays.toString(src.shape()));
            }
            float[] out = new float[Math.toIntExact(outD0)];
            for (int e = 0; e < out.length; e++) {
                long s = srcEntry.applyAsLong(e);
                if (s < 0) {
                    throw new IllegalArgumentException(src.describe() + ": new experts have no per-expert scale");
                }
                out[e] = a[(int) s];
            }
            return out;
        }

        /** {@code s} with its act-order permutation folded into the weight. */
        static RowSource withoutPerm(RowSource s) {
            if (s.inputPerm() == null) {
                return s;
            }
            try {
                return s.foldInputPerm();
            } catch (UnsupportedOperationException e) {
                return ColumnTransforms.foldPermGeneric(s);
            }
        }

        /**
         * {@code s} with its output scale folded into the weight: into the per-block scales when the format has them,
         * else through the source's own fold, else by dequantizing.
         */
        static RowSource withoutScale(RowSource s) {
            if (s.outputScale() == null) {
                return s;
            }
            if (ElementTransforms.foldsIntoBlocks(s)) {
                return ElementTransforms.foldScale(s);
            }
            try {
                return s.foldOutputScale();
            } catch (UnsupportedOperationException e) {
                return ElementTransforms.foldScale(s);
            }
        }

        /**
         * {@code parts} adjusted so their siblings merge under {@code mode}: differing act-order permutations are
         * folded, output scales that cannot be merged are folded, and parts whose format the folds changed are
         * brought to F32 together.
         */
        static List<RowSource> reconcile(List<RowSource> parts, Merge mode) {
            List<RowSource> ps = new ArrayList<>(parts);
            boolean sameBefore = sameFormat(ps);
            int[] perm0 = ps.get(0).inputPerm();
            boolean permsEqual = true;
            for (RowSource p : ps) {
                permsEqual &= Arrays.equals(perm0, p.inputPerm());
            }
            if (!permsEqual || (mode == Merge.CONCAT_COLS && perm0 != null)) {
                ps.replaceAll(Siblings::withoutPerm);
            }
            if (!scalesMerge(ps, mode)) {
                ps.replaceAll(Siblings::withoutScale);
            }
            if (sameBefore && !sameFormat(ps)) {
                LOG.warn("{}: folding sibling scales changed part formats; the parts are merged as F32",
                        ps.get(0).describe());
                ps.replaceAll(p -> ElementTransforms.convert(p, GGMLType.F32));
            }
            return ps;
        }

        private static boolean sameFormat(List<RowSource> ps) {
            ElementFormat f0 = ps.get(0).format();
            for (RowSource p : ps) {
                if (!p.format().equals(f0)) {
                    return false;
                }
            }
            return true;
        }

        private static boolean scalesMerge(List<RowSource> ps, Merge mode) {
            float[] s0 = ps.get(0).outputScale();
            boolean allNull = true;
            boolean allSingle = true;
            boolean allEqual = true;
            for (RowSource p : ps) {
                float[] s = p.outputScale();
                allNull &= s == null;
                allSingle &= s != null && s.length == 1;
                allEqual &= Arrays.equals(s, s0);
            }
            if (allNull) {
                return true;
            }
            switch (mode) {
                case STACK:
                    return allSingle;
                case CONCAT_COLS:
                    return allEqual;
                default:
                    if (ps.get(0).shape().length >= 3) {
                        for (RowSource p : ps) {
                            if (p.outputScale() == null) {
                                return false;
                            }
                        }
                        return true;
                    }
                    return allEqual;
            }
        }

        /** The siblings of a stack of reconciled parts: single values merged into one per expert. */
        static Siblings stacked(List<RowSource> ps) {
            float[] os = gatherSingles(ps, true);
            Activation a = Activation.merge(ps, Merge.STACK);
            return new Siblings(os, a.scale, a.zeroPoint, ps.get(0).inputPerm(), a.scheme);
        }

        /** The siblings of a concatenation along dim 0 of reconciled parts. */
        static Siblings concatDim0(List<RowSource> ps) {
            float[] os = ps.get(0).outputScale();
            if (os != null && ps.get(0).shape().length >= 3) {
                os = expandConcat(ps, true);
            }
            Activation a = Activation.merge(ps, Merge.CONCAT_ROWS);
            return new Siblings(os, a.scale, a.zeroPoint, ps.get(0).inputPerm(), a.scheme);
        }

        /** The siblings of a concatenation along the innermost dimension of reconciled parts. */
        static Siblings concatCols(List<RowSource> ps) {
            Activation a = Activation.merge(ps, Merge.CONCAT_COLS);
            return new Siblings(ps.get(0).outputScale(), a.scale, a.zeroPoint, null, a.scheme);
        }

        private static float[] gatherSingles(List<RowSource> ps, boolean output) {
            float[] out = new float[ps.size()];
            for (int i = 0; i < out.length; i++) {
                float[] s = output ? ps.get(i).outputScale() : ps.get(i).inputScale();
                if (s == null) {
                    return null;
                }
                out[i] = s[0];
            }
            return out;
        }

        /** Per-expert values of a dim-0 concatenation of 3-D parts, single values repeated per expert. */
        private static float[] expandConcat(List<RowSource> ps, boolean output) {
            int total = 0;
            for (RowSource p : ps) {
                total = Math.addExact(total, Math.toIntExact(p.shape()[0]));
            }
            float[] out = new float[total];
            int at = 0;
            for (RowSource p : ps) {
                float[] s = output ? p.outputScale() : p.inputScale();
                int e = (int) p.shape()[0];
                for (int i = 0; i < e; i++) {
                    out[at + i] = s.length == 1 ? s[0] : s[i];
                }
                at += e;
            }
            return out;
        }

        /** The merged activation values of parts, or none (with a warning) when they cannot be merged. */
        private static final class Activation {
            final float[] scale;
            final int[] zeroPoint;
            final ActQuantScheme scheme;

            Activation(float[] scale, int[] zeroPoint, ActQuantScheme scheme) {
                this.scale = scale;
                this.zeroPoint = zeroPoint;
                this.scheme = scheme;
            }

            static Activation merge(List<RowSource> ps, Merge mode) {
                RowSource p0 = ps.get(0);
                boolean anyNull = false;
                boolean anyPresent = false;
                boolean presenceEqual = true;
                boolean schemeEqual = true;
                boolean scaleEqual = true;
                boolean zpEqual = true;
                boolean allSingle = true;
                for (RowSource p : ps) {
                    boolean has = p.inputScale() != null || p.inputZeroPoint() != null || p.actQuant() != null;
                    anyNull |= !has;
                    anyPresent |= has;
                    schemeEqual &= p.actQuant() == p0.actQuant();
                    presenceEqual &= (p.inputScale() == null) == (p0.inputScale() == null)
                            && (p.inputZeroPoint() == null) == (p0.inputZeroPoint() == null);
                    scaleEqual &= Arrays.equals(p.inputScale(), p0.inputScale());
                    zpEqual &= Arrays.equals(p.inputZeroPoint(), p0.inputZeroPoint());
                    allSingle &= (p.inputScale() == null || p.inputScale().length == 1)
                            && (p.inputZeroPoint() == null || p.inputZeroPoint().length == 1);
                }
                if (!anyPresent) {
                    return new Activation(null, null, null);
                }
                if (anyNull || !schemeEqual || !presenceEqual) {
                    return drop(p0, "the parts carry different activation schemes");
                }
                switch (mode) {
                    case STACK:
                        if (!allSingle) {
                            return drop(p0, "a part carries more than one activation scale");
                        }
                        return new Activation(gatherSingles(ps, false), gatherZeroPoints(ps), p0.actQuant());
                    case CONCAT_COLS:
                        if (!scaleEqual || !zpEqual) {
                            return drop(p0, "the concatenated inputs carry different activation scales");
                        }
                        return new Activation(p0.inputScale(), p0.inputZeroPoint(), p0.actQuant());
                    default:
                        if (p0.shape().length >= 3) {
                            float[] s = p0.inputScale() == null ? null : expandConcat(ps, false);
                            if (!zpEqual) {
                                return drop(p0, "the concatenated experts carry different activation zero points");
                            }
                            return new Activation(s, p0.inputZeroPoint(), p0.actQuant());
                        }
                        if (!zpEqual || !allSingle) {
                            return drop(p0, "the fused parts carry different activation zero points");
                        }
                        return new Activation(maxScale(ps), p0.inputZeroPoint(), p0.actQuant());
                }
            }

            /**
             * The largest activation scale of fused parts, as vLLM requantizes fused modules to their largest
             * {@code input_scale} (V/model_executor/layers/quantization/fp8.py process_weights_after_loading).
             */
            private static float[] maxScale(List<RowSource> ps) {
                if (ps.get(0).inputScale() == null) {
                    return null;
                }
                float m = ps.get(0).inputScale()[0];
                for (RowSource p : ps) {
                    m = Math.max(m, p.inputScale()[0]);
                }
                return new float[]{m};
            }

            private static int[] gatherZeroPoints(List<RowSource> ps) {
                if (ps.get(0).inputZeroPoint() == null) {
                    return null;
                }
                int[] out = new int[ps.size()];
                for (int i = 0; i < out.length; i++) {
                    out[i] = ps.get(i).inputZeroPoint()[0];
                }
                return out;
            }

            private static Activation drop(RowSource p0, String why) {
                LOG.warn("{}: activation quantization dropped because {}; the merged weight loads weight-only",
                        p0.describe(), why);
                return new Activation(null, null, null);
            }
        }
    }
}
