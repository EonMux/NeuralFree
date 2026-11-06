package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.quant.IntWeightSpec.Formula;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.quant.ActQuantScheme;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * The ggml-typed rows of one quantized weight, produced on demand from its checkpoint tensors. Shape
 * {@code [N, K]}, or {@code [E, N, K]} for stacked experts. Immutable and thread-safe.
 */
final class IntGroupSource implements RowSource {

    /** Codes decoded per pass of a read. */
    private static final int CODE_BUDGET = 1 << 24;

    private final IntWeightSpec spec;
    private final GGMLType target;
    private final GGMLType fallback;
    private final int[] perm;
    private final Act act;
    private final float[] columnScale;
    private final float[] inputPreScale;
    private final String checkpoint;
    private final ElementFormat format;
    private final long rowBytes;

    IntGroupSource(IntWeightSpec spec, GGMLType target, GGMLType fallback, int[] perm, Act act, String checkpoint) {
        this(spec, target, fallback, perm, act, checkpoint, null);
    }

    IntGroupSource(IntWeightSpec spec, GGMLType target, GGMLType fallback, int[] perm, Act act, String checkpoint,
                   float[] columnScale) {
        this(spec, target, fallback, perm, act, checkpoint, columnScale, null);
    }

    /**
     * @param columnScale   a per-column multiplier folded into the produced values, or null
     * @param inputPreScale the per-column multiplier the runtime applies to the activation before its quantization
     *                      ({@link RowSource#inputPreScale()}), or null
     */
    IntGroupSource(IntWeightSpec spec, GGMLType target, GGMLType fallback, int[] perm, Act act, String checkpoint,
                   float[] columnScale, float[] inputPreScale) {
        this.spec = spec;
        this.target = target;
        this.fallback = fallback;
        this.perm = perm == null ? null : perm.clone();
        this.act = act;
        this.columnScale = columnScale == null ? null : columnScale.clone();
        this.inputPreScale = inputPreScale == null ? null : inputPreScale.clone();
        this.checkpoint = checkpoint;
        this.format = ElementFormat.of(target);
        this.rowBytes = format.rowBytes(spec.k);
    }

    @Override
    public String describe() {
        return spec.module + " " + spec.format + " b" + spec.bits + " g" + spec.groupSize + " -> " + target;
    }

    @Override
    public ElementFormat format() {
        return format;
    }

    @Override
    public long[] shape() {
        return spec.experts > 0 ? new long[]{spec.experts, spec.n, spec.k} : new long[]{spec.n, spec.k};
    }

    @Override
    public long rows() {
        return Math.max(1, spec.experts) * spec.n;
    }

    @Override
    public long rowBytes() {
        return rowBytes;
    }

    @Override
    public boolean rowLocal() {
        return spec.codes.rowLocal() && spec.scales.rowLocal() && (spec.zeros == null || spec.zeros.rowLocal());
    }

    @Override
    public float[] inputScale() {
        return act.scale();
    }

    @Override
    public int[] inputZeroPoint() {
        return act.zeroPoint();
    }

    @Override
    public int[] inputPerm() {
        return perm == null ? null : perm.clone();
    }

    @Override
    public ActQuantScheme actQuant() {
        return act.scheme;
    }

    @Override
    public float[] inputPreScale() {
        return inputPreScale == null ? null : inputPreScale.clone();
    }

    @Override
    public RowSource foldInputPerm() {
        if (perm == null) {
            throw new UnsupportedOperationException(describe() + " has no input permutation to fold");
        }
        return new IntGroupSource(spec, fallback, fallback, null, act, checkpoint, columnScale, inputPreScale);
    }

    @Override
    public void readRows(long first, int count, ByteBuffer dst) throws IOException {
        if (first < 0 || count < 0 || first + count > rows()) {
            throw new IllegalArgumentException(describe() + ": rows [" + first + ", " + (first + count)
                    + ") lie outside [0, " + rows() + ")");
        }
        long bytes = count * rowBytes;
        if (dst.remaining() < bytes) {
            throw new IllegalArgumentException(describe() + ": " + count + " rows need " + bytes + " bytes, "
                    + dst.remaining() + " remain");
        }
        ByteBuffer out = dst.slice().order(ByteOrder.LITTLE_ENDIAN);
        int chunk = Math.max(1, CODE_BUDGET / spec.k);
        int done = 0;
        while (done < count) {
            long r = first + done;
            int e = (int) (r / spec.n);
            long n0 = r - e * spec.n;
            int c = (int) Math.min(Math.min(count - done, chunk), spec.n - n0);
            encode(e, n0, c, out, (int) (done * rowBytes));
            done += c;
        }
        dst.position(dst.position() + (int) bytes);
    }

    private void encode(int e, long n0, int count, ByteBuffer out, int base) throws IOException {
        int k = spec.k;
        byte[] codes = new byte[count * k];
        spec.codes.read(e, n0, count, codes);
        float[] s = new float[count * spec.groups];
        spec.scales.read(e, n0, count, s);
        float[] z = null;
        if (spec.zeros != null) {
            z = new float[count * spec.groups];
            spec.zeros.read(e, n0, count, z);
        }
        int maxq = (1 << spec.bits) - 1;
        int[] q = new int[k];
        for (int i = 0; i < count; i++) {
            long row = n0 + i;
            for (int j = 0; j < k; j++) {
                int col = perm == null ? j : perm[j];
                int c = codes[i * k + col] & 0xFF;
                if (c > maxq) {
                    throw invalid(e, row, col, "code " + c + " exceeds " + spec.bits + " bits");
                }
                q[j] = c;
            }
            int off = base + (int) (i * rowBytes);
            if (target.isQuantized()) {
                encodeBlocks(e, row, q, s, z, i * spec.groups, out, off);
            } else {
                encodeFloats(e, row, q, s, z, i * spec.groups, out, off);
            }
        }
    }

    private void encodeFloats(int e, long row, int[] q, float[] s, float[] z, int si, ByteBuffer out, int off)
            throws IOException {
        for (int j = 0; j < spec.k; j++) {
            int g = spec.groupOf(e, perm == null ? j : perm[j]);
            float sc = scale(s, si + g, e, row, j);
            float zz = zero(z, si + g, e, row, j);
            float x = spec.formula == Formula.MUL_ADD ? q[j] * sc + zz : (q[j] - zz) * sc;
            if (columnScale != null) {
                x *= preScale(e, perm == null ? j : perm[j]);
            }
            if (target == GGMLType.F16) {
                fp16Field(x, "value", e, row, j);
            }
            IntPacking.putFloat(out, off, j, target, x);
        }
    }

    /** The per-column activation pre-scale of expert {@code e} at input column {@code col}. */
    private float preScale(int e, int col) {
        return columnScale.length == spec.k ? columnScale[col] : columnScale[e * spec.k + col];
    }

    private void encodeBlocks(int e, long row, int[] q, float[] s, float[] z, int si, ByteBuffer out, int off)
            throws IOException {
        int block = target.getBlockSize();
        int size = target.getTypeSize();
        for (int j0 = 0, b = 0; j0 < spec.k; j0 += block, b++) {
            int g = spec.groupOf(e, perm == null ? j0 : perm[j0]);
            float sc = scale(s, si + g, e, row, j0);
            float zz = zero(z, si + g, e, row, j0);
            int at = off + b * size;
            if (target != GGMLType.I8_B128) {
                fp16Field(sc, "scale", e, row, j0);
            }
            if (target == GGMLType.Q4_1 || target == GGMLType.Q5_1) {
                float m = fp16Field(spec.formula == Formula.MUL_ADD ? zz : -(zz * sc), "block minimum", e, row,
                        j0);
                if (target == GGMLType.Q4_1) {
                    IntPacking.putQ4_1(out, at, sc, m, q, j0);
                } else {
                    IntPacking.putQ5_1(out, at, sc, m, q, j0);
                }
                continue;
            }
            int lo = target == GGMLType.Q4_0 ? -8 : target == GGMLType.Q5_0 ? -16 : target == GGMLType.Q2_0 ? -2
                    : -128;
            for (int j = j0; j < j0 + block; j++) {
                int v = q[j] - (int) zz;
                if (v < lo || v > -lo - 1) {
                    throw invalid(e, row, perm == null ? j : perm[j], "value " + v + " does not fit " + target);
                }
                q[j] = target == GGMLType.Q2_0 ? 1 - v : v;
            }
            switch (target) {
                case Q4_0: IntPacking.putQ4_0(out, at, sc, q, j0); break;
                case Q5_0: IntPacking.putQ5_0(out, at, sc, q, j0); break;
                case Q8_0: IntPacking.putQ8_0(out, at, sc, q, j0); break;
                case Q2_0: IntPacking.putQ2_0(out, at, -sc, q, j0); break;
                case I8_B128: IntPacking.putI8B128(out, at, sc, q, j0); break;
                default: throw new IllegalStateException("unexpected target " + target);
            }
        }
    }

    private float scale(float[] s, int index, int e, long row, int j) throws IOException {
        float v = s[index];
        if (!Float.isFinite(v)) {
            throw invalid(e, row, perm == null ? j : perm[j], "scale " + v + " is not finite");
        }
        return v;
    }

    /** {@code v}, checked to stay finite when rounded into an fp16 field of the target. */
    private float fp16Field(float v, String what, int e, long row, int j) throws IOException {
        if (!Float.isFinite(FpBlockDecoders.roundToFp16(v))) {
            throw invalid(e, row, perm == null ? j : perm[j], what + " " + v + " overflows the fp16 range of "
                    + target);
        }
        return v;
    }

    private float zero(float[] z, int index, int e, long row, int j) throws IOException {
        if (z == null) {
            return spec.zeroDefault;
        }
        float v = z[index];
        if (!Float.isFinite(v)) {
            throw invalid(e, row, perm == null ? j : perm[j], "zero point " + v + " is not finite");
        }
        return v;
    }

    private IOException invalid(int e, long row, int col, String message) {
        return new IOException("Invalid Hugging Face checkpoint '" + checkpoint + "': " + describe() + ": "
                + (spec.experts > 0 ? "expert " + e + " " : "") + "row " + row + " column " + col + ": "
                + message);
    }

    /** The activation scheme of a weight and its static parameters. Immutable. */
    static final class Act {
        static final Act NONE = new Act(null, null, null);

        final ActQuantScheme scheme;
        private final float[] scale;
        private final int[] zeroPoint;

        Act(ActQuantScheme scheme, float[] scale, int[] zeroPoint) {
            this.scheme = scheme;
            this.scale = scale == null ? null : scale.clone();
            this.zeroPoint = zeroPoint == null ? null : zeroPoint.clone();
        }

        float[] scale() {
            return scale == null ? null : scale.clone();
        }

        int[] zeroPoint() {
            return zeroPoint == null ? null : zeroPoint.clone();
        }
    }
}
