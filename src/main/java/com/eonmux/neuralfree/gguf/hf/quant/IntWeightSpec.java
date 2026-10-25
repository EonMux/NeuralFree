package com.eonmux.neuralfree.gguf.hf.quant;

import com.eonmux.neuralfree.gguf.hf.config.QuantFormat;
import com.eonmux.neuralfree.gguf.hf.quant.IntPacking.CodeTable;
import com.eonmux.neuralfree.gguf.hf.quant.IntPacking.ParamTable;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.util.Objects;

/** One quantized weight: its geometry, code and parameter tables and group mapping. Immutable. */
final class IntWeightSpec {

    /** How codes and parameters combine into a weight value. */
    enum Formula {
        /** {@code (q - z) * s}: z is a zero point. */
        SUB_MUL,
        /** {@code q * s + z}: z is a bias (MLX affine). */
        MUL_ADD
    }

    final String module;
    final QuantFormat format;
    final int experts;
    final long n;
    final int k;
    final int bits;
    final int groupSize;
    final int groups;
    final CodeTable codes;
    final ParamTable scales;
    final StDType scaleDtype;
    final ParamTable zeros;
    final float zeroDefault;
    final int[][] gidx;
    final Formula formula;
    final boolean preferI8B128;

    private IntWeightSpec(Builder b, int[][] gidx) {
        this.module = b.module;
        this.format = b.format;
        this.experts = b.experts;
        this.n = b.n;
        this.k = b.k;
        this.bits = b.bits;
        this.groupSize = b.groupSize;
        this.groups = b.groups;
        this.codes = b.codes;
        this.scales = b.scales;
        this.scaleDtype = b.scaleDtype;
        this.zeros = b.zeros;
        this.zeroDefault = b.zeroDefault;
        this.gidx = gidx;
        this.formula = b.formula;
        this.preferI8B128 = b.preferI8B128;
    }

    /**
     * A builder for one module's weight of {@code n} output rows.
     *
     * @throws IOException when the stored tensors give no output rows
     */
    static Builder builder(IntModuleTensors m, int experts, long n, int k, int bits, int groupSize, int groups)
            throws IOException {
        if (n <= 0) {
            throw m.invalid("the stored tensors give " + n + " output rows");
        }
        return new Builder(m.name(), m.format(), experts, n, k, bits, groupSize, groups);
    }

    /** The group of input column {@code col} of expert {@code e}. */
    int groupOf(int e, int col) {
        return gidx == null ? col / groupSize : gidx[e][col];
    }

    /** The fields of a {@link IntWeightSpec}; {@link #build} validates and freezes them. */
    static final class Builder {
        private final String module;
        private final QuantFormat format;
        private final int experts;
        private final long n;
        private final int k;
        private final int bits;
        private final int groupSize;
        private final int groups;
        CodeTable codes;
        ParamTable scales;
        StDType scaleDtype;
        ParamTable zeros;
        float zeroDefault;
        int[][] gidx;
        Formula formula = Formula.SUB_MUL;
        boolean preferI8B128;

        private Builder(String module, QuantFormat format, int experts, long n, int k, int bits, int groupSize,
                        int groups) {
            this.module = module;
            this.format = format;
            this.experts = experts;
            this.n = n;
            this.k = k;
            this.bits = bits;
            this.groupSize = groupSize;
            this.groups = groups;
        }

        /** The spec; a group index equal to {@code k / groupSize} everywhere is dropped. */
        IntWeightSpec build() {
            Objects.requireNonNull(codes, "codes");
            Objects.requireNonNull(scales, "scales");
            Objects.requireNonNull(scaleDtype, "scaleDtype");
            if (n <= 0 || k <= 0 || groupSize <= 0 || groups != groupCount(k, groupSize) || bits < 1 || bits > 8) {
                throw new IllegalStateException(module + ": inconsistent geometry n=" + n + " k=" + k + " g="
                        + groupSize + " groups=" + groups + " bits=" + bits);
            }
            return new IntWeightSpec(this, gidx == null || trivial(gidx) ? null : deepCopy(gidx));
        }

        private boolean trivial(int[][] g) {
            for (int[] row : g) {
                for (int c = 0; c < row.length; c++) {
                    if (row[c] != c / groupSize) {
                        return false;
                    }
                }
            }
            return true;
        }

        private static int[][] deepCopy(int[][] g) {
            int[][] out = new int[g.length][];
            for (int i = 0; i < g.length; i++) {
                out[i] = g[i].clone();
            }
            return out;
        }
    }

    /** Groups of {@code g} along {@code k}: {@code ceil(k / g)}. */
    static int groupCount(int k, int g) {
        return (int) (((long) k + g - 1) / g);
    }
}
