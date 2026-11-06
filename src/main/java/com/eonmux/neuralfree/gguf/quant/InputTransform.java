package com.eonmux.neuralfree.gguf.quant;

import java.util.Arrays;
import java.util.Objects;

/**
 * The transform a projection applies to its input row before the unchanged weight kernel, made of up to four
 * optional parts applied in this order:
 * <ol>
 *   <li><b>PRE_SCALE</b>: the per-input-column multiplier of a quantized weight's {@code act_pre_scale}
 *       ({@link #preScale()}), applied before the quantization so the fake-quantization sees the scaled input, as
 *       torchao's linear does (torchao int8_tensor.py <i>activation_tensor = activation_tensor * act_pre_scale</i>
 *       before the activation quantizer);</li>
 *   <li><b>HADAMARD</b>: the PRISM or compressed-tensors input rotation ({@link HadamardSpec});</li>
 *   <li><b>ACT_QUANT</b>: the checkpoint's activation fake-quantization ({@link ActQuant}, {@link ActFakeQuant});</li>
 *   <li><b>PERMUTE</b>: the act-order gather {@code y[k] = x[perm[k]]} of a weight stored in permuted input order
 *       ({@code <base>.input_perm}).</li>
 * </ol>
 * The quantization order is vLLM's: the compressed-tensors input transform runs before the quantization method
 * (vllm model_executor/layers/quantization/compressed_tensors/transform/linear.py:150-160), which quantizes the
 * activation and hands it to the act-order kernel that permutes K
 * (vllm model_executor/layers/quantization/utils/marlin_utils.py:780-799). A weight stored in permuted K order thus
 * sees {@code perm} applied to the rotated, fake-quantized input. Immutable.
 */
public final class InputTransform {

    /** No transform. */
    public static final InputTransform IDENTITY = new InputTransform(null, null, null, null);

    private final HadamardSpec hadamard;
    private final ActQuant actQuant;
    private final int[] perm;
    private final float[] preScale;

    private InputTransform(HadamardSpec hadamard, ActQuant actQuant, int[] perm, float[] preScale) {
        this.hadamard = hadamard;
        this.actQuant = actQuant;
        this.perm = perm;
        this.preScale = preScale;
    }

    /**
     * The transform of the given parts; each may be null.
     *
     * @param perm the input permutation (copied): {@code perm[k]} is the input column read at position {@code k}
     * @throws IllegalArgumentException when {@code perm} is not a permutation of {@code [0, perm.length)}
     */
    public static InputTransform of(HadamardSpec hadamard, ActQuant actQuant, int[] perm) {
        return of(hadamard, actQuant, perm, null);
    }

    /**
     * The transform of the given parts, with the activation pre-scale; each may be null.
     *
     * @param preScale the per-input-column multiplier (copied): {@code K} values, or {@code E * K} for the experts of
     *                 a stacked tensor in expert-major order
     * @throws IllegalArgumentException when {@code perm} is not a permutation, or a pre-scale value is not finite
     */
    public static InputTransform of(HadamardSpec hadamard, ActQuant actQuant, int[] perm, float[] preScale) {
        int[] p = perm == null ? null : checkPermutation(perm.clone());
        float[] ps = preScale == null ? null : checkPreScale(preScale);
        if (hadamard == null && actQuant == null && p == null && ps == null) {
            return IDENTITY;
        }
        return new InputTransform(hadamard, actQuant, p, ps);
    }

    /** This transform with its HADAMARD part replaced ({@code null} removes it). */
    public InputTransform withHadamard(HadamardSpec h) {
        return of(h, actQuant, perm, preScale);
    }

    /** This transform with its ACT_QUANT part replaced ({@code null} removes it). */
    public InputTransform withActQuant(ActQuant a) {
        return of(hadamard, a, perm, preScale);
    }

    /**
     * This transform with its PERMUTE part replaced ({@code null} removes it).
     *
     * @throws IllegalArgumentException when {@code p} is not a permutation
     */
    public InputTransform withPerm(int[] p) {
        return of(hadamard, actQuant, p, preScale);
    }

    /**
     * This transform with its PRE_SCALE part replaced ({@code null} removes it).
     *
     * @throws IllegalArgumentException when a pre-scale value is not finite
     */
    public InputTransform withPreScale(float[] p) {
        return of(hadamard, actQuant, perm, p);
    }

    /** True when no part applies. */
    public boolean isIdentity() {
        return hadamard == null && actQuant == null && perm == null && preScale == null;
    }

    /** True when only a HADAMARD part (or nothing) applies. */
    public boolean isHadamardOnly() {
        return actQuant == null && perm == null && preScale == null;
    }

    /** The input rotation, or null. */
    public HadamardSpec hadamard() {
        return hadamard;
    }

    /** The activation fake-quantization, or null. */
    public ActQuant actQuant() {
        return actQuant;
    }

    /** True when an input permutation applies. */
    public boolean hasPerm() {
        return perm != null;
    }

    /** Copy of the input permutation, or null. */
    public int[] perm() {
        return perm == null ? null : perm.clone();
    }

    /**
     * The per-input-column multiplier applied before every other part ({@code act_pre_scale}), or null: {@code K}
     * values, or {@code E * K} for the experts of a stacked tensor in expert-major order.
     */
    public float[] preScale() {
        return preScale == null ? null : preScale.clone();
    }

    /** True when a per-input-column pre-scale applies. */
    public boolean hasPreScale() {
        return preScale != null;
    }

    /**
     * The number of pre-scale value sets for rows of {@code cols} elements: 0 when there is none, 1 for a shared
     * multiplier, or the number of experts of a stacked tensor.
     *
     * @throws IllegalArgumentException when the pre-scale does not cover whole rows
     */
    public int preScaleSets(int cols) {
        if (preScale == null) {
            return 0;
        }
        if (cols <= 0 || preScale.length % cols != 0) {
            throw new IllegalArgumentException("activation pre-scale of " + preScale.length
                    + " values does not cover whole rows of " + cols);
        }
        return preScale.length / cols;
    }

    /**
     * Multiplies {@code rows} contiguous rows of {@code cols} elements at {@code x[off ..]} in place by the pre-scale
     * of {@code expert} (0 when the values are shared), or leaves {@code x} unchanged when there is none.
     *
     * @throws IllegalArgumentException when the range lies outside {@code x}, or {@code expert} selects a value set
     *                                  that does not exist
     */
    public void applyPreScale(float[] x, int off, int rows, int cols, int expert) {
        if (preScale == null) {
            return;
        }
        Objects.requireNonNull(x, "x");
        if (rows < 0 || off < 0 || (long) off + (long) rows * cols > x.length) {
            throw new IllegalArgumentException("activation range of " + rows + " rows of " + cols + " at " + off
                    + " lies outside an array of " + x.length);
        }
        int sets = preScaleSets(cols);
        if (expert < 0 || (sets > 1 && expert >= sets)) {
            throw new IllegalArgumentException("expert " + expert + " is outside the " + sets
                    + " activation pre-scale value set(s)");
        }
        int base = sets > 1 ? expert * cols : 0;
        for (int r = 0; r < rows; r++) {
            int row = off + r * cols;
            for (int c = 0; c < cols; c++) {
                x[row + c] *= preScale[base + c];
            }
        }
    }

    /**
     * Checks that every part fits input rows of {@code cols} elements: the permutation has {@code cols} entries (it is
     * a permutation by construction), the rotation accepts the width ({@link HadamardSpec#checkWidth}), the row
     * splits into whole activation-quantization runs and the pre-scale covers whole rows.
     *
     * @return this transform
     * @throws IllegalArgumentException naming the part that does not fit
     */
    public InputTransform requireWidth(int cols) {
        if (perm != null && perm.length != cols) {
            throw new IllegalArgumentException("input permutation of " + perm.length + " columns does not fit rows of "
                    + cols);
        }
        if (hadamard != null) {
            hadamard.checkWidth(cols);
        }
        if (actQuant != null) {
            actQuant.scheme().runLength(cols);
        }
        preScaleSets(cols);
        return this;
    }

    private static float[] checkPreScale(float[] p) {
        float[] out = p.clone();
        for (int k = 0; k < out.length; k++) {
            if (!Float.isFinite(out[k])) {
                throw new IllegalArgumentException("activation pre-scale value " + out[k] + " at column " + k
                        + " is not finite");
            }
        }
        return out;
    }

    private static int[] checkPermutation(int[] p) {
        boolean[] seen = new boolean[p.length];
        for (int k = 0; k < p.length; k++) {
            int v = p[k];
            if (v < 0 || v >= p.length || seen[v]) {
                throw new IllegalArgumentException("input permutation entry " + k + " is " + v + "; expected a"
                        + " permutation of [0, " + p.length + ")");
            }
            seen[v] = true;
        }
        return p;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof InputTransform)) {
            return false;
        }
        InputTransform t = (InputTransform) o;
        return Objects.equals(hadamard, t.hadamard) && Objects.equals(actQuant, t.actQuant)
                && Arrays.equals(perm, t.perm) && Arrays.equals(preScale, t.preScale);
    }

    @Override
    public int hashCode() {
        return Objects.hash(hadamard, actQuant, Arrays.hashCode(perm), Arrays.hashCode(preScale));
    }

    @Override
    public String toString() {
        if (isIdentity()) {
            return "InputTransform{identity}";
        }
        StringBuilder b = new StringBuilder("InputTransform{");
        String sep = "";
        if (preScale != null) {
            b.append("preScale[").append(preScale.length).append(']');
            sep = ", ";
        }
        if (hadamard != null) {
            b.append(sep).append("hadamard=").append(hadamard);
            sep = ", ";
        }
        if (actQuant != null) {
            b.append(sep).append("actQuant=").append(actQuant);
            sep = ", ";
        }
        if (perm != null) {
            b.append(sep).append("perm[").append(perm.length).append(']');
        }
        return b.append('}').toString();
    }

    /**
     * The activation fake-quantization of one weight: its {@link ActQuantScheme} and the parameters the scheme reads,
     * one set shared by every expert or one per expert. The accessors give the {@code scale}, {@code zero_point} and
     * {@code global_scale} arguments of {@link ActFakeQuant#apply} and of the {@code LLM_ActFakeQuant} kernels.
     * Immutable.
     */
    public static final class ActQuant {
        private final ActQuantScheme scheme;
        private final float[] inputScales;
        private final int[] zeroPoints;
        private final float scaleUpperBound;

        private ActQuant(ActQuantScheme scheme, float[] inputScales, int[] zeroPoints, float scaleUpperBound) {
            this.scheme = scheme;
            this.inputScales = inputScales;
            this.zeroPoints = zeroPoints;
            this.scaleUpperBound = scaleUpperBound;
        }

        /**
         * The fake-quantization of {@code scheme}.
         *
         * @param inputScales     the {@code .input_scale} values (copied), one or one per expert: required by the
         *                        static schemes and NVFP4 (its global scale), null for the others
         * @param zeroPoints      the {@code .input_zero_point} values (copied), one or one per expert: required by
         *                        {@link ActQuantScheme#INT8_STATIC_ASYM}, null for the others
         * @param scaleUpperBound the {@code amax} bound of {@link ActQuantScheme#FP8_TOKEN} (positive,
         *                        {@link Float#POSITIVE_INFINITY} for none); ignored by the other schemes
         * @throws IllegalArgumentException when a parameter the scheme reads is missing or out of range, the
         *                                  per-expert counts of the two arrays disagree, or one the scheme does not
         *                                  read is given
         */
        public static ActQuant of(ActQuantScheme scheme, float[] inputScales, int[] zeroPoints,
                                  float scaleUpperBound) {
            Objects.requireNonNull(scheme, "scheme");
            boolean readsScale = scheme.isStatic() || scheme.needsGlobalScale();
            float[] scales = checkPresence(scheme, "input scales", inputScales, readsScale);
            int[] zps = checkPresence(scheme, "zero points", zeroPoints, scheme.needsZeroPoint());
            if (scales != null && zps != null && scales.length > 1 && zps.length > 1 && scales.length != zps.length) {
                throw new IllegalArgumentException(scheme + " has " + scales.length + " input scales but "
                        + zps.length + " zero points");
            }
            float ub = scheme == ActQuantScheme.FP8_TOKEN ? scaleUpperBound : Float.POSITIVE_INFINITY;
            ActQuant a = new ActQuant(scheme, scales, zps, ub);
            for (int e = 0; e < a.entries(); e++) {
                ActFakeQuant.checkParameters(scheme, a.scale(e), a.zeroPoint(e), a.globalScale(e));
            }
            return a;
        }

        private static float[] checkPresence(ActQuantScheme scheme, String what, float[] v, boolean required) {
            checkPresence(scheme, what, v == null ? -1 : v.length, required);
            return v == null ? null : v.clone();
        }

        private static int[] checkPresence(ActQuantScheme scheme, String what, int[] v, boolean required) {
            checkPresence(scheme, what, v == null ? -1 : v.length, required);
            return v == null ? null : v.clone();
        }

        private static void checkPresence(ActQuantScheme scheme, String what, int length, boolean required) {
            if (required && length < 1) {
                throw new IllegalArgumentException(scheme + " needs " + what);
            }
            if (!required && length >= 0) {
                throw new IllegalArgumentException(scheme + " reads no " + what);
            }
        }

        /** The scheme. */
        public ActQuantScheme scheme() {
            return scheme;
        }

        /** Parameter sets: 1 when every expert shares them, else the expert count. */
        public int entries() {
            return Math.max(inputScales == null ? 1 : inputScales.length, zeroPoints == null ? 1 : zeroPoints.length);
        }

        /**
         * The kernel {@code scale} argument for {@code expert}: the {@code .input_scale} of the static schemes, the
         * {@code amax} bound of {@link ActQuantScheme#FP8_TOKEN}, 0 otherwise.
         */
        public float scale(int expert) {
            if (scheme.isStatic()) {
                return inputScales[index(inputScales.length, expert)];
            }
            return scheme == ActQuantScheme.FP8_TOKEN ? scaleUpperBound : 0.0f;
        }

        /** The kernel {@code zero_point} argument for {@code expert}: the asymmetric int8 zero point, 0 otherwise. */
        public int zeroPoint(int expert) {
            return zeroPoints == null ? 0 : zeroPoints[index(zeroPoints.length, expert)];
        }

        /** The kernel {@code global_scale} argument for {@code expert}: the NVFP4 global scale, 0 otherwise. */
        public float globalScale(int expert) {
            return scheme.needsGlobalScale() ? inputScales[index(inputScales.length, expert)] : 0.0f;
        }

        /** The {@code amax} bound of {@link ActQuantScheme#FP8_TOKEN}; {@link Float#POSITIVE_INFINITY} for none. */
        public float scaleUpperBound() {
            return scaleUpperBound;
        }

        /**
         * Fake-quantizes {@code rows} contiguous rows of {@code cols} elements at {@code x[off ..]} in place with the
         * parameters of {@code expert} (0 for a weight without experts).
         *
         * @throws IllegalArgumentException when the range lies outside {@code x}, the row does not split into whole
         *                                  runs, or {@code expert} is out of range
         */
        public void apply(float[] x, int off, int rows, int cols, int expert) {
            ActFakeQuant.apply(scheme, scale(expert), zeroPoint(expert), globalScale(expert), x, off, rows, cols);
        }

        /** The array index of {@code expert}: shared parameters serve every expert. */
        private int index(int length, int expert) {
            int n = entries();
            if (expert < 0 || (n > 1 && expert >= n)) {
                throw new IllegalArgumentException("expert " + expert + " is outside the " + n
                        + " activation parameter set(s) of " + scheme);
            }
            return length == 1 ? 0 : expert;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof ActQuant)) {
                return false;
            }
            ActQuant a = (ActQuant) o;
            return scheme == a.scheme && Arrays.equals(inputScales, a.inputScales)
                    && Arrays.equals(zeroPoints, a.zeroPoints)
                    && Float.floatToIntBits(scaleUpperBound) == Float.floatToIntBits(a.scaleUpperBound);
        }

        @Override
        public int hashCode() {
            return Objects.hash(scheme, Arrays.hashCode(inputScales), Arrays.hashCode(zeroPoints), scaleUpperBound);
        }

        @Override
        public String toString() {
            return scheme + (inputScales == null ? "" : " scale" + Arrays.toString(inputScales))
                    + (zeroPoints == null ? "" : " zp" + Arrays.toString(zeroPoints))
                    + (scheme == ActQuantScheme.FP8_TOKEN && scaleUpperBound != Float.POSITIVE_INFINITY
                            ? " ub=" + scaleUpperBound : "");
        }
    }
}
