package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * The activation-quantization carrier of a model's metadata: {@code neuralfree.act_quant.version} (1),
 * {@code .weight_names} (GGUF weight names) with the parallel {@code .schemes} ({@link ActQuantScheme#code()}), and
 * the model-level {@code .scale_ub}, the upper bound on {@code max|x|} of every {@link ActQuantScheme#FP8_TOKEN}
 * weight (fbgemm_fp8 {@code activation_scale_ub}, vllm model_executor/layers/quantization/fbgemm_fp8.py:76,
 * :142-146). The static parameters stay in the weights' sibling tensors, which {@link InputTransformLoader} reads.
 * Immutable.
 */
public final class ActQuantPlan {

    /** The only carrier version. */
    public static final long VERSION = 1L;

    private static final String WEIGHT_SUFFIX = ".weight";

    /** A model without activation quantization. */
    public static final ActQuantPlan NONE = new ActQuantPlan(Collections.<String, ActQuantScheme>emptyMap(),
            Float.POSITIVE_INFINITY);

    private final Map<String, ActQuantScheme> schemes;
    private final float scaleUpperBound;

    private ActQuantPlan(Map<String, ActQuantScheme> schemes, float scaleUpperBound) {
        this.schemes = schemes;
        this.scaleUpperBound = scaleUpperBound;
    }

    /**
     * Reads and validates the carrier.
     *
     * @return the plan; {@link #NONE} (with the bound, when present) when the model carries no
     *         {@code neuralfree.act_quant.version}
     * @throws GgufFormatException naming the key or the entry: an unsupported version, missing or unequal arrays,
     *                             an empty array, a name that is not a {@code <base>.weight} name, a duplicate name, an
     *                             unknown scheme code, entries without a version, or a bound that is not finite and
     *                             positive
     */
    public static ActQuantPlan load(GgufKv kv) {
        Objects.requireNonNull(kv, "kv");
        float ub = readScaleUpperBound(kv);
        if (!kv.has(GgufKeys.NF_ACT_QUANT_VERSION)) {
            for (String key : new String[]{GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES, GgufKeys.NF_ACT_QUANT_SCHEMES}) {
                if (kv.has(key)) {
                    throw new GgufFormatException("metadata key '" + key + "' needs '"
                            + GgufKeys.NF_ACT_QUANT_VERSION + "'");
                }
            }
            return ub == Float.POSITIVE_INFINITY ? NONE : new ActQuantPlan(NONE.schemes, ub);
        }
        long version = kv.u32Long(GgufKeys.NF_ACT_QUANT_VERSION, 0L);
        if (version != VERSION) {
            throw new GgufFormatException("unsupported " + GgufKeys.NF_ACT_QUANT_VERSION + ": " + version
                    + "; this build reads version " + VERSION);
        }
        String[] names = kv.strArray(GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES);
        int[] codes = kv.i32Array(GgufKeys.NF_ACT_QUANT_SCHEMES);
        if (names == null || codes == null) {
            throw new GgufFormatException("required metadata key '" + (names == null
                    ? GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES : GgufKeys.NF_ACT_QUANT_SCHEMES) + "' is missing");
        }
        if (names.length != codes.length) {
            throw new GgufFormatException(GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES + " has " + names.length + " entries but "
                    + GgufKeys.NF_ACT_QUANT_SCHEMES + " has " + codes.length);
        }
        if (names.length == 0) {
            throw new GgufFormatException(GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES + " is empty");
        }
        Map<String, ActQuantScheme> out = new LinkedHashMap<>();
        for (int i = 0; i < names.length; i++) {
            String name = names[i];
            if (name == null || name.length() <= WEIGHT_SUFFIX.length() || !name.endsWith(WEIGHT_SUFFIX)) {
                throw new GgufFormatException(GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES + "[" + i + "] '" + name
                        + "' is not a <base>.weight tensor name");
            }
            ActQuantScheme scheme;
            try {
                scheme = ActQuantScheme.fromCode(codes[i]);
            } catch (GgufFormatException e) {
                throw new GgufFormatException(GgufKeys.NF_ACT_QUANT_SCHEMES + "[" + i + "] of '" + name + "': "
                        + e.getMessage(), e);
            }
            if (out.putIfAbsent(name, scheme) != null) {
                throw new GgufFormatException(GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES + " lists '" + name + "' twice");
            }
        }
        return new ActQuantPlan(Collections.unmodifiableMap(out), ub);
    }

    /** {@code neuralfree.act_quant.scale_ub}: finite and positive when present, {@code +inf} when absent. */
    private static float readScaleUpperBound(GgufKv kv) {
        if (!kv.has(GgufKeys.NF_ACT_QUANT_SCALE_UB)) {
            return Float.POSITIVE_INFINITY;
        }
        float ub = kv.f32(GgufKeys.NF_ACT_QUANT_SCALE_UB, Float.NaN);
        if (!(ub > 0.0f && ub < Float.POSITIVE_INFINITY)) {
            throw new GgufFormatException(GgufKeys.NF_ACT_QUANT_SCALE_UB + " is " + ub
                    + "; expected a finite positive bound");
        }
        return ub;
    }

    /** GGUF weight name ({@code <base>.weight}) to its scheme, in metadata order. Unmodifiable. */
    public Map<String, ActQuantScheme> schemes() {
        return schemes;
    }

    /** The scheme of {@code weightName}, or null. */
    public ActQuantScheme scheme(String weightName) {
        return schemes.get(weightName);
    }

    /**
     * The upper bound on {@code max|x|} of the {@link ActQuantScheme#FP8_TOKEN} weights;
     * {@link Float#POSITIVE_INFINITY} when the model sets none.
     */
    public float scaleUpperBound() {
        return scaleUpperBound;
    }

    /** True when no weight carries an activation scheme. */
    public boolean isEmpty() {
        return schemes.isEmpty();
    }

    @Override
    public String toString() {
        return "ActQuantPlan{" + schemes.size() + " weight(s)"
                + (scaleUpperBound == Float.POSITIVE_INFINITY ? "" : ", scale_ub=" + scaleUpperBound) + '}';
    }
}
