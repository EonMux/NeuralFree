package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorRef;

import java.util.Objects;
import java.util.function.Function;

/**
 * The input and output bounds of one Gemma 4 clippable linear ({@code Gemma4ClippableLinear}, llama.cpp
 * {@code clamp_info_map}): the scalar tensors {@code <w>.input_min}, {@code <w>.input_max}, {@code <w>.output_min}
 * and {@code <w>.output_max} beside a weight {@code <w>.weight}. The product clamps its input to
 * {@code [input_min, input_max]} and its output to {@code [output_min, output_max]} before any bias; an absent
 * bound is unbounded ({@code -FLT_MAX} / {@code FLT_MAX}). Immutable; the references view the file.
 */
final class ClipClamp {

    private static final String[] SUFFIXES = {".input_min", ".input_max", ".output_min", ".output_max"};

    final TensorRef inMin;
    final TensorRef inMax;
    final TensorRef outMin;
    final TensorRef outMax;

    private ClipClamp(TensorRef[] bounds) {
        this.inMin = bounds[0];
        this.inMax = bounds[1];
        this.outMin = bounds[2];
        this.outMax = bounds[3];
    }

    /**
     * The bounds of the weight named {@code weightName} ({@code <w>.weight}), or null when the file carries none.
     *
     * @throws IllegalArgumentException when a bound holds more than one value (clip.cpp {@code get_scalar})
     */
    static ClipClamp resolve(TensorCatalog tc, String weightName) {
        Objects.requireNonNull(weightName, "weightName");
        if (!weightName.endsWith(".weight")) {
            throw new IllegalArgumentException("clamp bounds belong to a '.weight' tensor, got " + weightName);
        }
        String base = weightName.substring(0, weightName.length() - ".weight".length());
        TensorRef[] bounds = new TensorRef[SUFFIXES.length];
        boolean any = false;
        for (int i = 0; i < SUFFIXES.length; i++) {
            bounds[i] = tc.get(base + SUFFIXES[i]);
            if (bounds[i] != null) {
                ClipShapes.elements(tc, bounds[i], 1);
                any = true;
            }
        }
        return any ? new ClipClamp(bounds) : null;
    }

    /** The four bounds as {@code {inMin, inMax, outMin, outMax}}, an absent one at {@code -/+Float.MAX_VALUE}. */
    float[] values(Function<String, float[]> floats) {
        return new float[]{value(inMin, -Float.MAX_VALUE, floats), value(inMax, Float.MAX_VALUE, floats),
                value(outMin, -Float.MAX_VALUE, floats), value(outMax, Float.MAX_VALUE, floats)};
    }

    private static float value(TensorRef r, float absent, Function<String, float[]> floats) {
        return r == null ? absent : floats.apply(r.name())[0];
    }
}
