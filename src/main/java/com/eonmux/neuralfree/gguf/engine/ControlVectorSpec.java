package com.eonmux.neuralfree.gguf.engine;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * One control vector to apply: the control-vector GGUF file and the user scale (llama.cpp
 * {@code common_control_vector_load_info} path and scaling; {@code common_control_vector_load} sums the vectors'
 * rows, which the per-file entries of the runtime set reproduce). Immutable.
 */
public final class ControlVectorSpec {
    private final String path;
    private final float scale;

    /**
     * @throws IllegalArgumentException for a blank path or a non-finite scale
     */
    public ControlVectorSpec(String path, float scale) {
        if (path == null || path.trim().isEmpty()) {
            throw new IllegalArgumentException("control vector path is empty");
        }
        if (!Float.isFinite(scale)) {
            throw new IllegalArgumentException("control vector " + path + " has a non-finite scale " + scale);
        }
        this.path = path.trim();
        this.scale = scale;
    }

    /** A vector at scale 1 ({@code --control-vector FNAME}). */
    public static ControlVectorSpec of(String path) {
        return new ControlVectorSpec(path, 1.0f);
    }

    /**
     * One {@code path[:scale]} entry: the text after the last {@code ':'} is the scale when it parses as a finite
     * number, otherwise the whole entry is the path (so a path containing {@code ':'} needs an explicit scale).
     *
     * @throws IllegalArgumentException for an empty entry
     */
    public static ControlVectorSpec parse(String entry) {
        String e = Objects.requireNonNull(entry, "entry").trim();
        int colon = e.lastIndexOf(':');
        if (colon > 0 && colon < e.length() - 1) {
            Float scale = finiteOrNull(e.substring(colon + 1).trim());
            if (scale != null) {
                return new ControlVectorSpec(e.substring(0, colon), scale);
            }
        }
        return of(e);
    }

    /**
     * A comma-separated list of {@link #parse} entries; empty entries are skipped.
     *
     * @throws IllegalArgumentException for an invalid entry
     */
    public static List<ControlVectorSpec> parseList(String value) {
        List<ControlVectorSpec> out = new ArrayList<>();
        if (value == null) {
            return out;
        }
        for (String entry : value.split(",")) {
            if (!entry.trim().isEmpty()) {
                out.add(parse(entry));
            }
        }
        return out;
    }

    private static Float finiteOrNull(String s) {
        try {
            float f = Float.parseFloat(s);
            return Float.isFinite(f) ? f : null;
        } catch (NumberFormatException ex) {
            return null;
        }
    }

    /** The control-vector GGUF file. */
    public String path() { return path; }
    /** The user scale. */
    public float scale() { return scale; }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ControlVectorSpec)) {
            return false;
        }
        ControlVectorSpec other = (ControlVectorSpec) o;
        return path.equals(other.path) && Float.compare(scale, other.scale) == 0;
    }

    @Override
    public int hashCode() {
        return 31 * path.hashCode() + Float.hashCode(scale);
    }

    @Override
    public String toString() {
        return path + ":" + scale;
    }
}
