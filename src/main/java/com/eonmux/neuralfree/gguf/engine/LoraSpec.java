package com.eonmux.neuralfree.gguf.engine;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * One LoRA adapter to apply: the adapter GGUF file and the user scale (llama.cpp {@code common_adapter_lora_info}
 * path and scale; the effective scale of a tensor is {@code scale * alpha / rank}, llama-adapter.h:53-57).
 * Immutable.
 */
public final class LoraSpec {
    private final String path;
    private final float scale;

    /**
     * @throws IllegalArgumentException for a blank path or a non-finite scale
     */
    public LoraSpec(String path, float scale) {
        if (path == null || path.trim().isEmpty()) {
            throw new IllegalArgumentException("LoRA adapter path is empty");
        }
        if (!Float.isFinite(scale)) {
            throw new IllegalArgumentException("LoRA adapter " + path + " has a non-finite scale " + scale);
        }
        this.path = path.trim();
        this.scale = scale;
    }

    /** An adapter at scale 1 ({@code --lora FNAME}). */
    public static LoraSpec of(String path) {
        return new LoraSpec(path, 1.0f);
    }

    /**
     * One {@code path[:scale]} entry: the text after the last {@code ':'} is the scale when it parses as a finite
     * number, otherwise the whole entry is the path (so a path containing {@code ':'} needs an explicit scale).
     *
     * @throws IllegalArgumentException for an empty entry
     */
    public static LoraSpec parse(String entry) {
        String e = Objects.requireNonNull(entry, "entry").trim();
        int colon = e.lastIndexOf(':');
        if (colon > 0 && colon < e.length() - 1) {
            Float scale = finiteOrNull(e.substring(colon + 1).trim());
            if (scale != null) {
                return new LoraSpec(e.substring(0, colon), scale);
            }
        }
        return of(e);
    }

    /**
     * A comma-separated list of {@link #parse} entries; empty entries are skipped.
     *
     * @throws IllegalArgumentException for an invalid entry
     */
    public static List<LoraSpec> parseList(String value) {
        List<LoraSpec> out = new ArrayList<>();
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

    /** The adapter GGUF file. */
    public String path() { return path; }
    /** The user scale. */
    public float scale() { return scale; }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof LoraSpec)) {
            return false;
        }
        LoraSpec other = (LoraSpec) o;
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
