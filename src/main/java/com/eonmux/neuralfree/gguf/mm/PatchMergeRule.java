package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;

import java.util.Objects;

/**
 * How a vision projector type reads its per-side patch merge ({@code n_merge} of llama.cpp {@code clip_hparams},
 * set per family by {@code clip_model_loader::load_hparams}, tools/mtmd/clip.cpp): a merge the family fixes, or
 * the one metadata key its converter writes, with the family default when the key is absent or with the key
 * required. The vision encoder hparams and the image preprocessing read the merge through the same rule. Immutable.
 */
public final class PatchMergeRule {

    private final String key;
    private final int defaultMerge;
    private final boolean required;

    private PatchMergeRule(String key, int defaultMerge, boolean required) {
        this.key = key;
        this.defaultMerge = defaultMerge;
        this.required = required;
    }

    /**
     * A merge the family fixes; no key is read.
     *
     * @throws IllegalArgumentException for a merge below 1
     */
    public static PatchMergeRule fixed(int merge) {
        requirePositive(merge);
        return new PatchMergeRule(null, merge, false);
    }

    /**
     * The merge under {@code key}, {@code defaultMerge} when the key is absent.
     *
     * @throws IllegalArgumentException for a blank key or a default below 1
     */
    public static PatchMergeRule optional(String key, int defaultMerge) {
        requirePositive(defaultMerge);
        return new PatchMergeRule(requireKey(key), defaultMerge, false);
    }

    /**
     * The merge under {@code key}, which the file must carry.
     *
     * @throws IllegalArgumentException for a blank key
     */
    public static PatchMergeRule required(String key) {
        return new PatchMergeRule(requireKey(key), 0, true);
    }

    /** {@code clip.vision.spatial_merge_size}, {@code defaultMerge} when absent (the qwen-style mergers). */
    public static PatchMergeRule spatialMergeSize(int defaultMerge) {
        return optional(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, defaultMerge);
    }

    /** {@code clip.vision.projector.scale_factor}, {@code defaultMerge} when absent. */
    public static PatchMergeRule projectorScaleFactor(int defaultMerge) {
        return optional(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, defaultMerge);
    }

    /**
     * The per-side merge of a file under this rule.
     *
     * @throws GgufFormatException naming the key when a required key is absent or the value is below 1
     */
    public int read(GgufKv kv) {
        Objects.requireNonNull(kv, "kv");
        if (key == null) {
            return defaultMerge;
        }
        if (required && !kv.has(key)) {
            throw new GgufFormatException("required metadata key '" + key + "' is missing");
        }
        int merge = kv.u32(key, defaultMerge);
        if (merge <= 0) {
            throw new GgufFormatException("metadata key '" + key + "' must be positive, got " + merge);
        }
        return merge;
    }

    /** The metadata key the merge is read from, or null when the family fixes it. */
    public String key() {
        return key;
    }

    /** The merge when the key is absent (the fixed merge for a fixed rule); 0 for a required key. */
    public int defaultMerge() {
        return defaultMerge;
    }

    /** True when the file must carry the key. */
    public boolean required() {
        return required;
    }

    private static void requirePositive(int merge) {
        if (merge <= 0) {
            throw new IllegalArgumentException("the patch merge must be positive, got " + merge);
        }
    }

    private static String requireKey(String key) {
        Objects.requireNonNull(key, "key");
        if (key.isBlank()) {
            throw new IllegalArgumentException("the patch merge key is blank");
        }
        return key;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof PatchMergeRule)) {
            return false;
        }
        PatchMergeRule other = (PatchMergeRule) o;
        return defaultMerge == other.defaultMerge && required == other.required && Objects.equals(key, other.key);
    }

    @Override
    public int hashCode() {
        return Objects.hash(key, defaultMerge, required);
    }

    @Override
    public String toString() {
        if (key == null) {
            return "fixed merge " + defaultMerge;
        }
        return "metadata key '" + key + "' (" + (required ? "required" : "default " + defaultMerge) + ")";
    }
}
