package com.eonmux.neuralfree.gguf.image;

import java.util.Objects;

/**
 * The per-image token budget ({@code --image-min-tokens} / {@code --image-max-tokens} of llama.cpp,
 * common/arg.cpp; {@code mtmd_context_params.image_min_tokens}, tools/mtmd/mtmd.h). A dynamic-resolution
 * preprocessor turns each limit into a pixel area through the merged patch area
 * ({@code patch_size * merge} squared, llama.cpp {@code clip_hparams::set_limit_image_tokens},
 * tools/mtmd/clip-model.h) and reads the result as its min/max pixel limits; a custom value in the GGUF
 * metadata stays out of the way (the CLI value takes precedence, llama.cpp {@code custom_image_min_tokens}).
 *
 * <p>Immutable. A limit of {@code 0} (the default) is unset.</p>
 */
public final class ImageTokenBudget {

    private final int minTokens;
    private final int maxTokens;

    private ImageTokenBudget(Builder b) {
        this.minTokens = b.minTokens;
        this.maxTokens = b.maxTokens;
    }

    /** An empty budget: both limits unset. */
    public static ImageTokenBudget none() {
        return builder().build();
    }

    /** A new builder with both limits unset. */
    public static Builder builder() {
        return new Builder();
    }

    /** The smallest accepted image token count; {@code 0} when unset. */
    public int minTokens() { return minTokens; }

    /** The largest accepted image token count; {@code 0} when unset. */
    public int maxTokens() { return maxTokens; }

    /** Whether no limit is set. */
    public boolean isEmpty() { return minTokens <= 0 && maxTokens <= 0; }

    /**
     * The smallest accepted pixel area under the merged patch area, or {@code 0} when the min limit is unset
     * ({@code custom_image_min_tokens * patch_area}, tools/mtmd/clip-model.h).
     *
     * @throws IllegalArgumentException when {@code mergedPatchArea} is not positive
     */
    public long minPixels(long mergedPatchArea) {
        if (mergedPatchArea <= 0) {
            throw new IllegalArgumentException("mergedPatchArea must be positive, got " + mergedPatchArea);
        }
        return minTokens > 0 ? (long) minTokens * mergedPatchArea : 0;
    }

    /**
     * The largest accepted pixel area under the merged patch area, or {@code 0} when the max limit is unset.
     *
     * @throws IllegalArgumentException when {@code mergedPatchArea} is not positive
     */
    public long maxPixels(long mergedPatchArea) {
        if (mergedPatchArea <= 0) {
            throw new IllegalArgumentException("mergedPatchArea must be positive, got " + mergedPatchArea);
        }
        return maxTokens > 0 ? (long) maxTokens * mergedPatchArea : 0;
    }

    /** Builder of an immutable {@link ImageTokenBudget}. */
    public static final class Builder {

        private int minTokens = 0;
        private int maxTokens = 0;

        private Builder() {
        }

        /**
         * Sets {@code --image-min-tokens}; zero unsets, negative values are refused.
         */
        public Builder minTokens(int minTokens) {
            if (minTokens < 0) throw new IllegalArgumentException("minTokens must be non-negative, got " + minTokens);
            this.minTokens = minTokens;
            return this;
        }

        /** Sets {@code --image-max-tokens}; zero unsets, negative values are refused. */
        public Builder maxTokens(int maxTokens) {
            if (maxTokens < 0) throw new IllegalArgumentException("maxTokens must be non-negative, got " + maxTokens);
            this.maxTokens = maxTokens;
            return this;
        }

        /**
         * Validates the pair: a set min must not exceed a set max (llama.cpp caps the min at the max when the
         * server clamps the budget).
         *
         * @throws IllegalArgumentException when both are set and the min exceeds the max
         */
        public ImageTokenBudget build() {
            if (minTokens > 0 && maxTokens > 0 && minTokens > maxTokens) {
                throw new IllegalArgumentException("image-min-tokens (" + minTokens + ") exceeds image-max-tokens ("
                        + maxTokens + ")");
            }
            return new ImageTokenBudget(this);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ImageTokenBudget)) return false;
        ImageTokenBudget that = (ImageTokenBudget) o;
        return minTokens == that.minTokens && maxTokens == that.maxTokens;
    }

    @Override
    public int hashCode() {
        return Objects.hash(minTokens, maxTokens);
    }

    @Override
    public String toString() {
        return "ImageTokenBudget(min=" + minTokens + ", max=" + maxTokens + ")";
    }
}
