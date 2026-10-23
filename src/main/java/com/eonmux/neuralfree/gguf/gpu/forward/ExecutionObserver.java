package com.eonmux.neuralfree.gguf.gpu.forward;

/**
 * Hooks the GPU forward path calls around each stage of a step, for profiling and diagnostics.
 *
 * <p>The executor calls {@link #onStepStart}, then for every stage {@link #onStageStart} and {@link #onStageEnd}
 * (layer stages carry the layer index, model-level stages -1), then {@link #onStepEnd}. Hooks run on the forward
 * thread between enqueues: an observer may synchronize the queue (a profiler timing each stage) or read the stage
 * result (a diagnostic), which serializes the step. {@link #NONE} observes nothing; executors skip the calls when
 * {@link #enabled()} is false, so an unobserved step has no overhead and no host sync.</p>
 */
public interface ExecutionObserver {

    /** Stages of one forward step. */
    enum Stage {
        /** Token embedding, scaling, padding and per-layer embedding inputs. Result: {@code X}. */
        EMBEDDING,
        /** Pre-mixer norm. Result: {@code NORMED}. */
        ATTN_NORM,
        /** Mixer (attention, recurrent, ...). Result: {@code BRANCH}. */
        MIXER,
        /** Residual after the mixer, including extras that consume it. Result: {@code X}. */
        MIXER_RESIDUAL,
        /** Pre-FFN norm. Result: {@code NORMED}. */
        FFN_NORM,
        /** Dense FFN or mixture of experts. Result: {@code BRANCH}. */
        FFN,
        /** Residual after the FFN, including extras that consume it. Result: {@code X}. */
        FFN_RESIDUAL,
        /** Whole layer including its extras. Result: {@code X}. */
        LAYER,
        /** Head preprocessing and output norm. Result: {@code NORMED}. */
        OUTPUT_NORM,
        /** Output projection, scaling, softcap and padding. Result: {@code LOGITS}. */
        LOGITS
    }

    /** Observer that observes nothing. */
    ExecutionObserver NONE = new ExecutionObserver() {
        @Override
        public boolean enabled() {
            return false;
        }

        @Override
        public String toString() {
            return "ExecutionObserver.NONE";
        }
    };

    /** False when the observer ignores every hook. */
    default boolean enabled() {
        return true;
    }

    /** A step over tokens {@code [pos0, pos0 + nTokens)} begins; nothing of it is enqueued yet. */
    default void onStepStart(int pos0, int nTokens) {
    }

    /** {@code stage} of layer {@code layer} (-1 for model-level stages) is about to be enqueued. */
    default void onStageStart(int layer, Stage stage) {
    }

    /**
     * Every dispatch of {@code stage} is enqueued. {@code result} holds its output for {@code nRows} rows of
     * {@code width} floats once the queue reaches this point.
     */
    default void onStageEnd(int layer, Stage stage, ScratchSlot result, int nRows, int width) {
    }

    /** Every dispatch of the step is enqueued. */
    default void onStepEnd(int pos0, int nTokens) {
    }
}
