package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;

import java.util.Objects;

/**
 * The explicit rope-section rule of a forward batch, shared by the CPU and GPU backends: a batch that carries
 * section-major positions ({@link ForwardBatch#ropePositions()}, the explicit sections of the embedding stage) must
 * carry exactly the sections the model's rope type reads per token ({@link RopeMath#positionsPerToken}). A
 * single-section model reads section 0 and a rope-less model reads none, so any other section would be silently
 * dropped; such a batch is refused by name before a backend uploads or stores anything.
 */
public final class RopeSectionSupport {

    private RopeSectionSupport() {
    }

    /**
     * Refuses the batch's explicit sections unless the model's rope reads exactly their count.
     *
     * @throws IllegalArgumentException naming the section count and the model's rope type
     */
    public static void check(RopeType type, ForwardBatch batch) {
        Objects.requireNonNull(batch, "batch");
        check(type, batch.ropePositions());
    }

    /**
     * Refuses {@code sections} unless it is null (no explicit sections) or holds exactly the model's positions per
     * token.
     *
     * @throws IllegalArgumentException naming the section count and the model's rope type
     */
    public static void check(RopeType type, int[][] sections) {
        String refusal = refusal(type, sections);
        if (refusal != null) {
            throw new IllegalArgumentException(refusal);
        }
    }

    /** Null when {@code sections} is null or holds the model's positions per token, else the refusal. */
    public static String refusal(RopeType type, int[][] sections) {
        Objects.requireNonNull(type, "type");
        if (sections == null) {
            return null;
        }
        int expected = RopeMath.positionsPerToken(type);
        if (sections.length == expected) {
            return null;
        }
        return "the batch carries " + sections.length + " rope sections, the model's " + type + " rope reads "
                + expected;
    }
}
