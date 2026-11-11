package com.eonmux.neuralfree.gguf.memory.gpu;

/**
 * Writes the per-step inputs of a one-token decode step on the device ({@link GpuStepInputs#uploadDecodeStep}):
 * the step's cache row into {@code cell_idx[0]}, its position into {@code kv_pos[row]} and {@code q_pos[0]}, and
 * the four M-RoPE sections of its rope position into {@code rope_pos[0..3]}, of the one KV group's buffers the
 * writer was bound to. The writes are enqueued in the queue's order, like the uploads they replace.
 */
@FunctionalInterface
public interface DecodeStepWriter {

    /**
     * Enqueues the writes of the step at {@code pos} stored in cache row {@code row}, with rope sections
     * {@code rope0 .. rope3}.
     */
    void write(int row, int pos, int rope0, int rope1, int rope2, int rope3);
}
