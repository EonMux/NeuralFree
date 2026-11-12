package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.memory.gpu.DecodeStepWriter;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;

import java.util.Objects;

/**
 * The one-token decode step's inputs written by {@value #KERNEL} ({@code kernels/llm/step_inputs.cl}, GLSL twin
 * {@code shaders/llm/step_inputs.comp}): one launch whose arguments carry the cache row, the position and the rope
 * sections, in place of the host uploads of {@code cell_idx}, the {@code kv_pos} row, {@code q_pos} and
 * {@code rope_pos}, each of which costs the step a host-to-device copy on the device's queue. Bound to the buffers
 * of a single-group step-input set; a backend without the kernel binds nothing and the step keeps its uploads. Not
 * thread-safe.
 */
final class GpuDecodeStepWriter implements DecodeStepWriter, AutoCloseable {

    static final String KERNEL = "LLM_DecodeStepInputs";
    static final int ARG_ROW = 0;
    static final int ARG_POS = 1;
    static final int ARG_ROPE0 = 2;

    private static final long[] ONE = {1};

    private final PreparedDispatch dispatch;

    private GpuDecodeStepWriter(PreparedDispatch dispatch) {
        this.dispatch = dispatch;
    }

    /**
     * The writer of {@code inputs}' buffers, or null when the backend provides no {@value #KERNEL} or the inputs
     * hold another number of KV groups than one.
     */
    static GpuDecodeStepWriter bind(GpuOps ops, GpuStepInputs inputs) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(inputs, "inputs");
        if (inputs.groups() != 1 || !ops.prewarm(KERNEL, ONE)) {
            return null;
        }
        return new GpuDecodeStepWriter(ops.prepare(KERNEL, ONE, ONE, 0, 0, 0, 0, 0, 0, inputs.cellIdx(0),
                inputs.kvPos(0), inputs.qPos(), inputs.ropePos()));
    }

    @Override
    public void write(int row, int pos, int rope0, int rope1, int rope2, int rope3) {
        dispatch.setInt(ARG_ROW, row);
        dispatch.setInt(ARG_POS, pos);
        dispatch.setInt(ARG_ROPE0, rope0);
        dispatch.setInt(ARG_ROPE0 + 1, rope1);
        dispatch.setInt(ARG_ROPE0 + 2, rope2);
        dispatch.setInt(ARG_ROPE0 + 3, rope3);
        dispatch.dispatch();
    }

    @Override
    public void close() {
        dispatch.close();
    }
}
