package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels.MixMode;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;

import java.util.Objects;

/**
 * The stream bookkeeping of a {@link LayerSchedule.Kind#REPEAT} schedule that combines its passes by
 * {@link LayerSchedule.LoopCombine#JUSTNORM} (Dory): the stream entering the repeated block and the state carried
 * between passes, kept in two scratch temps beside the executor's {@code X}. Each pass starts from
 * {@code justNorm(state + input)} and ends with {@code state = justNorm(x)}; the first output layer starts from the
 * last state. The executor calls {@link #beforeSlot} and {@link #afterSlot} around every slot of the layer loop.
 */
final class RepeatLoopStream {

    /** Repeat schedule: the stream entering the repeated block. */
    static final String TEMP_INPUT = "exec.repeat.input";
    /** Repeat schedule: the state carried between passes. */
    static final String TEMP_STATE = "exec.repeat.state";

    private final GpuOps ops;
    private final LayerSchedule schedule;
    private final ScratchSlot x;
    private final ScratchSlot input;
    private final ScratchSlot state;
    private final int width;
    private final int local;
    private final GpuBuffer placeholder;

    private RepeatLoopStream(GpuOps ops, LayerSchedule schedule, ScratchSlot x, ScratchSlot input,
                             ScratchSlot state, int width, GpuBuffer placeholder) {
        this.ops = ops;
        this.schedule = schedule;
        this.x = x;
        this.input = input;
        this.state = state;
        this.width = width;
        this.local = ops.local();
        this.placeholder = placeholder;
    }

    /**
     * The bookkeeping of {@code schedule} over the stream {@code x} of {@code width}-float rows, its temps sized for
     * {@code ubatch} rows; null when the schedule does not combine its passes by {@code JUSTNORM}.
     */
    static RepeatLoopStream of(GpuForwardContext ctx, LayerSchedule schedule, ScratchSlot x, int width,
                               int ubatch) {
        Objects.requireNonNull(ctx, "ctx");
        if (schedule == null || schedule.kind() != LayerSchedule.Kind.REPEAT
                || schedule.loopCombine() != LayerSchedule.LoopCombine.JUSTNORM) {
            return null;
        }
        GpuScratch scratch = ctx.scratch();
        return new RepeatLoopStream(ctx.ops(), schedule, x, scratch.temp(TEMP_INPUT, (long) ubatch * width),
                scratch.temp(TEMP_STATE, (long) ubatch * width), width, ctx.placeholders().readOnlyFloats());
    }

    /**
     * Before slot {@code i} over {@code n} rows: keep the stream entering the repeated block, and start each pass from
     * {@code justNorm(state + input)} (Dory's {@code combined = justnorm(state + input_h)}).
     */
    void beforeSlot(int i, int n) {
        int len = n * width;
        if (i == 0 && schedule.nInputLayers() == 0) {
            ops.copy(x.buffer(), x.offset(), input.buffer(), input.offset(), len);
        }
        if (i == schedule.recurrentStart()) {
            ops.copy(input.buffer(), input.offset(), state.buffer(), state.offset(), len);
        }
        if (schedule.loopStart(i)) {
            ops.enqueue(NormKernels.RESIDUAL_MIX, new long[]{(len + local - 1L) / local * local}, new long[]{local},
                    NormKernels.residualMixArgs(MixMode.ADD, len, 1.0f, state.buffer(), state.offset(),
                            input.buffer(), input.offset(), placeholder, placeholder, 1, x.buffer(), x.offset(),
                            placeholder));
            ops.norm(NormKind.L2_GDN, null, null, schedule.loopEps(), x.buffer(), x.offset(), x.buffer(), x.offset(),
                    n, width);
        }
    }

    /**
     * After slot {@code i} over {@code n} rows: a pass ends with {@code state = justNorm(x)}; the input stream is
     * captured after the last input layer, and the first output layer starts from the state.
     */
    void afterSlot(int i, int n) {
        int len = n * width;
        if (schedule.loopEnd(i)) {
            ops.norm(NormKind.L2_GDN, null, null, schedule.loopEps(), x.buffer(), x.offset(), x.buffer(), x.offset(),
                    n, width);
            ops.copy(x.buffer(), x.offset(), state.buffer(), state.offset(), len);
        }
        if (i == schedule.nInputLayers() - 1) {
            ops.copy(x.buffer(), x.offset(), input.buffer(), input.offset(), len);
        }
        if (i + 1 == schedule.recurrentEnd() && schedule.nOutputLayers() > 0) {
            ops.copy(state.buffer(), state.offset(), x.buffer(), x.offset(), len);
        }
    }
}
