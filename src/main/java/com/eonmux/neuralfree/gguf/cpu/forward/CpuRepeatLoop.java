package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

/**
 * The state a repeat schedule with {@link LayerSchedule.LoopCombine#JUSTNORM} carries between the passes of its
 * looped block: the stream entering the repeated block and the state the passes fuse it with. The backend calls
 * {@link #beforeSlot} before and {@link #afterSlot} after every slot; other schedules run nothing.
 */
final class CpuRepeatLoop {

    private static final String TEMP_REPEAT_INPUT = "CpuForwardBackend.repeatInput";
    private static final String TEMP_REPEAT_STATE = "CpuForwardBackend.repeatState";

    private final boolean combine;
    private final int width;
    private final float[] input;
    private final float[] state;

    /** The loop of {@code spec}'s repeat schedule over the step buffers of {@code scratch}; inert when absent. */
    CpuRepeatLoop(ModelSpec spec, CpuScratch scratch) {
        LayerSchedule schedule = spec.schedule();
        this.combine = schedule.kind() == LayerSchedule.Kind.REPEAT
                && schedule.loopCombine() == LayerSchedule.LoopCombine.JUSTNORM;
        if (combine) {
            this.width = spec.layer(0).residual().streamWidth();
            int len = Math.multiplyExact(scratch.ubatch(), width);
            this.input = scratch.temp(TEMP_REPEAT_INPUT, len);
            this.state = scratch.temp(TEMP_REPEAT_STATE, len);
        } else {
            this.width = 0;
            this.input = null;
            this.state = null;
        }
    }

    /**
     * Before slot {@code il}: the stream entering the repeated block is kept, the state is initialized from it at
     * the first pass, and each pass starts as {@code justNorm(state + input)} (Dory's {@code combined =
     * justnorm(state + input_h)}).
     */
    void beforeSlot(LayerSchedule schedule, int il, int n, float[] x) {
        if (!combine) {
            return;
        }
        int len = n * width;
        if (il == 0 && schedule.nInputLayers() == 0) {
            System.arraycopy(x, 0, input, 0, len);
        }
        if (il == schedule.recurrentStart()) {
            System.arraycopy(input, 0, state, 0, len);
        }
        if (schedule.loopStart(il)) {
            CpuOps.add(state, 0, input, 0, x, 0, len);
            CpuOps.norm(NormKind.L2_GDN, schedule.loopEps(), x, 0, x, 0, n, width, null, null);
        }
    }

    /**
     * After slot {@code il}: a pass ends by writing {@code state = justNorm(x)}; the input stream is captured
     * after the last input layer, and the first output layer starts from the carried state.
     */
    void afterSlot(LayerSchedule schedule, int il, int n, float[] x) {
        if (!combine) {
            return;
        }
        int len = n * width;
        if (schedule.loopEnd(il)) {
            CpuOps.norm(NormKind.L2_GDN, schedule.loopEps(), x, 0, x, 0, n, width, null, null);
            System.arraycopy(x, 0, state, 0, len);
        }
        if (il == schedule.nInputLayers() - 1) {
            System.arraycopy(x, 0, input, 0, len);
        }
        if (il + 1 == schedule.recurrentEnd() && schedule.nOutputLayers() > 0) {
            System.arraycopy(state, 0, x, 0, len);
        }
    }
}
