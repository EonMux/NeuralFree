package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.engine.HiddenTap;

import java.util.List;

/**
 * A {@link HiddenTap} that takes a tap point's rows on the device of the forward call: instead of reading every
 * row to the host, each capture enqueues one device-to-device copy into storage the tap owns, and the call's rows
 * are delivered as that storage once the queue finished. A drafter fed this way reads the target's hidden rows
 * without the host round-trip.
 *
 * <p>{@link GpuTapReads} asks {@link #wantsDevice} at a tap point's first capture of a call; the answer holds for
 * the whole call, and the point is captured on the device only when the call's rows run in call order (contiguous
 * steps, not batched rows). Every other tap keeps the host path.
 */
public interface GpuHiddenFeed {

    /**
     * Whether the rows at tap point {@code layer} of a call on {@code queue} are captured on the device. The same
     * answer must hold for every capture of the point within the call.
     */
    boolean wantsDevice(int layer, ComputeQueue queue);

    /**
     * Enqueues the copy of the rows at tap point {@code layer} into the tap's device storage, on {@code queue},
     * before the call's queue finished. Called once per capture in call order, only when {@link #wantsDevice} was
     * true; {@code rowOffset} is the element offset of the rows inside {@code rows}.
     */
    void captureDevice(int layer, GpuBuffer rows, long rowOffset, int nRows, int width);

    /**
     * The call's device rows are in the tap's storage: one delivery per tap point per call, in tap-point order,
     * once the queue finished, after every {@link #captureDevice} of the call. The tap matches the delivery to the
     * captures it enqueued (in capture order, their rows summing to {@code nRows}).
     */
    void onHiddenDevice(int layer, int nRows, int width);
}
