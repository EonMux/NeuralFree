package com.eonmux.neuralfree.acceleration.backend;

/**
 * Thrown when a command a capture is recording cannot be part of a replayable graph: a host transfer or read,
 * a marker, a synchronization, a dispatch that returns an event, or a cross-queue wait. The queue refuses the
 * command before the driver sees it, so the capture is still open and the caller can {@link
 * ComputeQueue#abortCapture()} it, discard the recorded commands and run the step directly.
 *
 * <p>It extends {@link IllegalStateException} because the capture contract documents these refusals as state
 * errors; a caller that records a step ({@link GpuGraph#launch}) distinguishes it from other failures to fall
 * back to a direct run instead of failing the step.</p>
 */
public final class GpuUncapturableCommandException extends IllegalStateException {

    private static final long serialVersionUID = 1L;

    public GpuUncapturableCommandException(String message) {
        super(message);
    }
}
