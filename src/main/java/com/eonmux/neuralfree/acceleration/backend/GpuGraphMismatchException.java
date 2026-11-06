package com.eonmux.neuralfree.acceleration.backend;

/**
 * A {@link GpuGraph#replay(Runnable)} whose step body does not match the captured commands: a different
 * kernel or order, a different command count, arguments that the replay cannot pick up, or an
 * uncapturable command. No device work happened when it is thrown, so the caller can capture the step
 * again.
 */
public final class GpuGraphMismatchException extends IllegalStateException {

    private static final long serialVersionUID = 1L;

    public GpuGraphMismatchException(String message) {
        super(message);
    }

    /** @param cause the patch or node update failure the replay refused with */
    public GpuGraphMismatchException(String message, RuntimeException cause) {
        super(message, cause);
    }
}
