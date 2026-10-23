package com.eonmux.neuralfree.acceleration;

/**
 * A GPU driver call failed with an error other than running out of memory
 * (which is {@link GpuOutOfMemoryException}).
 *
 * <p>Carries the backend name and the driver's numeric result code, so callers
 * can report the failure precisely.</p>
 */
public class GpuBackendException extends RuntimeException {

    private final String backend;
    private final int code;

    /**
     * @param backend backend name, for example {@code "CUDA"} or {@code "Vulkan"}
     * @param code    driver result code
     * @param message what was being done when the call failed
     */
    public GpuBackendException(String backend, int code, String message) {
        super(format(backend, code, message));
        this.backend = backend;
        this.code = code;
    }

    public GpuBackendException(String backend, int code, String message, Throwable cause) {
        super(format(backend, code, message), cause);
        this.backend = backend;
        this.code = code;
    }

    public String getBackend() {
        return backend;
    }

    public int getCode() {
        return code;
    }

    private static String format(String backend, int code, String message) {
        return backend + " error " + code + ": " + message;
    }
}
