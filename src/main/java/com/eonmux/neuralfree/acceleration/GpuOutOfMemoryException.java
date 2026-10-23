package com.eonmux.neuralfree.acceleration;

public final class GpuOutOfMemoryException extends RuntimeException {
    private final long requestedBytes;
    private final long availableBytes;
    private final String backendType;

    public GpuOutOfMemoryException(String message) {
        super(message);
        this.requestedBytes = -1;
        this.availableBytes = -1;
        this.backendType = "unknown";
    }

    public GpuOutOfMemoryException(String message, Throwable cause) {
        super(message, cause);
        this.requestedBytes = -1;
        this.availableBytes = -1;
        this.backendType = "unknown";
    }

    public GpuOutOfMemoryException(String message, long requestedBytes, long availableBytes, String backendType) {
        super(String.format("%s (requested=%d bytes, available=%d bytes, backend=%s)",
                message, requestedBytes, availableBytes, backendType));
        this.requestedBytes = requestedBytes;
        this.availableBytes = availableBytes;
        this.backendType = backendType;
    }

    public GpuOutOfMemoryException(String message, long requestedBytes, String backendType) {
        super(String.format("%s (requested=%d bytes, backend=%s)", message, requestedBytes, backendType));
        this.requestedBytes = requestedBytes;
        this.availableBytes = -1;
        this.backendType = backendType;
    }

    public long getRequestedBytes() {
        return requestedBytes;
    }

    public long getAvailableBytes() {
        return availableBytes;
    }

    public String getBackendType() {
        return backendType;
    }
}
