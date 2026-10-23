package com.eonmux.neuralfree.acceleration.backend;

public interface GpuBuffer extends AutoCloseable {
    long sizeInBytes();
    long bufferId();
    void close();
}
