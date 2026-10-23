package com.eonmux.neuralfree.acceleration.backend;

public interface KernelHandle extends AutoCloseable {
    String kernelName();
    void close();
}
