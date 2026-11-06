package com.eonmux.neuralfree.acceleration.backend.hip;

import com.eonmux.neuralfree.acceleration.hip.HipDriver;

/**
 * The calling thread's current HIP device: the HIP runtime has no explicit context — its
 * implicit per-device context follows the thread's device, which {@code hipSetDevice} selects.
 *
 * <p>{@link #makeCurrent()} keeps the last device it set per thread and calls
 * {@code hipSetDevice} only when the thread last ran another device, mirroring what
 * {@code CudaContext} does for CUDA primary contexts. Every public entry point of the HIP
 * backend, its queues, buffers, events and handles makes the backend's device current first.</p>
 */
public final class HipContext {

    /** The device each thread last made current through any {@code HipContext}. */
    private static final ThreadLocal<Integer> CURRENT = new ThreadLocal<>();

    private final int device;

    private HipContext(int device) {
        this.device = device;
    }

    /** The context of {@code device} (the implicit context the runtime keeps for it). */
    public static HipContext of(int device) {
        return new HipContext(device);
    }

    /** The device ordinal of this context. */
    public int device() {
        return device;
    }

    /**
     * Makes the device current on this thread; a thread-local identity check skips redundant
     * {@code hipSetDevice} calls.
     *
     * @throws RuntimeException the translated driver failure
     */
    public void makeCurrent(HipDriver driver) {
        Integer current = CURRENT.get();
        if (current != null && current == device) {
            return;
        }
        driver.setDevice(device);
        CURRENT.set(device);
    }

    /** True when this device is already the calling thread's current one. */
    public boolean isCurrent() {
        Integer current = CURRENT.get();
        return current != null && current == device;
    }

    /** Forgets the thread's current device (only for tests that swap drivers). */
    static void forgetForThread() {
        CURRENT.remove();
    }

    @Override
    public String toString() {
        return "HipContext{device=" + device + "}";
    }
}
