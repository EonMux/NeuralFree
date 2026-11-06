package com.eonmux.neuralfree.acceleration.metal;

import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;
import com.eonmux.neuralfree.utils.LoggingUtil;

import java.util.ArrayList;
import java.util.List;

/**
 * Enumeration of Apple GPUs through Metal: the devices the framework reports, as
 * {@link MetalDevice}s.
 *
 * <p>{@link #available()} is false without the JNI shim (a host that is not macOS, or a shim that
 * was not built); {@link #enumerate()} then throws with the reason. Enumeration creates no device
 * state and no backend: the Metal framework owns the devices, and a backend opens the command
 * queues it needs.</p>
 */
public final class MetalRuntime {

    private MetalRuntime() {
    }

    /** True when the JNI shim is loaded, so {@link #enumerate()} and every backend call can run. */
    public static boolean available() {
        return MetalNative.isAvailable();
    }

    /** Why the shim cannot load, or null when it can. */
    public static String unavailableReason() {
        String failure = MetalNative.loadFailure();
        return failure == null ? null : "the Metal shim is not available: " + failure
                + "; Metal exists only on macOS, so build the shim with clang (make native/metal)"
                + " and load it from the library path or -D" + MetalNative.SHIM_PATH_PROPERTY;
    }

    /**
     * Enumerates the devices the framework reports. A device without a usable name is dropped, as
     * every other backend does.
     *
     * @throws NoBackendAvailableException when the shim is unavailable or cannot count the devices
     */
    public static List<MetalDevice> enumerate() {
        if (!available()) {
            throw new NoBackendAvailableException(unavailableReason());
        }
        int count;
        try {
            count = MetalNative.getDeviceCount();
        } catch (RuntimeException e) {
            throw new NoBackendAvailableException("Metal device enumeration failed: " + e.getMessage(), e);
        }
        if (count < 0) {
            throw new NoBackendAvailableException("Metal device enumeration failed (status " + count + ")");
        }
        List<MetalDevice> devices = new ArrayList<>(count);
        for (int index = 0; index < count; index++) {
            try {
                devices.add(new MetalDevice(index));
            } catch (RuntimeException e) {
                System.out.println("Metal device " + index + " failed: " + e.getMessage());
            }
        }
        LoggingUtil.getLogger(MetalRuntime.class).info("Found " + devices.size() + " Metal capable device(s)");
        return devices;
    }
}
