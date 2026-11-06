package com.eonmux.neuralfree.acceleration.hip;

import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;
import com.eonmux.neuralfree.utils.LoggingUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Enumeration of the HIP runtime: the AMD devices the shim sees, as {@link HipDevice}s.
 *
 * <p>{@link #available()} is false without the JNI shim (never built, or no ROCm runtime
 * installed); {@link #enumerate(HipDriver)} then never runs. Enumeration is the only step that
 * touches the runtime outside a backend: the shim loads, the runtime initialises itself on the
 * first {@code hipGetDeviceCount}, and the result is plain device ordinals.</p>
 */
public final class HipRuntime {

    private HipRuntime() {
    }

    /**
     * True when the JNI shim is loaded, so {@link #enumerate} and every backend call can run.
     * The reason when false is in {@link HipNative#loadFailure()}.
     */
    public static boolean available() {
        return HipNative.isAvailable();
    }

    /** Why the shim cannot load, or null when it can. */
    public static String unavailableReason() {
        String failure = HipNative.loadFailure();
        return failure == null ? null : "the HIP shim is not available: " + failure
                + "; build it with hipcc (make native/hip) and load it from the library path"
                + " or -D" + HipNative.SHIM_PATH_PROPERTY;
    }

    /**
     * Enumerates the devices the driver counts ({@code hipGetDeviceCount}), naming-skipped like
     * every backend: a device without a usable name or processor name is dropped.
     *
     * @throws NoBackendAvailableException when the runtime cannot count its devices
     */
    public static List<HipDevice> enumerate(HipDriver driver) {
        Objects.requireNonNull(driver, "driver");
        int count;
        try {
            count = driver.deviceCount();
        } catch (RuntimeException e) {
            throw new NoBackendAvailableException("HIP device enumeration failed: " + e.getMessage(), e);
        }
        List<HipDevice> devices = new ArrayList<>();
        for (int index = 0; index < count; index++) {
            HipDevice device;
            try {
                device = new HipDevice(index, driver);
            } catch (RuntimeException e) {
                System.out.println("HIP device " + index + " failed: " + e.getMessage());
                continue;
            }
            devices.add(device);
        }
        LoggingUtil.getLogger(HipRuntime.class).info("Found " + devices.size() + " HIP capable device(s)");
        return devices;
    }
}
