package com.eonmux.neuralfree.acceleration.backend.spi;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;

import java.util.Objects;

/**
 * One device of an accelerator API, as its {@link AcceleratorBackendProvider} enumerates it.
 *
 * <p>An immutable value: the {@link DeviceCapabilities} describe the device before any backend is
 * opened, so the engine can rank devices and apply requirements without touching the vendor
 * runtime, and the {@link BackendFactory} opens the {@link ComputeBackend} on demand. The factory
 * must create a backend of the same {@link BackendType} as the device; {@link #openBackend()}
 * refuses one of another type by name.</p>
 */
public final class AcceleratorDevice {

    /** Opens the backend of this device; called at most once per backend the engine creates. */
    @FunctionalInterface
    public interface BackendFactory {
        ComputeBackend open();
    }

    private final BackendType backendType;
    private final DeviceCapabilities capabilities;
    private final BackendFactory factory;

    private AcceleratorDevice(BackendType backendType, DeviceCapabilities capabilities, BackendFactory factory) {
        this.backendType = backendType;
        this.capabilities = capabilities;
        this.factory = factory;
    }

    /**
     * A device of an accelerator API.
     *
     * @param backendType the API that enumerates the device; must be an accelerator type
     * @param capabilities what the device offers, including its name, identity and class
     * @param factory     opens the device's backend
     * @throws IllegalArgumentException when an argument is null or {@code backendType} is not an accelerator API
     */
    public static AcceleratorDevice of(BackendType backendType, DeviceCapabilities capabilities,
                                       BackendFactory factory) {
        Objects.requireNonNull(backendType, "backend type must not be null");
        if (!backendType.isAccelerator()) {
            throw new IllegalArgumentException(backendType + " is not an accelerator API; a provider serves only"
                    + " the accelerator types of " + BackendType.class.getSimpleName());
        }
        Objects.requireNonNull(capabilities, "device capabilities must not be null");
        Objects.requireNonNull(factory, "backend factory must not be null");
        return new AcceleratorDevice(backendType, capabilities, factory);
    }

    /** The accelerator API that enumerates the device. */
    public BackendType backendType() {
        return backendType;
    }

    /** What the device offers; its identity and class drive selection and ranking. */
    public DeviceCapabilities capabilities() {
        return capabilities;
    }

    /** The device name as the vendor runtime reports it. */
    public String name() {
        String name = capabilities.getDeviceName();
        return name == null ? "" : name;
    }

    /** The device's cross-API identity. */
    public DeviceIdentity identity() {
        return capabilities.getIdentity();
    }

    /** The device's physical class. */
    public DeviceClass deviceClass() {
        return capabilities.getDeviceClass();
    }

    /** Total device memory in bytes, or -1 when the runtime does not report it. */
    public long totalMemoryBytes() {
        long total = capabilities.getTotalVramBytes();
        return total > 0 ? total : -1L;
    }

    /**
     * Opens the device's backend.
     *
     * @throws IllegalStateException when the factory returns nothing or a backend of another API
     */
    public ComputeBackend openBackend() {
        ComputeBackend backend = factory.open();
        if (backend == null) {
            throw new IllegalStateException("the " + backendType.label() + " provider's factory for device '"
                    + name() + "' returned no backend");
        }
        if (backend.backendType() != backendType) {
            throw new IllegalStateException("the " + backendType.label() + " provider's factory for device '"
                    + name() + "' returned a " + backend.backendType().label() + " backend; a provider must create"
                    + " backends of the API it serves");
        }
        return backend;
    }

    @Override
    public String toString() {
        return backendType.label() + " device '" + name() + "' (" + deviceClass() + ")";
    }
}
