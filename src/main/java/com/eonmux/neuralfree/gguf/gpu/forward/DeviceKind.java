package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.gguf.ops.gemm.MmqConfig;

import java.util.Objects;

/**
 * The kind of device, as one backend's device report describes it, that a measured per-device lane
 * decision was taken on: the backend, the PCI vendor and physical class of the device plus, on CUDA,
 * which side of the int8 tensor path's compute-capability floor the device sits on ({@link
 * CudaBand#TENSOR} is sm_80 and above, where the MMQ kernels take the int8 tensor tile and the MMA
 * lanes live). Entries hold the kind they were measured on and lookups match it exactly; a device
 * whose vendor or class is unknown matches no entry and keeps the default dispatch, and a kind is
 * never inferred from a name or a missing field. Two backends describing the same physical device
 * still read as two kinds: their kernels and drivers differ, so their measurements never stand in
 * for each other.
 *
 * <p>Instances are immutable value objects; {@link #of(BackendType, DeviceCapabilities)} reads the
 * kind of a backend's device report.</p>
 */
public final class DeviceKind {

    /** PCI vendor of the device. */
    public enum Vendor {
        /** NVIDIA ({@code 0x10de}). */
        NVIDIA,
        /** Intel ({@code 0x8086}). */
        INTEL,
        /** AMD ({@code 0x1002}). */
        AMD,
        /** Any other vendor, or an unknown vendor id. */
        OTHER
    }

    /** The CUDA side of the int8 tensor path's compute-capability floor (sm_80, {@link
     * MmqConfig#TENSOR_MIN_COMPUTE_CAPABILITY}). */
    public enum CudaBand {
        /** Not a CUDA device, or a CUDA device that reports no compute capability. */
        NONE,
        /** A CUDA device below the floor, whose MMQ kernels take the packed-dot tile. */
        PACKED_DOT,
        /** A CUDA device at or above the floor. */
        TENSOR
    }

    private static final int PCI_VENDOR_NVIDIA = 0x10DE;
    private static final int PCI_VENDOR_INTEL = 0x8086;
    private static final int PCI_VENDOR_AMD = 0x1002;

    private final BackendType backend;
    private final Vendor vendor;
    private final DeviceClass deviceClass;
    private final CudaBand cudaBand;

    private DeviceKind(BackendType backend, Vendor vendor, DeviceClass deviceClass, CudaBand cudaBand) {
        this.backend = backend;
        this.vendor = vendor;
        this.deviceClass = deviceClass;
        this.cudaBand = cudaBand;
    }

    /**
     * The kind of the device {@code caps} reports on {@code backend}: its identity's PCI vendor, its
     * physical class and its compute-capability band. A null class reads as {@link DeviceClass#UNKNOWN}
     * and an unknown vendor id as {@link Vendor#OTHER}; both match no measured entry.
     *
     * @throws IllegalArgumentException when an argument is null
     */
    public static DeviceKind of(BackendType backend, DeviceCapabilities caps) {
        if (backend == null) {
            throw new IllegalArgumentException("backend must not be null");
        }
        if (caps == null) {
            throw new IllegalArgumentException("capabilities must not be null");
        }
        return of(backend, vendorOf(caps), classOf(caps), bandOf(caps.getComputeCapability()));
    }

    /** The kind of one backend, vendor, device class and CUDA band. */
    public static DeviceKind of(BackendType backend, Vendor vendor, DeviceClass deviceClass, CudaBand cudaBand) {
        if (backend == null) {
            throw new IllegalArgumentException("backend must not be null");
        }
        if (vendor == null) {
            throw new IllegalArgumentException("vendor must not be null");
        }
        if (deviceClass == null) {
            throw new IllegalArgumentException("device class must not be null");
        }
        if (cudaBand == null) {
            throw new IllegalArgumentException("CUDA band must not be null");
        }
        return new DeviceKind(backend, vendor, deviceClass, cudaBand);
    }

    public BackendType backend() {
        return backend;
    }

    public Vendor vendor() {
        return vendor;
    }

    public DeviceClass deviceClass() {
        return deviceClass;
    }

    public CudaBand cudaBand() {
        return cudaBand;
    }

    /** The vendor of {@code caps}' device identity: the PCI vendor id, or {@link Vendor#OTHER}. */
    private static Vendor vendorOf(DeviceCapabilities caps) {
        int vendorId = caps.getIdentity() != null ? caps.getIdentity().vendorId() : DeviceIdentity.UNKNOWN_VENDOR;
        switch (vendorId) {
            case PCI_VENDOR_NVIDIA:
                return Vendor.NVIDIA;
            case PCI_VENDOR_INTEL:
                return Vendor.INTEL;
            case PCI_VENDOR_AMD:
                return Vendor.AMD;
            default:
                return Vendor.OTHER;
        }
    }

    /** The physical class of {@code caps}' device; a null report reads as {@link DeviceClass#UNKNOWN}. */
    private static DeviceClass classOf(DeviceCapabilities caps) {
        return caps.getDeviceClass() != null ? caps.getDeviceClass() : DeviceClass.UNKNOWN;
    }

    /** The CUDA band of a compute capability ({@code major * 10 + minor}); 0 is unknown. */
    private static CudaBand bandOf(int computeCapability) {
        if (computeCapability <= 0) {
            return CudaBand.NONE;
        }
        return computeCapability >= MmqConfig.TENSOR_MIN_COMPUTE_CAPABILITY
                ? CudaBand.TENSOR
                : CudaBand.PACKED_DOT;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof DeviceKind)) {
            return false;
        }
        DeviceKind other = (DeviceKind) o;
        return backend == other.backend && vendor == other.vendor
                && deviceClass == other.deviceClass && cudaBand == other.cudaBand;
    }

    @Override
    public int hashCode() {
        return Objects.hash(backend, vendor, deviceClass, cudaBand);
    }

    @Override
    public String toString() {
        return "DeviceKind{backend=" + backend + ", vendor=" + vendor + ", class=" + deviceClass
                + ", cudaBand=" + cudaBand + '}';
    }
}
