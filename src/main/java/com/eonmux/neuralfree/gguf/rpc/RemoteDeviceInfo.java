package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;

import java.util.Objects;

/**
 * What a remote compute server reported at the handshake: the backend type its device runs through, the device's
 * capabilities (so the client plans work exactly as the server's device requires), whether it allocates host-mapped
 * buffers, and the endpoint it was reached at. Immutable.
 */
public final class RemoteDeviceInfo {

    private final ComputeBackend.BackendType backendType;
    private final DeviceCapabilities capabilities;
    private final boolean hostMappedBuffers;
    private final String endpoint;

    RemoteDeviceInfo(ComputeBackend.BackendType backendType, DeviceCapabilities capabilities,
                     boolean hostMappedBuffers, String endpoint) {
        this.backendType = Objects.requireNonNull(backendType, "backendType");
        this.capabilities = Objects.requireNonNull(capabilities, "capabilities");
        this.hostMappedBuffers = hostMappedBuffers;
        this.endpoint = Objects.requireNonNull(endpoint, "endpoint");
    }

    public ComputeBackend.BackendType backendType() {
        return backendType;
    }

    public DeviceCapabilities capabilities() {
        return capabilities;
    }

    public boolean hostMappedBuffers() {
        return hostMappedBuffers;
    }

    /** The {@code host:port} the connection was opened to. */
    public String endpoint() {
        return endpoint;
    }

    @Override
    public String toString() {
        return "RemoteDeviceInfo{" + endpoint + ", " + backendType + " '"
                + capabilities.getDeviceName() + "'}";
    }
}
