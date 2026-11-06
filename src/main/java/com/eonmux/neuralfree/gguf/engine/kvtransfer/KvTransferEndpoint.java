package com.eonmux.neuralfree.gguf.engine.kvtransfer;

import java.util.Objects;

/**
 * Where a producer's transfer connector listens, as its {@code kv_transfer_params} advertise it to a consumer: the
 * host a consumer connects to and the port. A shared-storage connector has no address, so both are {@code null} and
 * {@code -1}; the consumer then reads the configured storage instead.
 */
public final class KvTransferEndpoint {

    private final String host;
    private final int port;

    private KvTransferEndpoint(String host, int port) {
        this.host = host;
        this.port = port;
    }

    /** The address {@code host:port}. */
    public static KvTransferEndpoint of(String host, int port) {
        if (host == null || host.isBlank()) {
            throw new IllegalArgumentException("a KV transfer endpoint needs a host");
        }
        if (port < 1 || port > 65535) {
            throw new IllegalArgumentException("a KV transfer endpoint needs a port in [1, 65535], got " + port);
        }
        return new KvTransferEndpoint(host.trim(), port);
    }

    /** The endpoint of a connector that addresses no socket (shared storage). */
    public static KvTransferEndpoint none() {
        return new KvTransferEndpoint(null, -1);
    }

    /** The host, or null for a connector that addresses no socket. */
    public String host() {
        return host;
    }

    /** The port, or -1 for a connector that addresses no socket. */
    public int port() {
        return port;
    }

    /** Whether this endpoint names a socket. */
    public boolean addressed() {
        return host != null;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof KvTransferEndpoint)) {
            return false;
        }
        KvTransferEndpoint other = (KvTransferEndpoint) o;
        return port == other.port && Objects.equals(host, other.host);
    }

    @Override
    public int hashCode() {
        return Objects.hash(host, port);
    }

    @Override
    public String toString() {
        return addressed() ? host + ":" + port : "none";
    }
}
