package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;

import java.util.Objects;

/**
 * One remote compute server a layer split offloads layers to: its TCP endpoint, the shared key the connection
 * authenticates with, and whether the transport is TLS. The value is immutable and never prints the key.
 *
 * <p>The {@code neuralfree.rpc} entries take {@code host:port}, {@code host:port@key}, or the same with a
 * {@code tls://} scheme for a TLS-wrapped endpoint ({@code tls://host:port@key}); an entry without an inline
 * key reads it from {@value #KEY_ENV}. An IPv6 host is written in brackets, {@code tls://[::1]:50052}.</p>
 */
public final class RpcServerSpec {

    /** Environment variable holding the shared key of entries that carry none inline. */
    public static final String KEY_ENV = "NEURALFREE_RPC_KEY";
    /** The scheme prefix that wraps a connection in TLS; the HMAC handshake runs on top of either transport. */
    public static final String TLS_SCHEME = "tls://";

    private final String host;
    private final int port;
    private final String key;
    private final boolean tls;

    private RpcServerSpec(String host, int port, String key, boolean tls) {
        this.host = host;
        this.port = port;
        this.key = key;
        this.tls = tls;
    }

    /**
     * Parses one {@code neuralfree.rpc} entry.
     *
     * @throws IllegalArgumentException naming the entry when the endpoint is malformed or no key is available
     */
    public static RpcServerSpec parse(String entry) {
        Objects.requireNonNull(entry, "rpc entry");
        String s = entry.trim();
        if (s.isEmpty()) {
            throw new IllegalArgumentException(EngineConfig.PROP_RPC + " has an empty entry");
        }
        boolean tls = false;
        if (s.regionMatches(true, 0, TLS_SCHEME, 0, TLS_SCHEME.length())) {
            tls = true;
            s = s.substring(TLS_SCHEME.length()).trim();
            if (s.isEmpty()) {
                throw new IllegalArgumentException(EngineConfig.PROP_RPC + " entry '" + entry.trim()
                        + "' has no endpoint after " + TLS_SCHEME);
            }
        }
        String endpoint = s;
        String inline = null;
        int at = s.lastIndexOf('@');
        if (at >= 0) {
            endpoint = s.substring(0, at).trim();
            inline = s.substring(at + 1).trim();
            if (inline.isEmpty()) {
                throw new IllegalArgumentException(EngineConfig.PROP_RPC + " entry '" + masked(s)
                        + "' has an empty key after '@'");
            }
        }
        String key = inline != null ? inline : System.getenv(KEY_ENV);
        if (key == null || key.trim().isEmpty()) {
            throw new IllegalArgumentException(EngineConfig.PROP_RPC + " entry '" + endpoint + "' has no key:"
                    + " write it as host:port@key or set " + KEY_ENV + "; the transport is authenticated");
        }
        int colon;
        String host;
        if (endpoint.startsWith("[")) {
            int close = endpoint.indexOf(']');
            if (close < 0) {
                throw new IllegalArgumentException(EngineConfig.PROP_RPC + " entry '" + endpoint
                        + "' has an unterminated IPv6 address");
            }
            host = endpoint.substring(1, close);
            colon = close + 1 < endpoint.length() && endpoint.charAt(close + 1) == ':' ? close + 1 : -1;
        } else {
            colon = endpoint.lastIndexOf(':');
            host = colon < 0 ? "" : endpoint.substring(0, colon);
        }
        if (host.isEmpty() || host.indexOf(' ') >= 0) {
            throw new IllegalArgumentException(EngineConfig.PROP_RPC + " entry '" + endpoint
                    + "' is not host:port");
        }
        String portText = colon < 0 ? "" : endpoint.substring(colon + 1);
        int port;
        try {
            port = Integer.parseInt(portText);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(EngineConfig.PROP_RPC + " entry '" + endpoint
                    + "' has no numeric port", e);
        }
        if (port < 1 || port > 65535) {
            throw new IllegalArgumentException(EngineConfig.PROP_RPC + " entry '" + endpoint + "' has port "
                    + port + "; a TCP port is 1..65535");
        }
        return new RpcServerSpec(host, port, key.trim(), tls);
    }

    /**
     * The comma-separated {@code neuralfree.rpc} value with every inline key replaced by {@code ...}, for error
     * messages and logs: never prints a key.
     */
    public static String masked(String entries) {
        Objects.requireNonNull(entries, "rpc entries");
        StringBuilder out = new StringBuilder();
        String[] parts = entries.split(",", -1);
        for (int i = 0; i < parts.length; i++) {
            if (i > 0) {
                out.append(',');
            }
            String entry = parts[i];
            int at = entry.lastIndexOf('@');
            out.append(at < 0 ? entry : entry.substring(0, at) + "@...");
        }
        return out.toString();
    }

    public String host() {
        return host;
    }

    public int port() {
        return port;
    }

    /** The shared key; never part of {@link #toString()}. */
    public String key() {
        return key;
    }

    /** Whether the connection wraps the endpoint in TLS before the HMAC handshake. */
    public boolean tls() {
        return tls;
    }

    /** The endpoint as an optional {@code tls://} scheme plus {@code host:port}, an IPv6 host in brackets; never the key. */
    public String describe() {
        return (tls ? TLS_SCHEME : "") + (host.indexOf(':') >= 0 ? "[" + host + "]" : host) + ":" + port;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof RpcServerSpec)) {
            return false;
        }
        RpcServerSpec that = (RpcServerSpec) o;
        return port == that.port && tls == that.tls && host.equals(that.host) && key.equals(that.key);
    }

    @Override
    public int hashCode() {
        return Objects.hash(host, port, key, tls);
    }

    @Override
    public String toString() {
        return describe();
    }
}
