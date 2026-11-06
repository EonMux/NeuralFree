package com.eonmux.neuralfree.server;

import java.net.InetAddress;
import java.net.URI;
import java.net.UnknownHostException;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Set;

/** The tool surfaces' trusted Host names: literal loopbacks, the bind host and an explicit allowlist. Immutable. */
final class AllowedToolHosts {

    static final String PROPERTY = "neuralfree.server.allowed.hosts";
    static final String FLAG = "--allowed-hosts";
    static final String BIND_PROPERTY = "quarkus.http.host";
    private static final String IPV6_LOOPBACK = "[0:0:0:0:0:0:0:1]";

    private final Set<String> names;
    private final boolean disabled;

    AllowedToolHosts(String bindHost, String allowed) {
        Set<String> entries = entries(allowed, PROPERTY);
        Set<String> trusted = new LinkedHashSet<>(entries);
        String bind = normalizeName(bracketLiteral(bindHost));
        if (bind == null) {
            throw new IllegalArgumentException(BIND_PROPERTY + " must name a host or IP literal");
        }
        trusted.add(bind);
        names = Set.copyOf(trusted);
        disabled = entries.contains("*");
    }

    /** Checks the launch flag's entries with the same rules as the property, naming the flag on failure. */
    static String validate(String raw, String source) {
        entries(raw, source);
        return raw.trim();
    }

    boolean disabled() {
        return disabled;
    }

    /** Whether the Host authority is trusted. This never resolves a DNS name or trusts a forwarded header. */
    boolean accepts(String authority) {
        if (disabled) {
            return true;
        }
        String name = authorityName(authority);
        return name != null && (name.equals("localhost") || name.equals(IPV6_LOOPBACK)
                || ipv4Loopback(name) || names.contains(name));
    }

    private static Set<String> entries(String raw, String source) {
        if (raw == null || raw.isBlank()) {
            return Set.of();
        }
        Set<String> out = new LinkedHashSet<>();
        for (String entry : raw.split(",", -1)) {
            String value = entry.trim();
            String name = value.equals("*") ? "*" : normalizeName(value);
            if (name == null) {
                throw new IllegalArgumentException(source + " entry '" + value
                        + "' must be a host name or IP literal without a scheme, port or path, or '*'");
            }
            out.add(name);
        }
        return Set.copyOf(out);
    }

    /**
     * Whether a browser {@code Origin} is trusted: an absent origin (a non-browser client) is, and a present one
     * must be an {@code http} or {@code https} origin whose authority {@link #accepts(String)}. The opaque
     * {@code null} origin and anything carrying a path, query, fragment or user info fail.
     */
    boolean acceptsOrigin(String origin) {
        if (disabled || origin == null) {
            return true;
        }
        URI uri;
        try {
            uri = new URI(origin.trim());
        } catch (java.net.URISyntaxException e) {
            return false;
        }
        String scheme = uri.getScheme();
        boolean bare = (uri.getRawPath() == null || uri.getRawPath().isEmpty()) && uri.getRawQuery() == null
                && uri.getRawFragment() == null && uri.getRawUserInfo() == null;
        return bare && ("http".equalsIgnoreCase(scheme) || "https".equalsIgnoreCase(scheme))
                && uri.getRawAuthority() != null && accepts(uri.getRawAuthority());
    }

    /** A Host authority's name, with a valid optional port removed; malformed or ambiguous authorities fail. */
    private static String authorityName(String raw) {
        if (raw == null || raw.isBlank()) {
            return null;
        }
        String authority = raw.trim();
        int end = authority.startsWith("[") ? authority.indexOf(']') + 1 : authority.indexOf(':');
        if (end == 0) {
            return null;
        }
        if (end < 0) {
            end = authority.length();
        }
        if (end < authority.length() && !validPort(authority.substring(end))) {
            return null;
        }
        return normalizeName(authority.substring(0, end));
    }

    private static boolean validPort(String suffix) {
        if (!suffix.matches(":[0-9]{1,5}")) {
            return false;
        }
        int port = Integer.parseInt(suffix.substring(1));
        return port <= 65535;
    }

    /** A canonical DNS name or numeric literal. Only colon-bearing numeric IPv6 text reaches InetAddress. */
    private static String normalizeName(String raw) {
        if (raw == null || raw.isEmpty()) {
            return null;
        }
        if (raw.startsWith("[") && raw.endsWith("]")) {
            String literal = raw.substring(1, raw.length() - 1);
            if (!literal.contains(":") || !literal.matches("[0-9a-fA-F:.]+")) {
                return null;
            }
            try {
                return "[" + InetAddress.getByName(literal).getHostAddress().toLowerCase(Locale.ROOT) + "]";
            } catch (UnknownHostException e) {
                return null;
            }
        }
        if (raw.contains(":") || raw.contains("/") || raw.contains("@") || raw.contains("?") || raw.contains("#")) {
            return null;
        }
        try {
            String name = URI.create("http://" + raw).getHost();
            if (name == null || !name.equalsIgnoreCase(raw)) {
                return null;
            }
            name = name.toLowerCase(Locale.ROOT);
            return name.endsWith(".") ? name.substring(0, name.length() - 1) : name;
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    private static boolean ipv4Loopback(String name) {
        String[] octets = name.split("\\.", -1);
        if (octets.length != 4 || !octets[0].equals("127")) {
            return false;
        }
        for (String octet : octets) {
            if (!octet.matches("[0-9]{1,3}") || Integer.parseInt(octet) > 255) {
                return false;
            }
        }
        return true;
    }

    /** Quarkus's bind setting accepts unbracketed IPv6, unlike an HTTP Host authority. */
    private static String bracketLiteral(String host) {
        return host != null && host.contains(":") && !host.startsWith("[") ? "[" + host + "]" : host;
    }
}
