package com.eonmux.neuralfree.gguf.engine.kvtransfer;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * The remote addresses a KV transfer consumer may pull a chain from (the connector's
 * {@code kv_connector_extra_config} {@value #FIELD} list): the allowlist the decode side checks a request's
 * {@code remote_host}/{@code remote_port} against before it connects, so an API caller cannot point the instance
 * at an arbitrary host. Without the list the peers allowed are the loopback addresses and this instance's own
 * {@code kv_ip}.
 *
 * <p>An entry is a host name, an IP literal or a CIDR range ({@code 10.0.0.0/8}), each optionally followed by
 * {@code :port} (an IPv6 literal with a port is bracketed, {@code [::1]:14579}); an entry without a port allows
 * every port. A name entry allows the requests whose {@code remote_host} names it; an address or range entry
 * allows the addresses the name resolves to, checked on the resolution the connection itself uses — the name is
 * resolved once and the checked {@link InetAddress} is what the socket connects to, so DNS cannot answer the
 * check with one address and the connect with another. A list entry that is not a host name, IP literal or CIDR
 * range refuses the connector's start by name. Immutable.</p>
 */
public final class KvRemoteHosts {

    /** The extra-config key carrying the allowlist entries. */
    public static final String FIELD = "allowed_remote_hosts";

    /** One allowlist entry: an optional port and the addresses (or the one name) it allows. */
    private interface Rule {

        /** Whether the entry allows the port and the address or name a request names. */
        boolean allows(int port, InetAddress address, String host);

        /** The entry as the refusal message names it. */
        String text();
    }

    /** A CIDR range, an IP literal or the loopback predicate, with an optional port. */
    private static final class Range implements Rule {

        private final int port;
        private final byte[] network;
        private final int prefixBits;
        private final boolean loopback;
        private final String text;

        Range(int port, byte[] network, int prefixBits, boolean loopback, String text) {
            this.port = port;
            this.network = network;
            this.prefixBits = prefixBits;
            this.loopback = loopback;
            this.text = text;
        }

        @Override
        public boolean allows(int asked, InetAddress address, String host) {
            if (port >= 0 && port != asked) {
                return false;
            }
            if (loopback) {
                return address.isLoopbackAddress();
            }
            return inRange(address.getAddress());
        }

        /** Whether {@code bytes} sits in the rule's range. */
        private boolean inRange(byte[] bytes) {
            if (bytes.length != network.length) {
                return false;
            }
            int whole = prefixBits / 8;
            for (int i = 0; i < whole; i++) {
                if (bytes[i] != network[i]) {
                    return false;
                }
            }
            if (prefixBits % 8 == 0) {
                return true;
            }
            int mask = 0xFF << (8 - prefixBits % 8);
            return (bytes[whole] & mask) == (network[whole] & mask);
        }

        @Override
        public String text() {
            return text;
        }
    }

    /** A host name entry, with an optional port: allows the requests whose {@code remote_host} names it. */
    private static final class Name implements Rule {

        private final int port;
        private final String name;
        private final String text;

        Name(int port, String name, String text) {
            this.port = port;
            this.name = name;
            this.text = text;
        }

        @Override
        public boolean allows(int asked, InetAddress address, String host) {
            if (port >= 0 && port != asked) {
                return false;
            }
            return host != null && trimTrailingDot(host).equalsIgnoreCase(name);
        }

        @Override
        public String text() {
            return text;
        }
    }

    private final List<Rule> rules;
    private final boolean defaults;

    private KvRemoteHosts(List<Rule> rules, boolean defaults) {
        this.rules = Collections.unmodifiableList(rules);
        this.defaults = defaults;
    }

    /**
     * The allowlist of {@code config}: its {@value #FIELD} entries, or, when the key is absent, the loopback
     * addresses and the instance's own {@code kv_ip}.
     *
     * @throws IllegalArgumentException when a value is mistyped or an entry is not a host name, IP literal or
     *                                  CIDR range
     */
    public static KvRemoteHosts of(KvTransferConfig config) {
        Object raw = config.extra(FIELD, null);
        if (raw == null) {
            List<Rule> rules = new ArrayList<>();
            rules.add(new Range(-1, null, 0, true, "loopback"));
            rules.add(entry(config.kvIp()));
            return new KvRemoteHosts(rules, true);
        }
        List<Rule> rules = new ArrayList<>();
        for (String entry : entries(raw)) {
            rules.add(entry(entry));
        }
        return new KvRemoteHosts(rules, false);
    }

    /**
     * Whether {@code host} names only loopback addresses: the test that lets a producer without a shared token
     * bind. An unresolvable name is not loopback.
     */
    public static boolean isLoopback(String host) {
        Objects.requireNonNull(host, "host");
        try {
            for (InetAddress address : InetAddress.getAllByName(host.trim())) {
                if (!address.isLoopbackAddress()) {
                    return false;
                }
            }
            return true;
        } catch (UnknownHostException e) {
            return false;
        }
    }

    /**
     * The resolved address of {@code host} the connection must use: the first of the name's addresses the
     * allowlist allows for {@code port}. The caller connects to the returned address, so the check and the
     * connect see one resolution.
     *
     * @throws IllegalArgumentException naming {@code kv_connector_extra_config} {@value #FIELD} when no resolved
     *                                  address is allowed; this refusal is independent of
     *                                  {@code kv_load_failure_policy}
     * @throws UnknownHostException     when {@code host} does not resolve (a failed pull, not a refused one)
     */
    public InetAddress select(String host, int port) throws UnknownHostException {
        String asked = host == null ? "" : host.trim();
        for (InetAddress address : InetAddress.getAllByName(asked)) {
            for (Rule rule : rules) {
                if (rule.allows(port, address, trimTrailingDot(asked))) {
                    return address;
                }
            }
        }
        throw new IllegalArgumentException("the decode request's remote_host " + asked + ":" + port
                + " is not in kv_connector_extra_config." + FIELD + " (" + describe() + "); this instance pulls"
                + " KV only from the addresses that setting allows");
    }

    /** The entries as the refusal message shows them. */
    private String describe() {
        List<String> texts = new ArrayList<>();
        for (Rule rule : rules) {
            texts.add(rule.text());
        }
        return String.join(", ", texts);
    }

    /** The entry list of a raw extra-config value: an array of strings or one comma-separated string. */
    private static List<String> entries(Object raw) {
        List<String> out = new ArrayList<>();
        if (raw instanceof List) {
            for (Object item : (List<?>) raw) {
                if (!(item instanceof String)) {
                    throw new IllegalArgumentException("kv_connector_extra_config." + FIELD
                            + " holds a non-string entry, got " + describe(item));
                }
                out.add((String) item);
            }
            return out;
        }
        if (raw instanceof String) {
            out.addAll(Arrays.asList(((String) raw).split(",")));
            return out;
        }
        throw new IllegalArgumentException("kv_connector_extra_config." + FIELD + " must be a list of entries"
                + " or one comma-separated string, got " + describe(raw));
    }

    private static String describe(Object value) {
        return value == null ? "null" : value.getClass().getSimpleName();
    }

    /** The rule of one entry, refused by name when it names no host, IP literal or CIDR range. */
    private static Rule entry(String raw) {
        String entry = raw == null ? "" : raw.trim();
        if (entry.isEmpty()) {
            throw new IllegalArgumentException("kv_connector_extra_config." + FIELD + " has an empty entry");
        }
        if (entry.startsWith("[")) {
            return bracketed(entry);
        }
        int colon = entry.lastIndexOf(':');
        if (colonSplitsAPort(entry, colon)) {
            return target(entry.substring(0, colon), port(entry.substring(colon + 1), entry));
        }
        // any other colon belongs to an IPv6 literal or CIDR, which carries no port
        return target(entry, -1);
    }

    /** Whether the colon at {@code colon} separates a target from a port: it is the entry's only colon and the
     * text after it is digits. An IPv6 literal holds several colons, so only its last could be a port, and it
     * still holds an earlier one. */
    private static boolean colonSplitsAPort(String entry, int colon) {
        if (colon < 0 || entry.indexOf(':') != colon || colon + 1 == entry.length()) {
            return false;
        }
        for (int i = colon + 1; i < entry.length(); i++) {
            char c = entry.charAt(i);
            if (c < '0' || c > '9') {
                return false;
            }
        }
        return true;
    }

    /** The rule of a bracketed entry {@code [literal]} or {@code [literal]:port}. */
    private static Rule bracketed(String entry) {
        int close = entry.indexOf(']');
        if (close < 0) {
            throw refused(entry, "a bracketed IPv6 entry needs its ]");
        }
        String inside = entry.substring(1, close);
        if (!isNumericLiteral(inside)) {
            throw refused(entry, "a bracketed entry must hold an IPv6 literal");
        }
        int port = -1;
        if (close + 1 < entry.length()) {
            String rest = entry.substring(close + 1);
            if (!rest.startsWith(":")) {
                throw refused(entry, "the text after ] must be a :port");
            }
            port = port(rest.substring(1), entry);
        }
        return literal(inside, port, entry);
    }

    /** The rule of {@code target} (an IP literal, CIDR range or host name) at an optional port. */
    private static Rule target(String target, int port) {
        String text = target;
        if (target.indexOf('/') >= 0) {
            return cidr(target, port);
        }
        if (isNumericLiteral(target)) {
            return literal(target, port, text);
        }
        String name = trimTrailingDot(target);
        if (!isHostName(name)) {
            throw refused(text, "an entry must be a host name, an IP literal or a CIDR range");
        }
        return new Name(port, name.toLowerCase(Locale.ROOT), text + (port >= 0 ? ":" + port : ""));
    }

    /** The CIDR rule of {@code range} (an address, a / and a prefix length) at an optional port. */
    private static Rule cidr(String range, int port) {
        int slash = range.indexOf('/');
        String addressText = range.substring(0, slash);
        String bitsText = range.substring(slash + 1);
        if (!isNumericLiteral(addressText)) {
            throw refused(range, "a CIDR range needs an IP literal before the /");
        }
        InetAddress address;
        try {
            address = InetAddress.getByName(addressText);
        } catch (UnknownHostException e) {
            throw refused(range, "the address before the / does not parse", e);
        }
        int bits;
        try {
            bits = Integer.parseInt(bitsText);
        } catch (NumberFormatException e) {
            throw refused(range, "the prefix length after the / must be a whole number");
        }
        int maximum = address.getAddress().length * 8;
        if (bits < 0 || bits > maximum) {
            throw refused(range, "the prefix length must be in [0, " + maximum + "]");
        }
        return new Range(port, address.getAddress(), bits, false, range + (port >= 0 ? ":" + port : ""));
    }

    /** The address rule of the IP literal {@code literal} at an optional port. */
    private static Rule literal(String literal, int port, String text) {
        InetAddress address;
        try {
            address = InetAddress.getByName(literal);
        } catch (UnknownHostException e) {
            throw refused(text, "the IP literal does not parse", e);
        }
        return new Range(port, address.getAddress(), address.getAddress().length * 8, false,
                text + (port >= 0 ? ":" + port : ""));
    }

    /** The port of the digits after a colon, refused outside {@code [1, 65535]}. */
    private static int port(String digits, String entry) {
        int port;
        try {
            port = Integer.parseInt(digits);
        } catch (NumberFormatException e) {
            throw refused(entry, "the port must be a whole number");
        }
        if (port < 1 || port > 65535) {
            throw refused(entry, "the port must be in [1, 65535]");
        }
        return port;
    }

    /** Whether {@code text} is an IP literal, which {@link InetAddress#getByName} parses without DNS. */
    private static boolean isNumericLiteral(String text) {
        if (text.indexOf(':') >= 0) {
            return text.matches("[0-9a-fA-F:.]+");
        }
        if (!text.matches("[0-9.]{1,15}")) {
            return false;
        }
        String[] octets = text.split("\\.", -1);
        if (octets.length != 4) {
            return false;
        }
        for (String octet : octets) {
            if (octet.isEmpty() || octet.length() > 3 || Integer.parseInt(octet) > 255) {
                return false;
            }
        }
        return true;
    }

    /** Whether {@code name} is a host name an entry may carry. */
    private static boolean isHostName(String name) {
        if (name.isEmpty() || name.length() > 253 || !name.matches("[A-Za-z0-9._-]+")) {
            return false;
        }
        for (String label : name.split("\\.", -1)) {
            if (label.isEmpty() || label.startsWith("-") || label.endsWith("-")) {
                return false;
            }
        }
        return true;
    }

    private static String trimTrailingDot(String host) {
        return host.endsWith(".") ? host.substring(0, host.length() - 1) : host;
    }

    private static IllegalArgumentException refused(String entry, String why) {
        return new IllegalArgumentException("kv_connector_extra_config." + FIELD + " entry '" + entry + "': "
                + why);
    }

    private static IllegalArgumentException refused(String entry, String why, Exception cause) {
        return new IllegalArgumentException("kv_connector_extra_config." + FIELD + " entry '" + entry + "': "
                + why, cause);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof KvRemoteHosts)) {
            return false;
        }
        KvRemoteHosts other = (KvRemoteHosts) o;
        return defaults == other.defaults && rulesTexts().equals(other.rulesTexts());
    }

    @Override
    public int hashCode() {
        return Objects.hash(rulesTexts(), defaults);
    }

    private List<String> rulesTexts() {
        List<String> out = new ArrayList<>();
        for (Rule rule : rules) {
            out.add(rule.text());
        }
        return out;
    }

    @Override
    public String toString() {
        return "KvRemoteHosts{" + describe() + "}";
    }
}
