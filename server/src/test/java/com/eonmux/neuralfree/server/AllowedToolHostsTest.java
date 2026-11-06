package com.eonmux.neuralfree.server;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The tool surfaces' Host allowlist: literal loopbacks, explicit names and strict authority parsing without DNS. */
class AllowedToolHostsTest {

    @Test
    void loopbackNamesAndLiteralsPassWithoutResolvingNames() {
        AllowedToolHosts hosts = new AllowedToolHosts("127.0.0.1", null);
        for (String host : new String[]{"localhost", "LOCALHOST:8080", "localhost.", "127.0.0.1:8080",
                "127.255.0.2", "[::1]", "[0:0:0:0:0:0:0:1]:8080"}) {
            assertTrue(hosts.accepts(host), host);
        }
        for (String host : new String[]{"evil.example", "localhost.evil.example", "127.0.0.1.evil.example",
                "128.0.0.1", "127.0.0.999", "127.1", "2130706433", "[::2]"}) {
            assertFalse(hosts.accepts(host), host);
        }
    }

    @Test
    void bindHostAndExplicitNamesPassButWildcardBindIsNotAnAllowAll() {
        AllowedToolHosts hosts = new AllowedToolHosts("inference.internal", " tools.example, 192.0.2.4, [2001:db8::1] ");
        for (String host : new String[]{"INFERENCE.INTERNAL:443", "tools.example.:8080", "192.0.2.4",
                "[2001:0db8:0:0:0:0:0:1]:443"}) {
            assertTrue(hosts.accepts(host), host);
        }
        assertFalse(hosts.accepts("other.example"));
        assertFalse(new AllowedToolHosts("0.0.0.0", "").accepts("other.example"));
        assertFalse(new AllowedToolHosts("::", null).accepts("other.example"));
    }

    @Test
    void aMissingAmbiguousOrMalformedHostFailsClosed() {
        AllowedToolHosts hosts = new AllowedToolHosts("127.0.0.1", null);
        for (String host : new String[]{null, "", " ", "localhost,evil.example", "localhost@evil.example",
                "evil.example@localhost", "http://localhost", "localhost/path", "localhost?x", "localhost#x",
                "localhost:", "localhost:abc", "localhost:65536", "localhost:-1", "::1", "[::1]evil",
                "[::1%25lo]", "localhost\t.evil.example"}) {
            assertFalse(hosts.accepts(host), String.valueOf(host));
        }
    }

    @Test
    void aStarDisablesTheCheck() {
        AllowedToolHosts hosts = new AllowedToolHosts("127.0.0.1", "*");
        assertTrue(hosts.accepts("evil.example:8080"));
        assertTrue(hosts.accepts(null));
        assertTrue(hosts.disabled());
    }

    @Test
    void unsupportedAllowlistEntriesNameTheProperty() {
        for (String value : new String[]{"ok.example,", ",ok.example", "ok.example,,other.example",
                "https://example.com", "example.com:8080", "*.example.com", "[::1]:8080", "example.com/path"}) {
            IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                    () -> new AllowedToolHosts("127.0.0.1", value), value);
            assertTrue(error.getMessage().contains("neuralfree.server.allowed.hosts"), error.getMessage());
        }
    }
}
