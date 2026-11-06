package com.eonmux.neuralfree.gguf.engine.kvtransfer;

import org.junit.Test;

import java.net.InetAddress;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The remote-host allowlist of a KV transfer consumer ({@code kv_connector_extra_config.allowed_remote_hosts}):
 * the defaults (loopback and this instance's own {@code kv_ip}), the entry forms a list accepts (host names, IP
 * literals, CIDR ranges, each with an optional port), the resolved address a check returns, and the refusals that
 * name the setting.
 */
public class KvRemoteHostsTest {

    private static KvRemoteHosts hosts(String extraJson) {
        String json = "{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_consumer\", \"kv_ip\": \"10.0.0.5\","
                + " \"kv_connector_extra_config\": " + extraJson + "}";
        return KvRemoteHosts.of(KvTransferConfig.parse(json, "--kv-transfer-config"));
    }

    private static KvRemoteHosts defaultHosts() {
        String json = "{\"kv_connector\": \"TcpKvConnector\", \"kv_role\": \"kv_consumer\", \"kv_ip\": \"10.0.0.5\"}";
        return KvRemoteHosts.of(KvTransferConfig.parse(json, "--kv-transfer-config"));
    }

    private static InetAddress resolved(String host) throws Exception {
        return InetAddress.getByName(host);
    }

    @Test
    public void theDefaultsAllowLoopbackAndTheOwnKvIp() throws Exception {
        KvRemoteHosts hosts = defaultHosts();
        assertEquals(resolved("127.0.0.1"), hosts.select("127.0.0.1", 14579));
        assertEquals(resolved("::1"), hosts.select("::1", 1));
        assertEquals("the instance's own kv_ip", resolved("10.0.0.5"), hosts.select("10.0.0.5", 14579));
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> hosts.select("10.9.8.7", 14579));
        assertTrue(refused.getMessage(), refused.getMessage().contains("allowed_remote_hosts"));
        assertTrue(refused.getMessage(), refused.getMessage().contains("10.9.8.7"));
    }

    @Test
    public void aCidrRangeAllowsItsAddressesOnly() throws Exception {
        KvRemoteHosts hosts = hosts("{\"allowed_remote_hosts\": [\"10.0.0.0/8\"]}");
        assertEquals(resolved("10.1.2.3"), hosts.select("10.1.2.3", 14579));
        assertTrue(refused(hosts, "11.1.2.3").contains("allowed_remote_hosts"));
        assertTrue(refused(hosts, "127.0.0.1").contains("allowed_remote_hosts"));
        assertTrue("families do not mix", refused(hosts, "::1").contains("allowed_remote_hosts"));
    }

    @Test
    public void anExplicitListReplacesTheDefaults() throws Exception {
        KvRemoteHosts hosts = hosts("{\"allowed_remote_hosts\": [\"11.0.0.0/8\"]}");
        assertEquals(resolved("11.1.2.3"), hosts.select("11.1.2.3", 14579));
        assertTrue("the instance's own kv_ip is outside the list", refused(hosts, "10.0.0.5")
                .contains("allowed_remote_hosts"));
    }

    @Test
    public void anEntryWithAPortAllowsThatPortOnly() throws Exception {
        KvRemoteHosts hosts = hosts("{\"allowed_remote_hosts\": [\"127.0.0.1:14579\", \"[::1]:14580\"]}");
        assertEquals(resolved("127.0.0.1"), hosts.select("127.0.0.1", 14579));
        assertEquals(resolved("::1"), hosts.select("::1", 14580));
        assertTrue(refused(hosts, "127.0.0.1", 14580).contains("allowed_remote_hosts"));
        assertTrue(refused(hosts, "::1", 14579).contains("allowed_remote_hosts"));
    }

    @Test
    public void aNameEntryAllowsTheRequestsThatNameIt() throws Exception {
        KvRemoteHosts hosts = hosts("{\"allowed_remote_hosts\": [\"localhost\"]}");
        assertNotNull(hosts.select("localhost", 14579));
        assertTrue("a name entry does not cover another spelling of the same address",
                refused(hosts, "127.0.0.1").contains("allowed_remote_hosts"));
    }

    @Test
    public void aCommaSeparatedStringCarriesTheEntries() throws Exception {
        KvRemoteHosts hosts = hosts("{\"allowed_remote_hosts\": \" 10.0.0.0/8 , 127.0.0.1 \"}");
        assertNotNull(hosts.select("10.2.3.4", 14579));
        assertNotNull(hosts.select("127.0.0.1", 14579));
        assertTrue(refused(hosts, "11.1.1.1").contains("allowed_remote_hosts"));
    }

    @Test
    public void anExplicitEmptyListAllowsNoPeer() {
        KvRemoteHosts hosts = hosts("{\"allowed_remote_hosts\": []}");
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> hosts.select("127.0.0.1", 14579));
        assertTrue(refused.getMessage(), refused.getMessage().contains("allowed_remote_hosts"));
    }

    @Test
    public void aMistypedValueOrEntryRefusesByName() {
        for (String extra : new String[] {"{\"allowed_remote_hosts\": 42}",
                "{\"allowed_remote_hosts\": [\"10.0.0.0/8\", 7]}",
                "{\"allowed_remote_hosts\": [\"[::1\"]}",
                "{\"allowed_remote_hosts\": [\"[name]:1\"]}",
                "{\"allowed_remote_hosts\": [\"/24\"]}",
                "{\"allowed_remote_hosts\": [\"10.0.0.0/x\"]}",
                "{\"allowed_remote_hosts\": [\"10.0.0.0/33\"]}",
                "{\"allowed_remote_hosts\": [\"::1/129\"]}",
                "{\"allowed_remote_hosts\": [\"127.0.0.1:0\"]}",
                "{\"allowed_remote_hosts\": [\"127.0.0.1:99999\"]}",
                "{\"allowed_remote_hosts\": [\"127.0.0.1:x\"]}",
                "{\"allowed_remote_hosts\": [\"bad host\"]}",
                "{\"allowed_remote_hosts\": [\"\"]}",
                "{\"allowed_remote_hosts\": [\"a_b..c\"]}"}) {
            IllegalArgumentException refused = assertThrows(IllegalArgumentException.class, () -> hosts(extra));
            assertTrue(extra, refused.getMessage().contains("allowed_remote_hosts"));
        }
    }

    @Test
    public void loopbackIsWhatTheProducerBindChecks() {
        assertTrue(KvRemoteHosts.isLoopback("127.0.0.1"));
        assertTrue(KvRemoteHosts.isLoopback("127.9.9.9"));
        assertTrue(KvRemoteHosts.isLoopback("localhost"));
        assertTrue(KvRemoteHosts.isLoopback("::1"));
        assertFalse(KvRemoteHosts.isLoopback("0.0.0.0"));
        assertFalse(KvRemoteHosts.isLoopback("10.0.0.1"));
    }

    private static String refused(KvRemoteHosts hosts, String host) {
        return refused(hosts, host, 14579);
    }

    private static String refused(KvRemoteHosts hosts, String host, int port) {
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> hosts.select(host, port));
        return refused.getMessage();
    }
}
