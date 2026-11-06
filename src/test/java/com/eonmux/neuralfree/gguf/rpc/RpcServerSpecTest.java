package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import org.junit.Test;

import java.util.Collections;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** The {@code neuralfree.rpc} entry grammar and the configuration rules that pair it with a layer split. */
public class RpcServerSpecTest {

    private static RpcServerSpec spec(String host, int port, String key) {
        RpcServerSpec parsed = RpcServerSpec.parse(host + ":" + port + "@" + key);
        return parsed;
    }

    @Test
    public void parsesAnInlineKey() {
        RpcServerSpec spec = RpcServerSpec.parse("gpu-box:50052@s3cret");
        assertEquals("gpu-box", spec.host());
        assertEquals(50052, spec.port());
        assertEquals("s3cret", spec.key());
        assertEquals("gpu-box:50052", spec.describe());
        assertEquals("gpu-box:50052", spec.toString());
        assertFalse(spec.tls());
    }

    @Test
    public void parsesTheTlsScheme() {
        RpcServerSpec spec = RpcServerSpec.parse("tls://gpu-box:50052@s3cret");
        assertEquals("gpu-box", spec.host());
        assertEquals(50052, spec.port());
        assertEquals("s3cret", spec.key());
        assertTrue(spec.tls());
        assertEquals("tls://gpu-box:50052", spec.describe());
        // the scheme is case-insensitive and never prints the key
        assertTrue(RpcServerSpec.parse("TLS://gpu-box:50052@s3cret").tls());
        assertFalse(spec.describe().contains("s3cret"));
        // the scheme distinguishes two specs of one endpoint
        assertTrue(!spec.equals(RpcServerSpec.parse("gpu-box:50052@s3cret")));
    }

    @Test
    public void refusesAnEmptyTlsEndpoint() {
        for (String entry : new String[]{"tls://", "TLS://", "tls:// @key"}) {
            try {
                RpcServerSpec.parse(entry);
                fail("accepted '" + entry + "'");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains(EngineConfig.PROP_RPC));
            }
        }
    }

    @Test
    public void describeNeverPrintsTheKey() {
        RpcServerSpec spec = RpcServerSpec.parse("127.0.0.1:1@do-not-print");
        assertFalse(spec.toString().contains("do-not-print"));
        assertFalse(spec.describe().contains("do-not-print"));
    }

    @Test
    public void parsesAnIpv6Host() {
        RpcServerSpec spec = RpcServerSpec.parse("[::1]:50052@key");
        assertEquals("::1", spec.host());
        assertEquals(50052, spec.port());
        assertEquals("[::1]:50052", spec.describe());
    }

    @Test
    public void refusesAMalformedEndpoint() {
        for (String entry : new String[]{"host@key", "host:0@key", "host:70000@key", "host:abc@key",
                "[::1@key", ":50052@key", " :50052@key"}) {
            try {
                RpcServerSpec.parse(entry);
                fail("accepted '" + entry + "'");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains(EngineConfig.PROP_RPC));
            }
        }
    }

    @Test
    public void refusesAnEmptyInlineKey() {
        try {
            RpcServerSpec.parse("host:50052@");
            fail("accepted an empty key");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("empty key"));
        }
    }

    @Test
    public void maskedEntriesNeverPrintTheKey() {
        assertEquals("box-a:50052@...", RpcServerSpec.masked("box-a:50052@ka"));
        assertEquals("box-a:50052@...,box-b:50053@...",
                RpcServerSpec.masked("box-a:50052@ka,box-b:50053@kb"));
        assertEquals("[::1]:50052@...", RpcServerSpec.masked("[::1]:50052@ka"));
        assertEquals("box-a:50052", RpcServerSpec.masked("box-a:50052"));
    }

    @Test
    public void parsesAListFromProperties() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "layer");
        p.setProperty(EngineConfig.PROP_RPC, "box-a:50052@ka,box-b:50053@kb");
        EngineConfig cfg = EngineConfig.fromProperties(p);
        assertEquals(2, cfg.rpcServers().size());
        assertEquals("box-a", cfg.rpcServers().get(0).host());
        assertEquals("kb", cfg.rpcServers().get(1).key());
    }

    @Test
    public void refusesAnRpcServerWithoutALayerSplit() {
        EngineConfig.Builder builder = EngineConfig.builder()
                .rpcServers(Collections.singletonList(spec("box", 50052, "k")));
        try {
            builder.build();
            fail("accepted a remote device outside a layer split");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains(EngineConfig.PROP_RPC));
            assertTrue(expected.getMessage().contains(EngineConfig.PROP_SPLIT_MODE));
        }
    }

    @Test
    public void refusesRpcServersOnTheCpuEngine() {
        try {
            EngineConfig.builder()
                    .backend(EngineConfig.BackendMode.CPU)
                    .splitMode(EngineConfig.SplitMode.LAYER)
                    .addRpcServer(spec("box", 50052, "k"))
                    .build();
            fail("accepted remote devices on the CPU engine");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains(EngineConfig.PROP_RPC));
        }
    }

    @Test
    public void keepsRpcServersThroughToBuilder() {
        EngineConfig cfg = EngineConfig.builder()
                .splitMode(EngineConfig.SplitMode.LAYER)
                .addRpcServer(spec("box", 50052, "k"))
                .build();
        assertEquals(1, cfg.rpcServers().size());
        assertEquals(1, cfg.toBuilder().build().rpcServers().size());
        assertTrue(cfg.toString().contains("rpc="));
        assertFalse(cfg.toString().contains("@k"));
    }

    @Test
    public void refusesABlankPropertyListEntry() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_SPLIT_MODE, "layer");
        p.setProperty(EngineConfig.PROP_RPC, "box-a:50052@ka,,box-b:50053@kb");
        try {
            EngineConfig.fromProperties(p);
            fail("accepted an empty rpc entry");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains(EngineConfig.PROP_RPC));
            assertFalse("a blank-entry refusal masks the keys: " + expected.getMessage(),
                    expected.getMessage().contains("@ka") || expected.getMessage().contains("@kb"));
        }
    }

    @Test
    public void defaultsToNoRemoteServers() {
        assertTrue(EngineConfig.defaults().rpcServers().isEmpty());
        assertNull(EngineConfig.defaults().deviceFilter());
    }
}
