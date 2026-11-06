package com.eonmux.neuralfree.demo.tools;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/** The {@code nftool rpc-server} command line: the llama.cpp-shaped flags and the refusals. */
public class RpcServerCommandTest {

    @Test
    public void parsesTheReferenceFlags() throws Exception {
        RpcServerCommand.Options o = RpcServerCommand.parse(new String[]{
                "--host", "0.0.0.0", "--port", "50060", "--backend", "cuda", "--device", "1", "--key", "secret"});
        assertNotNull(o);
        assertEquals("0.0.0.0", o.host);
        assertEquals(50060, o.port);
        assertEquals(EngineConfig.BackendMode.CUDA, o.backend);
        assertEquals(java.util.Collections.singletonList("1"), o.devices);
        assertEquals("secret", o.key);
    }

    @Test
    public void defaultsMatchTheReferenceServer() throws Exception {
        RpcServerCommand.Options o = RpcServerCommand.parse(new String[]{"-H", "127.0.0.1", "-p", "50052",
                "-b", "opencl", "-d", "0", "--key", "k"});
        assertEquals(RpcServerCommand.DEFAULT_PORT, o.port);
        assertEquals(EngineConfig.BackendMode.OPENCL, o.backend);
    }

    @Test
    public void parsesACommaSeparatedDeviceList() throws Exception {
        RpcServerCommand.Options o = RpcServerCommand.parse(new String[]{
                "-d", "0,NVIDIA,A100", "--key", "k"});
        assertEquals(java.util.Arrays.asList("0", "NVIDIA", "A100"), o.devices);
    }

    @Test
    public void parsesTheCacheFlagWithAndWithoutADirectory() throws Exception {
        RpcServerCommand.Options plain = RpcServerCommand.parse(new String[]{"--cache", "--key", "k"});
        assertEquals(true, plain.cache);
        assertEquals(null, plain.cacheDir);
        RpcServerCommand.Options withDir = RpcServerCommand.parse(new String[]{"-c", "/tmp/rpc-cache",
                "--key", "k"});
        assertEquals(true, withDir.cache);
        assertEquals(java.nio.file.Paths.get("/tmp/rpc-cache"), withDir.cacheDir);
    }

    @Test
    public void parsesTheTlsFlags() throws Exception {
        RpcServerCommand.Options o = RpcServerCommand.parse(new String[]{"--tls-cert", "/etc/rpc/cert.pem",
                "--tls-key", "/etc/rpc/key.pem", "--key", "k"});
        assertEquals("/etc/rpc/cert.pem", o.tlsCert);
        assertEquals("/etc/rpc/key.pem", o.tlsKey);
    }

    @Test
    public void refusesAnUnknownOption() {
        try {
            RpcServerCommand.parse(new String[]{"--threads", "4", "--key", "k"});
            fail("accepted an unknown option");
        } catch (RpcServerCommand.UsageException expected) {
            assertEquals(true, expected.getMessage().contains("--threads"));
        }
    }

    @Test
    public void refusesAMissingValue() {
        try {
            RpcServerCommand.parse(new String[]{"--port"});
            fail("accepted a flag without a value");
        } catch (RpcServerCommand.UsageException expected) {
            assertEquals(true, expected.getMessage().contains("--port"));
        }
    }

    @Test
    public void refusesAPortOutsideTheTcpRange() {
        for (String port : new String[]{"0", "65536", "not-a-port"}) {
            try {
                RpcServerCommand.parse(new String[]{"--port", port, "--key", "k"});
                fail("accepted port " + port);
            } catch (RpcServerCommand.UsageException expected) {
                assertEquals(true, expected.getMessage().contains("--port"));
            }
        }
    }

    @Test
    public void refusesTheCpuEngine() {
        try {
            RpcServerCommand.parse(new String[]{"--backend", "cpu", "--key", "k"});
            fail("accepted the CPU engine");
        } catch (RpcServerCommand.UsageException expected) {
            assertEquals(true, expected.getMessage().contains("CPU"));
        }
    }

    @Test
    public void acceptsTheHipAndMetalBackends() throws Exception {
        assertEquals(EngineConfig.BackendMode.HIP,
                RpcServerCommand.parse(new String[]{"-b", "hip", "--key", "k"}).backend);
        assertEquals(EngineConfig.BackendMode.METAL,
                RpcServerCommand.parse(new String[]{"--backend", "Metal", "--key", "k"}).backend);
    }

    @Test
    public void theUsageLineNamesEveryBackendModeButTheCpuEngine() {
        for (EngineConfig.BackendMode mode : EngineConfig.BackendMode.values()) {
            if (mode == EngineConfig.BackendMode.CPU) {
                assertTrue("the usage does not offer the CPU engine: " + RpcServerCommand.USAGE,
                        !RpcServerCommand.USAGE.contains("cpu"));
                continue;
            }
            assertTrue("the usage names " + mode + ": " + RpcServerCommand.USAGE,
                    RpcServerCommand.USAGE.contains(mode.name().toLowerCase(java.util.Locale.ROOT)));
        }
    }

    @Test
    public void anUnknownBackendIsRefusedNamingEveryGpuBackend() {
        try {
            RpcServerCommand.parse(new String[]{"--backend", "tpu", "--key", "k"});
            fail("accepted tpu");
        } catch (RpcServerCommand.UsageException expected) {
            for (EngineConfig.BackendMode mode : EngineConfig.BackendMode.values()) {
                if (mode == EngineConfig.BackendMode.CPU) {
                    continue;
                }
                assertTrue("the refusal names " + mode + ": " + expected.getMessage(),
                        expected.getMessage()
                                .contains(mode.name().toLowerCase(java.util.Locale.ROOT)));
            }
        }
    }
}
