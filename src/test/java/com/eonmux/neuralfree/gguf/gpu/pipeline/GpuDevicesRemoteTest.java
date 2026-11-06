package com.eonmux.neuralfree.gguf.gpu.pipeline;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.rpc.RemoteComputeBackend;
import com.eonmux.neuralfree.gguf.rpc.RpcComputeServer;
import com.eonmux.neuralfree.gguf.rpc.RpcServerSpec;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The {@code neuralfree.rpc} servers in the device selection of a layer split: without selectors every server takes
 * part and contributes one device per view it serves ({@code rpc0:<device>}); a {@code neuralfree.gpu.devices}
 * selector names a server ({@code rpc0}) and one that names no server is refused with the remote names in the list.
 */
public class GpuDevicesRemoteTest {

    private static final String KEY = "gpu-devices-remote-key";

    private HWAcceleration acceleration;
    private RpcComputeServer server;

    @Before
    public void setUp() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        acceleration = HWAcceleration.forMode(ACCELERATION_MODE.OPENCL);
        GpuDevices.Selected served = GpuDevices.selectOne(acceleration, "0", "--device");
        server = new RpcComputeServer(served.backend(), KEY, "127.0.0.1", 0);
        server.start();
    }

    @After
    public void tearDown() {
        if (server != null) {
            server.close();
        }
        if (acceleration != null) {
            acceleration.shutdown();
        }
    }

    private EngineConfig cfgWithServer(String... devices) {
        EngineConfig.Builder b = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .splitMode(EngineConfig.SplitMode.LAYER)
                .addRpcServer(RpcServerSpec.parse("127.0.0.1:" + server.port() + "@" + KEY));
        for (String device : devices) {
            b.addGpuDevice(device);
        }
        return b.build();
    }

    @Test
    public void aServerWithoutSelectorsTakesPartAsOneDevicePerView() {
        List<GpuDevices.Selected> devices = GpuDevices.select(acceleration, cfgWithServer());
        assertEquals("every local device plus the server's one view", localCount() + 1, devices.size());
        boolean sawRemote = false;
        for (GpuDevices.Selected device : devices) {
            if (device.remote()) {
                sawRemote = true;
                assertEquals(RemoteComputeBackend.class, device.backend().getClass());
                assertTrue(device.name(), device.name().startsWith("rpc0:"));
            }
        }
        assertTrue(sawRemote);
        GpuDevices.closeRemote(devices);
    }

    /** The deduplicated local device count a split would pick without remotes. */
    private int localCount() {
        List<GpuDevices.Selected> locals = GpuDevices.select(acceleration,
                EngineConfig.builder()
                        .backend(EngineConfig.BackendMode.OPENCL)
                        .splitMode(EngineConfig.SplitMode.LAYER)
                        .build());
        GpuDevices.closeRemote(locals);
        return locals.size();
    }

    @Test
    public void aSelectorPicksTheServerAndCanLeaveItOut() {
        List<GpuDevices.Selected> picked = GpuDevices.select(acceleration, cfgWithServer("0", "rpc0"));
        assertEquals("the selector names the local device and the server", 2, picked.size());
        GpuDevices.closeRemote(picked);
        // without the server in the selector list the split is one device and selects nothing
        List<GpuDevices.Selected> without = GpuDevices.select(acceleration, cfgWithServer("0"));
        assertTrue("a selector that excludes the remote leaves a one-device load", without.isEmpty());
    }

    @Test
    public void aSelectorThatNamesNoDeviceListsTheServer() {
        try {
            GpuDevices.select(acceleration, cfgWithServer("no-such-device"));
            fail("accepted a selector that names no device");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("rpc0"));
            assertTrue(expected.getMessage(), expected.getMessage().contains("no-such-device"));
        }
    }

    @Test
    public void anUnreachableServerIsRefusedWithItsEndpoint() {
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL)
                .splitMode(EngineConfig.SplitMode.LAYER)
                .addRpcServer(RpcServerSpec.parse("127.0.0.1:1@" + KEY))
                .build();
        try {
            List<GpuDevices.Selected> devices = GpuDevices.select(acceleration, cfg);
            GpuDevices.closeRemote(devices);
            fail("reached a server on port 1");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("could not be reached"));
        }
    }
}
