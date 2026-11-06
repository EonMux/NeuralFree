package com.eonmux.neuralfree.gguf.gpu.pipeline;

import com.eonmux.neuralfree.Config;
import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.EnumSet;
import java.util.List;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Peer selection of a feed-forward row split on the OpenCL devices of the machine: the model's own device is never a
 * peer, a lone selector that names it leaves no peer (the model loads on one device), and a selector that names no
 * device is refused by name. Every enumerated device other than the model's is returned, so on a machine with one
 * OpenCL device the list is empty.
 */
public class GpuDevicesPeersTest {

    private HWAcceleration hw;
    private OclComputeBackend main;

    @Before
    public void open() {
        GpuTestUtil.assumeOpenCLAvailable();
        hw = new HWAcceleration(EnumSet.of(ComputeBackend.BackendType.OPENCL));
        main = new OclComputeBackend(hw.getCLDevices().get(0));
    }

    @After
    public void close() {
        if (main != null) {
            main.close();
        }
        if (hw != null) {
            hw.shutdown();
        }
    }

    private static EngineConfig cfg(String... devices) {
        EngineConfig.Builder b = EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL);
        for (String device : devices) {
            b.addGpuDevice(device);
        }
        return b.build();
    }

    private static HWAcceleration enumeration() {
        return HWAcceleration.forMode(Config.ACCELERATION_MODE.OPENCL);
    }

    @Test
    public void theModelDeviceIsNeverAPeer() {
        HWAcceleration enumeration = enumeration();
        try {
            List<GpuDevices.Selected> peers = GpuDevices.selectPeers(enumeration, cfg(), main);
            DeviceIdentity model = main.capabilities().getIdentity();
            for (GpuDevices.Selected peer : peers) {
                DeviceIdentity other = peer.backend().capabilities().getIdentity();
                assertFalse("the peer is a distinct device",
                        model != null && other != null && model.samePhysicalDevice(other));
            }
        } finally {
            enumeration.shutdown();
        }
    }

    @Test
    public void aSelectorNamingTheModelDeviceLeavesNoPeer() {
        HWAcceleration enumeration = enumeration();
        try {
            assertTrue("only the model's device is named, so no rank remains",
                    GpuDevices.selectPeers(enumeration, cfg("0"), main).isEmpty());
        } finally {
            enumeration.shutdown();
        }
    }

    @Test
    public void aSelectorThatNamesNoDeviceIsRefused() {
        HWAcceleration enumeration = enumeration();
        try {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> GpuDevices.selectPeers(enumeration, cfg("no-such-device"), main));
            assertTrue(e.getMessage(), e.getMessage().contains("names no device"));
        } finally {
            enumeration.shutdown();
        }
    }
}
