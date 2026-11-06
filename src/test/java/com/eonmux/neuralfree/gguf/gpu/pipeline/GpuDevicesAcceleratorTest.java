package com.eonmux.neuralfree.gguf.gpu.pipeline;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorBackendProvider;
import com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorDevice;
import com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorProviders;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import org.junit.Test;

import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Device selection over an accelerator API: {@link GpuDevices} enumerates the devices a provider
 * reported, names them by their API and index, orders them discrete first, and opens their
 * backends through {@link HWAcceleration}. The provider and its backends are test doubles.
 */
public class GpuDevicesAcceleratorTest {

    private static final long VRAM = 8L * 1024 * 1024 * 1024;

    private static DeviceCapabilities caps(String name, int computeUnits, DeviceClass deviceClass) {
        return new DeviceCapabilities.Builder()
                .deviceName(name)
                .identity(DeviceIdentity.ofName(name))
                .deviceClass(deviceClass)
                .totalVramBytes(VRAM)
                .computeUnits(computeUnits)
                .maxClockFrequencyMhz(1800)
                .build();
    }

    private static ComputeBackend backend(DeviceCapabilities caps) {
        return (ComputeBackend) Proxy.newProxyInstance(ComputeBackend.class.getClassLoader(),
                new Class<?>[]{ComputeBackend.class}, (proxy, method, args) -> {
                    switch (method.getName()) {
                        case "backendType":
                            return BackendType.SYCL;
                        case "capabilities":
                            return caps;
                        case "close":
                            return null;
                        case "toString":
                            return "FakeSYCL@" + caps.getDeviceName();
                        case "hashCode":
                            return System.identityHashCode(proxy);
                        case "equals":
                            return proxy == args[0];
                        default:
                            throw new UnsupportedOperationException(method.getName());
                    }
                });
    }

    private static AcceleratorDevice device(String name, int computeUnits, DeviceClass deviceClass) {
        DeviceCapabilities caps = caps(name, computeUnits, deviceClass);
        return AcceleratorDevice.of(BackendType.SYCL, caps, () -> backend(caps));
    }

    private static final class FakeProvider implements AcceleratorBackendProvider {
        private final List<AcceleratorDevice> devices;

        FakeProvider(List<AcceleratorDevice> devices) {
            this.devices = devices;
        }

        @Override
        public BackendType backendType() {
            return BackendType.SYCL;
        }

        @Override
        public List<AcceleratorDevice> devices() {
            return devices;
        }
    }

    private static HWAcceleration acceleration() {
        List<AcceleratorDevice> devices = new ArrayList<>();
        devices.add(device("Intel Arc B", 128, DeviceClass.DISCRETE));
        devices.add(device("Intel(R) Graphics", 96, DeviceClass.INTEGRATED));
        return new HWAcceleration(EnumSet.of(BackendType.SYCL),
                AcceleratorProviders.of(Collections.singletonList(new FakeProvider(devices))));
    }

    private static EngineConfig config() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.SYCL).build();
    }

    @Test
    public void aSplitSelectsTheAcceleratorDevicesDiscreteFirst() {
        HWAcceleration hw = acceleration();
        try {
            List<GpuDevices.Selected> selected = GpuDevices.select(hw, config());
            assertEquals(2, selected.size());
            assertTrue(selected.get(0).name(), selected.get(0).name().startsWith("sycl0 'Intel Arc B'"));
            assertTrue(selected.get(1).name(), selected.get(1).name().startsWith("sycl1 'Intel(R) Graphics'"));
            assertEquals(BackendType.SYCL, selected.get(0).backend().backendType());
        } finally {
            hw.shutdown();
        }
    }

    @Test
    public void aSelectorNamesAnAcceleratorDevice() {
        HWAcceleration hw = acceleration();
        try {
            GpuDevices.Selected picked = GpuDevices.selectOne(hw, "graphics", "--test-device");
            assertTrue(picked.name(), picked.name().contains("Intel(R) Graphics"));
            assertEquals(BackendType.SYCL, picked.backend().backendType());
        } finally {
            hw.shutdown();
        }
    }

    @Test
    public void theMainGpuIndexPicksAnAcceleratorDevice() {
        HWAcceleration hw = acceleration();
        try {
            EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.SYCL).mainGpu(1).build();
            GpuDevices.Selected main = GpuDevices.openMainGpu(hw, cfg);
            assertTrue(main.name(), main.name().contains("Intel(R) Graphics"));
        } finally {
            hw.shutdown();
        }
    }

    @Test
    public void thePeerSelectionSkipsTheMainDevice() {
        HWAcceleration hw = acceleration();
        try {
            GpuDevices.Selected main = GpuDevices.selectOne(hw, "arc b", "--test-device");
            List<GpuDevices.Selected> peers = GpuDevices.selectPeers(hw, config(), main.backend());
            assertEquals(1, peers.size());
            assertTrue(peers.get(0).name(), peers.get(0).name().contains("Intel(R) Graphics"));
        } finally {
            hw.shutdown();
        }
    }
}
