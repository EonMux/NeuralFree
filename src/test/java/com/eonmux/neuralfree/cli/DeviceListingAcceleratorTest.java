package com.eonmux.neuralfree.cli;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorBackendProvider;
import com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorDevice;
import com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorProviders;
import org.junit.Test;

import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@code --list-devices} over an accelerator backend: the devices a provider reported appear under
 * the API's label, numbered from zero like the built-in APIs, with the memory the provider reports
 * and no size when it reports none.
 */
public class DeviceListingAcceleratorTest {

    private static final long VRAM = 8L * 1024 * 1024 * 1024;
    private static final long MIB = 1024L * 1024L;

    private static DeviceCapabilities caps(String name, long totalBytes) {
        return new DeviceCapabilities.Builder()
                .deviceName(name)
                .identity(DeviceIdentity.ofName(name))
                .deviceClass(DeviceClass.DISCRETE)
                .totalVramBytes(totalBytes)
                .computeUnits(128)
                .maxClockFrequencyMhz(2100)
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

    private static AcceleratorDevice device(String name, long totalBytes) {
        DeviceCapabilities caps = caps(name, totalBytes);
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

    @Test
    public void acceleratorDevicesAreListedUnderTheirApi() {
        AcceleratorProviders providers = AcceleratorProviders.of(Collections.singletonList(
                new FakeProvider(Arrays.asList(device("Intel Arc B", VRAM), device("Intel Arc A", 0L)))));

        List<DeviceListing.Entry> sycl = new ArrayList<>();
        for (DeviceListing.Entry entry : DeviceListing.available(providers)) {
            if ("SYCL".equals(entry.api())) {
                sycl.add(entry);
            }
        }

        assertEquals(2, sycl.size());
        assertEquals(0, sycl.get(0).index());
        assertEquals("Intel Arc B", sycl.get(0).name());
        assertEquals(VRAM, sycl.get(0).totalBytes());
        assertEquals(1, sycl.get(1).index());
        assertEquals("Intel Arc A", sycl.get(1).name());
        assertEquals("a provider that reports no memory prints no size", -1, sycl.get(1).totalBytes());

        String lines = DeviceListing.lines(sycl);
        assertTrue(lines, lines.contains("  SYCL 0: Intel Arc B (" + VRAM / MIB + " MiB)\n"));
        assertTrue(lines, lines.contains("  SYCL 1: Intel Arc A\n"));
    }
}
