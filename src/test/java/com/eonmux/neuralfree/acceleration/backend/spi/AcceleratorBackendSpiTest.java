package com.eonmux.neuralfree.acceleration.backend.spi;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import com.eonmux.neuralfree.acceleration.BackendRequirement;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.NoBackendAvailableException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import org.junit.Test;

import java.io.IOException;
import java.lang.reflect.Proxy;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The accelerator backend SPI: API names and labels, the provider table, the device value object,
 * the device ranking, and the full path from a provider's devices through {@link HWAcceleration} to
 * an AUTO or explicit-mode backend. No vendor runtime is needed: the providers and backends are
 * test doubles that record what the engine did with them.
 */
public class AcceleratorBackendSpiTest {

    private static final long VRAM = 8L * 1024 * 1024 * 1024;

    private final List<String> closed = new ArrayList<>();

    private static DeviceCapabilities caps(String name, int computeUnits, int clockMhz, DeviceClass deviceClass) {
        return new DeviceCapabilities.Builder()
                .deviceName(name)
                .identity(DeviceIdentity.ofName(name))
                .deviceClass(deviceClass)
                .totalVramBytes(VRAM)
                .computeUnits(computeUnits)
                .maxClockFrequencyMhz(clockMhz)
                .build();
    }

    private ComputeBackend backend(BackendType type, DeviceCapabilities caps) {
        return (ComputeBackend) Proxy.newProxyInstance(ComputeBackend.class.getClassLoader(),
                new Class<?>[]{ComputeBackend.class}, (proxy, method, args) -> {
                    switch (method.getName()) {
                        case "backendType":
                            return type;
                        case "capabilities":
                            return caps;
                        case "close":
                            closed.add(caps.getDeviceName());
                            return null;
                        case "toString":
                            return "Fake" + type + "@" + caps.getDeviceName();
                        case "hashCode":
                            return System.identityHashCode(proxy);
                        case "equals":
                            return proxy == args[0];
                        default:
                            throw new UnsupportedOperationException(method.getName());
                    }
                });
    }

    private AcceleratorDevice device(BackendType type, DeviceCapabilities caps) {
        return AcceleratorDevice.of(type, caps, () -> backend(type, caps));
    }

    private static final class FakeProvider implements AcceleratorBackendProvider {
        private final BackendType type;
        private final List<AcceleratorDevice> devices;
        private final RuntimeException failure;

        FakeProvider(BackendType type, List<AcceleratorDevice> devices) {
            this(type, devices, null);
        }

        FakeProvider(BackendType type, List<AcceleratorDevice> devices, RuntimeException failure) {
            this.type = type;
            this.devices = devices;
            this.failure = failure;
        }

        @Override
        public BackendType backendType() {
            return type;
        }

        @Override
        public List<AcceleratorDevice> devices() {
            if (failure != null) {
                throw failure;
            }
            return devices;
        }

        @Override
        public String providerName() {
            return "fake-" + type.label();
        }
    }

    @Test
    public void apiNamesLabelsAndAliases() {
        assertEquals(BackendType.SYCL, BackendType.fromApiName(" sycl "));
        assertEquals(BackendType.SYCL, BackendType.fromApiName("level-zero"));
        assertEquals(BackendType.SYCL, BackendType.fromApiName("LEVEL_ZERO"));
        assertEquals(BackendType.SYCL, BackendType.fromApiName("levelzero"));
        assertEquals(BackendType.WEBGPU, BackendType.fromApiName("web-gpu"));
        assertEquals(BackendType.ZDNN, BackendType.fromApiName("z-dnn"));
        assertEquals(BackendType.ZENDNN, BackendType.fromApiName("Zen-DNN"));
        assertEquals(BackendType.XPU, BackendType.fromApiName("xpu"));
        assertEquals("zDNN", BackendType.ZDNN.label());
        assertEquals("ZenDNN", BackendType.ZENDNN.label());
        assertEquals("OpenCL", BackendType.OPENCL.label());
        assertEquals(BackendType.METAL, BackendType.fromApiName("METAL"));
        assertTrue(BackendType.HEXAGON.isAccelerator());
        assertFalse("CUDA is driven by the engine", BackendType.CUDA.isAccelerator());
        assertFalse("Metal is driven by the engine", BackendType.METAL.isAccelerator());
        try {
            BackendType.fromApiName("not-an-api");
            fail("an unknown API must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("not-an-api"));
            assertTrue(e.getMessage(), e.getMessage().contains("sycl"));
            assertTrue(e.getMessage(), e.getMessage().contains("metal"));
        }
    }

    /** A provider ServiceLoader can instantiate: public, with a public no-argument constructor. */
    public static class ServiceProvider implements AcceleratorBackendProvider {
        public ServiceProvider() {
        }

        @Override
        public BackendType backendType() {
            return BackendType.SYCL;
        }

        @Override
        public List<AcceleratorDevice> devices() {
            return Collections.emptyList();
        }
    }

    /** A second provider of the same API, to exercise the conflict path. */
    public static final class SecondServiceProvider extends ServiceProvider {
    }

    private static URLClassLoader loaderWithServices(String... providers) throws IOException {
        Path dir = Files.createTempDirectory("nf-accelerator-spi");
        Path services = dir.resolve("META-INF/services");
        Files.createDirectories(services);
        Files.write(services.resolve(AcceleratorBackendProvider.class.getName()), Arrays.asList(providers),
                StandardCharsets.UTF_8);
        return new URLClassLoader(new URL[]{dir.toUri().toURL()},
                AcceleratorBackendSpiTest.class.getClassLoader());
    }

    @Test
    public void theServiceLoaderFindsAnInstalledProvider() throws IOException {
        try (URLClassLoader loader = loaderWithServices(ServiceProvider.class.getName())) {
            AcceleratorProviders providers = AcceleratorProviders.load(loader);
            assertTrue(providers.providerFor(BackendType.SYCL) instanceof ServiceProvider);
            assertEquals(EnumSet.of(BackendType.SYCL), providers.installedTypes());
            assertTrue(providers.loadFailures().isEmpty());
        }
    }

    @Test
    public void twoInstalledProvidersOfOneApiConflictAndAreBothSkipped() throws IOException {
        try (URLClassLoader loader = loaderWithServices(ServiceProvider.class.getName(),
                SecondServiceProvider.class.getName())) {
            AcceleratorProviders providers = AcceleratorProviders.load(loader);
            assertNull(providers.providerFor(BackendType.SYCL));
            assertTrue(providers.failureReason(BackendType.SYCL),
                    providers.failureReason(BackendType.SYCL).contains("two SYCL backend providers"));
        }
    }

    @Test
    public void aBrokenServiceEntryDoesNotHideAWorkingProvider() throws IOException {
        try (URLClassLoader loader = loaderWithServices("com.example.NoSuchProvider",
                ServiceProvider.class.getName())) {
            AcceleratorProviders providers = AcceleratorProviders.load(loader);
            assertNotNull(providers.providerFor(BackendType.SYCL));
            assertFalse(providers.loadFailures().isEmpty());
        }
    }

    @Test
    public void providerTableFindsOneProviderPerAcceleratorApi() {
        AcceleratorDevice device = device(BackendType.SYCL, caps("Intel Arc B", 128, 2100, DeviceClass.DISCRETE));
        FakeProvider provider = new FakeProvider(BackendType.SYCL, Collections.singletonList(device));
        AcceleratorProviders providers = AcceleratorProviders.of(Collections.singletonList(provider));

        assertSame(provider, providers.providerFor(BackendType.SYCL));
        assertNull(providers.providerFor(BackendType.CANN));
        assertNull(providers.providerFor(BackendType.OPENCL));
        assertEquals(EnumSet.of(BackendType.SYCL), providers.installedTypes());
        assertNull(providers.failureReason(BackendType.SYCL));
    }

    @Test
    public void providerTableRefusesBuiltInAndDuplicateTypes() {
        FakeProvider cuda = new FakeProvider(BackendType.CUDA, Collections.<AcceleratorDevice>emptyList());
        try {
            AcceleratorProviders.of(Collections.singletonList(cuda));
            fail("a provider of a built-in API must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("CUDA"));
            assertTrue(e.getMessage(), e.getMessage().contains("accelerator"));
        }

        FakeProvider first = new FakeProvider(BackendType.SYCL, Collections.<AcceleratorDevice>emptyList());
        FakeProvider second = new FakeProvider(BackendType.SYCL, Collections.<AcceleratorDevice>emptyList());
        try {
            AcceleratorProviders.of(Arrays.asList(first, second));
            fail("two providers of one API must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("two SYCL backend providers"));
        }
    }

    @Test
    public void deviceRefusesABackendOfAnotherApiOrNoBackend() {
        DeviceCapabilities caps = caps("Intel Arc B", 128, 2100, DeviceClass.DISCRETE);
        AcceleratorDevice wrongType = AcceleratorDevice.of(BackendType.SYCL, caps,
                () -> backend(BackendType.CUDA, caps));
        try {
            wrongType.openBackend();
            fail("a provider must not return a backend of another API");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("SYCL"));
            assertTrue(e.getMessage(), e.getMessage().contains("CUDA"));
        }

        AcceleratorDevice nothing = AcceleratorDevice.of(BackendType.SYCL, caps, () -> null);
        try {
            nothing.openBackend();
            fail("a provider must not return nothing");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("Intel Arc B"));
        }
    }

    @Test
    public void deviceRefusesABuiltInApi() {
        try {
            AcceleratorDevice.of(BackendType.OPENCL, caps("d", 1, 1, DeviceClass.DISCRETE),
                    () -> backend(BackendType.OPENCL, caps("d", 1, 1, DeviceClass.DISCRETE)));
            fail("an accelerator device must not be declared for a built-in API");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("OPENCL"));
        }
    }

    @Test
    public void rankingPrefersDiscreteThenThroughput() {
        AcceleratorDevice integrated = device(BackendType.SYCL,
                caps("Intel(R) Graphics", 96, 2000, DeviceClass.INTEGRATED));
        AcceleratorDevice small = device(BackendType.SYCL, caps("Intel Arc A", 64, 1500, DeviceClass.DISCRETE));
        AcceleratorDevice fast = device(BackendType.SYCL, caps("Intel Arc B", 128, 2100, DeviceClass.DISCRETE));

        assertSame(fast, AcceleratorDeviceRanking.fastest(Arrays.asList(integrated, small, fast), BackendType.SYCL));
        assertSame(small, AcceleratorDeviceRanking.fastest(Arrays.asList(integrated, small), BackendType.SYCL));
        assertNull(AcceleratorDeviceRanking.fastest(Collections.<AcceleratorDevice>emptyList(), BackendType.SYCL));
    }

    @Test
    public void rankingHonoursTheApiDeviceProperty() {
        AcceleratorDevice small = device(BackendType.SYCL, caps("Intel Arc A", 64, 1500, DeviceClass.DISCRETE));
        AcceleratorDevice fast = device(BackendType.SYCL, caps("Intel Arc B", 128, 2100, DeviceClass.DISCRETE));
        System.setProperty("neuralfree.sycl.device", "arc a");
        try {
            assertSame(small, AcceleratorDeviceRanking.fastest(Arrays.asList(fast, small), BackendType.SYCL));
        } finally {
            System.clearProperty("neuralfree.sycl.device");
        }
        System.setProperty("neuralfree.device", "arc b");
        try {
            assertSame(fast, AcceleratorDeviceRanking.fastest(Arrays.asList(small, fast), BackendType.SYCL));
        } finally {
            System.clearProperty("neuralfree.device");
        }
    }

    @Test
    public void rankingRefusesABuiltInApi() {
        try {
            AcceleratorDeviceRanking.fastest(Collections.<AcceleratorDevice>emptyList(), BackendType.CUDA);
            fail("a built-in API has no accelerator devices");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("CUDA"));
        }
    }

    @Test
    public void hardwareAccelerationEnumeratesAndCreatesAcceleratorBackends() {
        AcceleratorDevice integrated = device(BackendType.SYCL,
                caps("Intel(R) Graphics", 96, 2000, DeviceClass.INTEGRATED));
        AcceleratorDevice discrete = device(BackendType.SYCL,
                caps("Intel Arc B", 128, 2100, DeviceClass.DISCRETE));
        AcceleratorProviders providers = AcceleratorProviders.of(Collections.singletonList(
                new FakeProvider(BackendType.SYCL, Arrays.asList(integrated, discrete))));

        HWAcceleration hw = new HWAcceleration(EnumSet.of(BackendType.SYCL), providers);
        try {
            assertEquals(EnumSet.of(BackendType.SYCL), hw.getEnumeratedApis());
            assertEquals(2, hw.getAcceleratorDevices(BackendType.SYCL).size());

            ComputeBackend explicit = hw.createBackend(ACCELERATION_MODE.SYCL);
            assertEquals(BackendType.SYCL, explicit.backendType());
            assertEquals("the ranking picks the discrete device", "Intel Arc B",
                    explicit.capabilities().getDeviceName());

            ComputeBackend auto = hw.createBackend(ACCELERATION_MODE.AUTO);
            assertEquals(BackendType.SYCL, auto.backendType());
            assertEquals("Intel Arc B", auto.capabilities().getDeviceName());
        } finally {
            hw.shutdown();
        }
        assertTrue("shutdown closes the backends it handed out", closed.contains("Intel Arc B"));
        assertEquals("the accelerator devices are released with the instance", 0,
                hw.getAcceleratorDevices(BackendType.SYCL).size());
    }

    @Test
    public void autoFallsBackToTheNextAcceleratorApiWhenTheRequirementRejectsOne() {
        AcceleratorDevice sycl = device(BackendType.SYCL, caps("Intel Arc B", 128, 2100, DeviceClass.DISCRETE));
        AcceleratorDevice cann = device(BackendType.CANN, caps("Ascend 910B", 64, 1800, DeviceClass.DISCRETE));
        AcceleratorProviders providers = AcceleratorProviders.of(Arrays.asList(
                new FakeProvider(BackendType.SYCL, Collections.singletonList(sycl)),
                new FakeProvider(BackendType.CANN, Collections.singletonList(cann))));
        BackendRequirement noSycl = b -> b.backendType() == BackendType.SYCL ? "LLM kernels unavailable" : null;

        HWAcceleration hw = new HWAcceleration(EnumSet.of(BackendType.SYCL, BackendType.CANN), providers);
        try {
            ComputeBackend backend = hw.createBackend(ACCELERATION_MODE.AUTO, noSycl);
            assertEquals(BackendType.CANN, backend.backendType());
            assertEquals("Ascend 910B", backend.capabilities().getDeviceName());
        } finally {
            hw.shutdown();
        }
        assertTrue("the rejected SYCL backend is closed", closed.contains("Intel Arc B"));
    }

    @Test
    public void explicitAcceleratorModeWithoutAProviderNamesTheApiAndTheSpi() {
        HWAcceleration hw = new HWAcceleration(EnumSet.of(BackendType.SYCL),
                AcceleratorProviders.of(Collections.<AcceleratorBackendProvider>emptyList()));
        try {
            hw.createBackend(ACCELERATION_MODE.SYCL);
            fail("a mode without a provider must not produce a backend");
        } catch (NoBackendAvailableException e) {
            assertNotNull(e.getMessage());
            assertTrue(e.getMessage(), e.getMessage().contains("SYCL"));
            assertTrue(e.getMessage(), e.getMessage().contains("no SYCL backend provider is installed"));
            assertTrue(e.getMessage(), e.getMessage().contains("META-INF/services"));
        } finally {
            hw.shutdown();
        }
    }

    @Test
    public void aProviderThatFailsToEnumerateIsReportedNotFatal() {
        FakeProvider broken = new FakeProvider(BackendType.SYCL, Collections.<AcceleratorDevice>emptyList(),
                new IllegalStateException("Level Zero runtime not found"));
        HWAcceleration hw = new HWAcceleration(EnumSet.of(BackendType.SYCL),
                AcceleratorProviders.of(Collections.singletonList(broken)));
        try {
            assertTrue(hw.getAcceleratorDevices(BackendType.SYCL).isEmpty());
            hw.createBackend(ACCELERATION_MODE.SYCL);
            fail("a failed enumeration must not produce a backend");
        } catch (NoBackendAvailableException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("Level Zero runtime not found"));
        } finally {
            hw.shutdown();
        }
    }

    @Test
    public void getAcceleratorDevicesRefusesABuiltInApi() {
        HWAcceleration hw = new HWAcceleration(EnumSet.of(BackendType.SYCL),
                AcceleratorProviders.of(Collections.<AcceleratorBackendProvider>emptyList()));
        try {
            hw.getAcceleratorDevices(BackendType.OPENCL);
            fail("OPENCL is not an accelerator API");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("OPENCL"));
        } finally {
            hw.shutdown();
        }
    }

    @Test
    public void createBackendRefusesADeviceOfAnApiTheInstanceDidNotEnumerate() {
        HWAcceleration hw = new HWAcceleration(EnumSet.of(BackendType.SYCL),
                AcceleratorProviders.of(Collections.<AcceleratorBackendProvider>emptyList()));
        AcceleratorDevice cann = device(BackendType.CANN, caps("Ascend 910B", 64, 1800, DeviceClass.DISCRETE));
        try {
            hw.createBackend(cann);
            fail("the instance never enumerated CANN");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("CANN"));
        } finally {
            hw.shutdown();
        }
    }
}
