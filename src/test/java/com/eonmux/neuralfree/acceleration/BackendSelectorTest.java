package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import org.junit.Test;

import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Pins the AUTO backend policy: device first, then API, with every rejected
 * backend closed and its reason reported. No GPU is needed: candidates build
 * proxy backends that record whether they were closed.
 */
public class BackendSelectorTest {

    private static final String INVENTORY_INCOMPLETE = "LLM kernel inventory incomplete: 12 kernels missing";

    private static final DeviceIdentity RTX_OPENCL = new DeviceIdentity(
            "NVIDIA GeForce RTX 5080 Laptop GPU", 0x10DE, "0000:01:00.0", null);
    private static final DeviceIdentity RTX_CUDA = new DeviceIdentity(
            "NVIDIA GeForce RTX 5080 Laptop GPU", 0x10DE, "01:00.0", "0f1e2d3c4b5a69788796a5b4c3d2e1f0");
    private static final DeviceIdentity RTX_VULKAN = new DeviceIdentity(
            "NVIDIA GeForce RTX 5080 Laptop GPU", 0x10DE, "0000:01:00.0", "0f1e2d3c4b5a69788796a5b4c3d2e1f0");
    private static final DeviceIdentity IGPU_OPENCL = new DeviceIdentity(
            "Intel(R) Graphics", 0x8086, "0000:00:02.0", null);
    private static final DeviceIdentity IGPU_VULKAN = new DeviceIdentity(
            "Intel(R) Graphics (ARL)", 0x8086, "0000:00:02.0", null);

    /** Records what the selector did with each backend it was handed. */
    private static final class Fake {
        final BackendType api;
        final String device;
        final List<String> closedLog;
        boolean created;
        boolean closed;
        ComputeBackend backend;

        Fake(BackendType api, String device, List<String> closedLog) {
            this.api = api;
            this.device = device;
            this.closedLog = closedLog;
        }

        ComputeBackend create() {
            created = true;
            backend = (ComputeBackend) Proxy.newProxyInstance(ComputeBackend.class.getClassLoader(),
                    new Class<?>[]{ComputeBackend.class}, (proxy, method, args) -> {
                        switch (method.getName()) {
                            case "backendType":
                                return api;
                            case "close":
                                closed = true;
                                closedLog.add(api + "@" + device);
                                return null;
                            case "toString":
                                return "Fake" + api + "@" + device;
                            case "hashCode":
                                return System.identityHashCode(proxy);
                            case "equals":
                                return proxy == args[0];
                            default:
                                throw new UnsupportedOperationException(method.getName());
                        }
                    });
            return backend;
        }
    }

    private final List<String> closedLog = new ArrayList<>();
    private final List<Fake> fakes = new ArrayList<>();

    private BackendSelector.Candidate candidate(BackendType api, DeviceIdentity id, DeviceClass cls) {
        Fake fake = new Fake(api, id.name(), closedLog);
        fakes.add(fake);
        return new BackendSelector.Candidate(api, id, cls, fake::create);
    }

    private Fake fakeOf(BackendType api, String deviceName) {
        for (Fake f : fakes) {
            if (f.api == api && f.device.equals(deviceName)) {
                return f;
            }
        }
        throw new AssertionError("no fake for " + api + " on " + deviceName);
    }

    private static BackendSelector.ApiView view(BackendType api, BackendSelector.Candidate winner,
                                                BackendSelector.Candidate... devices) {
        return new BackendSelector.ApiView(api, Arrays.asList(devices), winner, null);
    }

    /** Rejects every backend of the given APIs with the inventory reason. */
    private static BackendRequirement rejecting(BackendType... apis) {
        List<BackendType> rejected = Arrays.asList(apis);
        return b -> rejected.contains(b.backendType()) ? INVENTORY_INCOMPLETE : null;
    }

    /** A hybrid laptop: an RTX 5080 Laptop GPU and an Arrow Lake iGPU, each visible in all three APIs. */
    private List<BackendSelector.ApiView> laptop() {
        BackendSelector.Candidate clIgpu = candidate(BackendType.OPENCL, IGPU_OPENCL, DeviceClass.INTEGRATED);
        BackendSelector.Candidate clRtx = candidate(BackendType.OPENCL, RTX_OPENCL, DeviceClass.DISCRETE);
        BackendSelector.Candidate cuRtx = candidate(BackendType.CUDA, RTX_CUDA, DeviceClass.DISCRETE);
        BackendSelector.Candidate vkIgpu = candidate(BackendType.VULKAN, IGPU_VULKAN, DeviceClass.INTEGRATED);
        BackendSelector.Candidate vkRtx = candidate(BackendType.VULKAN, RTX_VULKAN, DeviceClass.DISCRETE);
        return Arrays.asList(
                view(BackendType.OPENCL, clRtx, clIgpu, clRtx),
                view(BackendType.CUDA, cuRtx, cuRtx),
                view(BackendType.VULKAN, vkRtx, vkIgpu, vkRtx));
    }

    @Test
    public void inventoryIncompleteOnCudaAndVulkanSelectsOpenClOnTheSameDevice() {
        BackendSelector.Selection s = BackendSelector.selectAuto(laptop(), BackendSelector.DEFAULT_ORDER, null,
                rejecting(BackendType.VULKAN, BackendType.CUDA));

        assertEquals(BackendType.OPENCL, s.backend.backendType());
        assertSame(RTX_OPENCL, s.chosen.identity);
        assertTrue(fakeOf(BackendType.VULKAN, RTX_VULKAN.name()).closed);
        assertTrue(fakeOf(BackendType.CUDA, RTX_CUDA.name()).closed);
        assertFalse(fakeOf(BackendType.OPENCL, RTX_OPENCL.name()).closed);
        assertFalse("the iGPU is never built while the target device can run",
                fakeOf(BackendType.VULKAN, IGPU_VULKAN.name()).created);
        assertEquals(2, s.reasons.size());
        assertEquals("Vulkan on '" + RTX_VULKAN.name() + "': " + INVENTORY_INCOMPLETE, s.reasons.get(0));
        assertEquals("CUDA on '" + RTX_CUDA.name() + "': " + INVENTORY_INCOMPLETE, s.reasons.get(1));
    }

    @Test
    public void vulkanIsPreferredOnTheTargetDevice() {
        BackendSelector.Selection s = BackendSelector.selectAuto(laptop(), BackendSelector.DEFAULT_ORDER, null,
                BackendRequirement.ANY);

        assertEquals(BackendType.VULKAN, s.backend.backendType());
        assertSame(RTX_VULKAN, s.chosen.identity);
        assertTrue(s.reasons.isEmpty());
        assertTrue(closedLog.isEmpty());
    }

    @Test
    public void targetIsMatchedAcrossApisEvenWhenThatApiRanksAnotherDeviceFirst() {
        BackendSelector.Candidate clRtx = candidate(BackendType.OPENCL, RTX_OPENCL, DeviceClass.DISCRETE);
        BackendSelector.Candidate vkIgpu = candidate(BackendType.VULKAN, IGPU_VULKAN, DeviceClass.INTEGRATED);
        BackendSelector.Candidate vkRtx = candidate(BackendType.VULKAN, RTX_VULKAN, DeviceClass.DISCRETE);
        List<BackendSelector.ApiView> views = Arrays.asList(
                view(BackendType.OPENCL, clRtx, clRtx),
                view(BackendType.VULKAN, vkIgpu, vkIgpu, vkRtx));

        BackendSelector.Selection s = BackendSelector.selectAuto(views, BackendSelector.DEFAULT_ORDER, null,
                BackendRequirement.ANY);

        assertSame(vkRtx, s.chosen);
        assertFalse(fakeOf(BackendType.VULKAN, IGPU_VULKAN.name()).created);
    }

    @Test
    public void deviceFilterChoosesTheTargetInAnyApi() {
        BackendSelector.Selection s = BackendSelector.selectAuto(laptop(), BackendSelector.DEFAULT_ORDER, "INTEL",
                BackendRequirement.ANY);

        assertEquals(BackendType.VULKAN, s.backend.backendType());
        assertSame(IGPU_VULKAN, s.chosen.identity);
    }

    @Test
    public void deviceFilterMatchingNothingSelectsNormally() {
        BackendSelector.Selection s = BackendSelector.selectAuto(laptop(), BackendSelector.DEFAULT_ORDER,
                "radeon", BackendRequirement.ANY);

        assertSame(RTX_VULKAN, s.chosen.identity);
    }

    @Test
    public void apiOrderIsHonoured() {
        BackendSelector.Selection s = BackendSelector.selectAuto(laptop(),
                Arrays.asList(BackendType.CUDA, BackendType.OPENCL), null, BackendRequirement.ANY);

        assertEquals(BackendType.CUDA, s.backend.backendType());
    }

    @Test
    public void fallbackTriesOtherDevicesWhenNoApiRunsTheTarget() {
        BackendSelector.Candidate clRtx = candidate(BackendType.OPENCL, RTX_OPENCL, DeviceClass.DISCRETE);
        BackendSelector.Candidate cuRtx = candidate(BackendType.CUDA, RTX_CUDA, DeviceClass.DISCRETE);
        BackendSelector.Candidate vkIgpu = candidate(BackendType.VULKAN, IGPU_VULKAN, DeviceClass.INTEGRATED);
        List<BackendSelector.ApiView> views = Arrays.asList(
                view(BackendType.OPENCL, clRtx, clRtx),
                view(BackendType.CUDA, cuRtx, cuRtx),
                view(BackendType.VULKAN, vkIgpu, vkIgpu));

        BackendSelector.Selection s = BackendSelector.selectAuto(views, BackendSelector.DEFAULT_ORDER, null,
                rejecting(BackendType.CUDA, BackendType.OPENCL));

        assertSame(vkIgpu, s.chosen);
        assertEquals(Arrays.asList("CUDA@" + RTX_CUDA.name(), "OPENCL@" + RTX_OPENCL.name()), closedLog);
    }

    @Test
    public void fallbackPrefersDiscreteDevicesOverApiOrder() {
        DeviceIdentity otherIgpu = new DeviceIdentity("Radeon 890M", 0x1002, "0000:c4:00.0", null);
        BackendSelector.Candidate clIgpu = candidate(BackendType.OPENCL, IGPU_OPENCL, DeviceClass.INTEGRATED);
        BackendSelector.Candidate vkOther = candidate(BackendType.VULKAN, otherIgpu, DeviceClass.INTEGRATED);
        BackendSelector.Candidate cuRtx = candidate(BackendType.CUDA, RTX_CUDA, DeviceClass.DISCRETE);
        List<BackendSelector.ApiView> views = Arrays.asList(
                view(BackendType.OPENCL, clIgpu, clIgpu),
                view(BackendType.VULKAN, vkOther, vkOther),
                view(BackendType.CUDA, cuRtx, cuRtx));

        BackendSelector.Selection s = BackendSelector.selectAuto(views, BackendSelector.DEFAULT_ORDER, null,
                rejecting(BackendType.OPENCL));

        assertSame(cuRtx, s.chosen);
        assertFalse(fakeOf(BackendType.VULKAN, otherIgpu.name()).created);
    }

    @Test
    public void fallbackPrefersAnUnknownClassDeviceOverACpuDevice() {
        DeviceIdentity llvmpipe = new DeviceIdentity("llvmpipe (LLVM 19.1.0, 256 bits)", 0x10005, null, null);
        DeviceIdentity unknownGpu = new DeviceIdentity("Mystery GPU", 0x1234, "0000:03:00.0", null);
        BackendSelector.Candidate clRtx = candidate(BackendType.OPENCL, RTX_OPENCL, DeviceClass.DISCRETE);
        BackendSelector.Candidate vkCpu = candidate(BackendType.VULKAN, llvmpipe, DeviceClass.CPU);
        BackendSelector.Candidate cuUnknown = candidate(BackendType.CUDA, unknownGpu, DeviceClass.UNKNOWN);
        List<BackendSelector.ApiView> views = Arrays.asList(
                view(BackendType.OPENCL, clRtx, clRtx),
                view(BackendType.VULKAN, vkCpu, vkCpu),
                view(BackendType.CUDA, cuUnknown, cuUnknown));

        BackendSelector.Selection s = BackendSelector.selectAuto(views, BackendSelector.DEFAULT_ORDER, null,
                rejecting(BackendType.OPENCL));

        assertSame(cuUnknown, s.chosen);
        assertFalse(fakeOf(BackendType.VULKAN, llvmpipe.name()).created);
        assertTrue(BackendSelector.classRank(DeviceClass.INTEGRATED) < BackendSelector.classRank(DeviceClass.UNKNOWN));
        assertTrue(BackendSelector.classRank(DeviceClass.UNKNOWN) < BackendSelector.classRank(DeviceClass.CPU));
    }

    @Test
    public void nothingAcceptedReportsEveryReasonAndClosesEveryBackend() {
        BackendSelector.Candidate clRtx = candidate(BackendType.OPENCL, RTX_OPENCL, DeviceClass.DISCRETE);
        BackendSelector.Candidate cuRtx = candidate(BackendType.CUDA, RTX_CUDA, DeviceClass.DISCRETE);
        List<BackendSelector.ApiView> views = Arrays.asList(
                view(BackendType.OPENCL, clRtx, clRtx),
                view(BackendType.CUDA, cuRtx, cuRtx),
                BackendSelector.ApiView.absent(BackendType.VULKAN, "Vulkan: no devices found"));

        BackendSelector.Selection s = BackendSelector.selectAuto(views, BackendSelector.DEFAULT_ORDER, null,
                b -> "rejected " + b.backendType());

        assertNull(s.backend);
        assertNull(s.chosen);
        assertEquals(Arrays.asList(
                "CUDA on '" + RTX_CUDA.name() + "': rejected CUDA",
                "OpenCL on '" + RTX_OPENCL.name() + "': rejected OPENCL",
                "Vulkan: no devices found"), s.reasons);
        assertTrue(fakeOf(BackendType.CUDA, RTX_CUDA.name()).closed);
        assertTrue(fakeOf(BackendType.OPENCL, RTX_OPENCL.name()).closed);
    }

    @Test
    public void apiMissingFromTheViewsIsReportedAsNotEnumerated() {
        BackendSelector.Selection s = BackendSelector.selectAuto(Collections.<BackendSelector.ApiView>emptyList(),
                Collections.singletonList(BackendType.CUDA), null, BackendRequirement.ANY);

        assertNull(s.backend);
        assertEquals(Collections.singletonList("CUDA: not enumerated"), s.reasons);
    }

    @Test
    public void creationFailureIsSkippedWithItsReason() {
        BackendSelector.Candidate vkBroken = new BackendSelector.Candidate(BackendType.VULKAN, RTX_VULKAN,
                DeviceClass.DISCRETE, () -> {
                    throw new IllegalStateException("vkCreateDevice failed: VK_ERROR_INITIALIZATION_FAILED");
                });
        BackendSelector.Candidate clRtx = candidate(BackendType.OPENCL, RTX_OPENCL, DeviceClass.DISCRETE);
        List<BackendSelector.ApiView> views = Arrays.asList(
                view(BackendType.VULKAN, vkBroken, vkBroken),
                view(BackendType.OPENCL, clRtx, clRtx));

        BackendSelector.Selection s = BackendSelector.selectAuto(views, BackendSelector.DEFAULT_ORDER, null,
                BackendRequirement.ANY);

        assertSame(clRtx, s.chosen);
        assertEquals(Collections.singletonList("Vulkan on '" + RTX_VULKAN.name()
                + "': backend creation failed: vkCreateDevice failed: VK_ERROR_INITIALIZATION_FAILED"), s.reasons);
    }

    @Test
    public void throwingRequirementCountsAsRejection() {
        BackendSelector.Candidate clRtx = candidate(BackendType.OPENCL, RTX_OPENCL, DeviceClass.DISCRETE);
        List<BackendSelector.ApiView> views = Collections.singletonList(view(BackendType.OPENCL, clRtx, clRtx));

        BackendSelector.Selection s = BackendSelector.selectAuto(views, BackendSelector.DEFAULT_ORDER, null, b -> {
            throw new IllegalStateException("index unreadable");
        });

        assertNull(s.backend);
        assertTrue(fakeOf(BackendType.OPENCL, RTX_OPENCL.name()).closed);
        assertEquals("OpenCL on '" + RTX_OPENCL.name() + "': requirement check failed: index unreadable",
                s.reasons.get(0));
    }

    @Test
    public void explicitModeRejectionClosesTheBackendAndThrows() {
        Fake fake = new Fake(BackendType.CUDA, RTX_CUDA.name(), closedLog);
        ComputeBackend backend = fake.create();
        try {
            BackendSelector.requireAccepted(BackendType.CUDA, backend, rejecting(BackendType.CUDA));
            fail("a rejected backend must not be returned");
        } catch (NoBackendAvailableException e) {
            assertEquals("CUDA backend unavailable: " + INVENTORY_INCOMPLETE, e.getMessage());
        }
        assertTrue(fake.closed);
    }

    @Test
    public void explicitModeAcceptanceReturnsTheBackend() {
        Fake fake = new Fake(BackendType.OPENCL, RTX_OPENCL.name(), closedLog);
        ComputeBackend backend = fake.create();

        assertSame(backend, BackendSelector.requireAccepted(BackendType.OPENCL, backend, BackendRequirement.ANY));
        assertFalse(fake.closed);
    }

    @Test
    public void orderParsing() {
        assertEquals(BackendSelector.DEFAULT_ORDER, BackendSelector.parseOrder(null));
        assertEquals(BackendSelector.DEFAULT_ORDER, BackendSelector.parseOrder("  "));
        assertEquals(Arrays.asList(BackendType.CUDA, BackendType.OPENCL), BackendSelector.parseOrder(" CUDA , opencl "));
        assertRejected("metal");
        assertRejected("cuda,Cuda");
        assertRejected(" , ");
    }

    private static void assertRejected(String spec) {
        try {
            BackendSelector.parseOrder(spec);
            fail("order '" + spec + "' must be rejected");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains(BackendSelector.ORDER_PROPERTY));
        }
    }

    @Test
    public void nameFilterNarrowsOrKeepsEverything() {
        List<String> names = Arrays.asList("Intel(R) Graphics", "NVIDIA GeForce RTX 5080 Laptop GPU");

        assertEquals(Collections.singletonList("NVIDIA GeForce RTX 5080 Laptop GPU"),
                BackendSelector.filterByName(names, n -> n, "rtx", "test"));
        assertEquals(names, BackendSelector.filterByName(names, n -> n, "radeon", "test"));
        assertEquals(names, BackendSelector.filterByName(names, n -> n, null, "test"));
    }

    @Test(expected = IllegalArgumentException.class)
    public void viewWinnerMustBeOneOfItsDevices() {
        BackendSelector.Candidate clRtx = candidate(BackendType.OPENCL, RTX_OPENCL, DeviceClass.DISCRETE);
        BackendSelector.Candidate clIgpu = candidate(BackendType.OPENCL, IGPU_OPENCL, DeviceClass.INTEGRATED);
        view(BackendType.OPENCL, clIgpu, clRtx);
    }

    @Test(expected = IllegalArgumentException.class)
    public void viewRejectsDevicesOfAnotherApi() {
        BackendSelector.Candidate cuRtx = candidate(BackendType.CUDA, RTX_CUDA, DeviceClass.DISCRETE);
        view(BackendType.OPENCL, cuRtx, cuRtx);
    }
}
