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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * AUTO backend selection with a Metal view: Metal is the first choice on the device it shares with
 * the Vulkan view (one Apple GPU), a requirement that rejects Metal falls back to Vulkan and closes
 * the rejected backend, and the API name parses and prints as Metal.
 */
public class MetalBackendSelectionTest {

    private static final DeviceIdentity APPLE_GPU = DeviceIdentity.ofName("Apple M3 Pro");

    @Test
    public void metalComesBeforeVulkanOnTheSameDevice() {
        List<String> closed = new ArrayList<>();
        ComputeBackend metal = backend(BackendType.METAL, closed);
        ComputeBackend vulkan = backend(BackendType.VULKAN, closed);

        List<BackendSelector.ApiView> views = Arrays.asList(
                view(BackendType.METAL, APPLE_GPU, metal),
                view(BackendType.VULKAN, APPLE_GPU, vulkan));

        BackendSelector.Selection selection = BackendSelector.selectAuto(views, BackendSelector.DEFAULT_ORDER,
                null, BackendRequirement.ANY);

        assertNotNull(selection.backend);
        assertSame(metal, selection.backend);
        assertEquals(BackendType.METAL, selection.chosen.api);
        assertTrue("the Vulkan view of the same GPU was never tried", closed.isEmpty());
    }

    @Test
    public void aRejectedMetalBackendFallsBackToVulkan() {
        List<String> closed = new ArrayList<>();
        ComputeBackend metal = backend(BackendType.METAL, closed);
        ComputeBackend vulkan = backend(BackendType.VULKAN, closed);

        List<BackendSelector.ApiView> views = Arrays.asList(
                view(BackendType.METAL, APPLE_GPU, metal),
                view(BackendType.VULKAN, APPLE_GPU, vulkan));

        BackendSelector.Selection selection = BackendSelector.selectAuto(views, BackendSelector.DEFAULT_ORDER,
                null, backend -> backend.backendType() == BackendType.METAL ? "the Metal kernels are unavailable"
                        : null);

        assertSame(vulkan, selection.backend);
        assertEquals(1, closed.size());
        assertTrue(selection.reasons.get(0).contains("the Metal kernels are unavailable"));
    }

    @Test
    public void theApiNameParsesAndPrints() {
        assertEquals("Metal", BackendSelector.label(BackendType.METAL));
        assertEquals(Collections.singletonList(BackendType.METAL), BackendSelector.parseOrder(" METAL "));
        assertEquals(BackendType.METAL, BackendSelector.DEFAULT_ORDER.get(0));
        assertEquals("the built-in APIs end with HIP", BackendType.HIP, BackendSelector.DEFAULT_ORDER.get(4));
        assertEquals("the accelerator APIs follow the built-in ones", BackendType.SYCL,
                BackendSelector.DEFAULT_ORDER.get(5));
    }

    private static BackendSelector.ApiView view(BackendType api, DeviceIdentity identity, ComputeBackend backend) {
        BackendSelector.Candidate candidate = new BackendSelector.Candidate(api, identity, DeviceClass.INTEGRATED,
                () -> backend);
        return new BackendSelector.ApiView(api, Collections.singletonList(candidate), candidate, null);
    }

    /** A backend of one API that records its close, as the other selector tests use. */
    private static ComputeBackend backend(BackendType api, List<String> closed) {
        return (ComputeBackend) Proxy.newProxyInstance(ComputeBackend.class.getClassLoader(),
                new Class<?>[]{ComputeBackend.class}, (proxy, method, args) -> {
                    if ("backendType".equals(method.getName())) {
                        return api;
                    }
                    if ("close".equals(method.getName())) {
                        closed.add(api.name());
                        return null;
                    }
                    if ("toString".equals(method.getName())) {
                        return api.name() + " stub";
                    }
                    if ("hashCode".equals(method.getName())) {
                        return System.identityHashCode(proxy);
                    }
                    if ("equals".equals(method.getName())) {
                        return proxy == args[0];
                    }
                    throw new UnsupportedOperationException(method.getName());
                });
    }
}
