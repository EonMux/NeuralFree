package com.eonmux.neuralfree.acceleration.backend.spi;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;

import java.util.List;

/**
 * Service-provider interface for an accelerator API the engine does not drive natively: SYCL/Level
 * Zero, Intel XPU, Huawei CANN, Qualcomm Hexagon, Intel OpenVINO, WebGPU, Moore Threads MUSA, IBM
 * zDNN or AMD ZenDNN.
 *
 * <p>A provider is a jar on the application class path that declares an implementation of this
 * interface in {@code META-INF/services/com.eonmux.neuralfree.acceleration.backend.spi.AcceleratorBackendProvider}.
 * The engine finds the installed providers through {@link java.util.ServiceLoader} once per process
 * ({@link AcceleratorProviders#installed()}), enumerates their devices when it builds an
 * {@code HWAcceleration}, and offers those devices to AUTO and to the explicit backend modes, as
 * llama.cpp loads an out-of-tree {@code ggml-*} backend from {@code GGML_BACKEND_PATH} and vLLM
 * loads a platform from its {@code vllm.platform_plugins} entry points.</p>
 *
 * <p>One provider serves exactly one {@link BackendType}; two providers for the same API are
 * ambiguous and that API reports no devices, naming both providers. A provider's
 * {@link #devices()} is called once per {@code HWAcceleration} instance and must be safe to call
 * repeatedly: enumeration creates no backend, and each device opens its own backend on demand.
 * A provider whose backend runs the engine's kernels also registers a
 * {@link com.eonmux.neuralfree.acceleration.kernels.KernelLibrary} for its API in
 * {@link com.eonmux.neuralfree.acceleration.kernels.KernelLibraryRegistry#global()}; without one
 * the engine's kernel requirement rejects the backend, exactly as it does for a built-in API whose
 * library is missing. Providers are trusted code the operator installed; nothing is read from a
 * model directory.</p>
 */
public interface AcceleratorBackendProvider {

    /** The accelerator API this provider serves; must be an accelerator type ({@link BackendType#isAccelerator()}). */
    BackendType backendType();

    /**
     * The devices of this API, in enumeration order. The engine ranks them itself
     * ({@link AcceleratorDeviceRanking}) and never assumes the first is the fastest.
     *
     * @return the devices; empty when the vendor runtime is installed but reports none
     */
    List<AcceleratorDevice> devices();

    /** The name reported in logs and error messages; the implementation class name by default. */
    default String providerName() {
        return getClass().getName();
    }
}
