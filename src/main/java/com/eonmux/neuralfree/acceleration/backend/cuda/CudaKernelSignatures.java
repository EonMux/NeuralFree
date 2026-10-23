package com.eonmux.neuralfree.acceleration.backend.cuda;

import com.eonmux.neuralfree.acceleration.ThreadSafe;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Build-time argument signatures of the kernels in prebuilt CUDA modules, keyed by the module's
 * classpath resource and the kernel name.
 *
 * <p>CUDA modules carry no parameter metadata the driver API of this binding can read (there is no
 * {@code cuFuncGetParamInfo}), so the kernel library that ships a module registers the signature its
 * kernel inventory records before the backend loads the kernel. {@link CudaComputeBackend#loadKernel}
 * attaches the registered signature to the handle, and every launch of that handle is then checked
 * against it ({@link CudaArgState}). Kernels without a registered signature (legacy PTX, images
 * without an inventory) launch unchecked.</p>
 *
 * <p>Signatures describe the classpath content, which does not change while the process runs, so
 * one registration per (resource, kernel) holds for every backend instance.</p>
 */
@ThreadSafe("Backed by a concurrent map; registrations are idempotent.")
public final class CudaKernelSignatures {

    private static final Map<String, KernelSignature> SIGNATURES = new ConcurrentHashMap<>();

    private CudaKernelSignatures() {
    }

    /**
     * Records the signature of {@code kernelName} in the module at {@code resourcePath}. Registering
     * an equal signature again does nothing.
     *
     * @throws IllegalArgumentException when an argument is null or empty
     * @throws IllegalStateException    when a different signature is already registered for the kernel,
     *                                  which means two builds disagree about one module
     */
    public static void register(String resourcePath, String kernelName, KernelSignature signature) {
        Objects.requireNonNull(signature, "signature must not be null");
        String key = key(resourcePath, kernelName);
        KernelSignature previous = SIGNATURES.putIfAbsent(key, signature);
        if (previous != null && !previous.equals(signature)) {
            throw new IllegalStateException("CUDA kernel " + kernelName + " in classpath:" + resourcePath
                    + " is registered with " + previous + " and again with " + signature
                    + "; the kernel inventories on the classpath come from different builds");
        }
    }

    /** The registered signature of {@code kernelName} in the module at {@code resourcePath}, or null. */
    public static KernelSignature lookup(String resourcePath, String kernelName) {
        return SIGNATURES.get(key(resourcePath, kernelName));
    }

    private static String key(String resourcePath, String kernelName) {
        if (resourcePath == null || resourcePath.trim().isEmpty()) {
            throw new IllegalArgumentException("module resource path must not be null or empty");
        }
        if (kernelName == null || kernelName.isEmpty()) {
            throw new IllegalArgumentException("kernel name must not be null or empty");
        }
        String path = resourcePath.trim();
        while (path.startsWith("/")) {
            path = path.substring(1);
        }
        return path + '#' + kernelName;
    }
}
