package com.eonmux.neuralfree.acceleration.backend;

import java.util.Collections;
import java.util.Objects;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.function.Supplier;

/**
 * Immutable description of where a backend loads one kernel from, produced by a
 * kernel library and consumed by {@link ComputeBackend#loadKernel(KernelArtifact)}.
 *
 * <ul>
 *   <li>{@link Kind#OPENCL_SOURCE}: OpenCL C source (supplied lazily and memoised)
 *       plus build options. Kernels of one program share a {@link #programKey()},
 *       which the backend uses as its program-cache key, so the source is built
 *       once per program.</li>
 *   <li>{@link Kind#CUDA_IMAGE}: a classpath fatbin, cubin or PTX resource.</li>
 *   <li>{@link Kind#SPIRV}: a classpath SPIR-V module with specialisation
 *       constants and the build-time {@link KernelSignature}.</li>
 * </ul>
 *
 * <p>For the binary kinds the program key is the resource path.</p>
 */
public final class KernelArtifact {

    /** The artifact format, which decides which backends can load it. */
    public enum Kind {
        OPENCL_SOURCE,
        CUDA_IMAGE,
        SPIRV
    }

    private final Kind kind;
    private final String kernelName;
    private final String programKey;
    private final Supplier<String> source;
    private final String buildOptions;
    private final String resourcePath;
    private final SortedMap<Integer, Integer> specConstants;
    private final KernelSignature signature;

    private KernelArtifact(Kind kind, String kernelName, String programKey, Supplier<String> source,
                           String buildOptions, String resourcePath, SortedMap<Integer, Integer> specConstants,
                           KernelSignature signature) {
        this.kind = kind;
        this.kernelName = requireText(kernelName, "kernelName");
        this.programKey = requireText(programKey, "programKey");
        this.source = source;
        this.buildOptions = buildOptions;
        this.resourcePath = resourcePath;
        this.specConstants = specConstants;
        this.signature = signature;
    }

    /**
     * An OpenCL kernel built from source.
     *
     * @param kernelName kernel function name
     * @param programKey unique id of the program (source plus options) this kernel belongs to
     * @param source     supplier of the full program source; evaluated at most once
     * @param options    build options; null means none
     */
    public static KernelArtifact openClSource(String kernelName, String programKey, Supplier<String> source,
                                              String options) {
        if (source == null) {
            throw new IllegalArgumentException("OpenCL source supplier for " + kernelName + " must not be null");
        }
        return new KernelArtifact(Kind.OPENCL_SOURCE, kernelName, programKey, new Memoized(source, kernelName),
                options == null ? "" : options, null, emptySpecs(), null);
    }

    /**
     * A CUDA kernel in a prebuilt module image on the classpath.
     *
     * @param resourcePath classpath path of the {@code .fatbin}, {@code .cubin} or {@code .ptx} file
     */
    public static KernelArtifact cudaImage(String kernelName, String resourcePath) {
        requireText(resourcePath, "resourcePath");
        return new KernelArtifact(Kind.CUDA_IMAGE, kernelName, resourcePath, null, "", resourcePath,
                emptySpecs(), null);
    }

    /**
     * A Vulkan kernel in a SPIR-V module on the classpath.
     *
     * @param resourcePath  classpath path of the {@code .spv} file
     * @param specConstants specialisation constant values by SpecId (copied); null means none
     * @param signature     build-time argument signature of the kernel; required
     */
    public static KernelArtifact spirv(String kernelName, String resourcePath,
                                       SortedMap<Integer, Integer> specConstants, KernelSignature signature) {
        requireText(resourcePath, "resourcePath");
        if (signature == null) {
            throw new IllegalArgumentException("SPIR-V artifact " + kernelName + " requires a kernel signature");
        }
        SortedMap<Integer, Integer> specs = new TreeMap<>();
        if (specConstants != null) {
            for (java.util.Map.Entry<Integer, Integer> e : specConstants.entrySet()) {
                if (e.getKey() == null || e.getValue() == null || e.getKey() < 0) {
                    throw new IllegalArgumentException("invalid specialisation constant " + e.getKey() + "="
                            + e.getValue() + " for " + kernelName);
                }
                specs.put(e.getKey(), e.getValue());
            }
        }
        return new KernelArtifact(Kind.SPIRV, kernelName, resourcePath, null, "", resourcePath,
                Collections.unmodifiableSortedMap(specs), signature);
    }

    public Kind kind() {
        return kind;
    }

    public String kernelName() {
        return kernelName;
    }

    /** Cache key of the program, module or SPIR-V file this kernel belongs to. */
    public String programKey() {
        return programKey;
    }

    /**
     * The OpenCL program source, computed on first call and memoised.
     *
     * @throws IllegalStateException when this is not an {@link Kind#OPENCL_SOURCE} artifact
     */
    public String sourceCode() {
        if (kind != Kind.OPENCL_SOURCE) {
            throw new IllegalStateException(kind + " artifact " + kernelName + " has no source code");
        }
        return source.get();
    }

    /** OpenCL build options; empty for the binary kinds. */
    public String buildOptions() {
        return buildOptions;
    }

    /** Classpath resource of a binary artifact, or null for OpenCL source. */
    public String resourcePath() {
        return resourcePath;
    }

    /** Unmodifiable specialisation constants by SpecId; empty unless SPIR-V. */
    public SortedMap<Integer, Integer> specConstants() {
        return specConstants;
    }

    /** Build-time argument signature; non-null for SPIR-V, null otherwise. */
    public KernelSignature signature() {
        return signature;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof KernelArtifact)) {
            return false;
        }
        KernelArtifact that = (KernelArtifact) o;
        return kind == that.kind
                && kernelName.equals(that.kernelName)
                && programKey.equals(that.programKey)
                && Objects.equals(resourcePath, that.resourcePath)
                && specConstants.equals(that.specConstants);
    }

    @Override
    public int hashCode() {
        return Objects.hash(kind, kernelName, programKey, resourcePath, specConstants);
    }

    @Override
    public String toString() {
        return "KernelArtifact{" + kind + " " + kernelName + " @ " + programKey
                + (specConstants.isEmpty() ? "" : " spec=" + specConstants) + "}";
    }

    private static SortedMap<Integer, Integer> emptySpecs() {
        return Collections.unmodifiableSortedMap(new TreeMap<Integer, Integer>());
    }

    private static String requireText(String value, String what) {
        if (value == null || value.isEmpty()) {
            throw new IllegalArgumentException(what + " must not be null or empty");
        }
        return value;
    }

    /** Thread-safe memoising wrapper; a failed or null supply is reported and retried on the next call. */
    private static final class Memoized implements Supplier<String> {
        private final Supplier<String> delegate;
        private final String kernelName;
        private volatile String value;

        Memoized(Supplier<String> delegate, String kernelName) {
            this.delegate = delegate;
            this.kernelName = kernelName;
        }

        @Override
        public String get() {
            String v = value;
            if (v != null) {
                return v;
            }
            synchronized (this) {
                if (value == null) {
                    String supplied = delegate.get();
                    if (supplied == null) {
                        throw new IllegalStateException("OpenCL source supplier for " + kernelName
                                + " returned null");
                    }
                    value = supplied;
                }
                return value;
            }
        }
    }
}
