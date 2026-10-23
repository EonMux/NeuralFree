package com.eonmux.neuralfree.acceleration.backend.vulkan;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.backend.KernelSignature;
import com.eonmux.neuralfree.acceleration.backend.WorkSizes;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.Collections;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * A Vulkan kernel: a program from the backend's {@link VkProgramCache}, the specialisation
 * constants of its artifact, its build-time {@link KernelSignature} (null for legacy shaders) and
 * its own sticky {@link VkArgState}.
 *
 * <p><b>Variant selection</b>. A handle loaded from a subgroup module
 * ({@code <name>__sg.spv}, resolved only when the device has subgroup arithmetic) runs that
 * module only when {@code local_x % maxSubgroupSize == 0}, and then requests
 * {@code REQUIRE_FULL_SUBGROUPS} when {@code computeFullSubgroups} is enabled. Any other local size
 * runs the plain module {@code <name>.spv}, loaded on first need. Both modules must have the same
 * argument interface, because they share this handle's argument state.</p>
 *
 * <p><b>Local size.</b> {@link #resolveLocalSize} takes the caller's local size for the
 * specialisable dimensions ({@link WorkSizes#defaultLocal} when the caller passes none) and the
 * module's value for fixed dimensions, warning once per handle when a caller's request differs
 * from a fixed size.</p>
 *
 * <p><b>Dispatch sequence</b> for a queue:</p>
 * <ol>
 *   <li>{@code argState().setAll(args)} when the dispatch carries arguments (pre-bound dispatches skip it);</li>
 *   <li>{@code int[] local3 = resolveLocalSize(global, local)} and {@code Variant v = select(local3)};</li>
 *   <li>{@code argState().materialize()}, then the hazard check with {@code readIds()}/{@code writeIds()};</li>
 *   <li>bind {@code v.pipeline()}, then {@code v.program().bindDescriptors(cb, argState(), frame)} and
 *       {@code v.program().pushConstants(cb, argState())};</li>
 *   <li>dispatch {@code ceil(global[d] / local3[d])} groups.</li>
 * </ol>
 *
 * <p>The program belongs to the cache; {@link #close()} releases only per-handle state. A clone
 * shares the program and specialisation constants and has fresh arguments.</p>
 */
@NotThreadSafe("Arguments and the last-variant cache are per handle; callers serialise dispatches of a handle.")
public final class VkKernelHandle implements KernelHandle {

    private static final Logger LOGGER = LoggingUtil.getLogger(VkKernelHandle.class);

    /** The program, pipeline and flags one dispatch runs with. */
    public static final class Variant {
        private final VkProgram program;
        private final long pipeline;
        private final boolean subgroupVariant;
        private final boolean fullSubgroups;

        Variant(VkProgram program, long pipeline, boolean subgroupVariant, boolean fullSubgroups) {
            this.program = program;
            this.pipeline = pipeline;
            this.subgroupVariant = subgroupVariant;
            this.fullSubgroups = fullSubgroups;
        }

        public VkProgram program() {
            return program;
        }

        /** The {@code VkPipeline} to bind. */
        public long pipeline() {
            return pipeline;
        }

        /** True when the subgroup-reduction module runs. */
        public boolean subgroupVariant() {
            return subgroupVariant;
        }

        /** True when the pipeline was created with {@code REQUIRE_FULL_SUBGROUPS}. */
        public boolean fullSubgroups() {
            return fullSubgroups;
        }
    }

    private final VkProgramCache cache;
    private final VkProgram program;
    private final String name;
    private final SortedMap<Integer, Integer> specs;
    private final KernelSignature signature;
    private final boolean isClone;
    private final boolean subgroupModule;
    private final VkArgState argState;

    private VkProgram plainProgram;
    private int lastX;
    private int lastY;
    private int lastZ;
    private Variant lastVariant;
    private boolean warnedFixedLocal;
    private boolean closed;

    /**
     * @param cache     the backend's program cache, which also loads the plain fallback module
     * @param program   the program resolved for this kernel
     * @param specs     specialisation constants by SpecId (copied); null means none
     * @param signature build-time argument signature, or null for a legacy shader
     * @throws UnsupportedKernelException when the module and the signature disagree
     */
    public VkKernelHandle(VkProgramCache cache, VkProgram program, String name, SortedMap<Integer, Integer> specs,
                          KernelSignature signature) {
        this(cache, program, name, copy(specs), signature, false);
    }

    private VkKernelHandle(VkProgramCache cache, VkProgram program, String name, SortedMap<Integer, Integer> specs,
                           KernelSignature signature, boolean isClone) {
        if (cache == null || program == null) {
            throw new IllegalArgumentException("program cache and program must not be null");
        }
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("kernel name must not be empty");
        }
        this.cache = cache;
        this.program = program;
        this.name = name;
        this.specs = specs;
        this.signature = signature;
        this.isClone = isClone;
        this.subgroupModule = VkProgramCache.isSubgroupVariant(program.resourcePath());
        this.argState = new VkArgState(name, program.info(), signature);
    }

    /** A handle sharing this kernel's program and specialisation constants, with fresh arguments. */
    public VkKernelHandle cloneHandle() {
        requireOpen();
        return new VkKernelHandle(cache, program, name, specs, signature, true);
    }

    @Override
    public String kernelName() {
        return name;
    }

    /** The program resolved for this kernel (the subgroup module when one was resolved). */
    public VkProgram program() {
        return program;
    }

    /** Unmodifiable specialisation constants by SpecId. */
    public SortedMap<Integer, Integer> specs() {
        return specs;
    }

    /** Build-time signature, or null for a legacy shader. */
    public KernelSignature signature() {
        return signature;
    }

    public VkArgState argState() {
        requireOpen();
        return argState;
    }

    public boolean isClone() {
        return isClone;
    }

    public boolean isClosed() {
        return closed;
    }

    /**
     * The local size a dispatch runs with (three entries).
     *
     * @param global total work-items per dimension
     * @param local  requested work-group size per dimension, or null for the backend default
     * @throws IllegalArgumentException when the sizes are malformed or exceed the device limits
     */
    public int[] resolveLocalSize(long[] global, long[] local) {
        requireOpen();
        if (global == null || global.length < 1 || global.length > 3) {
            throw new IllegalArgumentException(name + ": global work size must have 1 to 3 dimensions");
        }
        if (local != null && local.length != global.length) {
            throw new IllegalArgumentException(name + ": local work size has " + local.length
                    + " dimensions, global has " + global.length);
        }
        VkPhysicalDeviceInfo device = cache.deviceInfo();
        long[] maxSize = device.maxComputeWorkGroupSize();
        long maxInvocations = device.maxComputeWorkGroupInvocations() & 0xFFFFFFFFL;
        SpirvModuleInfo info = program.info();
        long[] requested = local;
        if (requested == null && info.hasAnySpecLocalSize()) {
            requested = WorkSizes.defaultLocal(global, Math.min(maxInvocations, maxSize[0]));
        }
        long[] padded = requested == null ? null : WorkSizes.pad3(requested, 1);
        int[] local3 = new int[3];
        boolean overridden = false;
        for (int d = 0; d < 3; d++) {
            if (info.localSizeSpecId(d) >= 0) {
                long v = padded[d];
                if (v < 1 || v > maxSize[d]) {
                    throw new IllegalArgumentException(name + ": local size " + Arrays.toString(padded)
                            + " exceeds maxComputeWorkGroupSize " + Arrays.toString(maxSize) + " or is not positive");
                }
                local3[d] = (int) v;
            } else {
                local3[d] = info.localSize(d);
                overridden |= local != null && padded[d] != local3[d];
            }
        }
        long invocations = (long) local3[0] * local3[1] * local3[2];
        if (invocations > maxInvocations) {
            throw new IllegalArgumentException(name + ": local size " + Arrays.toString(local3) + " has "
                    + invocations + " invocations, above maxComputeWorkGroupInvocations " + maxInvocations);
        }
        if (overridden && !warnedFixedLocal) {
            warnedFixedLocal = true;
            LOGGER.warn("Vulkan kernel {}: the shader fixes local size {}; requested {} is ignored and groups are"
                    + " counted with the shader's size", name, Arrays.toString(local3), Arrays.toString(local));
        }
        return local3;
    }

    /**
     * The variant a dispatch with this local size runs: the subgroup module with its pipeline when
     * the local-size rule allows it, otherwise the plain module. The last answer is cached.
     *
     * @param local3 local size from {@link #resolveLocalSize}
     * @throws UnsupportedKernelException when the plain module is missing or its interface differs
     */
    public Variant select(int[] local3) {
        requireOpen();
        Variant last = lastVariant;
        if (last != null && local3[0] == lastX && local3[1] == lastY && local3[2] == lastZ) {
            return last;
        }
        Variant v;
        if (runsSubgroupModule(local3)) {
            boolean full = cache.fullSubgroupsEnabled();
            v = new Variant(program, program.pipeline(local3, specs, full), true, full);
        } else {
            VkProgram plain = subgroupModule ? plainProgram() : program;
            v = new Variant(plain, plain.pipeline(local3, specs, false), false, false);
        }
        return remember(local3, v);
    }

    /** Creates the pipeline a dispatch with this local size would use. */
    public Variant prepare(long[] global, long[] local) {
        return select(resolveLocalSize(global, local));
    }

    /**
     * The variant {@link #select} returns for this local size when its module is loaded and its
     * pipeline already exists, or null when {@link #select} would first have to load the plain module
     * or create the pipeline. Never loads a module or creates a pipeline; a found variant is cached
     * as {@link #select} caches it.
     *
     * @param local3 local size from {@link #resolveLocalSize}
     */
    public Variant preparedVariant(int[] local3) {
        requireOpen();
        Variant last = lastVariant;
        if (last != null && local3[0] == lastX && local3[1] == lastY && local3[2] == lastZ) {
            return last;
        }
        boolean subgroups = runsSubgroupModule(local3);
        VkProgram target = subgroups || !subgroupModule ? program : plainProgram;
        if (target == null) {
            return null;
        }
        boolean full = subgroups && cache.fullSubgroupsEnabled();
        Long pipeline = target.existingPipeline(local3, specs, full);
        return pipeline == null ? null : remember(local3, new Variant(target, pipeline, subgroups, full));
    }

    /** Caches {@code v} as the variant of local size {@code local3} and returns it. */
    private Variant remember(int[] local3, Variant v) {
        lastX = local3[0];
        lastY = local3[1];
        lastZ = local3[2];
        lastVariant = v;
        return v;
    }

    /** The local-size rule of the subgroup module: loaded, and {@code local_x % maxSubgroupSize == 0}. */
    private boolean runsSubgroupModule(int[] local3) {
        int maxSubgroup = cache.maxSubgroupSize();
        return subgroupModule && maxSubgroup > 0 && local3[0] % maxSubgroup == 0;
    }

    /** Releases per-handle state; the program stays in the cache. Idempotent. */
    @Override
    public void close() {
        closed = true;
        lastVariant = null;
    }

    @Override
    public String toString() {
        return "VkKernelHandle{" + name + ", " + program.resourcePath() + (isClone ? ", clone" : "")
                + (closed ? ", closed" : "") + '}';
    }

    private VkProgram plainProgram() {
        if (plainProgram == null) {
            VkProgram plain = cache.get(VkProgramCache.plainVariantOf(program.resourcePath()));
            if (!plain.info().sameInterface(program.info())) {
                throw new UnsupportedKernelException(name + ": " + plain.resourcePath() + " and "
                        + program.resourcePath() + " have different bindings, push constants or local-size ids");
            }
            plainProgram = plain;
        }
        return plainProgram;
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("kernel handle " + name + " is closed");
        }
    }

    private static SortedMap<Integer, Integer> copy(SortedMap<Integer, Integer> specs) {
        return specs == null || specs.isEmpty() ? Collections.<Integer, Integer>emptySortedMap()
                : Collections.unmodifiableSortedMap(new TreeMap<>(specs));
    }
}
