package com.eonmux.neuralfree.gguf.kernels;

import com.eonmux.neuralfree.acceleration.BackendRequirement;
import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.KernelHandle;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraries;
import com.eonmux.neuralfree.acceleration.kernels.KernelLibraryRequirement;
import com.eonmux.neuralfree.acceleration.kernels.KernelSet;

import java.util.Collection;
import java.util.Objects;
import java.util.Set;

/**
 * The LLM inference kernels on one backend: a thin wrapper over
 * {@code KernelSet.open(backend, KernelLibraries.LLM)} that the GGUF engine uses for every kernel lookup.
 *
 * <p>Kernel names are identical on OpenCL, CUDA and Vulkan. Handles are owned by the backend.</p>
 */
public final class LlmKernels {

    /** PCI vendor id of NVIDIA devices. */
    private static final int PCI_VENDOR_NVIDIA = 0x10DE;

    private final KernelSet set;

    private LlmKernels(KernelSet set) {
        this.set = set;
    }

    /**
     * Opens the LLM kernels on a backend.
     *
     * @throws UnsupportedKernelException with the reason when the LLM kernels are unavailable on it
     */
    public static LlmKernels open(ComputeBackend backend) {
        Objects.requireNonNull(backend, "backend must not be null");
        return new LlmKernels(KernelSet.open(backend, KernelLibraries.LLM));
    }

    /** Whether the LLM kernels can be opened on a backend, with the reason when they cannot. */
    public static Availability availability(ComputeBackend backend) {
        return KernelSet.availability(backend, KernelLibraries.LLM);
    }

    /** Backend-selection requirement that the LLM kernels are available. */
    public static BackendRequirement requirement() {
        return KernelLibraryRequirement.of(KernelLibraries.LLM);
    }

    /**
     * The handle of a required kernel.
     *
     * @throws UnsupportedKernelException naming the kernel, the backend and the reason
     */
    public KernelHandle get(String name) {
        return set.get(name);
    }

    /** The handle of an optional kernel, or null when it is unavailable (logged once per name). */
    public KernelHandle tryGet(String name) {
        return set.tryGet(name);
    }

    /** Whether the kernel can be provided on this backend. */
    public boolean has(String name) {
        return set.has(name);
    }

    /** Loads every named kernel now; throws for the first that cannot be provided. */
    public void prewarm(Collection<String> names) {
        set.prewarm(names);
    }

    /** Loads a kernel and prepares it for dispatches with this work-group size. */
    public void prewarm(String name, long[] localWorkSize) {
        set.prewarm(name, localWorkSize);
    }

    /** {@code NF_FA_SMEM_FLOATS} of the flash kernels on this backend. */
    public int faSmemFloats() {
        return set.faSmemFloats();
    }

    /** Whether the kernels use subgroup reductions on this backend. */
    public boolean subgroups() {
        return set.subgroups();
    }

    /**
     * Whether the tensor-core prefill attention ({@code LLM_FlashAttnMma_F16} and its paged twin) can run on this
     * backend's device: the CUDA mma.sync form (kernels/llm/flash_mma.cl) needs {@link DeviceFeature#COOPERATIVE_MATRIX},
     * the Vulkan form (shaders/llm/flash_mma_cm2.comp) the workgroup-scope cooperative matrices of
     * {@link DeviceFeature#COOPERATIVE_MATRIX_2_ATTENTION}. OpenCL, HIP and Metal build no mma attention, so the
     * planner also checks that the kernel exists.
     */
    public boolean flashMmaPath() {
        DeviceFeature needed = set.backend().backendType() == ComputeBackend.BackendType.VULKAN
                ? DeviceFeature.COOPERATIVE_MATRIX_2_ATTENTION : DeviceFeature.COOPERATIVE_MATRIX;
        return set.caps().hasFeature(needed);
    }

    /**
     * Whether the flash-decode kernels run their warp fast path on this backend (flash_decode.cl and
     * flash_decode.comp {@code nf_fd_fast_ok}), which the decode planner serves with splits of at most one
     * workgroup of keys: CUDA and HIP compile it over the subgroup shuffle ({@link DeviceFeature#SUBGROUP_SHUFFLE});
     * the Vulkan {@code __sg} modules run it where every subgroup is 32 lanes wide, the warp the fast path reduces
     * over, so a device with subgroup reductions and a fixed 32-lane subgroup size.
     */
    public boolean decodeWarpPath() {
        return warpReductions();
    }

    /**
     * Whether kernels reduce over 32-lane warps in registers on this backend: the subgroup shuffle (CUDA, HIP), or
     * the Vulkan {@code __sg} modules on a device whose every subgroup is 32 lanes wide. The flash-decode fast path
     * ({@link #decodeWarpPath}) and the Q8_1 blocks of the row norms (norm_ext.cl / norm_ext.comp
     * {@code nf_norm_ext_q8}) take their register forms there; elsewhere they reduce through local memory.
     */
    public boolean warpReductions() {
        DeviceCapabilities caps = set.caps();
        if (caps.hasFeature(DeviceFeature.SUBGROUP_SHUFFLE)) {
            return true;
        }
        return set.backend().backendType() == ComputeBackend.BackendType.VULKAN && set.subgroups()
                && caps.getMinSubgroupSize() == 32 && caps.getMaxSubgroupSize() == 32;
    }

    /**
     * Whether the one-row flash decode ({@code LLM_FlashDecodeRow_F16}) may serve the plain decode step on this
     * backend's device. Off Vulkan it serves wherever the build defines it (the CUDA fatbins; the OpenCL, HIP and
     * Metal builds define none, so the planner also checks that the kernel exists). On Vulkan it serves NVIDIA devices
     * only, the device kind it was measured on: other vendors' subgroup sizes run its shared-memory reduction or
     * other lane counts, whose speed against the split flash decode no measurement covers.
     */
    public boolean flashDecodeRowPath() {
        if (set.backend().backendType() != ComputeBackend.BackendType.VULKAN) {
            return true;
        }
        DeviceCapabilities caps = set.caps();
        return caps.getIdentity() != null && caps.getIdentity().vendorId() == PCI_VENDOR_NVIDIA;
    }

    /** Preferred work-group size of matvec kernels on this backend. */
    public int preferredMatvecLocal() {
        return set.preferredMatvecLocal();
    }

    /** Work-group size of the matvec kernels: {@code min(LLM_DIMX, preferredMatvecLocal())}. */
    public int matvecLocal() {
        return Math.min(LlmKernelConstants.LLM_DIMX, set.preferredMatvecLocal());
    }

    /** Work-group size of the row-norm kernels: {@code LLM_DIMX} ({@code min(LLM_DIMX, rowReduceLocal())}). */
    public int rowReduceLocal() {
        return Math.min(LlmKernelConstants.LLM_DIMX, set.rowReduceLocal());
    }

    public ComputeBackend backend() {
        return set.backend();
    }

    public DeviceCapabilities caps() {
        return set.caps();
    }

    /** Kernels the LLM library lists for this backend type. */
    public Set<String> kernelNames() {
        return set.kernelNames();
    }

    @Override
    public String toString() {
        return "LlmKernels{" + set + "}";
    }
}
