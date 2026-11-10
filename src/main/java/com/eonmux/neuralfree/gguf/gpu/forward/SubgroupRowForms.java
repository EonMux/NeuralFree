package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.kernels.OpenClKernelLibrary;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;

import java.util.Objects;

/**
 * Whether a backend's LLM kernels include the pinned 16-lane subgroup forms: the subgroup-row MMQ GEMM
 * ({@link SubgroupMmqGemm}), the subgroup-row matvecs ({@code {T}_MatVecSg_Q8} and twins) and the subgroup flash
 * attention ({@code LLM_FlashDecodeSg_*}, {@code LLM_FlashAttnSg_*}). The forms were written for the Intel Xe GPUs,
 * whose 16-lane subgroups hold a 32-token accumulator tile per lane in registers; each fixes its subgroup width, so a
 * device must run it at exactly {@link #SIZE} lanes.
 *
 * <ul>
 *   <li>OpenCL: the program is built with {@code NF_SG_ROW_SIZE} ({@link OpenClKernelLibrary#subgroupRowSize}: the
 *       device accepts a required 16-lane subgroup).</li>
 *   <li>Vulkan: an Intel device whose subgroup sizes include {@link #SIZE} and whose kernels use subgroups. The
 *       forms are the {@code sg16} variants rows ({@code shaders/llm/matmul_mmq_sg.comp} and siblings), whose
 *       pipelines pin 16-lane full subgroups; a device that cannot pin them for compute shaders refuses the module,
 *       so the per-kernel lookups ({@link LlmKernels#has}, {@link GpuOps#optionalReady}) still decide each form. The
 *       vendor key keeps the forms on the hardware they were tuned for: other devices whose range includes 16 lanes
 *       (some mobile GPUs) have different register files and were never measured.</li>
 * </ul>
 *
 * <p>On the Arrow-Lake iGPU under Vulkan (Mesa ANV), the forms took a Qwen3-Embedding 0.6B Q8_0 512-token prefill
 * from 34-42 to 1338 tokens/s (llama.cpp Vulkan: 1034) and a Qwen3-Embedding 4B Q4_K_M one from 25.6 to 195
 * (llama.cpp: 193); 4B Q4_K_M decode runs 19.5 tokens/s against 4.4 (llama.cpp: 16.1).</p>
 */
public final class SubgroupRowForms {

    /** The subgroup width every form is written for ({@code NF_SG_ROW_SIZE}). */
    public static final int SIZE = OpenClKernelLibrary.SUBGROUP_ROW_SIZE;

    private SubgroupRowForms() {
    }

    /** Whether {@code kernels}' backend includes the forms. */
    public static boolean served(LlmKernels kernels) {
        Objects.requireNonNull(kernels, "kernels must not be null");
        return served(kernels.backend().backendType(), kernels.caps(), kernels.subgroups());
    }

    /**
     * Whether a backend of this type on a device with these capabilities includes the forms.
     *
     * @param subgroups whether the backend's kernels use subgroups ({@link LlmKernels#subgroups()})
     */
    public static boolean served(BackendType backend, DeviceCapabilities caps, boolean subgroups) {
        Objects.requireNonNull(backend, "backend must not be null");
        Objects.requireNonNull(caps, "capabilities must not be null");
        switch (backend) {
            case OPENCL:
                return OpenClKernelLibrary.subgroupRowSize(caps, subgroups) > 0;
            case VULKAN:
                return subgroups && DeviceKind.of(backend, caps).vendor() == DeviceKind.Vendor.INTEL
                        && caps.getMinSubgroupSize() <= SIZE && SIZE <= caps.getMaxSubgroupSize();
            default:
                return false;
        }
    }
}
