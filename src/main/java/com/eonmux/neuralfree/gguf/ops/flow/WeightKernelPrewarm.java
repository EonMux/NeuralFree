package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader.TensorPlan;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Prepares the weight kernels of the device types an uploader chose, so no kernel is built or (on Vulkan) no
 * pipeline is created during the first forward step that reads the weights. Projected tensors take every product
 * family of their type and embedding tables every lookup family, each prepared at the work-group size
 * {@link GpuOps} launches it with ({@link GpuOps#prewarmWeightKernels}). Every 3-D tensor (experts, and the per-head
 * products of MLA absorption and DSV4 {@code wo_a}) takes {@code {T}_MatVecId}; experts also take the fused gate-up
 * and batched id products. A model prepares the plans of its own weights; a drafter with a weight file of its own
 * (eagle3, gemma4-assistant, an MTP draft file) prepares the plans of the weights it places on the device, whose types
 * the target may not use. Stateless.
 */
public final class WeightKernelPrewarm {

    private WeightKernelPrewarm() {
    }

    /**
     * Prepares the weight kernels of {@code plans} on the runtime of {@code ops} and {@code kernels}.
     *
     * @param plans   the upload plans of the tensors the forward path reads on the device
     * @param ops     the dispatch helpers the weights are launched through
     * @param kernels the kernel library of the same backend
     * @throws com.eonmux.neuralfree.acceleration.UnsupportedKernelException when a kernel of a weight type cannot be
     *                                                                       prepared on the device
     */
    public static void prewarm(Iterable<TensorPlan> plans, GpuOps ops, LlmKernels kernels) {
        Objects.requireNonNull(plans, "plans");
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(kernels, "kernels");
        Set<GGMLType> projected = EnumSet.noneOf(GGMLType.class);
        Set<GGMLType> lookups = EnumSet.noneOf(GGMLType.class);
        Set<String> idKernels = new LinkedHashSet<>();
        for (TensorPlan p : plans) {
            collect(p, projected, lookups, idKernels);
        }
        Set<GGMLType> types = EnumSet.noneOf(GGMLType.class);
        types.addAll(projected);
        types.addAll(lookups);
        for (GGMLType t : types) {
            if (GpuOps.supportsWeightType(t)) {
                ops.prewarmWeightKernels(t, projected.contains(t), lookups.contains(t));
            }
        }
        List<String> present = new ArrayList<>(idKernels.size());
        for (String n : idKernels) {
            if (kernels.has(n)) {
                present.add(n);
            }
        }
        kernels.prewarm(present);
    }

    private static void collect(TensorPlan p, Set<GGMLType> projected, Set<GGMLType> lookups, Set<String> idKernels) {
        GGMLType t = p.deviceType();
        if (p.ne().length == 3 && MoeKernels.supports(t)) {
            idKernels.add(MoeKernels.matVecId(t));
        }
        switch (p.use()) {
            case MATMUL:
                projected.add(t);
                break;
            case LOOKUP:
            case ADDED_LOOKUP:
                lookups.add(t);
                break;
            case LOOKUP_MATMUL:
                lookups.add(t);
                projected.add(t);
                break;
            case EXPERT:
                if (MoeKernels.supports(t)) {
                    idKernels.add(MoeKernels.matVecId(t));
                    idKernels.add(MoeKernels.matVecIdGateUp(t));
                    idKernels.add(MoeKernels.batchMatMulId(t));
                }
                break;
            default:
                break;
        }
    }
}
