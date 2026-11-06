package com.eonmux.neuralfree.gguf.hybrid;

import com.eonmux.neuralfree.acceleration.GpuOutOfMemoryException;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightPlacement;
import com.eonmux.neuralfree.gguf.hparams.MropeTextPositions;
import com.eonmux.neuralfree.gguf.memory.ContextFitter;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

import java.util.Locale;
import java.util.function.IntPredicate;

/**
 * Device-side byte accounting and context fit of a hybrid CPU+GPU load: the context bytes the fit charges are the
 * plan bytes of the offloaded layers plus the per-step inputs, not the whole plan. The position and step buffers of
 * every KV group are allocated on the device ({@code DeviceModelMemory} addresses all groups of the shared plan), so
 * the sizing mirrors that allocation. Stateless.
 */
public final class HybridFit {

    private HybridFit() {
    }

    /**
     * The context bytes of the layers {@code deviceLayers} names over {@code plan}: their caches, recurrent state,
     * side caches and (when the plan reserved them) their recurrent-state snapshots, plus the per-step position and
     * identity buffers over every group of the plan.
     */
    public static long contextBytes(MemoryPlan plan, IntPredicate deviceLayers) {
        long bytes = 0;
        for (LayerMemoryPlan lp : plan.layers()) {
            if (!deviceLayers.test(lp.layer())) {
                continue;
            }
            bytes = Math.addExact(bytes, lp.kvBytes() + lp.stateBytes() + lp.sideCacheBytes());
        }
        if (plan.snapshotsReserved()) {
            for (LayerMemoryPlan lp : plan.layers()) {
                if (deviceLayers.test(lp.layer())) {
                    bytes = Math.addExact(bytes, lp.stateBytes());
                }
            }
        }
        return Math.addExact(bytes, stepInputBytes(plan));
    }

    /**
     * The {@link ContextFitter.PlanBytes} sizer of the device layers of {@code offload} over a plan: trunk layers
     * from {@link LayerOffload#firstGpuLayer()} on.
     */
    public static ContextFitter.PlanBytes contextSizer(LayerOffload offload) {
        return plan -> contextBytes(plan, deviceLayers(offload));
    }

    /**
     * The trunk layers whose caches the device memory allocates: the device layers of the offload, except those
     * whose KV cache is kept on the host ({@link LayerOffload#deviceLayer(int)}). The plan holds no NextN layer
     * caches.
     */
    public static IntPredicate deviceLayers(LayerOffload offload) {
        int first = offload.firstGpuLayer();
        int end = offload.nGpuLayers() + first;
        return il -> il >= first && il < end && offload.deviceLayer(il);
    }

    /**
     * Bytes of the per-step inputs {@code DeviceStepInputs} allocates over {@code plan}: per group the position
     * rows and the step cell indexes, plus the query positions, the RoPE sections, the recurrent-state row counter
     * and the identity rows.
     */
    public static long stepInputBytes(MemoryPlan plan) {
        int ubatch = plan.ubatch();
        long bytes = 0;
        for (MemoryPlan.KvGroup g : plan.groups()) {
            bytes = Math.addExact(bytes, (long) g.rows() * Integer.BYTES + (long) ubatch * Integer.BYTES);
        }
        bytes = Math.addExact(bytes, (long) ubatch * Integer.BYTES);
        bytes = Math.addExact(bytes, (long) MropeTextPositions.SECTIONS * ubatch * Integer.BYTES);
        bytes = Math.addExact(bytes, Integer.BYTES);
        bytes = Math.addExact(bytes, (long) ubatch * Integer.BYTES);
        return bytes;
    }

    /** The fitted split of one load: the context, the placement to upload with, and the split itself. */
    public static final class Fit {
        /** Context the fit chose. */
        public final int nCtx;
        /** The weight placement of the split (host tensors kept off the device). */
        public final WeightPlacement placement;
        /** The split that was fitted. */
        public final LayerOffload split;

        Fit(int nCtx, WeightPlacement placement, LayerOffload split) {
            this.nCtx = nCtx;
            this.placement = placement;
            this.split = split;
        }
    }

    /**
     * Fits a context beside the device weights of one candidate split, charging only the offloaded layers' plan
     * bytes to the fit ({@link #contextSizer}). Null when the split cannot serve and {@code required} is false; an
     * unfit required split fails with {@link GpuOutOfMemoryException} or
     * {@link ContextFitter.ContextBudgetException}.
     *
     * @param loadNextn whether the load places the model's NextN / MTP blocks
     * @param uploader  the weight uploader, whose plans size the device weights
     * @param usable    usable VRAM of the load
     * @param drafter   the memory a configured drafter reserves beside the context
     */
    public static Fit placeHybrid(ModelSpec initial, MemoryConfig mem,
                                  LayerOffload candidate, boolean loadNextn, GpuWeightUploader uploader, long usable,
                                  ContextFitter.Companion drafter, boolean required) {
        return placeHybrid(initial, mem, candidate, WeightPlacement.of(initial, loadNextn), uploader, usable,
                drafter, required);
    }

    /**
     * As {@link #placeHybrid(ModelSpec, MemoryConfig, LayerOffload, boolean, GpuWeightUploader, long,
     * ContextFitter.Companion, boolean)}, with {@code base} giving the placement the split adds its host tensors
     * to: the placement rules of the load stay in force beside the split.
     */
    public static Fit placeHybrid(ModelSpec initial, MemoryConfig mem,
                                  LayerOffload candidate, WeightPlacement base, GpuWeightUploader uploader, long usable,
                                  ContextFitter.Companion drafter, boolean required) {
        WeightPlacement placed = base.withHostTensors(candidate::hostTensor);
        long weightBytes = uploader.plannedBytes(placed::onDevice);
        if (usable - weightBytes < 0) {
            if (required) {
                throw new GpuOutOfMemoryException(initial.arch().ggufName() + ": the " + mib(weightBytes)
                        + " MiB of device weights of the requested layer split (" + candidate + ") do not fit the "
                        + mib(usable) + " MiB of usable VRAM; offload fewer layers", weightBytes, usable, "gpu");
            }
            return null;
        }
        try {
            int nCtx = ContextFitter.fit(initial, mem, usable - weightBytes, drafter, contextSizer(candidate));
            return new Fit(nCtx, placed, candidate);
        } catch (ContextFitter.ContextBudgetException e) {
            if (required) {
                throw e;
            }
            return null;
        }
    }

    private static String mib(long bytes) {
        return String.format(Locale.ROOT, "%.1f", bytes / (1024.0 * 1024.0));
    }
}
