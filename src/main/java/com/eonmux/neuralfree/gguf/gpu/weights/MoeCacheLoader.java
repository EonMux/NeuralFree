package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.acceleration.GpuModelUnsupportedException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.WeightDataProvider;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.ops.moe.MoeCacheRanking;
import com.eonmux.neuralfree.gguf.ops.moe.MoeKernels;
import com.eonmux.neuralfree.gguf.ops.moe.MoeSplitDispatcher;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Assembles the MoE expert cache of a GPU load: the trunk layers with routed experts it would serve
 * ({@link #routedMoeLayers}) and the cache itself over the VRAM left after loading ({@link #open}). The routed
 * expert stacks the cache serves are hosted by the one placement path ({@code neuralfree.moe.cache} and
 * {@code neuralfree.moe.cache.mb} resolve to a {@code neuralfree.cpu.moe} rule in {@code HostPlacement}).
 */
public final class MoeCacheLoader {

    private MoeCacheLoader() {
    }

    /**
     * The trunk layers whose routed experts the MoE expert cache would serve: every decoder layer of the routed
     * MoE kinds ({@link com.eonmux.neuralfree.gguf.arch.FfnKind#MOE},
     * {@link com.eonmux.neuralfree.gguf.arch.FfnKind#MOE_LATENT},
     * {@link com.eonmux.neuralfree.gguf.arch.FfnKind#MOE_CHUNKED} and
     * {@link com.eonmux.neuralfree.gguf.arch.FfnKind#MOE_PARALLEL_DENSE}), in execution order.
     */
    public static List<LayerSpec> routedMoeLayers(ModelSpec s) {
        List<LayerSpec> out = new ArrayList<>();
        for (LayerSpec l : s.layers()) {
            switch (l.ffnKind()) {
                case MOE:
                case MOE_LATENT:
                case MOE_CHUNKED:
                case MOE_PARALLEL_DENSE:
                    out.add(l);
                    break;
                default:
                    break;
            }
        }
        return out;
    }

    /**
     * Opens the MoE expert cache over {@code freeVramBytes} (an explicit budget is clamped to it), when
     * {@code cfg} enables it and {@code routed} is not empty; null otherwise. The layer plans are built from the
     * stored tensor shapes and the uploader's device types, and the cache is preloaded from the ranking the model's
     * {@link MoeCacheRanking} session read at load. The split dispatcher rides on the host team's cost model
     * ({@link CpuParallel#parallelism()}), as Avalon's dispatcher does.
     *
     * @param ranking the model's ranking session, or null for no preload
     * @throws GpuModelUnsupportedException when the device cannot hold the routed experts: a tensor's stored type
     *                                      does not match its device type, or its type has no device kernels
     */
    public static MoeExpertCache open(MoeCacheConfig cfg, List<GGUFTensorInfo> tensors, WeightStore store,
                                      GpuWeightUploader uploader, ComputeBackend backend, ComputeQueue queue,
                                      int ubatch, List<LayerSpec> routed, long freeVramBytes,
                                      MoeCacheRanking ranking, String modelName) {
        if (routed.isEmpty() || !cfg.enabled()) {
            return null;
        }
        List<MoeExpertCache.LayerPlan> plans = new ArrayList<>(routed.size());
        List<List<String>> namesByLayer = new ArrayList<>();
        int maxWeightLayer = 0;
        int maxK = 1;
        for (LayerSpec l : routed) {
            plans.add(layerPlan(l, tensors, uploader, backend));
            maxWeightLayer = Math.max(maxWeightLayer, l.weightLayer());
            maxK = Math.max(maxK, l.moe().nUsed());
        }
        for (int wl = 0; wl <= maxWeightLayer; wl++) {
            namesByLayer.add(Collections.emptyList());
        }
        for (MoeExpertCache.LayerPlan p : plans) {
            List<String> names = new ArrayList<>(p.tensors().size());
            for (MoeExpertCache.TensorPlan t : p.tensors()) {
                names.add(t.name());
            }
            namesByLayer.set(p.weightLayer(), names);
        }
        WeightDataProvider data = new WeightDataProvider(store, tensors, namesByLayer, 0, 0L);
        long budget = cfg.budgetMiB() == MoeCacheConfig.BUDGET_AUTO ? freeVramBytes
                : Math.min(Math.multiplyExact(cfg.budgetMiB(), 1024L * 1024L), freeVramBytes);
        MoeSplitDispatcher dispatcher = new MoeSplitDispatcher(
                MoeSplitDispatcher.Model.of(CpuParallel.parallelism()));
        try {
            return MoeExpertCache.open(backend, queue, data, plans, budget,
                    Math.multiplyExact(Math.max(1, ubatch), maxK), ranking, modelName, dispatcher);
        } catch (RuntimeException e) {
            data.close();
            throw e;
        }
    }

    /**
     * Opens the prefill expert prefetch over {@code cache} ({@code neuralfree.moe.cache.prefetch}, Avalon
     * {@code GGML_SCHED_PREFETCH_EXPERTS}): prefill-sized steps above {@code neuralfree.moe.cache.hosted.rows}
     * routed rows borrow that many layer stacks of slots from the cache's banks for the experts it does not hold.
     *
     * @return the pool, or null when there is no cache or {@code cfg} sets 0 zones
     */
    public static MoeExpertPrefetch openPrefetch(MoeCacheConfig cfg, MoeExpertCache cache, ComputeBackend backend,
                                                 String modelName) {
        if (cache == null || cfg.prefetchZones() <= 0) {
            return null;
        }
        return MoeExpertPrefetch.open(backend, cache, cache.layerPlans(), cfg.prefetchZones(), cfg.hostedMaxRows(),
                modelName);
    }

    /** The layer's plan in the cache: routed experts only, of stored types the device kernels run. */
    private static void checkDeviceType(LayerSpec l, MoeExpertCache.TensorPlan t, ComputeBackend backend) {
        if (!MoeKernels.supports(t.type())) {
            throw new GpuModelUnsupportedException("layer " + l.il() + ": " + t.name() + " has type " + t.type()
                    + ", which " + backend.backendType() + " cannot run on the device; clear"
                    + " neuralfree.moe.cache and neuralfree.moe.cache.mb to compute every pair on the host");
        }
    }

    /** The expert tensor plan of one cached layer, in the order the cached pass reads the banks. */
    private static MoeExpertCache.LayerPlan layerPlan(LayerSpec l, List<GGUFTensorInfo> tensors,
                                                      GpuWeightUploader uploader, ComputeBackend backend) {
        MoeSpec moe = l.moe();
        List<MoeExpertCache.TensorPlan> tensorPlans = new ArrayList<>(3);
        if (moe.fusedGateUp()) {
            tensorPlans.add(tensorPlan(l, TensorNames.FFN_GATE_UP_EXPS, tensors, uploader, backend));
        } else {
            if (moe.gated()) {
                tensorPlans.add(tensorPlan(l, TensorNames.FFN_GATE_EXPS, tensors, uploader, backend));
            }
            tensorPlans.add(tensorPlan(l, TensorNames.FFN_UP_EXPS, tensors, uploader, backend));
        }
        tensorPlans.add(tensorPlan(l, TensorNames.FFN_DOWN_EXPS, tensors, uploader, backend));
        return new MoeExpertCache.LayerPlan(l.weightLayer(), moe.nUsed(), tensorPlans);
    }

    /** One cached expert tensor: the stored shape and the device type the kernels run. */
    private static MoeExpertCache.TensorPlan tensorPlan(LayerSpec l, String suffix, List<GGUFTensorInfo> tensors,
                                                        GpuWeightUploader uploader, ComputeBackend backend) {
        String name = TensorNames.layerName(l.weightLayer(), TensorNames.weight(suffix));
        GGUFTensorInfo info = tensorInfo(tensors, name);
        if (info == null) {
            throw new IllegalStateException("layer " + l.il() + ": " + name + " is not in the model file");
        }
        long[] ne = info.dims();
        if (ne.length != 3) {
            throw new IllegalStateException("layer " + l.il() + ": " + name + " has rank " + ne.length
                    + ", expected the [cols, rows, n_expert] expert stack");
        }
        GGMLType device = uploader.deviceType(name);
        if (device != info.type()) {
            throw new GpuModelUnsupportedException("layer " + l.il() + ": " + name + " is stored as "
                    + info.type() + " but uploads as " + device + "; the MoE expert cache copies expert bytes"
                    + " unchanged, so the two must match");
        }
        MoeExpertCache.TensorPlan plan = new MoeExpertCache.TensorPlan(name, ne[0], ne[1], Math.toIntExact(ne[2]),
                device);
        checkDeviceType(l, plan, backend);
        return plan;
    }

    private static GGUFTensorInfo tensorInfo(List<GGUFTensorInfo> tensors, String name) {
        for (GGUFTensorInfo t : tensors) {
            if (t.name().equals(name)) {
                return t;
            }
        }
        return null;
    }
}
