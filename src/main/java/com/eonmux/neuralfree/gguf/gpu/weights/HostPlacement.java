package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.engine.TensorOverride;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.List;
import java.util.function.IntPredicate;
import java.util.function.LongSupplier;
import java.util.function.Predicate;

/**
 * The host placement of one GPU load: the per-tensor rules the engine settings resolve to, the MoE layers whose
 * whole routed expert set stays in RAM, and the byte accounting of both. The rules follow llama.cpp's flag
 * expansion (common/arg.cpp:2705-2747): the explicit {@code -ot} rules first, then {@code --n-cpu-ffn},
 * {@code --n-cpu-moe} and {@code --cpu-moe}; {@code neuralfree.moe.cache} / {@code neuralfree.moe.cache.mb}
 * host the routed experts through the same {@code --cpu-moe} rule (their budget configures the MoE expert cache).
 * Under Avalon's {@code --fit} (fit.cpp:614-646) the experts join them
 * when the dense weights alone fit the usable VRAM, and the fit falls back to the usual placement otherwise. A
 * layer with only part of its expert set on the host is refused: the block reads gate, up (or the fused gate-up)
 * and down from one tier. The tensors a hybrid layer split ({@code com.eonmux.neuralfree.gguf.hybrid}) keeps on
 * the host come in through {@code fixedHost}, so the rules and the fit see one placement. Stateless.
 */
public final class HostPlacement {

    private HostPlacement() {
    }

    /**
     * The placement rules of {@code cfg} for {@code spec}, probed against the VRAM {@code usable} reports when the
     * settings ask for a fit.
     *
     * @param loadNextn whether the NextN / MTP blocks are placed (their experts follow the same rules)
     */
    public static WeightPlacement placement(EngineConfig cfg, ModelSpec spec, GpuWeightUploader uploader,
                                            LongSupplier usableVram, boolean loadNextn) {
        return placement(cfg, spec, uploader, usableVram, loadNextn, n -> false);
    }

    /**
     * The placement rules of {@code cfg} for {@code spec}, with {@code fixedHost} naming further tensors the load
     * keeps on the host regardless of the rules.
     *
     * @param loadNextn whether the NextN / MTP blocks are placed (their experts follow the same rules)
     */
    public static WeightPlacement placement(EngineConfig cfg, ModelSpec spec, GpuWeightUploader uploader,
                                            LongSupplier usableVram, boolean loadNextn,
                                            Predicate<String> fixedHost) {
        if (fitApplies(cfg, spec)) {
            WeightPlacement candidate = fitCandidate(cfg, spec, loadNextn, fixedHost);
            if (uploader.plannedBytes(candidate::onDevice) <= usableVram.getAsLong()) {
                return candidate;
            }
        }
        return rulePlacement(cfg, spec, loadNextn, fixedHost);
    }

    /**
     * The placement of the explicit rules of {@code cfg} without the {@code --fit} probe: the {@code -ot} rules,
     * {@code --n-cpu-ffn}, {@code --n-cpu-moe}, then {@code --cpu-moe} or the MoE expert cache, with {@code fixedHost}
     * naming further host tensors. A load that fits its devices one by one (a layer split) starts from it.
     */
    public static WeightPlacement rulePlacement(EngineConfig cfg, ModelSpec spec, boolean loadNextn,
                                                Predicate<String> fixedHost) {
        return WeightPlacement.of(spec, loadNextn, rulesOf(cfg), fixedHost);
    }

    /**
     * Whether {@code --fit} probes the expert placement: it is set, neither {@code --cpu-moe} nor the MoE expert
     * cache already hosts every expert, and some trunk layer has routed experts to move.
     */
    public static boolean fitApplies(EngineConfig cfg, ModelSpec spec) {
        return cfg.fit() && !cfg.cpuMoe() && !moeCacheRequested(cfg) && hasExperts(spec);
    }

    /**
     * The placement {@code --fit} takes when the dense weights fit the device: the explicit rules with every routed
     * expert tensor on the host (Avalon fit.cpp:614-646).
     */
    public static WeightPlacement fitCandidate(EngineConfig cfg, ModelSpec spec, boolean loadNextn,
                                               Predicate<String> fixedHost) {
        return WeightPlacement.of(spec, loadNextn, rulesOf(cfg).with(TensorPlacementOverrides.cpuMoe()), fixedHost);
    }

    /**
     * Whether the load hosts the routed experts for the MoE expert cache: {@code neuralfree.moe.cache}
     * {@code on | auto | N} or a {@code neuralfree.moe.cache.mb} budget (llama.cpp {@code --moe-cache-mib}, Avalon
     * {@code --moe-cache}). Both resolve to the same settings: the rule above hosts the routed expert stacks as
     * {@code --cpu-moe} does, and the cache serves them from the budget.
     */
    public static boolean moeCacheRequested(EngineConfig cfg) {
        return cfg.moeCache().enabled() || cfg.moeCacheMb() > 0;
    }

    /**
     * The unified settings of the MoE expert cache: {@code neuralfree.moe.cache} and
     * {@code neuralfree.moe.cache.mb} resolve to one configuration. {@code on}, {@code auto} or a MiB value of
     * {@code neuralfree.moe.cache} hosts the routed experts (as {@code --cpu-moe} does) and sets the budget
     * ({@code auto} or unset: the VRAM left after loading); {@code neuralfree.moe.cache.mb} sets the same budget
     * and wins when both give one, and enables the cache on its own. The hosted tail-skip rank
     * ({@code neuralfree.moe.cache.tail.skip}) needs the cache: a rank above 0 with the cache off is refused.
     *
     * @throws IllegalArgumentException when a tail-skip rank is set without the cache enabled
     */
    public static MoeCacheConfig moeCacheSettings(EngineConfig cfg) {
        MoeCacheConfig c = cfg.moeCache();
        long mb = cfg.moeCacheMb();
        MoeCacheConfig out;
        if (!c.enabled()) {
            out = mb > 0 ? c.onWithBudgetMiB(mb)
                    : MoeCacheConfig.off().withRanking(c.ranking()).withTailSkipRank(c.tailSkipRank());
        } else {
            out = mb > 0 ? c.withBudgetMiB(mb) : c;
        }
        if (out.tailSkipRank() > 0 && !out.enabled()) {
            throw new IllegalArgumentException(MoeCacheConfig.PROP_MOE_CACHE_TAIL_SKIP + "=" + out.tailSkipRank()
                    + " needs the MoE expert cache: enable " + MoeCacheConfig.PROP_MOE_CACHE
                    + " (on, auto or a MiB budget) or set " + EngineConfig.PROP_MOE_CACHE_MB);
        }
        return out;
    }

    /**
     * The rules of {@code cfg} in match order: the user's {@code -ot} rules (a host match keeping its tensor in RAM),
     * then {@code --n-cpu-ffn}, {@code --n-cpu-moe} and {@code --cpu-moe} / the MoE expert cache.
     */
    private static TensorPlacementOverrides rulesOf(EngineConfig cfg) {
        TensorPlacementOverrides rules = TensorPlacementOverrides.none();
        for (TensorOverride o : cfg.tensorOverrides()) {
            rules = rules.with(TensorPlacementOverrides.builder()
                    .add(o.pattern(), o.host() ? TensorPlacementOverrides.Target.HOST
                            : TensorPlacementOverrides.Target.DEVICE)
                    .build());
        }
        if (cfg.nCpuFfn() > EngineConfig.N_CPU_LAYERS_UNSET) {
            rules = rules.with(TensorPlacementOverrides.nCpuFfn(cfg.nCpuFfn()));
        }
        if (cfg.nCpuMoe() > 0) {
            rules = rules.with(TensorPlacementOverrides.nCpuMoe(cfg.nCpuMoe()));
        }
        if (cfg.cpuMoe() || moeCacheRequested(cfg)) {
            rules = rules.with(TensorPlacementOverrides.cpuMoe());
        }
        return rules;
    }

    /** True when any trunk layer is a routed MoE layer, so a fit has experts to move. */
    private static boolean hasExperts(ModelSpec spec) {
        for (LayerSpec l : spec.layers()) {
            if (l.moe() != null) {
                return true;
            }
        }
        return false;
    }

    /**
     * The MoE layers whose whole routed expert set the placement keeps in RAM, in layer order: the candidates the
     * hybrid CPU+GPU expert path and the host-expert FFN decorator divide among themselves (the caller checks
     * which path serves each layer).
     *
     * @param deviceLayers whether a trunk weight layer computes on the device; a host layer of a hybrid split runs
     *                     whole on the host and is never a candidate
     * @throws UnsupportedOperationException for a partial expert set
     */
    public static List<LayerSpec> hostedMoeLayers(ModelSpec spec, TensorCatalog catalog, WeightPlacement placement,
                                                  IntPredicate deviceLayers) {
        List<LayerSpec> out = new ArrayList<>();
        List<LayerSpec> all = new ArrayList<>(spec.layers());
        all.addAll(spec.nextnLayers());
        for (LayerSpec l : all) {
            if (l.moe() == null || !deviceLayers.test(l.weightLayer())) {
                continue;
            }
            List<String> names = expertTensorNames(catalog, l);
            if (names.isEmpty()) {
                continue;
            }
            int hostCount = 0;
            for (String name : names) {
                if (!placement.onDeviceName(name)) {
                    hostCount++;
                }
            }
            if (hostCount == 0) {
                continue;
            }
            if (hostCount != names.size()) {
                throw new UnsupportedOperationException(spec.arch().ggufName() + " layer " + l.il() + ": the"
                        + " placement rules keep " + hostCount + " of the " + names.size() + " expert tensors in"
                        + " RAM; keep the whole set (" + names + ") off the device, or none of it");
            }
            out.add(l);
        }
        return out;
    }

    /** Stored names of the layer's routed expert tensors that the file holds. */
    private static List<String> expertTensorNames(TensorCatalog catalog, LayerSpec l) {
        String[] suffixes = {TensorNames.weight(TensorNames.FFN_UP_EXPS),
                TensorNames.weight(TensorNames.FFN_GATE_EXPS),
                TensorNames.weight(TensorNames.FFN_GATE_UP_EXPS),
                TensorNames.weight(TensorNames.FFN_DOWN_EXPS)};
        List<String> out = new ArrayList<>(suffixes.length);
        for (String s : suffixes) {
            String name = TensorNames.layerName(l.stack(), l.weightLayer(), s);
            if (catalog.has(name)) {
                out.add(name);
            }
        }
        return out;
    }

    /** Stored bytes of the hosted layers' expert tensors (their RAM residency). */
    public static long hostExpertBytes(List<LayerSpec> layers, TensorCatalog catalog) {
        long total = 0;
        for (LayerSpec l : layers) {
            for (String name : expertTensorNames(catalog, l)) {
                total = Math.addExact(total, catalog.get(name).sizeInBytes());
            }
        }
        return total;
    }
}
