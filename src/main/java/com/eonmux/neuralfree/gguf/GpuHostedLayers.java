package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.gpu.block.GpuBlockRegistry;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.weights.HostPlacement;
import com.eonmux.neuralfree.gguf.gpu.weights.WeightPlacement;
import com.eonmux.neuralfree.gguf.hybrid.HybridForwardBackend;
import com.eonmux.neuralfree.gguf.hybrid.LayerOffload;
import com.eonmux.neuralfree.gguf.ops.moe.GpuMoeFfn;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.function.IntPredicate;

/**
 * The MoE layers of a {@link GGUFModelGPU} load whose whole routed expert set the placement keeps in RAM, divided
 * between the hosted expert blocks ({@link GpuMoeFfn}'s cached MoE pass over {@link GpuForwardContext#hostWeights()}
 * and the MoE expert cache) and the host-expert FFN decorator, which runs the CPU MoE block of a hosted layer the
 * hosted path cannot serve. A layer split resolves them once for the whole model; every stage runs those of its own
 * layers. Immutable.
 */
public final class GpuHostedLayers {

    private static final Logger LOGGER = LogManager.getLogger(GpuHostedLayers.class);
    private static final double MIB = 1024.0 * 1024.0;

    /** No hosted layer: every expert set is on the device. */
    public static final GpuHostedLayers NONE = new GpuHostedLayers(Collections.<LayerSpec>emptyList(),
            Collections.<LayerSpec>emptyList());

    private final List<LayerSpec> cached;
    private final List<LayerSpec> decorated;

    private GpuHostedLayers(List<LayerSpec> cached, List<LayerSpec> decorated) {
        this.cached = Collections.unmodifiableList(new ArrayList<>(cached));
        this.decorated = Collections.unmodifiableList(new ArrayList<>(decorated));
    }

    /**
     * Resolves the hosted MoE layers of {@code placement}: a layer the hosted path cannot serve
     * ({@link GpuMoeFfn#checkSupported}) keeps its CPU MoE block when the MoE settings of {@code split} derive its
     * host experts ({@link LayerOffload#hostExperts(int)}).
     *
     * @param split the resolved hybrid split, or null; only its device layers are hosted-MoE candidates
     * @throws UnsupportedOperationException naming the layer when a placement rule alone hosted an expert set the
     *                                       hosted path cannot serve
     */
    public static GpuHostedLayers resolve(ModelSpec initial, TensorCatalog catalog, WeightPlacement placement,
                                   LayerOffload split) {
        List<LayerSpec> hosted = HostPlacement.hostedMoeLayers(initial, catalog, placement, deviceLayers(split));
        List<LayerSpec> cached = new ArrayList<>(hosted.size());
        List<LayerSpec> decorated = new ArrayList<>(hosted.size());
        for (LayerSpec l : hosted) {
            try {
                GpuMoeFfn.checkSupported(initial, l);
                cached.add(l);
            } catch (UnsupportedOperationException e) {
                if (decoratorServes(initial, split, l)) {
                    decorated.add(l);
                } else {
                    throw e;
                }
            }
        }
        return new GpuHostedLayers(cached, decorated);
    }

    /** Trunk weight layers the split computes on the device; every layer when no split is resolved. */
    private static IntPredicate deviceLayers(LayerOffload split) {
        return split != null ? il -> il >= split.firstGpuLayer() : il -> true;
    }

    /** True when the split derives hosted experts for the trunk layer, so its decorator keeps the CPU MoE block. */
    private static boolean decoratorServes(ModelSpec spec, LayerOffload split, LayerSpec l) {
        return split != null && l.weightLayer() < spec.nLayer() && split.hostExperts(l.il());
    }

    /** Trunk and NextN layers whose routed experts stay in RAM, served by the hosted expert blocks. */
    public List<LayerSpec> cached() {
        return cached;
    }

    /** True when some layer computes its MoE experts through the hosted expert blocks. */
    public boolean hasCached() {
        return !cached.isEmpty();
    }

    /** True when no layer keeps its expert set in RAM, cached or decorated ({@code == NONE} is unreliable: the resolver always builds a fresh instance). */
    public boolean isEmpty() {
        return cached.isEmpty() && decorated.isEmpty();
    }

    /**
     * The device blocks of {@code base}, with the host-expert FFN decorator over {@code cpuContext} on the hosted
     * layers the hosted path cannot serve; {@code base} itself when there are none.
     */
    public GpuBlockRegistry decorate(GpuBlockRegistry base, CpuForwardContext cpuContext) {
        if (decorated.isEmpty()) {
            return base;
        }
        return base.withFfnDecorator((layer, deviceBlock) -> decoratorCovers(layer.il())
                ? HybridForwardBackend.hostExpertFfn(cpuContext, layer) : deviceBlock);
    }

    /** True when the FFN of layer {@code il} is a decorator layer. */
    private boolean decoratorCovers(int il) {
        for (LayerSpec l : decorated) {
            if (l.il() == il) {
                return true;
            }
        }
        return false;
    }

    /**
     * Logs the hosted layers of the load of {@code modelName} and their RAM.
     *
     * @param cache the MoE expert cache serving the hosted blocks, or {@link MoeCacheConfig#off()} when none is
     *              configured
     */
    public void log(String modelName, TensorCatalog catalog, MoeCacheConfig cache) {
        Objects.requireNonNull(cache, "cache");
        if (!cached.isEmpty()) {
            LOGGER.info("{}: {} layer(s) compute their MoE experts on the host ({} MiB in RAM){}", modelName,
                    cached.size(), mib(HostPlacement.hostExpertBytes(cached, catalog)),
                    cache.enabled() ? ", expert cache " + cache : ", no expert cache");
        }
        if (!decorated.isEmpty()) {
            LOGGER.info("{}: {} layer(s) keep their expert set on the host beside the device layer through the"
                            + " host-expert FFN ({} MiB in RAM)", modelName, decorated.size(),
                    mib(HostPlacement.hostExpertBytes(decorated, catalog)));
        }
    }

    /** Mebibytes with one decimal. */
    private static String mib(long bytes) {
        return String.format(Locale.ROOT, "%.1f", bytes / MIB);
    }
}
