package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.engine.EplbConfig;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.ops.moe.EplbLoadWindow;
import com.eonmux.neuralfree.gguf.ops.moe.RankedExpert;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The runtime expert rearrangement of a tensor split (vLLM's {@code EplbState.step}): every forward pass counts one
 * step, the routed rows of the last {@code window-size} steps before a rearrangement boundary are collected in a
 * {@link EplbLoadWindow}, and at the boundary ({@code step-interval} steps) the window's per-layer loads are
 * re-balanced with {@link EplbPolicy#rebalance} and migrated to the ranks' physical banks in place. An expert that
 * keeps its rank keeps its slot where the policy can preserve it, so an unchanged slot is not rewritten.
 *
 * <p>The rearrangement follows the reference's schedule: the window records exactly the steps the next boundary
 * reads ({@code steps_remaining <= window_size} at the end of every step, vLLM's
 * {@code _should_record_current_step}); the first boundary falls on step {@code step-interval}. The physical slot
 * count never changes (the load-time placement's, logical experts plus the configured redundant slots), so a
 * rearrangement only moves experts between a rank's slots and between ranks. The per-layer balance is logged before
 * and after with the reference's {@code avg_tokens / max_tokens / balancedness} figures. Not thread-safe: the
 * forward path drives {@link #afterStep}.</p>
 */
public final class EplbRebalancer {

    /** The weight migration of one rearrangement: the group applies the new layouts to every rank's banks. */
    public interface Migrator {
        void migrate(Map<Integer, int[]> layouts);
    }

    private static final Logger LOGGER = LogManager.getLogger(EplbRebalancer.class);

    private final EplbConfig cfg;
    private final int ranks;
    private final EplbLoadWindow window;
    private final Migrator migrator;
    /** The rearranged layers by weight layer, in model order. */
    private final Map<Integer, TpMoePlan.Layer> layers = new LinkedHashMap<>();
    /** The layer specs by weight layer, for the hierarchical policy's group count. */
    private final Map<Integer, LayerSpec> specs = new LinkedHashMap<>();
    /** The physical layout the banks currently reproduce, per weight layer. */
    private final Map<Integer, int[]> layouts = new LinkedHashMap<>();
    private long step;

    private EplbRebalancer(EplbConfig cfg, int ranks, EplbLoadWindow window, Migrator migrator) {
        this.cfg = cfg;
        this.ranks = ranks;
        this.window = window;
        this.migrator = migrator;
    }

    /**
     * Opens the rearrangement of a rearrangeable expert split.
     *
     * @param cfg      the balancer configuration with {@code rebalance=on}
     * @param spec     the model spec
     * @param ranks    the split's rank count
     * @param plan     the rearrangeable expert split (its layers' layouts are the starting state)
     * @param migrator the weight migration, usually the tensor-parallel group
     * @return the rebalancer, or null when the configuration does not rearrange
     * @throws IllegalArgumentException when the model has no split routed-expert layer, or a layer's logical expert
     *                                  count does not form the configured expert groups
     */
    public static EplbRebalancer open(EplbConfig cfg, ModelSpec spec, int ranks, TpMoePlan plan, Migrator migrator) {
        if (cfg == null || !cfg.enabled() || !cfg.rebalance()) {
            return null;
        }
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(migrator, "migrator");
        if (plan.layers().isEmpty()) {
            throw new IllegalArgumentException(EplbConfig.PROP_EPLB_REBALANCE + "=on rearranges the routed experts"
                    + " of a tensor split, but the model has no split routed-expert layer; enable "
                    + EngineConfig.PROP_EXPERT_PARALLEL + "=true with a MoE model");
        }
        EplbRebalancer rebalancer = new EplbRebalancer(cfg, ranks,
                new EplbLoadWindow(cfg.windowSize()), migrator);
        rebalancer.takeLayers(spec, plan, ranks);
        rebalancer.window.record(cfg.stepInterval() <= cfg.windowSize());
        LOGGER.info("expert rearrangement: every {} step(s) over a {} step load window, {} layer(s)", cfg.stepInterval(),
                cfg.windowSize(), rebalancer.layers.size());
        return rebalancer;
    }

    private void takeLayers(ModelSpec spec, TpMoePlan plan, int ranks) {
        for (TpMoePlan.Layer layer : plan.layers()) {
            if (!layer.expertParallel() || !layer.rearrangeable() || layer.phyToLog() == null) {
                continue;
            }
            LayerSpec layerSpec = spec.layer(layer.il());
            Objects.requireNonNull(layerSpec, "layer " + layer.il());
            int groups = groups(layerSpec);
            int logical = layer.nExpert();
            if (logical % groups != 0) {
                throw new IllegalArgumentException("layer " + layer.il() + ": " + logical + " logical experts do not"
                        + " form " + groups + " expert group(s) for " + EplbConfig.PROP_EPLB_NUM_GROUPS + "="
                        + cfg.numGroups() + "; the group count must divide the experts");
            }
            layers.put(layer.weightLayer(), layer);
            specs.put(layer.weightLayer(), layerSpec);
            layouts.put(layer.weightLayer(), layer.phyToLog());
        }
    }

    /** The steps the rearrangement has counted so far. */
    public long steps() {
        return step;
    }

    /** The live routed-row window the boundaries read. */
    public EplbLoadWindow window() {
        return window;
    }

    /** The physical layout the banks currently reproduce, per weight layer. */
    public Map<Integer, int[]> layouts() {
        Map<Integer, int[]> out = new LinkedHashMap<>();
        for (Map.Entry<Integer, int[]> entry : layouts.entrySet()) {
            out.put(entry.getKey(), entry.getValue().clone());
        }
        return out;
    }

    /**
     * One forward pass ended: counts the step, rearranges on a {@code step-interval} boundary from the window's
     * recorded loads, then opens the window's recording for the steps the next boundary reads.
     */
    public void afterStep() {
        step++;
        if (step % cfg.stepInterval() == 0) {
            rearrange();
        }
        long remaining = cfg.stepInterval() - (step % cfg.stepInterval());
        window.record(remaining <= cfg.windowSize());
    }

    /**
     * Rearranges now from the window's recorded routed rows, whether or not a boundary has been reached: the window
     * is drained, every layer's load is re-balanced and the changed layouts are migrated. Layers whose layout the
     * policy keeps are not migrated.
     */
    public void rearrange() {
        List<RankedExpert> rows = window.drain();
        Map<Integer, double[]> loads = new HashMap<>();
        for (RankedExpert row : rows) {
            TpMoePlan.Layer layer = layers.get(row.layer());
            if (layer == null) {
                continue;
            }
            double[] w = loads.computeIfAbsent(row.layer(), k -> new double[layer.nExpert()]);
            if (row.expert() >= 0 && row.expert() < w.length) {
                w[row.expert()] += row.heat();
            }
        }
        Map<Integer, int[]> changed = new LinkedHashMap<>();
        for (Map.Entry<Integer, TpMoePlan.Layer> entry : layers.entrySet()) {
            rearrangeLayer(entry.getKey(), entry.getValue(), loads.getOrDefault(entry.getKey(),
                    new double[entry.getValue().nExpert()]), changed);
        }
        if (!changed.isEmpty()) {
            migrator.migrate(changed);
            for (Map.Entry<Integer, int[]> entry : changed.entrySet()) {
                layouts.put(entry.getKey(), entry.getValue().clone());
            }
        }
    }

    /** Re-balances one layer from its window load and logs the balance figures; registers a changed layout. */
    private void rearrangeLayer(int weightLayer, TpMoePlan.Layer layer, double[] load, Map<Integer, int[]> changed) {
        int[] current = layouts.get(weightLayer);
        double[] rankLoadBefore = EplbPolicy.rankLoads(load, current, ranks);
        String before = EplbPolicy.balanceLine(new double[][]{rankLoadBefore});
        int[][] packed = EplbPolicy.rebalance(new double[][]{load}, current.length, groups(specs.get(weightLayer)),
                cfg.numNodes(), ranks, new int[][]{current});
        int[] next = packed[0];
        if (Arrays.equals(current, next)) {
            LOGGER.info("eplb: layer {} keeps its layout after {} step(s) ({})", layer.il(), step, before);
            return;
        }
        changed.put(weightLayer, next);
        LOGGER.info("eplb: layer {} rearranged after {} step(s): {} -> {}", layer.il(), step, before,
                EplbPolicy.balanceLine(new double[][]{EplbPolicy.rankLoads(load, next, ranks)}));
    }

    /** The hierarchical policy's group count of a layer: the configured one, else the router's groups. */
    private int groups(LayerSpec layer) {
        int configured = cfg.numGroups();
        if (configured > 0) {
            return configured;
        }
        int routerGroups = layer.moe().nGroups();
        return Math.max(1, routerGroups);
    }

    @Override
    public String toString() {
        return "EplbRebalancer{" + layers.size() + " layer(s), step " + step + ", every " + cfg.stepInterval()
                + " step(s)}";
    }
}
