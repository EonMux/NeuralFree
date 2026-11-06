package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.engine.EplbConfig;
import com.eonmux.neuralfree.gguf.ops.moe.RankedExpert;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;

import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The load-balanced placement of a tensor split's routed experts (vLLM's EPLB, {@code vllm/distributed/eplb}): every
 * layer's logical experts are replicated into physical slots and the slots are packed over the ranks by their
 * recorded routed-row load, so each rank holds {@code num_physical / ranks} slots whose total load is as close as
 * possible to every other rank's.
 *
 * <p>A layer's physical slots are contiguous per rank: rank {@code r} holds physical slots
 * {@code [r * slots_per_rank, (r + 1) * slots_per_rank)}, and slot {@code s} reproduces logical expert
 * {@code phyToLog[s]}. With no redundant experts the physical count equals the logical count and every logical
 * expert appears once; {@code num_redundant_experts} adds replicas of the heaviest logical experts, which
 * {@link EplbPolicy} picks and packs. The router keeps selecting logical experts, and the per-pair replica choice
 * (which of an expert's physical copies computes a routed row) is made on the device from the token index, so a pair
 * is computed by exactly one rank.</p>
 *
 * <p>The load of an expert is the number of routed rows counted for it (a {@code moe-cache-ranking.bin} profile,
 * {@link RankedExpert} entries keyed by weight layer). A layer without counted load (no profile entry, or all zero)
 * has no placement and keeps the contiguous expert block of {@code rank * n_expert / ranks}. The placement is
 * computed once at load; {@link EplbPolicy#rebalance} computes a new one for a runtime rearrangement. Immutable.</p>
 */
public final class EplbPlacement {

    /** The policy settings shared by every layer. */
    public static final class Options {
        private final EplbConfig.Policy policy;
        private final int numRedundant;
        private final int numGroups;
        private final int numNodes;

        private Options(EplbConfig.Policy policy, int numRedundant, int numGroups, int numNodes) {
            this.policy = Objects.requireNonNull(policy, "policy");
            if (numRedundant < 0) {
                throw new IllegalArgumentException("num_redundant_experts must be >= 0, got " + numRedundant);
            }
            if (numGroups < 0) {
                throw new IllegalArgumentException("num_groups must be >= 0 (0 derives it per layer), got "
                        + numGroups);
            }
            if (numNodes < 1) {
                throw new IllegalArgumentException("num_nodes must be >= 1, got " + numNodes);
            }
            this.numRedundant = numRedundant;
            this.numGroups = numGroups;
            this.numNodes = numNodes;
        }

        /** The default policy: no replicas, the layer's own expert groups, one node. */
        public static Options defaults() {
            return new Options(EplbConfig.Policy.DEFAULT, 0, 0, 1);
        }

        /** The options of a configuration. */
        public static Options of(EplbConfig.Policy policy, int numRedundant, int numGroups, int numNodes) {
            return new Options(policy, numRedundant, numGroups, numNodes);
        }

        /** The placement policy. */
        public EplbConfig.Policy policy() { return policy; }

        /** Redundant physical slots added to every layer's logical experts. */
        public int numRedundant() { return numRedundant; }

        /** The configured group count, or 0 to take each layer's {@link MoeSpec#nGroups()}. */
        public int numGroups() { return numGroups; }

        /** Server nodes the hierarchical policy packs groups into. */
        public int numNodes() { return numNodes; }

        @Override
        public String toString() {
            return "Options{" + policy.label() + ", redundant " + numRedundant + ", groups "
                    + (numGroups == 0 ? "layer" : Integer.toString(numGroups)) + ", nodes " + numNodes + '}';
        }
    }

    /** One layer's physical layout. */
    private static final class LayerPlacement {
        private final int[] phyToLog;
        private final int[][] logToPhy;
        private final int[] replicaCount;
        private final int slotsPerRank;
        private final int maxReplicas;

        private LayerPlacement(int[] phyToLog, int[][] logToPhy, int[] replicaCount, int slotsPerRank,
                               int maxReplicas) {
            this.phyToLog = phyToLog;
            this.logToPhy = logToPhy;
            this.replicaCount = replicaCount;
            this.slotsPerRank = slotsPerRank;
            this.maxReplicas = maxReplicas;
        }
    }

    private final int ranks;
    private final Map<String, LayerPlacement> placements;

    private EplbPlacement(int ranks, Map<String, LayerPlacement> placements) {
        this.ranks = ranks;
        this.placements = placements;
    }

    /** The rank count the placement packs over. */
    public int ranks() { return ranks; }

    /** True when the layer's experts are packed by recorded load rather than in one contiguous block per rank. */
    public boolean covers(LayerSpec layer) {
        return placements.containsKey(key(layer));
    }

    /** The layer count the placement balances. */
    public int layered() { return placements.size(); }

    /** The physical slot count of {@code layer} (logical plus redundant), or 0 when it has no placement. */
    public int physicalExperts(LayerSpec layer) {
        LayerPlacement p = placements.get(key(layer));
        return p == null ? 0 : p.phyToLog.length;
    }

    /** The physical slots each rank holds of {@code layer}, or 0 when it has no placement. */
    public int slotsPerRank(LayerSpec layer) {
        LayerPlacement p = placements.get(key(layer));
        return p == null ? 0 : p.slotsPerRank;
    }

    /** True when {@code layer} has more physical slots than logical experts. */
    public boolean replicates(LayerSpec layer) {
        LayerPlacement p = placements.get(key(layer));
        return p != null && p.phyToLog.length > p.replicaCount.length;
    }

    /**
     * The physical slots rank {@code rank} holds of {@code layer}, {@code [rank * slots, (rank + 1) * slots)}, or null
     * when the layer has no placement (its experts divide into contiguous blocks).
     */
    public int[] physicalSlots(LayerSpec layer, int rank) {
        LayerPlacement p = placements.get(key(layer));
        if (p == null) {
            return null;
        }
        checkRank(rank);
        int[] out = new int[p.slotsPerRank];
        for (int i = 0; i < out.length; i++) {
            out[i] = rank * p.slotsPerRank + i;
        }
        return out;
    }

    /**
     * The logical experts rank {@code rank} holds of {@code layer}, in physical slot order (duplicates appear once per
     * replica), or null when the layer has no placement.
     */
    public int[] experts(LayerSpec layer, int rank) {
        LayerPlacement p = placements.get(key(layer));
        if (p == null) {
            return null;
        }
        checkRank(rank);
        int[] out = new int[p.slotsPerRank];
        for (int i = 0; i < out.length; i++) {
            out[i] = p.phyToLog[rank * p.slotsPerRank + i];
        }
        return out;
    }

    /** The logical expert physical slot {@code slot} reproduces. */
    public int logicalOf(LayerSpec layer, int slot) {
        LayerPlacement p = require(layer);
        if (slot < 0 || slot >= p.phyToLog.length) {
            throw new IllegalArgumentException("physical slot " + slot + " outside [0, " + p.phyToLog.length + ")");
        }
        return p.phyToLog[slot];
    }

    /** {@code [logical][maxReplicas]}: the physical slots of each logical expert, -1 where unused. */
    public int[][] logicalToPhysical(LayerSpec layer) {
        int[][] in = require(layer).logToPhy;
        int[][] out = new int[in.length][];
        for (int i = 0; i < in.length; i++) {
            out[i] = in[i].clone();
        }
        return out;
    }

    /** {@code [logical]}: the number of physical replicas of each logical expert. */
    public int[] replicaCount(LayerSpec layer) {
        return require(layer).replicaCount.clone();
    }

    /** The largest replica count over the layer's logical experts. */
    public int maxReplicas(LayerSpec layer) {
        return require(layer).maxReplicas;
    }

    private LayerPlacement require(LayerSpec layer) {
        LayerPlacement p = placements.get(key(layer));
        if (p == null) {
            throw new IllegalArgumentException("layer " + layer.il() + " has no expert placement");
        }
        return p;
    }

    private void checkRank(int rank) {
        if (rank < 0 || rank >= ranks) {
            throw new IllegalArgumentException("rank " + rank + " outside the placement's [0, " + ranks + ")");
        }
    }

    @Override
    public String toString() {
        int redundant = 0;
        for (LayerPlacement p : placements.values()) {
            redundant += p.phyToLog.length - p.replicaCount.length;
            break;
        }
        return "EplbPlacement{" + placements.size() + " balanced layer(s) over " + ranks + " rank(s)"
                + (redundant > 0 ? ", " + redundant + " redundant slot(s)" : "") + '}';
    }

    /**
     * Packs the routed experts of every trunk MoE layer of {@code spec} by their recorded load with no redundant
     * replicas and vLLM's global policy over one node.
     *
     * @param loads per (weight layer, expert) routed rows, as a ranking profile holds them
     * @throws IllegalArgumentException for a rank count below two, an expert count that does not divide, or a load
     *                                  entry naming an expert outside its layer
     */
    public static EplbPlacement of(ModelSpec spec, int ranks, List<RankedExpert> loads) {
        return of(spec, ranks, loads, Options.defaults());
    }

    /**
     * Packs the routed experts of every trunk MoE layer of {@code spec} by their recorded load.
     *
     * @param loads   per (weight layer, expert) routed rows, as a ranking profile holds them
     * @param options the policy, redundant expert count, group count and node count
     * @throws IllegalArgumentException for a rank count below two, an expert or physical count that does not divide,
     *                                  a load entry naming an expert outside its layer, or a redundant count with the
     *                                  default policy's divisibility rules
     */
    public static EplbPlacement of(ModelSpec spec, int ranks, List<RankedExpert> loads, Options options) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(loads, "loads");
        Objects.requireNonNull(options, "options");
        if (ranks < 2) {
            throw new IllegalArgumentException("an expert placement needs at least two ranks, got " + ranks);
        }
        Map<Long, Double> heat = new HashMap<>();
        Map<Integer, Integer> maxExpertOfLayer = new HashMap<>();
        for (RankedExpert e : loads) {
            heat.put(index(e.layer(), e.expert()), (double) e.heat());
            maxExpertOfLayer.merge(e.layer(), e.expert() + 1, Math::max);
        }
        Map<String, LayerPlacement> placements = new LinkedHashMap<>();
        for (LayerSpec layer : spec.layers()) {
            if (!bearsMoe(layer)) {
                continue;
            }
            int nExpert = layer.moe().nExpert();
            Integer largest = maxExpertOfLayer.get(layer.weightLayer());
            if (largest != null && largest > nExpert) {
                throw new IllegalArgumentException("the expert-load profile names experts up to " + (largest - 1)
                        + " of layer " + layer.il() + ", which has " + nExpert + " experts");
            }
            double[] weight = new double[nExpert];
            boolean counted = false;
            for (int e = 0; e < nExpert; e++) {
                Double h = heat.get(index(layer.weightLayer(), e));
                if (h != null) {
                    if (!Double.isFinite(h) || h < 0) {
                        throw new IllegalArgumentException("layer " + layer.il() + " expert " + e + " has load " + h);
                    }
                    weight[e] = h;
                    counted |= h > 0;
                }
            }
            if (counted) {
                placements.put(key(layer), place(layer, weight, ranks, options));
            }
        }
        return new EplbPlacement(ranks, Collections.unmodifiableMap(placements));
    }

    private static LayerPlacement place(LayerSpec layer, double[] weight, int ranks, Options options) {
        int logical = weight.length;
        int numPhysical = logical + options.numRedundant();
        int numGroups = options.numGroups() == 0 ? layer.moe().nGroups() : options.numGroups();
        if (numGroups < 1) {
            numGroups = 1;
        }
        if (numPhysical % ranks != 0) {
            throw new IllegalArgumentException("layer " + layer.il() + ": " + logical + " logical expert(s) plus "
                    + options.numRedundant() + " redundant do not divide over " + ranks + " rank(s)");
        }
        if (options.policy() == EplbConfig.Policy.DEFAULT && options.numNodes() > 1) {
            throw new IllegalArgumentException(EplbConfig.PROP_EPLB_POLICY + "=default packs over the ranks as one"
                    + " node; give " + EplbConfig.PROP_EPLB_POLICY + "=hierarchical to use "
                    + EplbConfig.PROP_EPLB_NUM_NODES);
        }
        int[][] phyToLog = EplbPolicy.rebalance(new double[][]{weight}, numPhysical, numGroups, options.numNodes(),
                ranks, null);
        int[][] logToPhy = EplbPolicy.layout(phyToLog, logical).logicalToPhysical(0);
        int[][] counts = EplbPolicy.layout(phyToLog, logical).replicaCount();
        return new LayerPlacement(phyToLog[0], logToPhy, counts[0], numPhysical / ranks,
                EplbPolicy.maxReplicas(phyToLog, logical));
    }

    private static boolean bearsMoe(LayerSpec layer) {
        return layer.moe() != null && (layer.ffnKind() == FfnKind.MOE || layer.ffnKind() == FfnKind.MOE_LATENT);
    }

    /**
     * Places every trunk MoE layer of {@code spec} with zero counted load, so a runtime rebalancing that starts
     * without a profile still holds its redundant physical slots from the first step: the packing keeps the logical
     * order and the redundant slots land on the lowest expert id, exactly as the reference packs an empty profile.
     *
     * @throws IllegalArgumentException from {@link #place} for a physical count that does not divide over the ranks
     */
    static EplbPlacement uniform(ModelSpec spec, int ranks, Options options) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(options, "options");
        if (ranks < 2) {
            throw new IllegalArgumentException("an expert placement needs at least two ranks, got " + ranks);
        }
        if (options.numRedundant() < 1) {
            throw new IllegalArgumentException("a zero-load placement is only built for redundant experts; without"
                    + " them the split keeps its contiguous expert blocks");
        }
        Map<String, LayerPlacement> placements = new LinkedHashMap<>();
        for (LayerSpec layer : spec.layers()) {
            if (!bearsMoe(layer)) {
                continue;
            }
            placements.put(key(layer), place(layer, new double[layer.moe().nExpert()], ranks, options));
        }
        return new EplbPlacement(ranks, Collections.unmodifiableMap(placements));
    }

    private static String key(LayerSpec layer) {
        return layer.stack() + ":" + layer.weightLayer();
    }

    private static long index(int layer, int expert) {
        return ((long) layer << 32) | (expert & 0xFFFFFFFFL);
    }

    /**
     * The physical placement of {@code load}'s experts over {@code ranks} with no redundant replicas (vLLM's
     * {@code balanced_packing}): slot {@code rank * (n / ranks) + j} holds the logical expert of that physical
     * position.
     *
     * <p>With one expert per rank the packing cannot balance anything and the experts keep their order, as the
     * reference does. Otherwise the heaviest expert takes the lightest rank with a free slot, ties going to the lower
     * expert id and the lower rank, until every rank holds {@code n / ranks} experts.</p>
     *
     * @throws IllegalArgumentException for a rank count or expert count that does not divide the experts, or a
     *                                  negative or non-finite load
     */
    static int[] phy2log(double[] load, int ranks) {
        Objects.requireNonNull(load, "load");
        if (ranks < 1 || load.length == 0 || load.length % ranks != 0) {
            throw new IllegalArgumentException("cannot pack " + load.length + " experts into " + ranks + " rank(s)");
        }
        int[][] packed = EplbPolicy.rebalance(new double[][]{load}, load.length, 1, 1, ranks, null);
        return packed[0];
    }
}
