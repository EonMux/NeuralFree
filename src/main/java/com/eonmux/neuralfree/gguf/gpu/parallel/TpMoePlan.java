package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;
import com.eonmux.neuralfree.gguf.quant.InputTransform;
import com.eonmux.neuralfree.gguf.quant.InputTransformLoader;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The routed-expert half of a tensor-parallel split (vLLM's tensor-parallel fused MoE): every ranked MoE trunk layer
 * shards each expert's intermediate width over the ranks. Rank {@code r} holds hidden rows
 * {@code [r * n_ff_exp / ranks, (r + 1) * n_ff_exp / ranks)} of every expert's gate and up projections and the
 * matching columns of its down projection, so each rank computes {@code down_r(act(gate_r x, up_r x))} for every
 * routed expert and an all-reduce sums the partials. The router ({@code ffn_gate_inp}), the per-expert scales and the
 * routing itself stay whole and identical on every rank, so every rank selects the same experts. A shared expert is
 * sharded like a dense FFN, including its hidden-width biases; its down bias stays on rank 0 because the all-reduce
 * sums the partials.
 *
 * <p>With expert parallel ({@code neuralfree.expert.parallel=true}, vLLM's {@code --enable-expert-parallel}) the
 * routed experts are placed whole instead: rank {@code r} holds the contiguous experts
 * {@code [r * n_expert / ranks, (r + 1) * n_expert / ranks)} of every expert tensor at their full intermediate width.
 * The router still runs on every rank and selects the same experts; a rank computes only the pairs of its experts,
 * zeroes the others and the group all-reduce sums the ranks' partial outputs. The shared expert is sharded like a
 * dense FFN in both forms, and the router, the per-expert scales, the latent projections and the norms stay whole and
 * identical on every rank.</p>
 *
 * <p>An {@link EplbPlacement} replaces the contiguous expert block with the load-balanced packing of EPLB: every rank
 * holds the same number of experts, chosen so that the recorded routed-row load of each rank is as even as possible
 * (a rank's experts need not be adjacent, so each becomes one run of whole experts). The experts keep their global
 * ids, so the router selects the same logical experts as without the placement and only the rank that computes each
 * pair changes. A layer without recorded load keeps its contiguous block.</p>
 *
 * <p>The plan refuses a chunked (grovemoe), zero-expert, expert-bias or post-expert-norm MoE, and an intermediate
 * width (or, with expert parallel, an expert count) that does not divide over the ranks. The plan is immutable.</p>
 */
public final class TpMoePlan {

    /** How one MoE tensor moves to a rank. */
    public enum Shard {
        /** The rank's intermediate-width rows of every expert's gate or up projection. */
        EXPERT_ROWS,
        /** The rank's gate and up segments of every expert of a fused {@code ffn_gate_up_exps}. */
        EXPERT_FUSED,
        /** The rank's intermediate-width columns of every expert's down projection. */
        EXPERT_DOWN,
        /** The rank's intermediate-width rows of the shared expert's gate or up projection. */
        SHARED_ROWS,
        /** The rank's segments of the shared expert's fused up projection. */
        SHARED_FUSED,
        /** The rank's intermediate-width columns of the shared expert's down projection. */
        SHARED_DOWN,
        /** The rank's slice of a shared-expert hidden-width bias. */
        SHARED_HIDDEN_VECTOR,
        /** The whole shared-expert output-width down bias, uploaded to rank 0 only. */
        SHARED_OUTPUT_VECTOR,
        /** The rank's whole experts of a routed expert tensor (expert parallel). */
        EXPERT_SLICE
    }

    /** One run of adjacent whole experts a rank holds (expert parallel and its EPLB packing). */
    public static final class ExpertRun {
        private final int expert;
        private final int count;

        /**
         * @throws IllegalArgumentException for a negative expert or a count below one
         */
        public ExpertRun(int expert, int count) {
            if (expert < 0) {
                throw new IllegalArgumentException("run expert must be >= 0, got " + expert);
            }
            if (count < 1) {
                throw new IllegalArgumentException("run count must be positive, got " + count);
            }
            this.expert = expert;
            this.count = count;
        }

        /** First expert of the run (its global expert id). */
        public int expert() { return expert; }

        /** Experts the run holds. */
        public int count() { return count; }

        /** One past the run's last expert. */
        public int end() { return expert + count; }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof ExpertRun)) {
                return false;
            }
            ExpertRun other = (ExpertRun) o;
            return expert == other.expert && count == other.count;
        }

        @Override
        public int hashCode() {
            return 31 * expert + count;
        }

        @Override
        public String toString() {
            return "experts [" + expert + ", " + end() + ")";
        }
    }

    /** The split experts of one layer. */
    public static final class Layer {
        private final String stack;
        private final int weightLayer;
        private final int il;
        private final int nFf;
        private final int shard;
        private final int nFfShared;
        private final int sharedShard;
        private final int nExpert;
        private final int expertCount;
        private final boolean expertParallel;
        private final boolean balanced;
        private final boolean rearrangeable;
        private final int physicalExperts;
        private final int[] phyToLog;
        private final int[][] logToPhy;
        private final int[] replicaCount;
        private final int maxReplicas;
        private final List<List<ExpertRun>> rankRuns;
        private final Map<String, Shard> tensors;
        private final List<String> support;

        Layer(String stack, int weightLayer, int il, int nFf, int shard, int nFfShared, int sharedShard,
              int nExpert, int expertCount, boolean expertParallel, boolean balanced, boolean rearrangeable,
              int physicalExperts, int[] phyToLog, int[][] logToPhy, int[] replicaCount, int maxReplicas,
              List<List<ExpertRun>> rankRuns, Map<String, Shard> tensors, List<String> support) {
            this.stack = stack;
            this.weightLayer = weightLayer;
            this.il = il;
            this.nFf = nFf;
            this.shard = shard;
            this.nFfShared = nFfShared;
            this.sharedShard = sharedShard;
            this.nExpert = nExpert;
            this.expertCount = expertCount;
            this.expertParallel = expertParallel;
            this.balanced = balanced;
            this.rearrangeable = rearrangeable;
            this.physicalExperts = physicalExperts;
            this.phyToLog = phyToLog == null ? null : phyToLog.clone();
            this.logToPhy = copyMatrix(logToPhy);
            this.replicaCount = replicaCount == null ? null : replicaCount.clone();
            this.maxReplicas = maxReplicas;
            this.rankRuns = Collections.unmodifiableList(copyRuns(rankRuns));
            this.tensors = Collections.unmodifiableMap(new LinkedHashMap<>(tensors));
            this.support = Collections.unmodifiableList(new ArrayList<>(support));
        }

        private static List<List<ExpertRun>> copyRuns(List<List<ExpertRun>> runs) {
            List<List<ExpertRun>> out = new ArrayList<>(runs.size());
            for (List<ExpertRun> rank : runs) {
                out.add(Collections.unmodifiableList(new ArrayList<>(rank)));
            }
            return out;
        }

        private static int[][] copyMatrix(int[][] values) {
            if (values == null) {
                return null;
            }
            int[][] out = new int[values.length][];
            for (int i = 0; i < values.length; i++) {
                out[i] = values[i].clone();
            }
            return out;
        }

        public String stack() { return stack; }
        public int weightLayer() { return weightLayer; }
        public int il() { return il; }
        /** Routed expert width {@code n_ff_exp}. */
        public int nFf() { return nFf; }
        /** Intermediate rows each rank holds of every routed expert. */
        public int shard() { return shard; }
        /** First intermediate row rank {@code rank} holds of every routed expert. */
        public int start(int rank) { return rank * shard; }
        /** Shared-expert width, or 0 without a shared expert. */
        public int nFfShared() { return nFfShared; }
        /** Intermediate rows each rank holds of the shared expert. */
        public int sharedShard() { return sharedShard; }
        /** First shared-expert row rank {@code rank} holds. */
        public int sharedStart(int rank) { return rank * sharedShard; }
        /** True when the routed experts are placed whole ({@link Shard#EXPERT_SLICE}). */
        public boolean expertParallel() { return expertParallel; }
        /** True when the whole experts are packed by recorded load ({@link EplbPlacement}) rather than in a block. */
        public boolean balanced() { return balanced; }
        /** True when the runtime rearrangement may re-pack this layer in place ({@code neuralfree.eplb.rebalance}). */
        public boolean rearrangeable() { return rearrangeable; }
        /** The model's routed expert count. */
        public int nExpert() { return nExpert; }
        /** Whole experts each rank holds under expert parallel, or 0 for the intermediate-width split. */
        public int expertCount() { return expertCount; }
        /** Physical slots of the layer's routed experts (logical experts plus redundant replicas), or 0. */
        public int physicalExperts() { return physicalExperts; }
        /** The physical slots each rank holds (its bank width). */
        public int slotsPerRank() { return expertCount; }
        /** True when a logical router id must be remapped to a physical replica slot before the experts. */
        public boolean remaps() { return phyToLog != null && physicalExperts > 0
                && (balanced || physicalExperts != nExpert || rearrangeable); }
        /** The physical slot layout: the logical expert each slot reproduces; null without a physical layout. */
        public int[] phyToLog() { return phyToLog == null ? null : phyToLog.clone(); }
        /** The logical expert of physical slot {@code slot}; null without a physical layout. */
        public int logicalOfSlot(int slot) {
            if (phyToLog == null) {
                throw new IllegalStateException("layer " + il + " has no physical expert layout");
            }
            return phyToLog[slot];
        }
        /** {@code [logical][maxReplicas]}: the physical slots of each logical expert, -1 where unused. */
        public int[][] logicalToPhysical() { return copyMatrix(logToPhy); }
        /** {@code [logical]}: the physical replica count of each logical expert; null without a physical layout. */
        public int[] replicaCount() { return replicaCount == null ? null : replicaCount.clone(); }
        /** The largest replica count of a logical expert. */
        public int maxReplicas() { return maxReplicas; }
        /** The physical slot of every logical expert's mapping, {@code [logical * maxReplicas]}, -1 unused. */
        public int[] logicalToPhysicalFlat() {
            int[] out = new int[logToPhy.length * maxReplicas];
            for (int e = 0; e < logToPhy.length; e++) {
                System.arraycopy(logToPhy[e], 0, out, e * maxReplicas, maxReplicas);
            }
            return out;
        }
        /**
         * First whole expert rank {@code rank} holds under the contiguous expert-parallel block; meaningless for a
         * {@link #balanced()} layer, whose experts are not adjacent.
         */
        public int expertStart(int rank) { return rank * expertCount; }
        /** First physical slot rank {@code rank} holds. */
        public int slotStart(int rank) { return rank * expertCount; }
        /** The runs of adjacent whole experts rank {@code rank} holds (empty without expert parallel). */
        public List<ExpertRun> expertRuns(int rank) {
            if (rank < 0 || rank >= rankRuns.size()) {
                throw new IllegalArgumentException("rank " + rank + " outside [0, " + rankRuns.size() + ")");
            }
            return rankRuns.get(rank);
        }
        /** The tensor suffixes this layer shards, in model order. */
        public Map<String, Shard> tensors() { return tensors; }
        /** Full names of the layer's MoE tensors every rank holds whole (router, scales, latent projections). */
        public List<String> support() { return support; }
        /** The full tensor name of a suffix. */
        public String tensor(String suffix) { return TensorNames.layerName(stack, weightLayer, suffix); }

        /** The layer's MoE spec with this rank's intermediate widths. */
        public MoeSpec shardSpec(MoeSpec of, int rank) {
            MoeSpec.Builder b = MoeSpec.builder(of).experts(of.nExpert(), of.nUsed(),
                    expertParallel ? of.nFfExp() : shard);
            FfnSpec shared = of.sharedExpert();
            if (shared != null) {
                b.sharedExpert(sharedSpec(shared, sharedShard), of.sharedExpertGated());
            }
            return b.build();
        }

        @Override
        public String toString() {
            return stack + "." + il + (balanced
                    ? " (experts " + nExpert + " -> " + expertCount + " balanced per rank, shared " + nFfShared
                    + " -> " + sharedShard + ")"
                    : (expertParallel
                    ? " (experts " + nExpert + " -> " + expertCount + " whole per rank, shared " + nFfShared
                    + " -> " + sharedShard + ")"
                    : " (n_ff_exp " + nFf + " -> " + shard + ", shared " + nFfShared + " -> " + sharedShard + ")"));
        }
    }

    private final int ranks;
    private final List<Layer> layers;
    private final Map<String, Layer> byLayer;
    private final Map<String, Layer> byTensor;

    private TpMoePlan(int ranks, List<Layer> layers, Map<String, Layer> byLayer, Map<String, Layer> byTensor) {
        this.ranks = ranks;
        this.layers = layers;
        this.byLayer = byLayer;
        this.byTensor = byTensor;
    }

    /** Number of ranks the expert width divides over. */
    public int ranks() { return ranks; }

    /** The split layers, in model order (unmodifiable). */
    public List<Layer> layers() { return Collections.unmodifiableList(layers); }

    /** The split MoE of a layer, or null for a layer the plan does not cover. */
    public Layer layerOf(LayerSpec layer) { return byLayer.get(layerKey(layer)); }

    /** True when a tensor uploads as per-rank expert shards. */
    public boolean splits(String tensorName) { return byTensor.containsKey(tensorName); }

    /**
     * Plans the routed-expert split of {@code spec}'s trunk over {@code ranks} ranks.
     *
     * @throws UnsupportedOperationException when a MoE layer cannot shard; the message names the first disqualifier
     * @throws IllegalArgumentException      for fewer than two ranks or null arguments
     */
    public static TpMoePlan of(ModelSpec spec, int ranks, GpuWeightUploader uploader) {
        return of(spec, ranks, uploader, false, null);
    }

    /**
     * Plans the routed-expert split of {@code spec}'s trunk over {@code ranks} ranks.
     *
     * @param expertParallel whether each rank holds whole experts (expert parallel) instead of an intermediate-width
     *                       slice of every expert
     * @throws UnsupportedOperationException when a MoE layer cannot shard; the message names the first disqualifier
     * @throws IllegalArgumentException      for fewer than two ranks or null arguments
     */
    public static TpMoePlan of(ModelSpec spec, int ranks, GpuWeightUploader uploader, boolean expertParallel) {
        return of(spec, ranks, uploader, expertParallel, null);
    }

    /**
     * Plans the routed-expert split of {@code spec}'s trunk over {@code ranks} ranks, placing the whole experts of
     * the layers the placement covers by its load-balanced packing (EPLB).
     *
     * @param expertParallel whether each rank holds whole experts (expert parallel) instead of an intermediate-width
     *                       slice of every expert
     * @param placement      the load-balanced expert placement, or null for the contiguous whole-expert blocks
     * @throws UnsupportedOperationException when a MoE layer cannot shard; the message names the first disqualifier
     * @throws IllegalArgumentException      for fewer than two ranks, null arguments, a placement over another rank
     *                                       count, or a placement without expert parallel
     */
    public static TpMoePlan of(ModelSpec spec, int ranks, GpuWeightUploader uploader, boolean expertParallel,
                               EplbPlacement placement) {
        return of(spec, ranks, uploader, expertParallel, placement, false);
    }

    /**
     * Plans the routed-expert split of {@code spec}'s trunk over {@code ranks} ranks, placing the whole experts of
     * the layers the placement covers by its load-balanced packing (EPLB).
     *
     * @param expertParallel whether each rank holds whole experts (expert parallel) instead of an intermediate-width
     *                       slice of every expert
     * @param placement      the load-balanced expert placement, or null for the contiguous whole-expert blocks
     * @param rearrangeable  whether the runtime expert rearrangement may re-pack the layers: every expert-parallel
     *                       layer is built remapped (identity initially) so a rearrangement can rewrite its
     *                       routing map and migrate expert rows in place
     * @throws UnsupportedOperationException when a MoE layer cannot shard; the message names the first disqualifier
     * @throws IllegalArgumentException      for fewer than two ranks, null arguments, a placement over another rank
     *                                       count, a placement without expert parallel, or a rearrangeable split
     *                                       without expert parallel
     */
    public static TpMoePlan of(ModelSpec spec, int ranks, GpuWeightUploader uploader, boolean expertParallel,
                               EplbPlacement placement, boolean rearrangeable) {
        if (ranks < 2) {
            throw new IllegalArgumentException("a tensor split needs at least two ranks, got " + ranks);
        }
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(uploader, "uploader");
        if (placement != null) {
            if (!expertParallel) {
                throw new IllegalArgumentException("an expert placement needs expert parallel: every rank holds"
                        + " whole experts, so set " + EngineConfig.PROP_EXPERT_PARALLEL + "=true");
            }
            if (placement.ranks() != ranks) {
                throw new IllegalArgumentException("the expert placement packs over " + placement.ranks()
                        + " rank(s), the split over " + ranks);
            }
        }
        if (rearrangeable && !expertParallel) {
            throw new IllegalArgumentException("a rearrangeable expert split needs expert parallel: every rank"
                    + " holds whole experts a rearrangement can move, so set " + EngineConfig.PROP_EXPERT_PARALLEL
                    + "=true");
        }
        List<Layer> layers = new ArrayList<>();
        Map<String, Layer> byLayer = new LinkedHashMap<>();
        for (LayerSpec l : spec.layers()) {
            if (!bearsMoe(l)) {
                continue;
            }
            String reason = ineligible(l, ranks, uploader, expertParallel);
            if (reason != null) {
                throw new UnsupportedOperationException("split-mode tensor cannot split the expert FFN of "
                        + spec.arch().ggufName() + " layer " + describe(l) + ": " + reason);
            }
            Layer layer = layer(l, ranks, uploader, expertParallel, placement, rearrangeable);
            layers.add(layer);
            byLayer.put(layerKey(l), layer);
        }
        return new TpMoePlan(ranks, layers, byLayer, byTensor(layers));
    }

    private static boolean bearsMoe(LayerSpec l) {
        return l.ffnKind() == FfnKind.MOE || l.ffnKind() == FfnKind.MOE_LATENT;
    }

    private static String layerKey(LayerSpec l) {
        return l.stack() + ":" + l.weightLayer();
    }

    private static String describe(LayerSpec l) {
        return l.stack() + "." + l.il();
    }

    /** Why the layer's routed experts cannot shard, or null when they can. */
    private static String ineligible(LayerSpec l, int ranks, GpuWeightUploader uploader, boolean expertParallel) {
        if (!TensorNames.STACK_BLK.equals(l.stack())) {
            return "only the main trunk splits; the " + l.stack() + " stack keeps its experts whole";
        }
        MoeSpec moe = l.moe();
        if (moe == null) {
            return "the layer has no MoE spec";
        }
        if (moe.chunked()) {
            return "grovemoe chunk experts are not tensor-split";
        }
        if (moe.zeroExperts() > 0) {
            return "zero experts duplicate the expert input and would be summed once per rank";
        }
        if (moe.expertBiases()) {
            return "per-expert biases are not tensor-split";
        }
        if (moe.postExpertNorm()) {
            return "a norm over each rank's expert partial is not the norm of the summed output";
        }
        GpuWeightUploader.TensorPlan router = uploader.plan(layerTensor(l, TensorNames.weight(TensorNames.FFN_GATE_INP)));
        if (router == null) {
            return "the model has no " + TensorNames.weight(TensorNames.FFN_GATE_INP);
        }
        GpuWeightUploader.TensorPlan down = uploader.plan(layerTensor(l, TensorNames.weight(TensorNames.FFN_DOWN_EXPS)));
        if (down == null) {
            return "the model has no " + TensorNames.weight(TensorNames.FFN_DOWN_EXPS);
        }
        String reason = expertParallel
                ? wholeExpertShapes(l, moe, down, ranks, uploader)
                : expertShapes(l, moe, down, ranks, uploader);
        if (reason != null) {
            return reason;
        }
        FfnSpec shared = moe.sharedExpert();
        if (shared != null) {
            String sharedReason = sharedShapes(l, shared, ranks, uploader);
            if (sharedReason != null) {
                return sharedReason;
            }
        }
        return null;
    }

    /**
     * Why the routed experts cannot be placed whole over the ranks, or null: the expert count must divide, every
     * expert tensor must hold the spec's full expert count and rows, and the width stays whole so no block alignment
     * applies. A PRISM transform stays valid because each rank keeps whole experts.
     */
    private static String wholeExpertShapes(LayerSpec l, MoeSpec moe, GpuWeightUploader.TensorPlan down, int ranks,
                                            GpuWeightUploader uploader) {
        int nFf = moe.nFfExp();
        if (nFf <= 0 || down.ne().length != 3 || down.ne()[0] != nFf) {
            return down.name() + " does not hold the layer's n_ff_exp " + nFf;
        }
        if (down.ne()[2] != moe.nExpert()) {
            return down.name() + " holds " + down.ne()[2] + " experts, the spec says " + moe.nExpert();
        }
        if (moe.nExpert() % ranks != 0) {
            return "n_expert " + moe.nExpert() + " does not divide into " + ranks + " rank(s)";
        }
        if (moe.fusedGateUp()) {
            GpuWeightUploader.TensorPlan fused = uploader.plan(
                    layerTensor(l, TensorNames.weight(TensorNames.FFN_GATE_UP_EXPS)));
            if (fused == null || fused.ne().length != 3 || fused.ne()[1] != 2L * nFf
                    || fused.ne()[2] != moe.nExpert()) {
                return "the fused " + TensorNames.weight(TensorNames.FFN_GATE_UP_EXPS) + " does not hold 2 * n_ff_exp"
                        + " rows for each of " + moe.nExpert() + " experts";
            }
        } else {
            GpuWeightUploader.TensorPlan up = uploader.plan(layerTensor(l, TensorNames.weight(TensorNames.FFN_UP_EXPS)));
            if (up == null || up.ne().length != 3 || up.ne()[1] != nFf || up.ne()[2] != moe.nExpert()) {
                return "the " + TensorNames.weight(TensorNames.FFN_UP_EXPS) + " does not hold n_ff_exp rows for each"
                        + " of " + moe.nExpert() + " experts";
            }
            if (moe.gated()) {
                GpuWeightUploader.TensorPlan gate = uploader.plan(layerTensor(l, TensorNames.weight(TensorNames.FFN_GATE_EXPS)));
                if (gate == null || gate.ne().length != 3 || gate.ne()[1] != nFf
                        || gate.ne()[2] != moe.nExpert()) {
                    return "the " + TensorNames.weight(TensorNames.FFN_GATE_EXPS) + " does not hold n_ff_exp rows"
                            + " for each of " + moe.nExpert() + " experts";
                }
            }
        }
        return null;
    }

    private static String expertShapes(LayerSpec l, MoeSpec moe, GpuWeightUploader.TensorPlan down, int ranks,
                                       GpuWeightUploader uploader) {
        int nFf = moe.nFfExp();
        if (down.ne().length != 3 || down.ne()[0] != nFf) {
            return down.name() + " reads " + (down.ne().length == 3 ? down.ne()[0] : 0) + " values per expert,"
                    + " expected n_ff_exp " + nFf;
        }
        if (down.ne()[2] != moe.nExpert()) {
            return down.name() + " holds " + down.ne()[2] + " experts, the spec says " + moe.nExpert();
        }
        String reason = divides(nFf, ranks, down, "n_ff_exp");
        if (reason != null) {
            return reason;
        }
        if (moe.fusedGateUp()) {
            GpuWeightUploader.TensorPlan fused = uploader.plan(
                    layerTensor(l, TensorNames.weight(TensorNames.FFN_GATE_UP_EXPS)));
            if (fused == null || fused.ne().length != 3 || fused.ne()[1] != 2L * nFf) {
                return "the fused " + TensorNames.weight(TensorNames.FFN_GATE_UP_EXPS) + " does not hold 2 * n_ff_exp"
                        + " rows per expert";
            }
            reason = transform(uploader, fused);
            if (reason != null) {
                return reason;
            }
        } else {
            GpuWeightUploader.TensorPlan up = uploader.plan(
                    layerTensor(l, TensorNames.weight(TensorNames.FFN_UP_EXPS)));
            if (up == null || up.ne().length != 3 || up.ne()[1] != nFf) {
                return "the " + TensorNames.weight(TensorNames.FFN_UP_EXPS) + " does not hold n_ff_exp rows per"
                        + " expert";
            }
            reason = transform(uploader, up);
            if (reason != null) {
                return reason;
            }
            if (moe.gated()) {
                GpuWeightUploader.TensorPlan gate = uploader.plan(
                        layerTensor(l, TensorNames.weight(TensorNames.FFN_GATE_EXPS)));
                if (gate == null || gate.ne().length != 3 || gate.ne()[1] != nFf) {
                    return "the " + TensorNames.weight(TensorNames.FFN_GATE_EXPS) + " does not hold n_ff_exp rows"
                            + " per expert";
                }
                reason = transform(uploader, gate);
                if (reason != null) {
                    return reason;
                }
            }
        }
        return transform(uploader, down);
    }

    private static String sharedShapes(LayerSpec l, FfnSpec shared, int ranks, GpuWeightUploader uploader) {
        GpuWeightUploader.TensorPlan down = uploader.plan(layerTensor(l, shared.downWeight()));
        if (down == null || down.ne().length < 2 || down.ne()[0] != shared.nFf()) {
            return "the shared expert's " + shared.downWeight() + " does not read n_ff " + shared.nFf();
        }
        String reason = divides(shared.nFf(), ranks, down, "the shared expert's n_ff");
        if (reason != null) {
            return reason;
        }
        GpuWeightUploader.TensorPlan up = uploader.plan(layerTensor(l, shared.upWeight()));
        if (up == null || up.ne().length < 2
                || (up.ne()[1] != shared.nFf() && up.ne()[1] != 2L * shared.nFf())) {
            return "the shared expert's " + shared.upWeight() + " does not hold n_ff or 2 * n_ff rows";
        }
        if (shared.gated() && up.ne()[1] == shared.nFf()) {
            GpuWeightUploader.TensorPlan gate = uploader.plan(layerTensor(l, shared.gateWeight()));
            if (gate == null || gate.ne().length < 2 || gate.ne()[1] != shared.nFf()) {
                return "the shared expert's " + shared.gateWeight() + " does not hold n_ff rows";
            }
        }
        for (GpuWeightUploader.TensorPlan p : new GpuWeightUploader.TensorPlan[]{up, down}) {
            reason = transform(uploader, p);
            if (reason != null) {
                return reason;
            }
        }
        return null;
    }

    private static String divides(int nFf, int ranks, GpuWeightUploader.TensorPlan down, String what) {
        long block = down.deviceType().getBlockSize();
        if (nFf % (ranks * block) != 0) {
            return what + " " + nFf + " does not divide into " + ranks + " rank shard(s) aligned to whole "
                    + down.deviceType() + " blocks of " + block + " element(s)";
        }
        return null;
    }

    private static String transform(GpuWeightUploader uploader, GpuWeightUploader.TensorPlan p) {
        InputTransformLoader.Result transforms = uploader.transforms();
        if (transforms == null || transforms.isEmpty()) {
            return null;
        }
        InputTransform t = transforms.forward(p.name());
        if (t != null && !t.isIdentity()) {
            return "the PRISM input transform of " + p.name() + " is built for the full expert width";
        }
        HadamardSpec hadamard = transforms.hadamard(p.name());
        if (hadamard != null) {
            return "the Hadamard rotation of " + p.name() + " is built for the full expert width";
        }
        return null;
    }

    /** The split experts of one eligible layer. */
    private static Layer layer(LayerSpec l, int ranks, GpuWeightUploader uploader, boolean expertParallel,
                               EplbPlacement placement, boolean rearrangeable) {
        MoeSpec moe = l.moe();
        int nFf = moe.nFfExp();
        int shard = expertParallel ? nFf : nFf / ranks;
        Map<String, Shard> tensors = new LinkedHashMap<>();
        if (moe.fusedGateUp()) {
            tensors.put(TensorNames.weight(TensorNames.FFN_GATE_UP_EXPS),
                    expertParallel ? Shard.EXPERT_SLICE : Shard.EXPERT_FUSED);
        } else {
            tensors.put(TensorNames.weight(TensorNames.FFN_UP_EXPS),
                    expertParallel ? Shard.EXPERT_SLICE : Shard.EXPERT_ROWS);
            if (moe.gated()) {
                tensors.put(TensorNames.weight(TensorNames.FFN_GATE_EXPS),
                        expertParallel ? Shard.EXPERT_SLICE : Shard.EXPERT_ROWS);
            }
        }
        tensors.put(TensorNames.weight(TensorNames.FFN_DOWN_EXPS),
                expertParallel ? Shard.EXPERT_SLICE : Shard.EXPERT_DOWN);
        int nFfShared = 0;
        int sharedShard = 0;
        FfnSpec shared = moe.sharedExpert();
        if (shared != null) {
            nFfShared = shared.nFf();
            sharedShard = nFfShared / ranks;
            GpuWeightUploader.TensorPlan up = uploader.plan(layerTensor(l, shared.upWeight()));
            if (up.ne()[1] == 2L * nFfShared) {
                tensors.put(shared.upWeight(), Shard.SHARED_FUSED);
            } else {
                tensors.put(shared.upWeight(), Shard.SHARED_ROWS);
                if (shared.gated()) {
                    tensors.put(shared.gateWeight(), Shard.SHARED_ROWS);
                }
            }
            tensors.put(shared.downWeight(), Shard.SHARED_DOWN);
            if (shared.biases()) {
                addVector(tensors, TensorNames.bias(shared.role().up()), uploader, l, Shard.SHARED_HIDDEN_VECTOR);
                if (shared.gated() && up.ne()[1] == nFfShared) {
                    addVector(tensors, TensorNames.bias(shared.role().gate()), uploader, l,
                            Shard.SHARED_HIDDEN_VECTOR);
                }
                String downBias = TensorNames.bias(shared.role().down());
                if (uploader.plan(layerTensor(l, downBias)) != null) {
                    tensors.put(downBias, Shard.SHARED_OUTPUT_VECTOR);
                }
            }
        }
        boolean covered = expertParallel && placement != null && placement.covers(l);
        int nExpert = moe.nExpert();
        int physical = expertParallel ? (covered ? placement.physicalExperts(l) : nExpert) : 0;
        int[] phyToLog = null;
        int[][] logToPhy = null;
        int[] replicaCount = null;
        int maxReplicas = 1;
        if (expertParallel) {
            if (covered) {
                phyToLog = new int[physical];
                for (int slot = 0; slot < physical; slot++) {
                    phyToLog[slot] = placement.logicalOf(l, slot);
                }
                logToPhy = placement.logicalToPhysical(l);
                replicaCount = placement.replicaCount(l);
                maxReplicas = placement.maxReplicas(l);
            } else {
                phyToLog = new int[nExpert];
                logToPhy = new int[nExpert][1];
                replicaCount = new int[nExpert];
                for (int e = 0; e < nExpert; e++) {
                    phyToLog[e] = e;
                    logToPhy[e][0] = e;
                    replicaCount[e] = 1;
                }
            }
        }
        return new Layer(l.stack(), l.weightLayer(), l.il(), nFf, shard, nFfShared, sharedShard, nExpert,
                expertParallel ? physical / ranks : 0, expertParallel, covered, expertParallel && rearrangeable,
                physical, phyToLog, logToPhy,
                replicaCount, maxReplicas, rankRuns(ranks, expertParallel, phyToLog, physical),
                tensors, support(l, tensors, uploader));
    }

    /** The runs of adjacent logical experts each rank holds, empty without expert parallel. */
    private static List<List<ExpertRun>> rankRuns(int ranks, boolean expertParallel, int[] phyToLog,
                                                  int physical) {
        List<List<ExpertRun>> out = new ArrayList<>(ranks);
        if (!expertParallel) {
            for (int rank = 0; rank < ranks; rank++) {
                out.add(Collections.<ExpertRun>emptyList());
            }
            return out;
        }
        int count = physical / ranks;
        for (int rank = 0; rank < ranks; rank++) {
            int[] experts = new int[count];
            for (int i = 0; i < count; i++) {
                experts[i] = phyToLog[rank * count + i];
            }
            Arrays.sort(experts);
            out.add(runs(experts));
        }
        return out;
    }

    /** The runs of adjacent experts of an ascending expert list. */
    private static List<ExpertRun> runs(int[] experts) {
        List<ExpertRun> out = new ArrayList<>();
        int i = 0;
        while (i < experts.length) {
            int start = experts[i];
            int n = 1;
            while (i + n < experts.length && experts[i + n] == start + n) {
                n++;
            }
            out.add(new ExpertRun(start, n));
            i += n;
        }
        return out;
    }

    /** The layer's MoE tensors the split leaves whole on every rank, in model order. */
    private static List<String> support(LayerSpec l, Map<String, Shard> tensors, GpuWeightUploader uploader) {
        String prefix = l.stack() + "." + l.weightLayer() + ".";
        List<String> out = new ArrayList<>();
        for (String name : uploader.plans().keySet()) {
            if (!name.startsWith(prefix)) {
                continue;
            }
            String suffix = name.substring(prefix.length());
            if (!tensors.containsKey(suffix)
                    && (suffix.startsWith("ffn_") || suffix.startsWith(TensorNames.FFN_EXP_PROBS_B))) {
                out.add(name);
            }
        }
        return out;
    }

    private static void addVector(Map<String, Shard> tensors, String suffix,
                                  GpuWeightUploader uploader, LayerSpec l, Shard shard) {
        GpuWeightUploader.TensorPlan p = uploader.plan(layerTensor(l, suffix));
        if (p != null) {
            tensors.put(suffix, shard);
        }
    }

    /** A shared expert's FfnSpec with this rank's {@code n_ff}. */
    static FfnSpec sharedSpec(FfnSpec of, int nFf) {
        FfnSpec.Builder b = FfnSpec.builder(of.role()).nFf(nFf);
        if (of.usesGlu()) {
            b.glu(of.gating(), of.glu());
        } else {
            b.act(of.gated(), of.act());
        }
        b.params(of.params()).biases(of.biases()).activationSparsityStdMul(of.activationSparsityStdMul());
        return b.build();
    }

    private static String layerTensor(LayerSpec l, String suffix) {
        return TensorNames.layerName(l.stack(), l.weightLayer(), suffix);
    }


    private static GpuWeightUploader.TensorPlan plan(GpuWeightUploader uploader, String name) {
        return uploader.plan(name);
    }

    private static Map<String, Layer> byTensor(List<Layer> layers) {
        Map<String, Layer> out = new LinkedHashMap<>();
        for (Layer layer : layers) {
            for (String suffix : layer.tensors.keySet()) {
                out.put(layer.tensor(suffix), layer);
            }
        }
        return Collections.unmodifiableMap(out);
    }
}
