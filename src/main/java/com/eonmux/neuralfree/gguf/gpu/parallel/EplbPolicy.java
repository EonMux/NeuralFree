package com.eonmux.neuralfree.gguf.gpu.parallel;

import java.util.Objects;

/**
 * vLLM's expert-placement policies ({@code vllm/distributed/eplb/policy/default.py}): the weighted packing of experts
 * into packs of equal size, the replication of heavy logical experts into redundant physical slots, and the
 * hierarchical rearrangement that keeps each node's groups together before spreading the physical slots over the
 * ranks.
 *
 * <p>Every method works on {@code [layers][experts]} rows and follows the reference's arithmetic: the packing takes
 * the heaviest item first into the lightest pack that still has a free slot, replication adds each redundant slot to
 * the logical expert with the largest load per replica, and the hierarchical policy packs groups to nodes, replicates
 * within each node, then packs the node's physical slots to its ranks. Ties go to the lower expert, pack, node or
 * rank, which makes the result deterministic where NumPy's quicksort order is unspecified.</p>
 *
 * <p>The public entry point is {@link #rebalance(double[][], int, int, int, int, int[][])}: it chooses the
 * hierarchical policy when the group count divides over the node count (as the reference does), else the global
 * policy over one group and one node, and optionally preserves the slots of experts that stay on the same rank. All
 * methods are pure and stateless.</p>
 */
public final class EplbPolicy {

    private EplbPolicy() {
    }

    /** The result of {@link #balancedPacking}: each item's pack and its slot inside that pack. */
    public static final class Packing {
        private final int[][] packIndex;
        private final int[][] rankInPack;

        private Packing(int[][] packIndex, int[][] rankInPack) {
            this.packIndex = packIndex;
            this.rankInPack = rankInPack;
        }

        /** {@code [layers][items]}: the pack of each item. */
        public int[][] packIndex() { return copy(packIndex); }

        /** {@code [layers][items]}: the position of each item inside its pack. */
        public int[][] rankInPack() { return copy(rankInPack); }

        int packOf(int layer, int item) { return packIndex[layer][item]; }

        int slotOf(int layer, int item) { return rankInPack[layer][item]; }
    }

    /** The physical layout of one layer's experts: each physical slot's logical expert and each expert's replicas. */
    public static final class Layout {
        private final int[][] phyToLog;
        private final int[][][] logToPhy;
        private final int[][] replicaCount;

        private Layout(int[][] phyToLog, int[][][] logToPhy, int[][] replicaCount) {
            this.phyToLog = phyToLog;
            this.logToPhy = logToPhy;
            this.replicaCount = replicaCount;
        }

        /** {@code [layers][physical]}: the logical expert of each physical slot. */
        public int[][] phyToLog() { return copy(phyToLog); }

        /** {@code [logical][replica]}: the physical slots of each logical expert of one layer, -1 where unused. */
        public int[][] logicalToPhysical(int layer) {
            int[][] in = logToPhy[layer];
            int[][] out = new int[in.length][];
            for (int i = 0; i < in.length; i++) {
                out[i] = in[i].clone();
            }
            return out;
        }

        /** {@code [layers][logical]}: the number of physical replicas of each logical expert. */
        public int[][] replicaCount() { return copy(replicaCount); }

        /** The physical slot count of every layer. */
        public int physical() { return phyToLog[0].length; }

        /** The logical expert count of every layer. */
        public int logical() { return logToPhy[0].length; }
    }

    /**
     * Packs the items of every layer into {@code numPacks} packs of equal size, balanced by weight
     * ({@code DefaultEplbPolicy.balanced_packing}).
     *
     * @param weight    {@code [layers][items]} non-negative item weights
     * @param numPacks  packs per layer; must divide the item count
     * @throws IllegalArgumentException when the shapes are inconsistent, the packs do not divide, or a weight is
     *                                  negative or non-finite
     */
    public static Packing balancedPacking(double[][] weight, int numPacks) {
        requireRectangular(weight, "weight");
        int layers = weight.length;
        int items = weight[0].length;
        if (numPacks < 1 || items % numPacks != 0) {
            throw new IllegalArgumentException("cannot pack " + items + " items into " + numPacks + " pack(s)");
        }
        int groupsPerPack = items / numPacks;
        int[][] packIndex = new int[layers][items];
        int[][] rankInPack = new int[layers][items];
        if (groupsPerPack == 1) {
            for (int l = 0; l < layers; l++) {
                for (int i = 0; i < items; i++) {
                    packIndex[l][i] = i;
                }
            }
            return new Packing(packIndex, rankInPack);
        }
        int[][] order = descendingOrder(weight);
        for (int l = 0; l < layers; l++) {
            double[] packWeight = new double[numPacks];
            int[] packItems = new int[numPacks];
            for (int item : order[l]) {
                int pack = lowest(packWeight);
                packIndex[l][item] = pack;
                rankInPack[l][item] = packItems[pack];
                packWeight[pack] += weight[l][item];
                packItems[pack]++;
                if (packItems[pack] == groupsPerPack) {
                    packWeight[pack] = Double.POSITIVE_INFINITY;
                }
            }
        }
        return new Packing(packIndex, rankInPack);
    }

    /**
     * Replicates every layer's logical experts into {@code numPhysical} physical slots so that the largest load of a
     * replica is as small as possible ({@code DefaultEplbPolicy.replicate_experts}): the first slots hold the logical
     * experts in order and each additional slot becomes a replica of the expert with the largest load per replica.
     *
     * @param weight      {@code [layers][logical]} routed-row load
     * @param numPhysical physical slots per layer, at least the logical count
     * @return {@code [layers][numPhysical]} the logical expert of each physical slot
     * @throws IllegalArgumentException when a layer is shorter than the logical count or a weight is invalid
     */
    public static int[][] replicateExperts(double[][] weight, int numPhysical) {
        requireRectangular(weight, "weight");
        int layers = weight.length;
        int logical = weight[0].length;
        if (numPhysical < logical) {
            throw new IllegalArgumentException("replication needs at least " + logical + " physical slots, got "
                    + numPhysical);
        }
        int[][] phyToLog = new int[layers][numPhysical];
        int[][] count = new int[layers][logical];
        for (int l = 0; l < layers; l++) {
            for (int p = 0; p < numPhysical; p++) {
                phyToLog[l][p] = p;
            }
            for (int e = 0; e < logical; e++) {
                count[l][e] = 1;
            }
        }
        for (int p = logical; p < numPhysical; p++) {
            for (int l = 0; l < layers; l++) {
                int best = 0;
                double bestLoad = weight[l][0] / count[l][0];
                for (int e = 1; e < logical; e++) {
                    double load = weight[l][e] / count[l][e];
                    if (load > bestLoad) {
                        bestLoad = load;
                        best = e;
                    }
                }
                phyToLog[l][p] = best;
                count[l][best]++;
            }
        }
        return phyToLog;
    }

    /**
     * The hierarchical rearrangement ({@code DefaultEplbPolicy.rebalance_experts_hierarchical}): pack the logical
     * expert groups to nodes, replicate within each node, pack each node's physical slots to its ranks, and return the
     * logical expert of every physical slot in global physical order.
     *
     * @param weight      {@code [layers][logical]} routed-row load
     * @param numPhysical physical slots per layer, a multiple of {@code numRanks}
     * @param numGroups   logical expert groups, a multiple of {@code numNodes}
     * @param numNodes    server nodes; the intra-node network is faster
     * @param numRanks    ranks (GPUs), a multiple of {@code numNodes}
     * @throws IllegalArgumentException for a shape or divisibility the policy cannot satisfy
     */
    public static int[][] rebalanceHierarchical(double[][] weight, int numPhysical, int numGroups, int numNodes,
                                                int numRanks) {
        requireRectangular(weight, "weight");
        int layers = weight.length;
        int logical = weight[0].length;
        requirePositive("num_physical", numPhysical);
        requirePositive("num_groups", numGroups);
        requirePositive("num_nodes", numNodes);
        requirePositive("num_ranks", numRanks);
        if (logical % numGroups != 0) {
            throw new IllegalArgumentException(logical + " logical experts do not form " + numGroups + " groups");
        }
        int groupSize = logical / numGroups;
        if (numGroups % numNodes != 0) {
            throw new IllegalArgumentException(numGroups + " groups do not divide over " + numNodes + " node(s)");
        }
        int groupsPerNode = numGroups / numNodes;
        if (numRanks % numNodes != 0) {
            throw new IllegalArgumentException(numRanks + " ranks do not divide over " + numNodes + " node(s)");
        }
        if (numPhysical % numRanks != 0) {
            throw new IllegalArgumentException(numPhysical + " physical experts do not divide over " + numRanks
                    + " rank(s)");
        }
        int phyPerRank = numPhysical / numRanks;
        int phyPerNode = numPhysical / numNodes;

        // Step 1: pack the logical expert groups to nodes, and reorder the logical experts node-locally.
        double[][] tokensPerGroup = new double[layers][numGroups];
        for (int l = 0; l < layers; l++) {
            for (int g = 0; g < numGroups; g++) {
                double sum = 0;
                for (int x = 0; x < groupSize; x++) {
                    sum += weight[l][g * groupSize + x];
                }
                tokensPerGroup[l][g] = sum;
            }
        }
        Packing groupPacking = balancedPacking(tokensPerGroup, numNodes);
        int[][] logToLocal = new int[layers][logical];
        for (int l = 0; l < layers; l++) {
            for (int g = 0; g < numGroups; g++) {
                int localGroup = groupPacking.packOf(l, g) * groupsPerNode + groupPacking.slotOf(l, g);
                for (int x = 0; x < groupSize; x++) {
                    logToLocal[l][g * groupSize + x] = localGroup * groupSize + x;
                }
            }
        }
        int[][] localToLog = inverse(logToLocal);

        // Step 2: replicate within each node over the node-local logical order.
        int rows = layers * numNodes;
        int localLogical = logical / numNodes;
        double[][] tokensPerLocal = new double[rows][localLogical];
        int[][] physicalOfLocal = new int[rows][phyPerNode];
        int[][] replicaCount = new int[rows][localLogical];
        for (int l = 0; l < layers; l++) {
            for (int node = 0; node < numNodes; node++) {
                int row = l * numNodes + node;
                for (int j = 0; j < localLogical; j++) {
                    tokensPerLocal[row][j] = weight[l][localToLog[l][node * localLogical + j]];
                }
            }
        }
        int[][] localPhy = replicateExperts(tokensPerLocal, phyPerNode);
        for (int r = 0; r < rows; r++) {
            System.arraycopy(localPhy[r], 0, physicalOfLocal[r], 0, phyPerNode);
        }
        double[][] loadPerLocal = new double[rows][localLogical];
        for (int r = 0; r < rows; r++) {
            for (int j = 0; j < localLogical; j++) {
                int copies = 0;
                for (int p = 0; p < phyPerNode; p++) {
                    if (physicalOfLocal[r][p] == j) {
                        copies++;
                    }
                }
                replicaCount[r][j] = copies;
                loadPerLocal[r][j] = tokensPerLocal[r][j] / copies;
            }
        }

        // Step 3: pack each node's physical slots to its ranks and map node-local logical ids back to global ids.
        double[][] tokensPerPhysical = new double[rows][phyPerNode];
        for (int r = 0; r < rows; r++) {
            for (int p = 0; p < phyPerNode; p++) {
                tokensPerPhysical[r][p] = loadPerLocal[r][physicalOfLocal[r][p]];
            }
        }
        Packing physicalPacking = balancedPacking(tokensPerPhysical, numRanks / numNodes);
        int[][] physicalToPacked = new int[rows][phyPerNode];
        for (int r = 0; r < rows; r++) {
            for (int p = 0; p < phyPerNode; p++) {
                physicalToPacked[r][p] = physicalPacking.packOf(r, p) * phyPerRank + physicalPacking.slotOf(r, p);
            }
        }
        int[][] packedToPhysical = inverse(physicalToPacked);
        int[][] phyToLog = new int[layers][numPhysical];
        for (int l = 0; l < layers; l++) {
            for (int node = 0; node < numNodes; node++) {
                int row = l * numNodes + node;
                for (int packed = 0; packed < phyPerNode; packed++) {
                    int physical = packedToPhysical[row][packed];
                    int local = physicalOfLocal[row][physical];
                    int globalLocal = local + node * localLogical;
                    phyToLog[l][node * phyPerNode + packed] = localToLog[l][globalLocal];
                }
            }
        }
        return phyToLog;
    }

    /**
     * Reorders a new physical mapping per rank so experts that stay on the same rank keep their previous slot when
     * possible ({@code DefaultEplbPolicy.preserve_intragpu_slots}); incoming experts fill the remaining slots.
     *
     * @param phyToLog the new mapping, {@code [layers][physical]}
     * @param numRanks ranks the physical slots divide over
     * @param oldPhyToLog the previous mapping of the same shape, or null to keep {@code phyToLog}
     * @return the reordered mapping
     * @throws IllegalArgumentException when the two mappings disagree in shape or do not divide over the ranks
     */
    public static int[][] preserveIntraRankSlots(int[][] phyToLog, int numRanks, int[][] oldPhyToLog) {
        Objects.requireNonNull(phyToLog, "phyToLog");
        if (oldPhyToLog == null) {
            return copy(phyToLog);
        }
        requireRectangular(phyToLog, "phyToLog");
        requireRectangular(oldPhyToLog, "oldPhyToLog");
        int layers = phyToLog.length;
        int physical = phyToLog[0].length;
        if (oldPhyToLog.length != layers || oldPhyToLog[0].length != physical) {
            throw new IllegalArgumentException("the old mapping is " + oldPhyToLog.length + " x "
                    + oldPhyToLog[0].length + ", the new one " + layers + " x " + physical);
        }
        if (numRanks < 1 || physical % numRanks != 0) {
            return copy(phyToLog);
        }
        int slots = physical / numRanks;
        int[][] out = copy(phyToLog);
        for (int rank = 0; rank < numRanks; rank++) {
            int start = rank * slots;
            for (int l = 0; l < layers; l++) {
                boolean[] used = new boolean[slots];
                boolean[] kept = new boolean[slots];
                for (int slot = 0; slot < slots; slot++) {
                    int target = oldPhyToLog[l][start + slot];
                    for (int candidate = 0; candidate < slots; candidate++) {
                        if (!used[candidate] && phyToLog[l][start + candidate] == target) {
                            out[l][start + slot] = target;
                            used[candidate] = true;
                            kept[slot] = true;
                            break;
                        }
                    }
                }
                int candidate = 0;
                for (int slot = 0; slot < slots; slot++) {
                    if (kept[slot]) {
                        continue;
                    }
                    while (candidate < slots && used[candidate]) {
                        candidate++;
                    }
                    if (candidate >= slots) {
                        break;
                    }
                    out[l][start + slot] = phyToLog[l][start + candidate];
                    used[candidate] = true;
                }
            }
        }
        return out;
    }

    /**
     * The policy entry point ({@code DefaultEplbPolicy.rebalance_experts}): the hierarchical rearrangement when
     * {@code numGroups} divides over {@code numNodes}, else the global policy over one group and one node, then the
     * optional preservation of the slots of experts that stay on the same rank.
     *
     * @param weight        {@code [layers][logical]} routed-row load
     * @param numPhysical   physical slots per layer, a multiple of {@code numRanks}
     * @param numGroups     logical expert groups
     * @param numNodes      server nodes
     * @param numRanks      ranks
     * @param oldPhyToLog   the previous mapping for slot preservation, or null
     * @return {@code [layers][numPhysical]} the logical expert of each physical slot
     */
    public static int[][] rebalance(double[][] weight, int numPhysical, int numGroups, int numNodes, int numRanks,
                                    int[][] oldPhyToLog) {
        requireRectangular(weight, "weight");
        requirePositive("num_physical", numPhysical);
        requirePositive("num_groups", numGroups);
        requirePositive("num_nodes", numNodes);
        requirePositive("num_ranks", numRanks);
        int[][] phyToLog;
        if (numGroups % numNodes == 0) {
            phyToLog = rebalanceHierarchical(weight, numPhysical, numGroups, numNodes, numRanks);
        } else {
            phyToLog = rebalanceHierarchical(weight, numPhysical, 1, 1, numRanks);
        }
        return preserveIntraRankSlots(phyToLog, numRanks, oldPhyToLog);
    }

    /** {@code [layers][logical][replica]} from a physical mapping, -1 for unused replica slots. */
    public static Layout layout(int[][] phyToLog, int logical) {
        requireRectangular(phyToLog, "phyToLog");
        requirePositive("logical", logical);
        int layers = phyToLog.length;
        int physical = phyToLog[0].length;
        int maxReplicas = 1;
        int[][] counts = new int[layers][logical];
        for (int l = 0; l < layers; l++) {
            for (int p = 0; p < physical; p++) {
                int e = phyToLog[l][p];
                if (e < 0 || e >= logical) {
                    throw new IllegalArgumentException("physical slot " + p + " of layer " + l + " names logical"
                            + " expert " + e + " outside [0, " + logical + ")");
                }
                counts[l][e]++;
                maxReplicas = Math.max(maxReplicas, counts[l][e]);
            }
        }
        int[][] logToPhy = new int[layers][logical * maxReplicas];
        for (int l = 0; l < layers; l++) {
            java.util.Arrays.fill(logToPhy[l], -1);
            int[] next = new int[logical];
            for (int p = 0; p < physical; p++) {
                int e = phyToLog[l][p];
                logToPhy[l][e * maxReplicas + next[e]++] = p;
            }
        }
        int[][][] shaped = new int[layers][logical][maxReplicas];
        for (int l = 0; l < layers; l++) {
            for (int e = 0; e < logical; e++) {
                System.arraycopy(logToPhy[l], e * maxReplicas, shaped[l][e], 0, maxReplicas);
            }
        }
        return new Layout(copy(phyToLog), shaped, counts);
    }

    /** The replica slots per logical expert, the largest over every layer and expert. */
    public static int maxReplicas(int[][] phyToLog, int logical) {
        int[][] counts = layout(phyToLog, logical).replicaCount();
        int max = 1;
        for (int[] row : counts) {
            for (int c : row) {
                max = Math.max(max, c);
            }
        }
        return max;
    }

    /**
     * The routed rows each rank computes of one layer: rank {@code r} sums the physical slots
     * {@code [r * physical / numRanks, (r + 1) * physical / numRanks)}, and a slot holding a logical expert with
     * {@code n} replicas counts the expert's load over {@code n} (the replica the hash picks is uniformly one of
     * them, so each carries its share).
     *
     * @param logicalLoad {@code [logical]} routed rows per logical expert
     * @param phyToLog    {@code [physical]} the logical expert of each physical slot
     * @throws IllegalArgumentException when the arrays disagree in length or the ranks do not divide the slots
     */
    public static double[] rankLoads(double[] logicalLoad, int[] phyToLog, int numRanks) {
        Objects.requireNonNull(logicalLoad, "logicalLoad");
        Objects.requireNonNull(phyToLog, "phyToLog");
        if (numRanks < 1 || phyToLog.length % numRanks != 0) {
            throw new IllegalArgumentException("cannot spread " + phyToLog.length + " slot(s) over " + numRanks
                    + " rank(s)");
        }
        if (logicalLoad.length < max(phyToLog) + 1) {
            throw new IllegalArgumentException("the load holds " + logicalLoad.length + " expert(s), the mapping"
                    + " names " + (max(phyToLog) + 1));
        }
        int[] replicas = new int[logicalLoad.length];
        for (int slot : phyToLog) {
            replicas[slot]++;
        }
        double[] out = new double[numRanks];
        int slots = phyToLog.length / numRanks;
        for (int slot = 0; slot < phyToLog.length; slot++) {
            int logical = phyToLog[slot];
            out[slot / slots] += logicalLoad[logical] / replicas[logical];
        }
        return out;
    }

    /**
     * The load-balance line of per-layer rank loads (vLLM's {@code _compute_eplb_load_stats}): the tokens are the
     * per-layer mean over the ranks, the maximum the per-layer maximum, both summed across the layers.
     * {@code balancedness} is the mean over the maximum.
     */
    public static String balanceLine(double[][] rankLoadsPerLayer) {
        Objects.requireNonNull(rankLoadsPerLayer, "rankLoadsPerLayer");
        if (rankLoadsPerLayer.length == 0) {
            throw new IllegalArgumentException("balanceLine of no layer");
        }
        int ranks = rankLoadsPerLayer[0].length;
        double avgSum = 0;
        double maxSum = 0;
        for (double[] layer : rankLoadsPerLayer) {
            if (layer.length != ranks) {
                throw new IllegalArgumentException("layer rank loads hold " + layer.length + " rank(s), the first"
                        + " layer " + ranks);
            }
            double sum = 0;
            double max = 0;
            for (double v : layer) {
                if (!Double.isFinite(v) || v < 0) {
                    throw new IllegalArgumentException("a rank load is " + v);
                }
                sum += v;
                max = Math.max(max, v);
            }
            avgSum += ranks == 0 ? 0 : sum / ranks;
            maxSum += max;
        }
        double balancedness = maxSum > 0 ? avgSum / maxSum : 1.0;
        return String.format(java.util.Locale.ROOT, "avg_tokens=%.2f, max_tokens=%.0f, balancedness=%.4f",
                avgSum, maxSum, balancedness);
    }

    /** The largest value of an int array, or -1 when it is empty. */
    private static int max(int[] values) {
        int out = -1;
        for (int v : values) {
            out = Math.max(out, v);
        }
        return out;
    }

    // ------------------------------------------------------------------ helpers

    /** Per layer, the item ids heaviest first, equal loads by ascending id. */
    private static int[][] descendingOrder(double[][] weight) {
        int layers = weight.length;
        int items = weight[0].length;
        int[][] order = new int[layers][items];
        for (int l = 0; l < layers; l++) {
            for (int i = 0; i < items; i++) {
                if (!Double.isFinite(weight[l][i]) || weight[l][i] < 0) {
                    throw new IllegalArgumentException("layer " + l + " item " + i + " has weight " + weight[l][i]);
                }
                order[l][i] = i;
            }
            insertionSort(order[l], weight[l]);
        }
        return order;
    }

    private static void insertionSort(int[] ids, double[] load) {
        for (int i = 1; i < ids.length; i++) {
            int item = ids[i];
            int j = i - 1;
            while (j >= 0 && (load[ids[j]] < load[item] || (load[ids[j]] == load[item] && ids[j] > item))) {
                ids[j + 1] = ids[j];
                j--;
            }
            ids[j + 1] = item;
        }
    }

    /** The lowest index of the smallest value (NumPy's {@code argmin}). */
    private static int lowest(double[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] < values[best]) {
                best = i;
            }
        }
        return best;
    }

    /** The inverse permutation of every row: {@code out[row][perm[i]] = i}. */
    private static int[][] inverse(int[][] perm) {
        int layers = perm.length;
        int n = perm[0].length;
        int[][] out = new int[layers][n];
        for (int l = 0; l < layers; l++) {
            for (int i = 0; i < n; i++) {
                int p = perm[l][i];
                if (p < 0 || p >= n) {
                    throw new IllegalArgumentException("permutation value " + p + " outside [0, " + n + ")");
                }
                out[l][p] = i;
            }
        }
        return out;
    }

    private static void requireRectangular(double[][] values, String name) {
        Objects.requireNonNull(values, name);
        if (values.length == 0 || values[0] == null || values[0].length == 0) {
            throw new IllegalArgumentException(name + " has no rows");
        }
        int width = values[0].length;
        for (int i = 0; i < values.length; i++) {
            if (values[i] == null || values[i].length != width) {
                throw new IllegalArgumentException(name + " row " + i + " is not " + width + " values wide");
            }
            for (int j = 0; j < width; j++) {
                if (!Double.isFinite(values[i][j]) || values[i][j] < 0) {
                    throw new IllegalArgumentException(name + " row " + i + " item " + j + " has weight "
                            + values[i][j]);
                }
            }
        }
    }

    private static void requireRectangular(int[][] values, String name) {
        Objects.requireNonNull(values, name);
        if (values.length == 0 || values[0] == null || values[0].length == 0) {
            throw new IllegalArgumentException(name + " has no rows");
        }
        int width = values[0].length;
        for (int i = 0; i < values.length; i++) {
            if (values[i] == null || values[i].length != width) {
                throw new IllegalArgumentException(name + " row " + i + " is not " + width + " values wide");
            }
        }
    }

    private static void requirePositive(String name, int value) {
        if (value < 1) {
            throw new IllegalArgumentException(name + " must be positive, got " + value);
        }
    }

    private static int[][] copy(int[][] values) {
        int[][] out = new int[values.length][];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i].clone();
        }
        return out;
    }
}
