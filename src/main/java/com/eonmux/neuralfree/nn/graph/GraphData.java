package com.eonmux.neuralfree.nn.graph;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Tensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * A graph: node features {@code [numNodes, featureDim]}, a COO edge index {@code [2, numEdges]} (row 0 the
 * sources, row 1 the targets), optional edge features {@code [numEdges, edgeFeatureDim]} and an optional batch index
 * {@code [numNodes]} assigning each node to a graph of a batch. Index tensors are INT32, INT64 (narrowed by Java
 * {@code (int)}) or FLOAT32 (converted by Java {@code (int)}).
 *
 * <p><b>Structure metadata</b>. The graph layers aggregate through CSR edge lists. Each
 * instance reads its edge index and batch index to the host once, on first use, builds the structure there and
 * uploads it as INT32 tensors to the node features' backend, where it stays cached until {@link #close()}:</p>
 * <ul>
 *   <li>{@link #sourceIndex()} and {@link #targetIndex()}: the two rows of the edge index, unclamped;</li>
 *   <li>{@link #csrOffsets()} and {@link #csrEdges()}: the edges grouped by target node. Only edges whose target is
 *       in {@code [0, numNodes)} are listed, grouped in ascending target order and, within a target, in ascending
 *       edge id, so an aggregation over a segment adds the messages in the order the edges appear;</li>
 *   <li>{@link #inDegree()}: the number of listed edges per target;</li>
 *   <li>{@link #symmetricNormalization()}: GCN's per-edge coefficient (FLOAT32);</li>
 *   <li>{@link #graphOffsets()} and {@link #graphNodes()}: the nodes grouped by graph of the batch.</li>
 * </ul>
 * <p>The CSR arrays satisfy the precondition of {@code TensorOps.segmentReduce}: offsets start at 0, never decrease
 * and end at the listed edge count, and every listed edge id names a row of a {@code [numEdges, ...]} message
 * tensor (every listed node id a row of the node features). They also satisfy the precondition of its backward
 * ({@code TensorOps.segmentReduceBackward}): {@link #csrEdges()} lists each edge id at most once and
 * {@link #graphNodes()} each node id at most once, because both are built by one pass over the ids that places each
 * id in at most one slot. The returned tensors are owned by this instance: callers must not close them. They are
 * never written after they are built, so the gradient functions of a recorded forward may save them; a saved handle
 * shares the storage, which stays valid until the backward even when this instance is closed first.</p>
 */
public final class GraphData implements AutoCloseable {
    private final Tensor nodeFeatures;
    private final Tensor edgeIndex;
    private final Tensor edgeFeatures;
    private final Tensor batchIndex;

    private volatile boolean closed = false;

    /** Edge structure, built on first use; guarded by {@code this}. */
    private EdgeTopology edges;
    /** Batch structure, built on first use; guarded by {@code this}. */
    private BatchTopology batches;

    public GraphData(Tensor nodeFeatures, Tensor edgeIndex) {
        this(nodeFeatures, edgeIndex, null, null);
    }

    public GraphData(Tensor nodeFeatures, Tensor edgeIndex, Tensor edgeFeatures, Tensor batchIndex) {
        Objects.requireNonNull(nodeFeatures, "nodeFeatures cannot be null");
        Objects.requireNonNull(edgeIndex, "edgeIndex cannot be null");

        validateNodeFeatures(nodeFeatures);
        validateEdgeIndex(edgeIndex);
        if (edgeFeatures != null) {
            validateEdgeFeatures(edgeFeatures, edgeIndex);
        }
        if (batchIndex != null) {
            validateBatchIndex(batchIndex, nodeFeatures);
        }

        this.nodeFeatures = nodeFeatures;
        this.edgeIndex = edgeIndex;
        this.edgeFeatures = edgeFeatures;
        this.batchIndex = batchIndex;
    }

    private void validateNodeFeatures(Tensor nodeFeatures) {
        if (nodeFeatures.shape().rank() != 2) {
            throw new IllegalArgumentException(
                "nodeFeatures must be 2D [numNodes, featureDim], got shape: " + nodeFeatures.shape());
        }
    }

    private void validateEdgeIndex(Tensor edgeIndex) {
        if (edgeIndex.shape().rank() != 2) {
            throw new IllegalArgumentException(
                "edgeIndex must be 2D [2, numEdges], got shape: " + edgeIndex.shape());
        }
        if (edgeIndex.shape().dim(0) != 2) {
            throw new IllegalArgumentException(
                "edgeIndex first dimension must be 2 (COO format), got: " + edgeIndex.shape().dim(0));
        }
        validateIndexDType(edgeIndex, "edgeIndex");
    }

    private void validateEdgeFeatures(Tensor edgeFeatures, Tensor edgeIndex) {
        if (edgeFeatures.shape().rank() != 2) {
            throw new IllegalArgumentException(
                "edgeFeatures must be 2D [numEdges, edgeFeatureDim], got shape: " + edgeFeatures.shape());
        }
        long numEdges = edgeIndex.shape().dim(1);
        if (edgeFeatures.shape().dim(0) != numEdges) {
            throw new IllegalArgumentException(
                "edgeFeatures first dimension (" + edgeFeatures.shape().dim(0) +
                ") must match number of edges (" + numEdges + ")");
        }
    }

    private void validateBatchIndex(Tensor batchIndex, Tensor nodeFeatures) {
        if (batchIndex.shape().rank() != 1) {
            throw new IllegalArgumentException(
                "batchIndex must be 1D [numNodes], got shape: " + batchIndex.shape());
        }
        long numNodes = nodeFeatures.shape().dim(0);
        if (batchIndex.shape().dim(0) != numNodes) {
            throw new IllegalArgumentException(
                "batchIndex length (" + batchIndex.shape().dim(0) +
                ") must match number of nodes (" + numNodes + ")");
        }
        validateIndexDType(batchIndex, "batchIndex");
    }

    private static void validateIndexDType(Tensor index, String name) {
        DType d = index.dtype();
        if (d != DType.INT32 && d != DType.INT64 && d != DType.FLOAT32) {
            throw new IllegalArgumentException(name + " must be INT32, INT64 or FLOAT32, got " + d);
        }
    }

    public Tensor nodeFeatures() {
        ensureOpen();
        return nodeFeatures;
    }

    public Tensor edgeIndex() {
        ensureOpen();
        return edgeIndex;
    }

    public Tensor edgeFeatures() {
        ensureOpen();
        return edgeFeatures;
    }

    public Tensor batchIndex() {
        ensureOpen();
        return batchIndex;
    }

    public long numNodes() {
        return nodeFeatures.shape().dim(0);
    }

    public long numEdges() {
        return edgeIndex.shape().dim(1);
    }

    public long featureDim() {
        return nodeFeatures.shape().dim(1);
    }

    public long edgeFeatureDim() {
        return edgeFeatures != null ? edgeFeatures.shape().dim(1) : 0;
    }

    public boolean hasEdgeFeatures() {
        return edgeFeatures != null;
    }

    public boolean isBatched() {
        return batchIndex != null;
    }

    /** The backend every cached structure tensor lives on: the node features' backend. */
    public ComputeBackend backend() {
        return nodeFeatures.backend();
    }

    /**
     * The number of graphs in the batch: one more than the largest batch index, and at least 1 (1 when the graph is
     * not batched). Nodes with a negative batch index belong to no graph.
     *
     * @throws IllegalArgumentException when the largest batch index is {@link Integer#MAX_VALUE}
     */
    public synchronized int numGraphs() {
        return batchTopology().numGraphs;
    }

    // ------------------------------------------------------------------ edge structure

    /** INT32 {@code [numEdges]}: the source node of each edge, as stored (not clamped). */
    public synchronized Tensor sourceIndex() {
        return edgeTopology().sources;
    }

    /** INT32 {@code [numEdges]}: the target node of each edge, as stored (not clamped). */
    public synchronized Tensor targetIndex() {
        return edgeTopology().targets;
    }

    /** INT32 {@code [numNodes + 1]}: target node {@code t} owns slots {@code [csrOffsets[t], csrOffsets[t + 1])}. */
    public synchronized Tensor csrOffsets() {
        return edgeTopology().csrOffsets;
    }

    /** INT32: the edge ids with a target in {@code [0, numNodes)}, by ascending target, then ascending edge id. */
    public synchronized Tensor csrEdges() {
        return edgeTopology().csrEdges;
    }

    /** INT32 {@code [numNodes]}: the number of edges into each node. */
    public synchronized Tensor inDegree() {
        EdgeTopology t = edgeTopology();
        if (t.inDegree == null) {
            int[] degree = new int[t.numNodes];
            for (int v = 0; v < t.numNodes; v++) {
                degree[v] = t.offsets[v + 1] - t.offsets[v];
            }
            t.inDegree = Tensor.fromArray(backend(), degree);
        }
        return t.inDegree;
    }

    /** Whether every edge target lies in {@code [0, numNodes)}. */
    public synchronized boolean targetsInRange() {
        return edgeTopology().targetsInRange;
    }

    /** Whether every edge source and target lies in {@code [0, numNodes)}. */
    public synchronized boolean indicesInRange() {
        EdgeTopology t = edgeTopology();
        return t.sourcesInRange && t.targetsInRange;
    }

    /**
     * FLOAT32 {@code [numEdges]}: GCN's symmetric normalization {@code deg(s)^-1/2 * deg(t)^-1/2} of each edge
     * {@code s -> t}, where {@code deg(v)} counts the edges with {@code v} as source plus those with {@code v} as
     * target. Each factor is {@code (float) (1.0 / Math.sqrt(deg))} in double precision and the product is a float
     * multiply.
     *
     * @throws IndexOutOfBoundsException when an edge names a node outside {@code [0, numNodes)}
     */
    public synchronized Tensor symmetricNormalization() {
        EdgeTopology t = edgeTopology();
        if (!(t.sourcesInRange && t.targetsInRange)) {
            throw new IndexOutOfBoundsException("GCN normalization needs every edge index in [0, " + t.numNodes
                + "), the edge index of this graph has one outside");
        }
        if (t.symmetricNorm == null) {
            int[] degree = new int[t.numNodes];
            for (int e = 0; e < t.src.length; e++) {
                degree[t.src[e]]++;
                degree[t.dst[e]]++;
            }
            float[] norm = new float[t.src.length];
            for (int e = 0; e < norm.length; e++) {
                norm[e] = inverseSqrt(degree[t.src[e]]) * inverseSqrt(degree[t.dst[e]]);
            }
            t.symmetricNorm = Tensor.fromArray(backend(), norm);
        }
        return t.symmetricNorm;
    }

    private static float inverseSqrt(int degree) {
        return degree > 0 ? (float) (1.0 / Math.sqrt(degree)) : 0.0f;
    }

    // ------------------------------------------------------------------ batch structure

    /**
     * INT32 {@code [numGraphs() + 1]}: graph {@code g} owns slots {@code [graphOffsets[g], graphOffsets[g + 1])} of
     * {@link #graphNodes()}. An unbatched graph is one graph of every node.
     */
    public synchronized Tensor graphOffsets() {
        return batchTopology().offsets;
    }

    /** INT32: the node ids with a batch index in {@code [0, numGraphs())}, by ascending graph, then ascending id. */
    public synchronized Tensor graphNodes() {
        return batchTopology().nodes;
    }

    // ------------------------------------------------------------------ validation for the graph layers

    /**
     * Checks that {@code x} can stand for this graph's node features in a graph layer: rank 2 with one row per node,
     * on the backend of the cached structure.
     *
     * @throws IllegalArgumentException naming {@code op} otherwise
     */
    void requireNodeFeatures(Tensor x, String op) {
        Objects.requireNonNull(x, "x cannot be null");
        if (x.shape().rank() != 2) {
            throw new IllegalArgumentException(op + " needs node features [numNodes, features], got shape "
                + x.shape());
        }
        if (x.shape().dim(0) != numNodes()) {
            throw new IllegalArgumentException(op + " needs one feature row per node: the graph has " + numNodes()
                + " nodes, the features have " + x.shape().dim(0) + " rows");
        }
        if (x.backend() != backend()) {
            throw new IllegalArgumentException(op + " needs the node features on the graph's backend ("
                + backend().backendType() + "), got " + x.backend().backendType());
        }
    }

    // ------------------------------------------------------------------ building

    private EdgeTopology edgeTopology() {
        ensureOpen();
        if (edges == null) {
            edges = EdgeTopology.build(backend(), readIndices(edgeIndex), (int) numEdges(), (int) numNodes());
        }
        return edges;
    }

    private BatchTopology batchTopology() {
        ensureOpen();
        if (batches == null) {
            batches = BatchTopology.build(backend(), batchIndex != null ? readIndices(batchIndex) : null,
                (int) numNodes());
        }
        return batches;
    }

    /** The index values in row-major order: Java {@code (int)} of each INT64 or FLOAT32 element. */
    private static int[] readIndices(Tensor index) {
        if (index.dtype() != DType.FLOAT32) {
            return index.toIntArray();
        }
        float[] values = index.toFloatArray();
        int[] result = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = (int) values[i];
        }
        return result;
    }

    /** Host copies of the edge index and the device CSR built from them. */
    private static final class EdgeTopology implements AutoCloseable {
        final int numNodes;
        final int[] src;
        final int[] dst;
        final int[] offsets;
        final boolean sourcesInRange;
        final boolean targetsInRange;
        Tensor sources;
        Tensor targets;
        Tensor csrOffsets;
        Tensor csrEdges;
        Tensor inDegree;
        Tensor symmetricNorm;

        private EdgeTopology(int numNodes, int[] src, int[] dst, int[] offsets, boolean sourcesInRange,
                             boolean targetsInRange) {
            this.numNodes = numNodes;
            this.src = src;
            this.dst = dst;
            this.offsets = offsets;
            this.sourcesInRange = sourcesInRange;
            this.targetsInRange = targetsInRange;
        }

        static EdgeTopology build(ComputeBackend backend, int[] index, int numEdges, int numNodes) {
            int[] src = Arrays.copyOfRange(index, 0, numEdges);
            int[] dst = Arrays.copyOfRange(index, numEdges, 2 * numEdges);
            int[] offsets = new int[numNodes + 1];
            boolean sourcesOk = true;
            boolean targetsOk = true;
            for (int e = 0; e < numEdges; e++) {
                sourcesOk &= inRange(src[e], numNodes);
                if (inRange(dst[e], numNodes)) {
                    offsets[dst[e] + 1]++;
                } else {
                    targetsOk = false;
                }
            }
            for (int v = 0; v < numNodes; v++) {
                offsets[v + 1] += offsets[v];
            }
            int[] cursor = Arrays.copyOf(offsets, numNodes);
            int[] listed = new int[offsets[numNodes]];
            for (int e = 0; e < numEdges; e++) {
                if (inRange(dst[e], numNodes)) {
                    listed[cursor[dst[e]]++] = e;
                }
            }
            EdgeTopology t = new EdgeTopology(numNodes, src, dst, offsets, sourcesOk, targetsOk);
            try {
                t.sources = Tensor.fromArray(backend, src);
                t.targets = Tensor.fromArray(backend, dst);
                t.csrOffsets = Tensor.fromArray(backend, offsets);
                t.csrEdges = Tensor.fromArray(backend, listed);
                return t;
            } catch (RuntimeException ex) {
                t.close();
                throw ex;
            }
        }

        @Override
        public void close() {
            closeAll(sources, targets, csrOffsets, csrEdges, inDegree, symmetricNorm);
        }
    }

    /** The nodes grouped by graph of the batch. */
    private static final class BatchTopology implements AutoCloseable {
        final int numGraphs;
        Tensor offsets;
        Tensor nodes;

        private BatchTopology(int numGraphs) {
            this.numGraphs = numGraphs;
        }

        /** {@code batch} null: one graph of every node. */
        static BatchTopology build(ComputeBackend backend, int[] batch, int numNodes) {
            int numGraphs = 1;
            if (batch != null) {
                int maxIdx = 0;
                for (int g : batch) {
                    maxIdx = Math.max(maxIdx, g);
                }
                if (maxIdx == Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("batch index " + maxIdx + " leaves no room for the graph count");
                }
                numGraphs = maxIdx + 1;
            }
            int[] offsets = new int[numGraphs + 1];
            for (int n = 0; n < numNodes; n++) {
                int g = batch == null ? 0 : batch[n];
                if (inRange(g, numGraphs)) {
                    offsets[g + 1]++;
                }
            }
            for (int g = 0; g < numGraphs; g++) {
                offsets[g + 1] += offsets[g];
            }
            int[] cursor = Arrays.copyOf(offsets, numGraphs);
            int[] listed = new int[offsets[numGraphs]];
            for (int n = 0; n < numNodes; n++) {
                int g = batch == null ? 0 : batch[n];
                if (inRange(g, numGraphs)) {
                    listed[cursor[g]++] = n;
                }
            }
            BatchTopology t = new BatchTopology(numGraphs);
            try {
                t.offsets = Tensor.fromArray(backend, offsets);
                t.nodes = Tensor.fromArray(backend, listed);
                return t;
            } catch (RuntimeException ex) {
                t.close();
                throw ex;
            }
        }

        @Override
        public void close() {
            closeAll(offsets, nodes);
        }
    }

    private static boolean inRange(int index, int size) {
        return index >= 0 && index < size;
    }

    private static void closeAll(Tensor... tensors) {
        for (Tensor t : tensors) {
            if (t != null) {
                t.close();
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("GraphData has been closed");
        }
    }

    /** Closes the graph's tensors and every cached structure tensor. */
    @Override
    public synchronized void close() {
        if (!closed) {
            closed = true;
            if (edges != null) {
                edges.close();
                edges = null;
            }
            if (batches != null) {
                batches.close();
                batches = null;
            }
            if (nodeFeatures != null) nodeFeatures.close();
            if (edgeIndex != null) edgeIndex.close();
            if (edgeFeatures != null) edgeFeatures.close();
            if (batchIndex != null) batchIndex.close();
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "GraphData{closed=true}";
        }
        return "GraphData{" +
            "numNodes=" + numNodes() +
            ", numEdges=" + numEdges() +
            ", featureDim=" + featureDim() +
            ", edgeFeatureDim=" + edgeFeatureDim() +
            ", isBatched=" + isBatched() +
            '}';
    }
}
